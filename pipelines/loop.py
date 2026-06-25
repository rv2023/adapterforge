"""M4.5 — the automated loop: drift -> retrain -> gate -> auto-promote. Hands-off.

Run with the control plane up on :8000 and the production model set.

    drift_detected? --no--> nothing to do
                     --yes--> retrain + register a candidate
                              -> ask the control-plane gate to promote it
                              -> the M3 gate auto-promotes ONLY if better

No human in the loop. The gate is the safety arbiter.
"""

import os
from pathlib import Path

import mlflow
import pandas as pd
import requests

from baseline import build_model, load_data, split_data
from drift import PSI_THRESHOLD, REGIME_CSV, oov_rate, psi, reference_analyzer_vocab
from register_baseline import BEST_C, MODEL_NAME, register_adapter, register_sklearn

CONTROL_PLANE = "http://127.0.0.1:8000"
_DB = Path(__file__).resolve().parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{_DB}")


def drift_detected() -> bool:
    """OOV/PSI drift check on the regime batch vs the in-distribution reference."""
    analyzer, vocab = reference_analyzer_vocab()
    _, _, test_df = split_data(load_data())
    regime_df = pd.read_csv(REGIME_CSV)
    reference_oov = [oov_rate(t, analyzer, vocab) for t in test_df["text"]]
    current_oov = [oov_rate(t, analyzer, vocab) for t in regime_df["text"]]
    return psi(reference_oov, current_oov) > PSI_THRESHOLD


def production_model_kind() -> str:
    """Read production's model_kind from the control plane (same source serving uses)."""
    resp = requests.get(f"{CONTROL_PLANE}/models/{MODEL_NAME}/production")
    resp.raise_for_status()
    return resp.json()["model_kind"]


def retrain_sklearn() -> str:
    """Legacy sklearn retrain (kept; can't beat the LLM at the gate — never wins now)."""
    train_df, _, test_df = split_data(load_data())
    model = build_model(BEST_C)
    model.fit(train_df["text"], train_df["label"])
    return register_sklearn(model, test_df)


def retrain_lora() -> str:
    """Retrain the LLM adapter (QLoRA) and register it as a candidate. GPU-BOUND.

    Chains the M5 pipeline; runs on a GPU runner (prod: the retrain.yml workflow). Wired
    here so the loop is model-aware; the actual run is a paid GPU session.
    """
    from pipelines.instruction_format import main as ensure_instruction_data
    from pipelines.eval_adapter import main as eval_adapter

    ensure_instruction_data()
    os.environ["AF_MODE"] = "real"
    from pipelines.finetune import main as finetune_adapter

    finetune_adapter()
    eval_adapter()
    _, _, test_df = split_data(load_data())
    return register_adapter("models/fpb-lora", test_df)


# model-aware: retrain produces a candidate of the SAME kind as production.
RETRAIN_BY_KIND = {
    "sklearn": retrain_sklearn,
    "lora_adapter": retrain_lora,
    # "distilbert": retrain_student,  # re-distill from the updated teacher (later)
}


def retrain_and_register() -> str:
    """Dispatch the retrain matching production's kind -> a candidate version."""
    kind = production_model_kind()
    fn = RETRAIN_BY_KIND.get(kind)
    if fn is None:
        raise RuntimeError(f"no retrain path for model_kind={kind!r}")
    return fn()


def request_promotion(version: str) -> requests.Response:
    """Ask the control-plane gate to promote the candidate (the automation 'approves')."""
    return requests.post(
        f"{CONTROL_PLANE}/models/{MODEL_NAME}/promote",
        json={"version": version, "approved_by": "auto-retrain-bot"},
    )


def run_loop() -> None:
    """The whole chain, hands-off."""
    if not drift_detected():
        print("no drift — model healthy, nothing to do")
        return
    print("DRIFT detected -> retraining")
    version = retrain_and_register()
    print(f"registered candidate v{version} -> asking the gate to promote")
    resp = request_promotion(version)
    print(f"gate decision: HTTP {resp.status_code} -> {resp.json()}")


if __name__ == "__main__":
    run_loop()
