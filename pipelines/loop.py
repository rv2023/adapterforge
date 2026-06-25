"""M4.5 — the automated loop: drift -> retrain -> gate -> auto-promote. Hands-off.

Run with the control plane up on :8000 and the production model set.

    drift_detected? --no--> nothing to do
                     --yes--> retrain + register a candidate
                              -> ask the control-plane gate to promote it
                              -> the M3 gate auto-promotes ONLY if better

No human in the loop. The gate is the safety arbiter.
"""

from pathlib import Path

import mlflow
import pandas as pd
import requests

from baseline import build_model, load_data, split_data
from drift import PSI_THRESHOLD, REGIME_CSV, oov_rate, psi, reference_analyzer_vocab
from register_baseline import BEST_C, MODEL_NAME, register_sklearn

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


def retrain_and_register() -> str:
    """Retrain the baseline and register a NEW candidate version with its dossier."""
    train_df, _, test_df = split_data(load_data())
    model = build_model(BEST_C)
    model.fit(train_df["text"], train_df["label"])
    return register_sklearn(model, test_df)


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
