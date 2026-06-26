"""M8 — register the trained summarization adapter as `fpb-summarizer` + its dossier.

Mirrors register_student.py (a NEW registered-model name) but it's a LoRA adapter (like
register_adapter) scored on the ECTSum test set. The eval_set_hash it prints is what you
put in the control-plane gate (FPB_SUMMARIZER_EXPECTED_HASH) so promote() can govern it.

    python -m pipelines.register_summarizer        # after training + eval on the GPU pod
"""

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import register_baseline as registry
from summarize_format import load_ectsum

SUMMARIZER_DIR = "models/fpb-summarizer"
MODEL_NAME = "fpb-summarizer"          # NEW registered model (heterogeneous registry)


def ectsum_test_df() -> pd.DataFrame:
    """The frozen ECTSum TEST set as a DataFrame — the summarizer's exam (for the hash)."""
    _, _, test = load_ectsum()
    return pd.DataFrame(test, columns=["transcript", "summary"])


def register_summarizer():
    """Register models/fpb-summarizer as a new `fpb-summarizer` version with its dossier.

    Blend of register_adapter (LoRA artifact logging, model_kind="lora_adapter") and
    register_student (NEW model name -> create_registered_model first + MODEL_NAME swap).
    """
    summarizer_dir = Path(SUMMARIZER_DIR)
    metrics_path = summarizer_dir / "eval_metrics.json"
    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)

    test_f1 = metrics["test_f1"]
    n_test = metrics["n_test"]
    test_df = ectsum_test_df()
    expected_n_test = len(test_df)
    if n_test != expected_n_test:
        raise ValueError(
            f"Summarizer eval metrics n_test mismatch in {metrics_path}: "
            f"metrics n_test={n_test}, test_df rows={expected_n_test}"
        )

    def log_and_register():
        mlflow.log_artifacts(str(summarizer_dir), "adapter")
        source = mlflow.get_artifact_uri("adapter")
        run_id = mlflow.active_run().info.run_id
        client = mlflow.MlflowClient()
        try:
            client.create_registered_model(MODEL_NAME)
        except mlflow.exceptions.MlflowException as exc:
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        client.create_model_version(name=MODEL_NAME, source=source, run_id=run_id)

    previous_model_name = registry.MODEL_NAME
    try:
        registry.MODEL_NAME = MODEL_NAME
        version = registry.register_model_with_dossier(
            test_df, test_f1, log_and_register, model_kind="lora_adapter"
        )
    finally:
        registry.MODEL_NAME = previous_model_name
    print(
        f"Registered {MODEL_NAME} version {version} with dossier. "
        f"eval_set_hash={registry.compute_eval_hash(test_df)}"
    )
    return version


if __name__ == "__main__":
    register_summarizer()
