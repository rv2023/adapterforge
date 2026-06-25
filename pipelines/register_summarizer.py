"""M8 — register the trained summarization adapter as `fpb-summarizer` + its dossier.

Mirrors register_student.py (a NEW registered-model name) but it's a LoRA adapter (like
register_adapter) scored on the ECTSum test set. The eval_set_hash it prints is what you
put in the control-plane gate (FPB_SUMMARIZER_EXPECTED_HASH) so promote() can govern it.

    python -m pipelines.register_summarizer        # after training + eval on the GPU pod
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# (when implementing, re-add: import json, mlflow; import register_baseline as registry;
#  from summarize_format import load_ectsum — see the TODOs below.)

SUMMARIZER_DIR = "models/fpb-summarizer"
MODEL_NAME = "fpb-summarizer"          # NEW registered model (heterogeneous registry)


def ectsum_test_df() -> pd.DataFrame:
    """The frozen ECTSum TEST set as a DataFrame — the summarizer's exam (for the hash)."""
    # TODO: _, _, test = load_ectsum(); return pd.DataFrame(test, columns=["transcript", "summary"])
    raise NotImplementedError


def register_summarizer():
    """Register models/fpb-summarizer as a new `fpb-summarizer` version with its dossier.

    Blend of register_adapter (LoRA artifact logging, model_kind="lora_adapter") and
    register_student (NEW model name -> create_registered_model first + MODEL_NAME swap).
    """
    # TODO: read SUMMARIZER_DIR/eval_metrics.json -> test_f1 (=ROUGE-L), n_test
    # TODO: test_df = ectsum_test_df(); cross-check n_test == len(test_df) (495)
    # TODO: define log_and_register():
    #         mlflow.log_artifacts(SUMMARIZER_DIR, "adapter")
    #         source = mlflow.get_artifact_uri("adapter"); run_id = mlflow.active_run().info.run_id
    #         client = mlflow.MlflowClient()
    #         try: client.create_registered_model(MODEL_NAME)        # NEW name must exist first
    #         except mlflow.exceptions.MlflowException: pass
    #         client.create_model_version(name=MODEL_NAME, source=source, run_id=run_id)
    # TODO: temporarily set registry.MODEL_NAME = MODEL_NAME (restore in finally), then
    #         version = registry.register_model_with_dossier(test_df, test_f1, log_and_register,
    #                                                        model_kind="lora_adapter")
    # TODO: print the version AND the eval_set_hash (registry.compute_eval_hash(test_df)) ->
    #         set that as FPB_SUMMARIZER_EXPECTED_HASH in the control plane before promoting.
    raise NotImplementedError


if __name__ == "__main__":
    register_summarizer()
