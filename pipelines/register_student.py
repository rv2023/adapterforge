"""M5 Piece 5, Step 5 — register the DistilBERT student in the MLflow registry.

Adds the third, heterogeneous model (66M encoder) alongside the sklearn baseline and the
1.5B LLM adapter — the raw material for M8's cost-aware router. Reuse the centralized
register_model_with_dossier / register_adapter pattern from register_baseline.py.

    python -m pipelines.register_student
"""

import json
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent))

import register_baseline as registry
from baseline import load_data, split_data

STUDENT_DIR = "models/distilbert-student"
MODEL_NAME = "fpb-student"          # new registered model name (heterogeneous registry)


def register_student():
    """Register the student dir as a new model version, with its dossier.

    Mirror register_adapter (pipelines/register_baseline.py):
      - read models/distilbert-student/eval_metrics.json (test_f1, n_test)
      - n_test cross-check against the frozen test set
      - recompute eval_set_hash fresh on this box (trust-boundary: never trust a remote hash)
      - log the student dir as artifacts; MlflowClient.create_model_version(
            name=MODEL_NAME, source=get_artifact_uri(...), run_id=...)
      - stamp dossier tags (test_f1, eval_set_hash, schema, commit)
    """
    student_dir = Path(STUDENT_DIR)
    metrics_path = student_dir / "eval_metrics.json"
    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)

    test_f1 = metrics["test_f1"]
    n_test = metrics["n_test"]
    _, _, test_df = split_data(load_data())
    expected_n_test = len(test_df)
    if n_test != expected_n_test:
        raise ValueError(
            f"Student eval metrics n_test mismatch in {metrics_path}: "
            f"metrics n_test={n_test}, test_df rows={expected_n_test}"
        )

    def log_and_register():
        mlflow.log_artifacts(str(student_dir), "student")
        run_id = mlflow.active_run().info.run_id
        source = mlflow.get_artifact_uri("student")
        client = mlflow.MlflowClient()
        # fpb-student is a NEW registered model name -> create it first (create_model_version
        # requires it to exist; unlike fpb-sentiment which the baseline already created).
        try:
            client.create_registered_model(MODEL_NAME)
        except mlflow.exceptions.MlflowException:
            pass  # already exists
        client.create_model_version(name=MODEL_NAME, source=source, run_id=run_id)

    previous_model_name = registry.MODEL_NAME
    try:
        registry.MODEL_NAME = MODEL_NAME
        version = registry.register_model_with_dossier(test_df, test_f1, log_and_register)
    finally:
        registry.MODEL_NAME = previous_model_name

    print(f"Registered {MODEL_NAME} version {version} with dossier.")
    return version


if __name__ == "__main__":
    register_student()
