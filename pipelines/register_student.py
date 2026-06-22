"""M5 Piece 5, Step 5 — register the DistilBERT student in the MLflow registry.

Adds the third, heterogeneous model (66M encoder) alongside the sklearn baseline and the
1.5B LLM adapter — the raw material for M8's cost-aware router. Reuse the centralized
register_model_with_dossier / register_adapter pattern from register_baseline.py.

    python -m pipelines.register_student
"""

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
    # TODO
    ...


if __name__ == "__main__":
    register_student()
