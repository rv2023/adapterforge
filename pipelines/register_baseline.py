"""M3 — register a model into the MLflow registry WITH its dossier.

The control plane's promotion gates can only work if every model carries the
metadata they need to read later. `register_model_with_dossier` is the single
source of truth for that — it scores a model on the frozen test set and attaches
its "dossier" as tags:

    test_f1        the bar (gate #1 reads this)
    eval_set_hash  fingerprint of the frozen test set (gate #2 reads this)
    schema_version the data contract it was trained against (gate #3)
    code_commit    which code built it (lineage)

It does NOT assign Production — that's the control plane's promote job. The Dagster
DAG (dag.py) and the retrain loop (loop.py) both reuse this so the dossier is
created exactly one way, everywhere.
"""

import hashlib
import subprocess
from pathlib import Path

import mlflow
import pandas as pd

from baseline import build_model, evaluate, load_data, split_data

MODEL_NAME = "fpb-sentiment"
SCHEMA_VERSION = "v1"   # the Pandera schema the data was validated against
BEST_C = 10             # the winner we locked in M2
EXPERIMENT = "m2-baseline"

# absolute path so it resolves no matter which dir (script, dagster, loop) calls in
_DB = Path(__file__).resolve().parent.parent / "mlflow.db"


def compute_eval_hash(test_df: pd.DataFrame) -> str:
    """Return a SHA-256 fingerprint of the frozen test set.

    Same rows -> same hash, always. Any change -> different hash. This is the
    machine-checkable proof that two models were scored on the identical exam.
    """
    canonical = test_df.to_csv(index=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def get_git_commit() -> str:
    """Return the short git commit hash of the current code (provenance)."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def register_model_with_dossier(model, test_df: pd.DataFrame) -> str:
    """Score `model` on the frozen TEST set, register it, attach its dossier.

    The ONE place a candidate gets registered. Returns the new registry version
    (as a string). Used by the standalone script, the Dagster DAG, and the loop.
    """
    mlflow.set_tracking_uri(f"sqlite:///{_DB}")
    mlflow.set_experiment(EXPERIMENT)

    test_f1 = evaluate(model, test_df["text"], test_df["label"])
    eval_hash = compute_eval_hash(test_df)
    commit = get_git_commit()

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME)
        client = mlflow.MlflowClient()
        # query the registry for the version we just created (robust: the
        # log_model return's registered_model_version can be None / lag)
        version = max(
            (mv.version for mv in client.search_model_versions(f"name='{MODEL_NAME}'")),
            key=int,
        )
        client.set_model_version_tag(MODEL_NAME, version, "test_f1", str(test_f1))
        client.set_model_version_tag(MODEL_NAME, version, "eval_set_hash", eval_hash)
        client.set_model_version_tag(MODEL_NAME, version, "schema_version", SCHEMA_VERSION)
        client.set_model_version_tag(MODEL_NAME, version, "code_commit", commit)
        mlflow.log_metric("test_f1", test_f1)

    return str(version)


def register_baseline() -> None:
    """Train the locked baseline and register it with its dossier (M3.1 bootstrap)."""
    train_df, _val_df, test_df = split_data(load_data())
    model = build_model(BEST_C)
    model.fit(train_df["text"], train_df["label"])
    version = register_model_with_dossier(model, test_df)
    print(f"Registered {MODEL_NAME} version {version} with dossier.")


if __name__ == "__main__":
    register_baseline()
