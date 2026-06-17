"""M3 — register the baseline as a governed model WITH its dossier.

The control plane's promotion gates can only work if every model carries the
metadata they need to read later. So this script registers the M2 baseline into
the MLflow Model Registry and attaches its "dossier" as tags:

    test_f1        the bar (gate #1 reads this)
    eval_set_hash  fingerprint of the frozen test set (gate #2 reads this)
    schema_version the data contract it was trained against (gate #3)
    code_commit    which code built it (lineage)

This does NOT assign Production — that's the control plane's promote job. It only
creates version 1 with a full dossier, so promotion has something to govern.
"""

import hashlib
import subprocess

import mlflow
import pandas as pd

from baseline import RANDOM_SEED, build_model, evaluate, load_data, split_data

MODEL_NAME = "fpb-sentiment"
SCHEMA_VERSION = "v1"   # the Pandera schema the data was validated against
BEST_C = 10             # the winner we locked in M2


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


def register_baseline() -> None:
    """Train the locked baseline, score it on TEST (the bar), register with dossier."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("m2-baseline")

    df = load_data()
    train_df, _val_df, test_df = split_data(df)

    model = build_model(BEST_C)
    model.fit(train_df["text"], train_df["label"])
    test_f1 = evaluate(model, test_df["text"], test_df["label"])
    eval_hash = compute_eval_hash(test_df)
    commit = get_git_commit()

    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
        )
        version = info.registered_model_version

        client = mlflow.MlflowClient()
        client.set_model_version_tag(MODEL_NAME, version, "test_f1", str(test_f1))
        client.set_model_version_tag(MODEL_NAME, version, "eval_set_hash", eval_hash)
        client.set_model_version_tag(MODEL_NAME, version, "schema_version", SCHEMA_VERSION)
        client.set_model_version_tag(MODEL_NAME, version, "code_commit", commit)
        mlflow.log_metric("test_f1", test_f1)

        print(
            f"Registered {MODEL_NAME} version {version} with dossier: "
            f"test_f1={test_f1}, eval_set_hash={eval_hash}, "
            f"schema_version={SCHEMA_VERSION}, code_commit={commit}"
        )


if __name__ == "__main__":
    register_baseline()
