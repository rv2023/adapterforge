"""M4.4 — Dagster orchestration: ingest -> train -> register, as a dependency graph.

Three assets, each wrapping code from earlier milestones. Dagster reads the
dependencies (via the function parameter names) and runs them in order:

    financial_phrasebank  ->  baseline_model  ->  registered_model

Run:  dagster dev -f pipelines/dag.py   (UI at http://127.0.0.1:3000)
"""

import sys
from pathlib import Path

# Dagster loads this file with cwd at the repo root, so pipelines/ isn't on the
# import path. Add it here so `from baseline import ...` works, while cwd stays at
# the repo root (needed so data/ paths resolve correctly).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
from dagster import Definitions, asset

from adapter_sdk.adapters.hf import HFDatasetAdapter
from baseline import build_model, evaluate, load_data, split_data
from register_baseline import (
    BEST_C,
    MODEL_NAME,
    SCHEMA_VERSION,
    compute_eval_hash,
    get_git_commit,
)

_DB = Path(__file__).resolve().parent.parent / "mlflow.db"


@asset
def financial_phrasebank() -> str:
    """Land the validated (silver) PhraseBank via the SDK. Returns the parquet path."""
    HFDatasetAdapter().run()
    return "data/validated/financial_phrasebank.parquet"


@asset
def baseline_model(financial_phrasebank: str):
    """Train the locked baseline (C=BEST_C) on the silver data. Returns the fitted model.

    The `financial_phrasebank` param makes Dagster run that asset FIRST (dependency).
    """
    train_df, _, _ = split_data(load_data())
    model = build_model(BEST_C)
    model.fit(train_df["text"], train_df["label"])
    return model


@asset
def registered_model(baseline_model) -> str:
    """Score on TEST, register the model with its dossier in MLflow. Returns the version."""
    mlflow.set_tracking_uri(f"sqlite:///{_DB}")
    mlflow.set_experiment("m2-baseline")
    _, _, test_df = split_data(load_data())
    test_f1 = evaluate(baseline_model, test_df["text"], test_df["label"])
    eval_hash = compute_eval_hash(test_df)
    commit = get_git_commit()
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            baseline_model,
            name="model",
            registered_model_name=MODEL_NAME,
        )
        version = info.registered_model_version
        client = mlflow.MlflowClient()
        client.set_model_version_tag(MODEL_NAME, version, "test_f1", str(test_f1))
        client.set_model_version_tag(MODEL_NAME, version, "eval_set_hash", eval_hash)
        client.set_model_version_tag(MODEL_NAME, version, "schema_version", SCHEMA_VERSION)
        client.set_model_version_tag(MODEL_NAME, version, "code_commit", commit)
        return str(version)


defs = Definitions(assets=[financial_phrasebank, baseline_model, registered_model])
