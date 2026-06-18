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

from dagster import Definitions, asset

from adapter_sdk.adapters.hf import HFDatasetAdapter
from baseline import build_model, load_data, split_data
from register_baseline import BEST_C, register_model_with_dossier


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
    """Register the trained model with its dossier in MLflow. Returns the version."""
    _, _, test_df = split_data(load_data())
    return register_model_with_dossier(baseline_model, test_df)


defs = Definitions(assets=[financial_phrasebank, baseline_model, registered_model])
