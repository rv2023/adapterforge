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

import json
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
MODEL_KINDS = frozenset({"sklearn", "lora_adapter", "distilbert"})

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


def register_model_with_dossier(
    test_df: pd.DataFrame, test_f1: float, log_and_register, model_kind: str
) -> str:
    """Register a NEW version under MODEL_NAME and stamp its dossier. Model-agnostic.

    test_f1:          the score, precomputed by the caller (the gate trusts this tag).
    log_and_register: zero-arg callable that logs the artifact AND registers a new
                      version under MODEL_NAME. Sklearn and LLM pass different ones.
    model_kind:       'sklearn' | 'lora_adapter' | 'distilbert' — tells the serving
                      plane HOW to load + run this version (serving's dispatch reads it).
    """
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind must be one of {sorted(MODEL_KINDS)}; got {model_kind!r}")

    mlflow.set_tracking_uri(f"sqlite:///{_DB}")
    mlflow.set_experiment(EXPERIMENT)

    eval_hash = compute_eval_hash(test_df)
    commit = get_git_commit()

    with mlflow.start_run():
        log_and_register()
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
        client.set_model_version_tag(MODEL_NAME, version, "model_kind", model_kind)
        mlflow.log_metric("test_f1", test_f1)

    return str(version)


def register_sklearn(model, test_df: pd.DataFrame) -> str:
    """The old sklearn path, expressed through the agnostic core."""
    return register_model_with_dossier(
        test_df,
        evaluate(model, test_df["text"], test_df["label"]),
        lambda: mlflow.sklearn.log_model(
            model, name="model", registered_model_name=MODEL_NAME
        ),
        model_kind="sklearn",
    )


def register_adapter(adapter_dir, test_df: pd.DataFrame) -> str:
    """Register a trained LoRA adapter as a new fpb-sentiment version + dossier."""
    adapter_dir = Path(adapter_dir)
    metrics_path = adapter_dir / "eval_metrics.json"
    with metrics_path.open() as f:
        metrics = json.load(f)

    test_f1 = metrics["test_f1"]
    n_test = metrics["n_test"]
    expected_n_test = len(test_df)
    if n_test != expected_n_test:
        raise ValueError(
            f"Adapter eval metrics n_test mismatch in {metrics_path}: "
            f"metrics n_test={n_test}, test_df rows={expected_n_test}"
        )

    def log_and_register():
        mlflow.log_artifacts(str(adapter_dir), "adapter")
        run_id = mlflow.active_run().info.run_id
        source = mlflow.get_artifact_uri("adapter")
        mlflow.MlflowClient().create_model_version(
            name=MODEL_NAME, source=source, run_id=run_id
        )

    return register_model_with_dossier(
        test_df, test_f1, log_and_register, model_kind="lora_adapter"
    )


def register_baseline() -> None:
    """Train the locked baseline and register it with its dossier (M3.1 bootstrap)."""
    train_df, _val_df, test_df = split_data(load_data())
    model = build_model(BEST_C)
    model.fit(train_df["text"], train_df["label"])
    version = register_sklearn(model, test_df)
    print(f"Registered {MODEL_NAME} version {version} with dossier.")


if __name__ == "__main__":
    register_baseline()
