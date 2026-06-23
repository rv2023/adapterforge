"""M6 Piece 0 — one-off backfill: stamp `model_kind` on versions registered before
the tag existed.

New registrations get `model_kind` automatically (register_baseline.py). But the
versions already in the registry (the LLM `fpb-sentiment` v14, the `fpb-student`
version) predate the tag, so serving's dispatch would find nothing on them. This
script tags them after the fact.

Idempotent by design: re-running just re-sets the same value (set_model_version_tag
overwrites). Run once locally:

    python -m pipelines.backfill_model_kind
"""

from pathlib import Path

import mlflow

# absolute path so it resolves regardless of cwd (same convention as register_baseline)
_DB = Path(__file__).resolve().parent.parent / "mlflow.db"


def backfill() -> None:
    """Stamp model_kind on the pre-existing versions.

    Targets:
      - fpb-sentiment v14  -> "lora_adapter"   (the promoted LLM)
      - fpb-student <ver>  -> "distilbert"     (find its version, don't hardcode if unsure)
    """
    mlflow.set_tracking_uri(f"sqlite:///{_DB}")
    client = mlflow.MlflowClient()

    targets = [
        ("fpb-sentiment", "14", "lora_adapter"),
        (
            "fpb-student",
            max(
                (mv.version for mv in client.search_model_versions("name='fpb-student'")),
                key=int,
            ),
            "distilbert",
        ),
    ]

    for name, version, kind in targets:
        client.set_model_version_tag(name, version, "model_kind", kind)
        print(f"Tagged {name} version {version} with model_kind={kind}")


if __name__ == "__main__":
    backfill()
