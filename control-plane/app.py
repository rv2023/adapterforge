"""M3 control plane — governed model promotion.

A thin FastAPI service that OWNS promotion policy. It does not store models
(MLflow does); it reads each model's dossier (its registry tags) and enforces the
four gates before letting a version become Production. A rejected promotion
changes NOTHING and returns the reason — fail-closed, like an admission controller.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- policy constants the gates enforce ---
MODEL_NAME = "fpb-sentiment"
PRODUCTION_ALIAS = "production"
MARGIN = 0.01          # a candidate must beat the incumbent's F1 by at least this
MIN_F1_FLOOR = 0.40    # first model (empty throne) must at least clear this
EXPECTED_SCHEMA = "v1"
# the ONE true frozen exam — gate 2 pins every model to this, incumbent or not
EXPECTED_HASH = "145f42323f06bb454c002cb740019e3f8b87d3e755c4e4fbdc37e1604af9f0b2"

GATE_CONFIG = {
    MODEL_NAME: {
        "expected_hash": EXPECTED_HASH,
        "expected_schema": EXPECTED_SCHEMA,
        "margin": MARGIN,
        "floor": MIN_F1_FLOOR,
        "metric_label": "F1",
    },
    "fpb-summarizer": {
        "expected_hash": os.getenv("FPB_SUMMARIZER_EXPECTED_HASH"),
        "expected_schema": EXPECTED_SCHEMA,
        "margin": MARGIN,
        "floor": 0.0,
        "metric_label": "ROUGE-L",
    },
}

# point at the repo-root mlflow.db with an ABSOLUTE path, so it works no matter
# which directory uvicorn is launched from.
_DB = Path(__file__).resolve().parent.parent / "mlflow.db"
AUDIT_PATH = Path(__file__).resolve().parent / "audit.jsonl"
mlflow.set_tracking_uri(f"sqlite:///{_DB}")

app = FastAPI(title="AdapterForge Control Plane")
client = mlflow.MlflowClient()


class PromoteRequest(BaseModel):
    """Body for a promote call: which version, and who approved it."""

    version: str
    approved_by: str


def get_dossier(name: str, version: str) -> dict:
    """Read a model version's dossier (its tags) from the registry.

    NOTE: tag VALUES are always strings — cast test_f1 to float before comparing.
    """
    mv = client.get_model_version(name, version)
    return mv.tags


def get_production_version(name: str) -> str | None:
    """Return the version currently aliased 'production', or None if throne empty."""
    try:
        return client.get_model_version_by_alias(name, PRODUCTION_ALIAS).version
    except MlflowException:
        return None


def write_audit(record: dict) -> None:
    record["ts"] = datetime.now(timezone.utc).isoformat()
    with AUDIT_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def reject(base: dict, status: int, reason: str) -> None:
    write_audit({**base, "decision": "rejected", "status": status, "reason": reason})
    raise HTTPException(status_code=status, detail=reason)


@app.post("/models/{name}/promote")
def promote(name: str, req: PromoteRequest) -> dict:
    """Run the 4 gates. All pass -> alias candidate 'production'. Any fail -> 4xx + reason.

    Reject with HTTPException(status_code=409, detail="<why>") so the caller sees the
    exact policy that blocked it. A rejection must change nothing.
    """
    base = {"model": name, "version": req.version, "approved_by": req.approved_by}
    cfg = GATE_CONFIG.get(name)
    if cfg is None:
        reject(base, 404, "no gate config for model")

    if not cfg["expected_hash"]:
        reject(base, 409, f"expected eval-set hash not configured for {name}")

    candidate = get_dossier(name, req.version)
    metric_label = cfg["metric_label"]

    if not req.approved_by:
        reject(base, 400, "approved_by required")

    if candidate["eval_set_hash"] != cfg["expected_hash"]:
        reject(base, 409, "eval-set hash mismatch — not the canonical frozen test set")

    if candidate["schema_version"] != cfg["expected_schema"]:
        reject(
            base,
            409,
            f"schema {candidate['schema_version']} incompatible with {cfg['expected_schema']}",
        )

    cand_f1 = float(candidate["test_f1"])
    prod_version = get_production_version(name)
    if prod_version is None:
        if cand_f1 < cfg["floor"]:
            reject(
                base,
                409,
                f"candidate {metric_label} {cand_f1:.4f} below minimum floor {cfg['floor']:.4f}",
            )
    else:
        prod_f1 = float(get_dossier(name, prod_version)["test_f1"])
        required_f1 = prod_f1 + cfg["margin"]
        if cand_f1 < required_f1:
            reject(
                base,
                409,
                (
                    f"candidate {metric_label} {cand_f1:.4f} must be at least {required_f1:.4f} "
                    f"(production v{prod_version} {metric_label} {prod_f1:.4f} "
                    f"+ margin {cfg['margin']:.4f})"
                ),
            )

    client.set_registered_model_alias(name, PRODUCTION_ALIAS, req.version)
    write_audit({**base, "decision": "promoted", "previous_production": prod_version})
    return {"promoted": req.version, "approved_by": req.approved_by, "test_f1": cand_f1}


@app.get("/models/{name}/production")
def get_production(name: str) -> dict:
    """What version is currently Production, with its dossier. 404 if none."""
    version = get_production_version(name)
    if version is None:
        raise HTTPException(status_code=404, detail="no production model")
    return {"version": version, **get_dossier(name, version)}


@app.get("/models/{name}/lineage/{version}")
def get_lineage(name: str, version: str) -> dict:
    """Provenance for a specific version: what code/schema/exam produced it."""
    d = get_dossier(name, version)
    return {
        "version": version,
        "code_commit": d.get("code_commit"),
        "schema_version": d.get("schema_version"),
        "eval_set_hash": d.get("eval_set_hash"),
    }
