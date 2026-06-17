"""M3.5 serving — serves the PRODUCTION model.

The data plane. It does NOT decide what's live — at startup it ASKS the control
plane "what version is Production?", loads exactly that from the MLflow registry,
and answers predictions. Promote a new version through the gate + restart this,
and it serves the new one. The brain decides; this obeys.
"""

from pathlib import Path

import mlflow
import requests
from fastapi import FastAPI
from pydantic import BaseModel

CONTROL_PLANE = "http://127.0.0.1:8000"   # where the control plane lives
MODEL_NAME = "fpb-sentiment"

# same registry the control plane uses (absolute path -> cwd-independent)
_DB = Path(__file__).resolve().parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{_DB}")

app = FastAPI(title="AdapterForge Serving")


class PredictRequest(BaseModel):
    text: str


def load_production_model():
    """Ask the control plane what's Production, load that version, return (model, version)."""
    resp = requests.get(f"{CONTROL_PLANE}/models/{MODEL_NAME}/production")
    resp.raise_for_status()
    version = resp.json()["version"]
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    return model, version


# load ONCE at startup (module import). If the control plane is down, this fails fast.
model, model_version = load_production_model()


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    """Classify one statement with the currently-loaded production model."""
    pred = model.predict([req.text])[0]
    confidence = float(model.predict_proba([req.text]).max())
    return {"label": pred, "confidence": confidence, "model_version": model_version}
