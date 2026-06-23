"""M3.5 serving — serves the PRODUCTION model. (M6: now model-aware.)

The data plane. It does NOT decide what's live — at startup it ASKS the control
plane "what version is Production?", loads exactly that from the MLflow registry,
and answers predictions. Promote a new version through the gate + restart this,
and it serves the new one. The brain decides; this obeys.

M6 change: production is no longer always sklearn. The registry now holds
heterogeneous kinds (lora_adapter / distilbert), so this reads the `model_kind`
tag (returned by the control plane's /production endpoint) and dispatches to the
matching loader + predictor. See docs/m6-serving-concepts.md §7c.
"""

import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel

# repo root on path so we can reuse the eval-script loaders/predictors (DRY)
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

CONTROL_PLANE = "http://127.0.0.1:8000"   # where the control plane lives
MODEL_NAME = "fpb-sentiment"

# same registry the control plane uses (absolute path -> cwd-independent)
_DB = _REPO / "mlflow.db"
PRED_LOG = Path(__file__).resolve().parent / "predictions.jsonl"
mlflow.set_tracking_uri(f"sqlite:///{_DB}")

class PredictRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------
def download_version_artifacts(name: str, version: str) -> Path:
    """Download a registered version's artifacts to a local dir; return that dir.

    The eval-script loaders read from a local directory, but registered versions
    live in the registry. So: get the version's artifact source, download it, hand
    the local path to the loader.
    """
    source = mlflow.MlflowClient().get_model_version(name, version).source
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=source)
    return Path(local_path)


# ---------------------------------------------------------------------------
# Per-kind predictor builders.  Each returns a predict_fn(text) -> (label, confidence)
# confidence may be None where the kind has no clean probability (TODO(decide) below).
# ---------------------------------------------------------------------------
def build_lora_predictor(name: str, version: str):
    """Load the LoRA adapter version; return predict_fn(text) -> (label, confidence).

    Reuse eval_adapter.load_model_and_tokenizer (point it at the downloaded dir) and
    eval_adapter.predict_one. Build the chat messages from the SAME strings training
    used (instruction_format.INSTRUCTION + the system prompt) — a drifted prompt
    silently tanks accuracy.
    """
    from pipelines import eval_adapter, instruction_format

    local_dir = download_version_artifacts(name, version)
    model, tokenizer = eval_adapter.load_model_and_tokenizer(adapter_dir=local_dir)

    def predict_fn(text: str):
        messages = [
            {"role": "system", "content": "You are a financial sentiment classifier."},
            {"role": "user", "content": instruction_format.INSTRUCTION + text},
        ]
        with eval_adapter.torch.inference_mode():
            label = eval_adapter.predict_one(model, tokenizer, messages)
        return label, None

    return predict_fn


def build_distilbert_predictor(name: str, version: str):
    """Load the DistilBERT student version; return predict_fn(text) -> (label, confidence).

    Reuse eval_student.predict (+ eval_student.LABELS). Runs on CPU — this is the
    path you can smoke-test locally.
    """
    from pipelines import eval_student
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    local_dir = download_version_artifacts(name, version)
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForSequenceClassification.from_pretrained(local_dir)
    model.eval()

    def predict_fn(text: str):
        label = eval_student.predict(model, tokenizer, [text])[0]
        return label, None

    return predict_fn


PREDICTOR_BUILDERS = {
    "lora_adapter": build_lora_predictor,
    "distilbert": build_distilbert_predictor,
    # note: "sklearn" intentionally absent — retired, never served (docs §7c)
}


def log_prediction(record: dict) -> None:
    record["ts"] = datetime.now(timezone.utc).isoformat()
    with PRED_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_production_model():
    """Ask the control plane what's Production, read its model_kind, build the predictor.

    Returns (predict_fn, version, model_kind).
    """
    resp = requests.get(f"{CONTROL_PLANE}/models/{MODEL_NAME}/production")
    resp.raise_for_status()
    production = resp.json()

    try:
        version = production["version"]
        model_kind = production["model_kind"]
    except KeyError as exc:
        raise RuntimeError(
            f"production response missing required field {exc.args[0]!r}: {production}"
        ) from exc

    builder = PREDICTOR_BUILDERS.get(model_kind)
    if builder is None:
        raise RuntimeError(
            f"unsupported production model_kind {model_kind!r}; "
            f"expected one of {sorted(PREDICTOR_BUILDERS)}"
        )

    predict_fn = builder(MODEL_NAME, str(version))
    return predict_fn, str(version), model_kind


@asynccontextmanager
async def lifespan(app: FastAPI):
    predict_fn, version, kind = load_production_model()
    app.state.predict_fn = predict_fn
    app.state.model_version = version
    app.state.model_kind = kind
    yield


app = FastAPI(title="AdapterForge Serving", lifespan=lifespan)


@app.post("/predict")
def predict(req: PredictRequest, request: Request) -> dict:
    """Classify one statement with the currently-loaded production model."""
    state = request.app.state
    label, confidence = state.predict_fn(req.text)
    response = {
        "label": label,
        "confidence": confidence,
        "model_version": state.model_version,
        "model_kind": state.model_kind,
    }
    log_prediction({"text": req.text, **response})
    return response
