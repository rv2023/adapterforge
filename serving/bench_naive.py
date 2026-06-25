"""M6 Piece 1 — the NAÏVE serving baseline (bf16), for benchmarking only.

Deliberately naïve: FastAPI + HF `generate()` per request. This is the "before" in
the vLLM benchmark — same model, NO continuous batching. It isolates the serving
*mechanism*: it loads the adapter directory directly (no control plane, no registry),
so the only thing being compared against vLLM is how requests are handled.

    # on the pod, after dvc-pulling models/fpb-lora:
    ADAPTER_DIR=models/fpb-lora uvicorn serving.bench_naive:app --port 8001

Then point the harness at it:
    python -m pipelines.benchmark_serving --server naive --url http://localhost:8001
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from pydantic import BaseModel

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

ADAPTER_DIR = os.getenv("ADAPTER_DIR", "models/fpb-lora")


class PredictRequest(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the adapter ONCE at startup, in bf16, and stash it on app.state."""
    from pipelines import eval_adapter, instruction_format

    model, tokenizer = eval_adapter.load_model_and_tokenizer(ADAPTER_DIR, use_4bit=False)
    app.state.eval_adapter = eval_adapter
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.build_messages = instruction_format.build_chat_messages  # shared prompt contract
    yield


app = FastAPI(title="AdapterForge Naive Serving (benchmark)", lifespan=lifespan)


@app.post("/predict")
def predict(req: PredictRequest, request: Request) -> dict:
    """One statement -> one label. Same wire format as serving/app.py (/predict)."""
    state = request.app.state
    messages = state.build_messages(req.text)
    with state.eval_adapter.torch.inference_mode():
        label = state.eval_adapter.predict_one(
            state.model,
            state.tokenizer,
            messages,
        )
    return {"label": label}
