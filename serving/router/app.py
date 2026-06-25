"""M8 router service — thin FastAPI wrapper. Loads backends once (lifespan), delegates
the decision to routing.route(). All the smarts live in routing.py (testable separately).

    uvicorn serving.router.app:app --port 8090
    curl -s localhost:8090/predict -H 'content-type: application/json' \
      -d '{"text":"the company posted record profits","task":"classify","tier":"escalate"}'
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from serving.router import backends as backends_mod
from serving.router.routing import route


class RouteRequest(BaseModel):
    text: str
    task: str = "classify"
    tier: str = "escalate"  # default = the cascade


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.backends = backends_mod.default_backends()
    yield


app = FastAPI(title="AdapterForge Router", lifespan=lifespan)


@app.post("/predict")
def predict(req: RouteRequest, request: Request) -> dict:
    return route(req.task, req.tier, req.text, request.app.state.backends)
