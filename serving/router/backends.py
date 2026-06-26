"""M8 router backends — each backend is a callable: text -> (output, confidence|None).

The routing DECISION (routing.py) is transport-agnostic; this file WIRES the actual
backends. Locally: `student` is real (CPU); `llm_sentiment` is a MOCK (no GPU);
`summarizer` is a stub. In prod these slots become HTTP clients to vLLM / student
service / KServe — routing.py doesn't change, only what these callables are.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root


def build_llm_sentiment():
    """MOCK LLM sentiment backend for local dev (no GPU).

    Prod: replace with an HTTP client to vLLM (/v1/chat/completions, the fpb adapter).
    """
    def predict(text: str):
        return ("bullish", None)  # canned; confidence None (a generate() model)

    return predict


def build_student():
    """Real DistilBERT student (CPU), reusing the M6 model-aware loader."""
    from serving.app import build_distilbert_predictor

    predict_fn = build_distilbert_predictor("fpb-student", "2")
    return predict_fn


def build_summarizer():
    """Real summarization LoRA adapter (slow on CPU locally; prod -> vLLM HTTP).

    Delegates to serving.app, exactly like build_student delegates the distilbert load.
    """
    from serving.app import build_summary_predictor

    return build_summary_predictor("fpb-summarizer", "1")


def default_backends() -> dict:
    """The real backend registry the router service loads at startup."""
    return {
        "llm_sentiment": build_llm_sentiment(),
        "student": build_student(),
        "summarizer": build_summarizer(),
    }
