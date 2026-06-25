"""M8 router — the routing DECISION (task + tier, incl. the escalate cascade).

Tutor-protected: Karthik writes route(). It's transport-agnostic — it receives a
`backends` dict {name: callable(text)->(output, confidence)}, so the SAME logic is
unit-testable with mock backends (no models, no GPU). See docs/m8-concepts.md §1-3.
"""

THRESHOLD = 0.70  # escalate tier: go to the LLM when student confidence < THRESHOLD


def route(task: str, tier: str, text: str, backends: dict) -> dict:
    """Pick a backend by (task, tier), call it, return the result + provenance.

    Routing table (docs/m8-concepts §1):
      task=summarize            -> "summarizer"
      task=classify, accurate   -> "llm_sentiment"
      task=classify, cheap      -> "student"
      task=classify, escalate   -> "student"; if confidence < THRESHOLD -> "llm_sentiment"

    Return shape: {"output": ..., "served_by": <backend key>, "escalated": bool}
    """
    if task == "summarize":
        output, _ = backends["summarizer"](text)
        return {"output": output, "served_by": "summarizer", "escalated": False}

    if task == "classify":
        tier = tier or "escalate"

        if tier == "accurate":
            output, _ = backends["llm_sentiment"](text)
            return {"output": output, "served_by": "llm_sentiment", "escalated": False}

        if tier == "cheap":
            output, _ = backends["student"](text)
            return {"output": output, "served_by": "student", "escalated": False}

        if tier == "escalate":
            output, confidence = backends["student"](text)
            if confidence is not None and confidence < THRESHOLD:
                output, _ = backends["llm_sentiment"](text)
                return {"output": output, "served_by": "llm_sentiment", "escalated": True}
            return {"output": output, "served_by": "student", "escalated": False}

        raise ValueError(f"Unknown tier for task='classify': {tier!r}")

    raise ValueError(f"Unknown task: {task!r}")
