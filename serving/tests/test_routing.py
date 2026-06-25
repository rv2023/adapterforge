"""M8 router routing-logic tests — pure decision testing with MOCK backends.

No models, no GPU: route() takes a `backends` dict of callables, so every routing path
is verified in milliseconds. (This is the payoff of the backend abstraction.)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from serving.router.routing import THRESHOLD, route


def _backends(student_label="bullish", student_conf=0.99, llm_label="bearish"):
    """Mock backends: each is text -> (output, confidence)."""
    return {
        "student": lambda t: (student_label, student_conf),
        "llm_sentiment": lambda t: (llm_label, None),
        "summarizer": lambda t: ("a brief", None),
    }


def test_summarize_routes_to_summarizer():
    r = route("summarize", "escalate", "x", _backends())
    assert r["served_by"] == "summarizer"
    assert r["escalated"] is False


def test_classify_accurate_routes_to_llm():
    r = route("classify", "accurate", "x", _backends())
    assert r["served_by"] == "llm_sentiment"
    assert r["escalated"] is False


def test_classify_cheap_never_escalates_even_when_unsure():
    # cheap must stay on the student regardless of (low) confidence
    r = route("classify", "cheap", "x", _backends(student_conf=0.10))
    assert r["served_by"] == "student"
    assert r["escalated"] is False


def test_escalate_keeps_student_when_confident():
    r = route("classify", "escalate", "x", _backends(student_conf=0.99))
    assert r["served_by"] == "student"
    assert r["escalated"] is False


def test_escalate_escalates_to_llm_when_unsure():
    r = route(
        "classify", "escalate", "x",
        _backends(student_label="neutral", student_conf=0.48, llm_label="bullish"),
    )
    assert r["served_by"] == "llm_sentiment"
    assert r["escalated"] is True
    assert r["output"] == "bullish"  # the LLM's answer, not the student's


def test_escalate_boundary_at_threshold_does_not_escalate():
    # strict <: confidence == THRESHOLD should KEEP the student
    r = route("classify", "escalate", "x", _backends(student_conf=THRESHOLD))
    assert r["served_by"] == "student"
    assert r["escalated"] is False


def test_empty_tier_defaults_to_escalate():
    r = route("classify", "", "x", _backends(student_conf=0.10))
    assert r["escalated"] is True


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        route("translate", "escalate", "x", _backends())


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        route("classify", "fancy", "x", _backends())
