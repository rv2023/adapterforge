"""Tests for RestAPIAdapter.

The live API free tier allows only ~25 calls/day, so these tests must NOT hit
the network. We replace requests.get with a fake that returns a saved JSON
fixture, then check that read() parses + normalizes it into our standard table.
"""

import json
from pathlib import Path

from adapter_sdk.adapters.rest import RestAPIAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "alphavantage_sample.json"


class _FakeResponse:
    """Stands in for a requests Response — only needs .json()."""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_read_normalizes_api_response(monkeypatch):
    payload = json.loads(FIXTURE.read_text())

    # Replace requests.get inside the rest module so no real HTTP call happens.
    def fake_get(*args, **kwargs):
        return _FakeResponse(payload)

    monkeypatch.setattr("adapter_sdk.adapters.rest.requests.get", fake_get)

    adapter = RestAPIAdapter(topics="financial_markets")
    df = adapter.read()

    assert list(df.columns) == ["text", "label"]
    assert len(df) == 5
    # all 5 source flavours folded into our 3-word vocabulary
    assert set(df["label"].unique()) <= {"bullish", "bearish", "neutral"}
    adapter.validate(df)  # end-to-end: normalized live data passes schema v1
