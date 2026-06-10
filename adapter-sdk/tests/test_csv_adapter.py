"""Tests for CSVAdapter.

Reads a small sample CSV fixture (no network), proves read() produces our
standard normalized table and that it passes the v1 rulebook end-to-end.
"""

from pathlib import Path

from adapter_sdk.adapters.csv import CSVAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "twitter_financial_news_sample.csv"


def test_read_returns_standard_table():
    df = CSVAdapter(str(FIXTURE)).read()
    assert list(df.columns) == ["text", "label"]
    assert len(df) == 6


def test_labels_normalized_and_valid():
    adapter = CSVAdapter(str(FIXTURE))
    df = adapter.read()
    # integers became our words, nothing left over
    assert set(df["label"].unique()) <= {"bullish", "bearish", "neutral"}
    adapter.validate(df)  # end-to-end: normalized CSV data passes schema v1
