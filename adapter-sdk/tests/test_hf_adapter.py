"""Tests for HFDatasetAdapter.

These actually download the dataset (a few MB, cached after the first run), so
they're slower than the other tests. They prove read() returns our standard,
normalized table — and that the normalized data passes the v1 rulebook end-to-end.
"""

from adapter_sdk.adapters.hf import HFDatasetAdapter


def test_read_returns_standard_table():
    df = HFDatasetAdapter().read()
    # Exactly our two columns, in order.
    assert list(df.columns) == ["text", "label"]
    # Got some rows.
    assert len(df) > 0


def test_read_labels_are_in_our_vocabulary():
    df = HFDatasetAdapter().read()
    # Every label was translated into our vocabulary — no leftover positive/negative.
    assert set(df["label"].unique()) <= {"bullish", "bearish", "neutral"}


def test_normalized_data_passes_v1_schema():
    # End-to-end: read() output flows straight through validate() with no error.
    adapter = HFDatasetAdapter()
    df = adapter.read()
    adapter.validate(df)
