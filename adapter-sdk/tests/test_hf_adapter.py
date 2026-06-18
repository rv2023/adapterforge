"""Tests for HFDatasetAdapter.

These MOCK the HuggingFace download (``load_dataset``) so they run offline, fast,
and deterministically — CI must never depend on the Hub being reachable. They prove
read() returns our standard, normalized table and that the normalized data passes
the v1 rulebook end-to-end. That is: they test OUR logic, not the network.

(A separate, network-touching integration test could be added and marked to run only
locally/nightly — but it must not gate CI.)
"""

import pandas as pd
import pytest

from adapter_sdk.adapters import hf
from adapter_sdk.adapters.hf import HFDatasetAdapter


class _FakeDataset:
    """Stand-in for a HuggingFace dataset — read() only ever calls .to_pandas()."""

    def to_pandas(self):
        # mimics flare-fpb's shape: a `text` column + an `answer` column whose
        # values are the UPSTREAM labels (positive/negative/neutral) that read()
        # must translate into our vocabulary.
        return pd.DataFrame(
            {
                "text": [
                    "shares jumped on strong guidance",
                    "the firm posted a deep loss",
                    "the board meeting is on tuesday",
                ],
                "answer": ["positive", "negative", "neutral"],
            }
        )


def _fake_load_dataset(*args, **kwargs):
    return _FakeDataset()


@pytest.fixture
def mock_hf_download(monkeypatch):
    """Replace the real HF download with the offline fake (patched where it's USED)."""
    monkeypatch.setattr(hf, "load_dataset", _fake_load_dataset)


def test_read_returns_standard_table(mock_hf_download):
    df = HFDatasetAdapter().read()
    # Exactly our two columns, in order.
    assert list(df.columns) == ["text", "label"]
    # Got some rows.
    assert len(df) > 0


def test_read_labels_are_in_our_vocabulary(mock_hf_download):
    df = HFDatasetAdapter().read()
    # Every label was translated into our vocabulary — no leftover positive/negative.
    assert set(df["label"].unique()) <= {"bullish", "bearish", "neutral"}


def test_normalized_data_passes_v1_schema(mock_hf_download):
    # End-to-end: read() output flows straight through validate() with no error.
    adapter = HFDatasetAdapter()
    df = adapter.read()
    adapter.validate(df)
