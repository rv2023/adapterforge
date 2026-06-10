"""Tests for BaseAdapter.validate() running the adapter's declared schema.

Proves the wiring: an adapter declares `schema = schema_v1`, and the shared
validate() uses it to reject a bad batch and accept a clean one.
"""

import pandas as pd
import pandera.errors
import pytest

from adapter_sdk.base import BaseAdapter
from adapter_sdk.schemas.v1 import schema_v1


class _SchemaAdapter(BaseAdapter):
    """Minimal adapter that declares the v1 rulebook. read() is unused here."""

    schema = schema_v1

    def read(self) -> pd.DataFrame:
        raise NotImplementedError  # not exercised in these tests


def test_validate_accepts_clean_batch():
    adapter = _SchemaAdapter()
    clean = pd.DataFrame({"text": ["Fed signals rate cut"], "label": ["bullish"]})
    adapter.validate(clean)  # should not raise


def test_validate_rejects_bad_label():
    adapter = _SchemaAdapter()
    bad = pd.DataFrame({"text": ["Fed signals rate cut"], "label": ["positive"]})
    with pytest.raises(pandera.errors.SchemaError):
        adapter.validate(bad)
