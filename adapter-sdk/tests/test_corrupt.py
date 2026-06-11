"""Tests for the corrupt.py chaos tool.

Proves the SDK's validation rejects each kind of deliberately-broken batch,
while the clean batch passes.
"""

import pandas as pd
import pandera.errors
import pytest

from adapter_sdk.corrupt import corrupt_drop_column, corrupt_label, corrupt_null_text
from adapter_sdk.schemas.v1 import schema_v1


def clean_batch() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["Fed cuts rates", "Bank posts record loss"],
            "label": ["bullish", "bearish"],
        }
    )


def test_clean_batch_passes():
    # sanity check: the un-corrupted batch is valid
    schema_v1.validate(clean_batch())


@pytest.mark.parametrize(
    "corrupt",
    [corrupt_label, corrupt_null_text, corrupt_drop_column],
)
def test_corruption_is_rejected(corrupt):
    bad = corrupt(clean_batch())
    with pytest.raises(pandera.errors.SchemaError):
        schema_v1.validate(bad)
