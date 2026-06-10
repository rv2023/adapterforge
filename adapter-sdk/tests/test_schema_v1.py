"""Tests for schema v1.

Two things to prove:
- a clean batch passes validation (no error)
- a batch with a bad label is rejected (Pandera raises)
"""

import pandas as pd
import pandera.errors
import pytest

from adapter_sdk.schemas.v1 import schema_v1


def test_good_batch_passes():
    df = pd.DataFrame(
        {
            "text": ["Fed signals rate cut", "Bank reports record loss"],
            "label": ["bullish", "bearish"],
        }
    )
    # Should return the table unchanged and NOT raise.
    schema_v1.validate(df)


def test_bad_label_is_rejected():
    df = pd.DataFrame(
        {
            "text": ["Company holds dividend flat"],
            "label": ["positive"],  # not one of bullish/bearish/neutral
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        schema_v1.validate(df)
