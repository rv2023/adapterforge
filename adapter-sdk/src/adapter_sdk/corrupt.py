"""corrupt.py — a chaos tool that deliberately breaks a clean batch.

Each function takes a clean (text, label) DataFrame and returns a COPY broken in
one specific way. Running those broken batches through validate() proves the SDK
rejects bad data — turning "validation should work" into "watch it work".
"""

import pandas as pd


def corrupt_label(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where the first row's label is an invalid value."""
    out = df.copy()
    out.loc[df.index[0], "label"] = "garbage"
    return out


def corrupt_null_text(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where the first row's text is null (missing)."""
    out = df.copy()
    out.loc[df.index[0], "text"] = None
    return out


def corrupt_drop_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the "label" column removed entirely."""
    return df.drop(columns=["label"])
