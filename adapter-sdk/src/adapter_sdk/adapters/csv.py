"""CSVAdapter — loads financial sentiment data from a local CSV file.

Same shape as HFDatasetAdapter: it declares the v1 schema and fills in read().
The only real difference is read() itself — it reads a FILE (pd.read_csv) and
the source labels are integers. Everything else (validate/version/land/run) is
inherited unchanged.
"""

import pandas as pd

from adapter_sdk.base import BaseAdapter
from adapter_sdk.schemas.v1 import schema_v1

# Source integer labels -> our vocabulary (Twitter Financial News convention).
LABEL_MAP = {0: "bearish", 1: "bullish", 2: "neutral"}


class CSVAdapter(BaseAdapter):
    """Adapter for a Twitter-Financial-News-style CSV (text + integer label)."""

    schema = schema_v1

    def __init__(self, path: str):
        self.path = path

    def read(self) -> pd.DataFrame:
        """Load the CSV file and normalize it to our (text, label) table."""
        df = pd.read_csv(self.path, usecols=["text", "label"])
        df["label"] = df["label"].map(LABEL_MAP)
        return df
