"""HFDatasetAdapter — loads Financial PhraseBank from HuggingFace.

The first real adapter. Its read() pulls the dataset and NORMALIZES it into our
standard (text, label) table, with labels in our vocabulary (bullish / bearish /
neutral). After read(), the rest of the lifecycle (validate / version / land)
treats it like any other batch.
"""

import pandas as pd
from datasets import load_dataset

from adapter_sdk.base import BaseAdapter
from adapter_sdk.schemas.v1 import schema_v1

DATASET_NAME = "ChanceFocus/flare-fpb"

LABEL_MAP = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}


class HFDatasetAdapter(BaseAdapter):
    schema = schema_v1

    def read(self) -> pd.DataFrame:
        """Load Financial PhraseBank and normalize it to our (text, label) table."""

        ds = load_dataset(DATASET_NAME, split="train")
        df = ds.to_pandas()  # turn the HF dataset into a pandas table
        df = df[["text", "answer"]]  # a LIST of names inside [] → a smaller table
        df = df.rename(columns={"answer": "label"})
        df["label"] = df["label"].map(LABEL_MAP)

        return df
