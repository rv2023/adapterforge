"""Tests for version() + land().

version() builds a provenance record on self.metadata; land() writes the batch
(parquet) AND the provenance (.meta.json) into the raw + validated zones.
DATA_ROOT is redirected to a temp dir so the real ./data folder is untouched.
"""

import json

import pandas as pd

from adapter_sdk.base import BaseAdapter
from adapter_sdk.schemas.v1 import schema_v1


class _LandAdapter(BaseAdapter):
    """Minimal adapter with a name + schema, for exercising version()/land()."""

    name = "test_dataset"
    schema = schema_v1

    def read(self) -> pd.DataFrame:
        return pd.DataFrame({"text": ["hello world"], "label": ["bullish"]})


def test_land_writes_data_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr("adapter_sdk.base.DATA_ROOT", tmp_path)

    adapter = _LandAdapter()
    df = adapter.read()
    adapter.version(df)  # build the provenance record
    adapter.land(df)  # write data + provenance into both zones

    for zone in ("raw", "validated"):
        parquet = tmp_path / zone / "test_dataset.parquet"
        meta = tmp_path / zone / "test_dataset.meta.json"

        assert parquet.exists(), f"missing parquet in {zone} zone"
        assert meta.exists(), f"missing provenance sidecar in {zone} zone"

        reloaded = pd.read_parquet(parquet)
        assert list(reloaded.columns) == ["text", "label"]
        assert len(reloaded) == 1

        record = json.loads(meta.read_text())
        assert record["schema_version"] == "v1"
        assert record["rows"] == 1
        assert "sdk_version" in record
