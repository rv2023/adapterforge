"""BaseAdapter: the parent class every data source is built on.

Every adapter follows the same lifecycle, always in this order:

    read -> validate -> version -> land

The base owns that order (in ``run``), so an adapter can never skip validation
or run the steps out of order. Each source only has to fill in ``read`` — the
one abstract "blank".
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from adapter_sdk import __version__

# Root of the local data lake (the medallion zones live under here).
# Tests monkeypatch this to a temp directory so they don't pollute the repo.
DATA_ROOT = Path("data")


class BaseAdapter(ABC):
    """Parent class for all data adapters.

    To add a new data source, subclass this and fill in ``read``. The shared
    steps (``validate`` / ``version`` / ``land``) are inherited as-is, and the
    order is fixed by ``run``.
    """

    # Each adapter sets this to its rulebook (e.g. schema_v1). validate() runs it.
    schema = None

    # Each adapter sets this to a short name used for its file in the data lake.
    name = None

    def run(self) -> None:
        """Run the full lifecycle in order: read -> validate -> version -> land.

        This is the single method that owns the sequence. It is written once
        here and every adapter inherits it, so the steps can never be reordered
        or skipped.
        """
        df = self.read()
        self.validate(df)
        self.version(df)
        self.land(df)

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """Fetch data from this source and return it as a table (DataFrame).

        This is the "blank": every adapter MUST write its own ``read``, because
        a CSV file, a web API, and a HuggingFace dataset are all fetched in
        completely different ways.
        """
        ...

    def validate(self, df: pd.DataFrame) -> None:
        """Check the table against this adapter's schema; reject it if it's bad.

        Shared by every adapter. Each adapter declares ``schema`` (a class
        attribute); this method runs it. Pandera raises if a rule is broken.
        """
        self.schema.validate(df)

    def version(self, df: pd.DataFrame) -> None:
        """Record provenance for this batch: which schema + SDK version produced it.

        Shared by every adapter. Stores the record on ``self.metadata``; ``land``
        writes it next to the data as a .meta.json sidecar.
        """
        self.metadata = {
            "schema_version": self.schema.name,
            "sdk_version": __version__,
            "rows": len(df),
        }

    def land(self, df: pd.DataFrame) -> None:
        """Save the batch into the data lake — raw and validated zones — as parquet.

        Shared by every adapter. Writes one file per zone, named after the
        adapter (e.g. data/validated/financial_phrasebank.parquet).

        """

        for zone in ("raw", "validated"):
            path = DATA_ROOT / zone / f"{self.name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            meta_path = path.parent / f"{self.name}.meta.json"
            meta_path.write_text(json.dumps(self.metadata, indent=2))
