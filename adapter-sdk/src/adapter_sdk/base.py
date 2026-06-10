"""BaseAdapter: the parent class every data source is built on.

Every adapter follows the same lifecycle, always in this order:

    read -> validate -> version -> land

The base owns that order (in ``run``), so an adapter can never skip validation
or run the steps out of order. Each source only has to fill in ``read`` — the
one abstract "blank".
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseAdapter(ABC):
    """Parent class for all data adapters.

    To add a new data source, subclass this and fill in ``read``. The shared
    steps (``validate`` / ``version`` / ``land``) are inherited as-is, and the
    order is fixed by ``run``.
    """

    # Each adapter sets this to its rulebook (e.g. schema_v1). validate() runs it.
    schema = None

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

    def version(self, df: pd.DataFrame) -> None:  # noqa: B027  (intentional no-op hook for now)
        """Stamp the batch with its schema version + SDK version.

        Shared by every adapter. (Built out in a later component.)
        """
        # TODO (later component): record schema version + SDK version.
        ...

    def land(self, df: pd.DataFrame) -> None:  # noqa: B027  (intentional no-op hook for now)
        """Save the table to storage (the S3 medallion zones).

        Shared by every adapter. (Built out in a later component.)
        """
        # TODO (later component): write df to S3.
        ...
