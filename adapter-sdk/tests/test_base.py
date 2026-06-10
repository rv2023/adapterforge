"""Tests for BaseAdapter.

The key behaviour to prove: run() calls the four lifecycle steps in the
correct, fixed order. We build a throwaway "fake" adapter that records the
order steps were called in, then assert on it.
"""

import pandas as pd

from adapter_sdk.base import BaseAdapter


class _FakeAdapter(BaseAdapter):
    """Minimal adapter that records which steps ran, in order."""

    def __init__(self):
        self.calls: list[str] = []

    def read(self) -> pd.DataFrame:
        self.calls.append("read")
        return pd.DataFrame({"text": ["hello"]})

    def validate(self, df: pd.DataFrame) -> None:
        self.calls.append("validate")

    def version(self, df: pd.DataFrame) -> None:
        self.calls.append("version")

    def land(self, df: pd.DataFrame) -> None:
        self.calls.append("land")


def test_run_calls_steps_in_order():
    adapter = _FakeAdapter()
    adapter.run()
    assert adapter.calls == ["read", "validate", "version", "land"]
