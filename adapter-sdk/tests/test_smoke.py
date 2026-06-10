"""Smoke test: proves the package is importable and installed.

Replaced/expanded as real components land. Exists so `make test` is green
from day one and CI has something to run.
"""

import adapter_sdk


def test_package_imports():
    assert adapter_sdk.__version__ == "0.1.0"
