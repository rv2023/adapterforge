"""Schema v1 — the rulebook for financial sentiment data (text + label).

This is what validate() checks each batch against. It is VERSIONED: when the
rules need to change, we add a v2.py rather than editing this file, so any
dataset stamped "schema v1" can always be re-checked against these exact rules.
"""

import pandera.pandas as pa

# The only labels a valid batch may contain.
ALLOWED_LABELS = ["bullish", "bearish", "neutral"]


schema_v1 = pa.DataFrameSchema(
    {"text": pa.Column(str), "label": pa.Column(str, checks=pa.Check.isin(ALLOWED_LABELS))},
    name="v1",  # version label; version() stamps this onto each landed batch
)
