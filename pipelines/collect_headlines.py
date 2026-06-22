"""M5 Piece 5, Step 1 — pull UNLABELED financial headlines via the RestAPIAdapter.

Fan across several topic/ticker queries (Alpha Vantage free tier: ~25 calls/day,
limit up to 1000), keep ONLY the headline text (the teacher labels these, not AV),
dedup, and save to data/distill/headlines.jsonl.

    python -m pipelines.collect_headlines

Design + rationale: docs/practicum-vision.md is the v3 vision; this is the core
Piece-5 distillation data step.
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from adapter_sdk.adapters.rest import RestAPIAdapter

OUT = Path("data/distill/headlines.jsonl")   # data/ is gitignored
REQUEST_SLEEP_SECONDS = 1.0


def build_windows(n=25, days_each=14):
    """Return disjoint UTC (time_from, time_to) pairs formatted YYYYMMDDTHHMM."""
    fmt = "%Y%m%dT%H%M"
    now = datetime.now(timezone.utc)
    windows = []
    for i in range(n):
        end = now - timedelta(days=i * days_each)
        start = end - timedelta(days=days_each)
        windows.append((start.strftime(fmt), end.strftime(fmt)))
    return windows


# Each entry = one API call. Mix topics + tickers for variety. Keep within the
# free-tier daily call budget (~25). Valid topics include: financial_markets,
# economy_macro, economy_fiscal, economy_monetary, earnings, ipo,
# mergers_and_acquisitions, technology, finance, energy_transportation, ...
QUERIES = [
    {"topics": "financial_markets", "time_from": time_from, "time_to": time_to}
    for time_from, time_to in build_windows(n=25, days_each=14)
]


def collect() -> pd.DataFrame:
    """Pull all QUERIES, keep text only, dedup. Returns the unlabeled DataFrame.

    Resilience: one bad call (rate-limit note instead of a 'feed') shouldn't kill
    the whole run — skip it and keep going.
    """
    frames = []
    for q in QUERIES:
        try:
            adapter = RestAPIAdapter(**q, limit=1000)
            df = adapter.read()
            frames.append(df[["text"]])
        except Exception as e:
            print(f"skip {q}: {e}")
        time.sleep(REQUEST_SLEEP_SECONDS)

    if not frames:
        return pd.DataFrame(columns=["text"])

    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["text"])
        .drop_duplicates("text")
        .reset_index(drop=True)
    )


def main() -> None:
    """Collect and save the unlabeled headline pool as JSONL."""
    new = collect()
    if OUT.exists():
        old = pd.read_json(OUT, lines=True)
        new = pd.concat([old, new], ignore_index=True)

    new = (
        new.dropna(subset=["text"])
        .drop_duplicates("text")
        .reset_index(drop=True)
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new.to_json(OUT, orient="records", lines=True)
    print(f"total {len(new)} unique headlines -> {OUT}")


if __name__ == "__main__":
    main()
