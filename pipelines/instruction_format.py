"""M5 Piece 1 — reshape PhraseBank (text,label) into chat-format instruction data.

Reuses M2's frozen split so the LLM is evaluated on the exact same sealed test set
as the 0.6885 baseline. Writes train/val/test as JSONL under data/instruction/.
"""

from pathlib import Path
import json

OUT_DIR = Path("data/instruction")

INSTRUCTION = (
    "Classify the financial sentiment of the following statement as exactly one "
    "of bullish, bearish, or neutral.\n\nStatement: "
)
# Single source of truth for the prompt contract. Training, serving (app.py /
# bench_naive), and the benchmark client all import these so the prompt can never
# drift between train and inference (a drifted prompt silently tanks accuracy).
SYSTEM_PROMPT = "You are a financial sentiment classifier."


def build_chat_messages(text: str) -> list[dict]:
    """The [system, user] chat turns for one statement (no assistant turn)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCTION + text},
    ]


def to_chat_example(text: str, label: str) -> dict:
    """One (text, label) row -> a chat-messages dict (adds the assistant label turn)."""
    return {"messages": build_chat_messages(text) + [{"role": "assistant", "content": label}]}


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write a list of dicts to `path`, one compact JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    from pipelines.baseline import load_data, split_data  # reuse — same frozen split

    df = load_data()
    splits = zip(("train", "val", "test"), split_data(df))

    for split_name, split_df in splits:
        rows = [
            to_chat_example(row.text, row.label)
            for row in split_df[["text", "label"]].itertuples(index=False)
        ]
        write_jsonl(rows, OUT_DIR / f"{split_name}.jsonl")
        print(f"{split_name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
