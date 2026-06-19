"""M5 Piece 1 — reshape PhraseBank (text,label) into chat-format instruction data.

Reuses M2's frozen split so the LLM is evaluated on the exact same sealed test set
as the 0.6885 baseline. Writes train/val/test as JSONL under data/instruction/.
"""

from pathlib import Path
import json

from pipelines.baseline import load_data, split_data  # reuse — same frozen split

OUT_DIR = Path("data/instruction")

INSTRUCTION = (
    "Classify the financial sentiment of the following statement as exactly one "
    "of bullish, bearish, or neutral.\n\nStatement: "
)


def to_chat_example(text: str, label: str) -> dict:
    """Turn one (text, label) row into a chat-messages dict.

    Target shape:
        {"messages": [
            {"role": "user",      "content": <INSTRUCTION + the statement>},
            {"role": "assistant", "content": <the bare label word>},
        ]}
    """
    user_content = INSTRUCTION + text
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a financial sentiment classifier.",
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": label},
        ]
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write a list of dicts to `path`, one compact JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
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
