"""M8 — reshape ECTSum (earnings-call transcript, summary) into chat-format SFT data.

The 2nd task adapter: summarization. Mirrors instruction_format.py (the sentiment task)
but with a summarization prompt + ECTSum data. Output feeds the SAME finetune.py
(task-agnostic SFT on chat-messages JSONL) — just a different DATA_DIR/ADAPTER_DIR.

    python -m pipelines.summarize_format
"""

from pathlib import Path
import json

OUT_DIR = Path("data/instruction_summ")

# Summarization prompt contract (separate task -> its own prompt; cf. instruction_format).
SYSTEM_PROMPT = "You are an earnings-call summarizer."
INSTRUCTION = "Summarize the following earnings call as concise bullet points.\n\nTranscript: "


def build_chat_messages(transcript: str) -> list[dict]:
    """[system, user] chat turns for one transcript (no assistant turn)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCTION + transcript},
    ]


def to_chat_example(transcript: str, summary: str) -> dict:
    """One (transcript, summary) row -> chat-messages dict (adds the assistant summary)."""
    return {"messages": build_chat_messages(transcript) + [{"role": "assistant", "content": summary}]}


def load_ectsum():
    """Return (train, val, test) of (transcript, summary) pairs from ECTSum.

    TODO: load ECTSum (HF `datasets`), find the transcript + summary columns, return the
    three splits. Confirm the exact HF dataset id + column names when you run it.
    """
    # TODO: from datasets import load_dataset
    # TODO: ds = load_dataset("<ectsum-hf-id>")   # confirm id + split names
    # TODO: map each split to (transcript, summary) using the right column names
    # TODO: return train, val, test
    raise NotImplementedError


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    """ECTSum -> data/instruction_summ/{train,val,test}.jsonl (chat-messages for SFT)."""
    # TODO: train, val, test = load_ectsum()
    # TODO: for split_name, pairs in [("train",train),("val",val),("test",test)]:
    #           rows = [to_chat_example(t, s) for (t, s) in pairs]
    #           write_jsonl(rows, OUT_DIR / f"{split_name}.jsonl")
    #           print(split_name, len(rows))
    raise NotImplementedError


if __name__ == "__main__":
    main()
