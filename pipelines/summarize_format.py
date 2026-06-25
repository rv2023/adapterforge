"""M8 — reshape ECTSum (earnings-call transcript, summary) into chat-format SFT data.

The 2nd task adapter: summarization. Mirrors instruction_format.py (the sentiment task)
but with a summarization prompt + ECTSum data. Output feeds the SAME finetune.py
(task-agnostic SFT on chat-messages JSONL) — just a different DATA_DIR/ADAPTER_DIR.

    python -m pipelines.summarize_format
"""

from pathlib import Path
import json
import urllib.request
import zipfile

OUT_DIR = Path("data/instruction_summ")
ECTSUM_ZIP_URL = "https://codeload.github.com/rajdeep345/ECTSum/zip/refs/heads/main"
ECTSUM_ZIP_PATH = Path("/tmp/ectsum-main.zip")

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
    """Return (train, val, test) of (transcript, summary) pairs from ECTSum."""
    if not ECTSUM_ZIP_PATH.exists():
        urllib.request.urlretrieve(ECTSUM_ZIP_URL, ECTSUM_ZIP_PATH)

    def pairs(zf: zipfile.ZipFile, split_name: str) -> list[tuple[str, str]]:
        prefix = f"ECTSum-main/data/final/{split_name}/ects/"
        transcript_names = sorted(
            name
            for name in zf.namelist()
            if name.startswith(prefix) and name.endswith(".txt")
        )
        rows = []
        for transcript_name in transcript_names:
            summary_name = transcript_name.replace("/ects/", "/gt_summaries/")
            transcript = zf.read(transcript_name).decode("utf-8").strip()
            summary = zf.read(summary_name).decode("utf-8").strip()
            rows.append((transcript, summary))
        return rows

    with zipfile.ZipFile(ECTSUM_ZIP_PATH) as zf:
        return pairs(zf, "train"), pairs(zf, "val"), pairs(zf, "test")


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    """ECTSum -> data/instruction_summ/{train,val,test}.jsonl (chat-messages for SFT)."""
    train, val, test = load_ectsum()
    for split_name, pairs in (("train", train), ("val", val), ("test", test)):
        rows = [to_chat_example(t, s) for t, s in pairs]
        write_jsonl(rows, OUT_DIR / f"{split_name}.jsonl")
        print(split_name, len(rows))


if __name__ == "__main__":
    main()
