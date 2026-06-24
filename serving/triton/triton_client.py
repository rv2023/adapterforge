"""M6 Piece 2 — minimal Triton client for the DistilBERT student (KServe v2 protocol).

Triton serves raw tensors, so the CLIENT owns tokenization (text -> input_ids/
attention_mask) and the argmax -> label mapping. This proves the student is served
through Triton end-to-end.

    # with Triton running (see export_student_onnx.py docstring for the docker cmd):
    python serving/triton/triton_client.py "the company posted record profits"
"""

import sys

import requests
from transformers import AutoTokenizer

STUDENT_DIR = "models/distilbert-student"
TRITON_URL = "http://localhost:8000/v2/models/distilbert-student/infer"
LABELS = ["bullish", "bearish", "neutral"]   # index order must match eval_student.LABELS


def classify(text: str) -> tuple[str, list[float]]:
    """Tokenize, POST to Triton's v2 infer endpoint, return (label, logits)."""
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_DIR)
    enc = tokenizer([text], return_tensors="np")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # v2 request: each input = {name, shape, datatype, data(flat list)}.
    # shape is [batch, seq]; datatype "INT64" matches config.pbtxt.
    payload = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": list(input_ids.shape),
                "datatype": "INT64",
                "data": input_ids.ravel().tolist(),
            },
            {
                "name": "attention_mask",
                "shape": list(attention_mask.shape),
                "datatype": "INT64",
                "data": attention_mask.ravel().tolist(),
            },
        ],
        "outputs": [{"name": "logits"}],
    }

    resp = requests.post(TRITON_URL, json=payload)
    resp.raise_for_status()
    logits = resp.json()["outputs"][0]["data"]
    label = LABELS[max(range(len(logits)), key=logits.__getitem__)]
    return label, logits


def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else "the company posted record profits"
    label, logits = classify(text)
    print(f"{label}  logits={logits}")


if __name__ == "__main__":
    main()
