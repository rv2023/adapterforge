"""M5 Piece 5, Step 4 — evaluate the DistilBERT student on the FROZEN test set.

Same sealed FPB test exam as the baseline (0.6885) and the teacher (0.8477) — that's how
all three are comparable. SEQ_CLS eval is simple: forward -> argmax -> class. Bar: within
2-3 F1 pts of the teacher's 0.8477. Concepts: docs/m5-training-concepts.md (train/val/test).

    python -m pipelines.eval_student
"""

# MUST match Step 2/3 index<->label order, or argmax maps to the wrong class.
LABELS = ["bullish", "bearish", "neutral"]
STUDENT_DIR = "models/distilbert-student"
OUT_METRICS = "models/distilbert-student/eval_metrics.json"


def load_test_set():
    """The frozen FPB test split (gold labels) — reuse baseline.load_data + split_data."""
    from pipelines.baseline import load_data, split_data

    _, _, test_df = split_data(load_data())
    return test_df


def predict(model, tokenizer, texts: list[str]) -> list[str]:
    """Batched forward -> argmax -> map class index to LABELS[idx]. Returns predicted words."""
    import torch

    device = next(model.parameters()).device
    preds: list[str] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), 128):
            batch = tokenizer(
                texts[start : start + 128],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            pred_ids = model(**batch).logits.argmax(dim=-1).cpu().tolist()
            preds.extend(LABELS[idx] for idx in pred_ids)
    return preds


def main() -> float:
    """Score the student on the frozen test set; print + persist macro-F1. Returns it."""
    import json
    from pathlib import Path

    import torch
    from sklearn.metrics import f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT_DIR)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    test_df = load_test_set()
    preds = predict(model, tokenizer, test_df["text"].tolist())
    f1 = f1_score(test_df["label"], preds, average="macro")

    print(f"student test macro-F1: {f1:.4f} (teacher 0.8477, baseline 0.6885)")
    metrics = {"test_f1": float(f1), "n_test": int(len(test_df))}
    out_path = Path(OUT_METRICS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return float(f1)


if __name__ == "__main__":
    main()
