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
    # TODO: from pipelines.baseline import load_data, split_data
    #       _, _, test_df = split_data(load_data()); return test_df   (columns: text,label)
    ...


def predict(model, tokenizer, texts: list[str]) -> list[str]:
    """Batched forward -> argmax -> map class index to LABELS[idx]. Returns predicted words."""
    # TODO: tokenize (batched), forward (no_grad), logits.argmax(-1) -> LABELS[idx]
    ...


def main() -> float:
    """Score the student on the frozen test set; print + persist macro-F1. Returns it."""
    # TODO:
    #   load student (AutoModelForSequenceClassification + tokenizer from STUDENT_DIR)
    #   test_df = load_test_set(); preds = predict(...)
    #   f1 = f1_score(test_df["label"], preds, average="macro")
    #   print f1 vs 0.8477 (teacher) / 0.6885 (baseline)
    #   write {"test_f1": f1, "n_test": len(test_df)} to OUT_METRICS; return f1
    ...


if __name__ == "__main__":
    main()
