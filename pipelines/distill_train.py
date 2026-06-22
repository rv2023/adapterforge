"""M5 Piece 5, Step 3 — train the DistilBERT student on the teacher's SOFT labels (KL).

Student = DistilBERT SEQ_CLS (encoder + 3-class head), trained to MATCH the teacher's
soft distribution via KL-divergence (classic distillation). Runs on the GPU pod (or CPU,
slow). Concepts: docs/m5-distillation-concepts.md (§3–4).

    python -m pipelines.distill_train
"""

# Must match Step 2's LABELS order (index <-> class) exactly.
LABELS = ["bullish", "bearish", "neutral"]
TEMPERATURE = 2.0          # same T the teacher probs were softened with (Step 2)
STUDENT_MODEL = "distilbert-base-uncased"
IN = "data/distill/labeled.jsonl"        # {"text", "probs":[3]}
STUDENT_DIR = "models/distilbert-student"


def load_labeled():
    """Load the teacher-labeled data and split into train/val (for early stopping).

    Returns HF datasets (or DataFrames) with columns: text, probs (3-vector).
    """
    # TODO: read IN jsonl; train/val split (e.g. 90/10, seed=42); return them
    ...


def build_student():
    """DistilBERT for sequence classification, 3 labels, + its tokenizer."""
    # TODO: AutoTokenizer + AutoModelForSequenceClassification.from_pretrained(
    #         STUDENT_MODEL, num_labels=len(LABELS)); return (model, tokenizer)
    ...


def tokenize(ds, tokenizer):
    """Tokenize text; keep `probs` as the soft-label target column."""
    # TODO: map tokenizer over text (truncation, max_length ~128); carry probs through
    ...


class DistillTrainer:
    """HF Trainer subclass with a KL-distillation compute_loss.

    Classic Hinton distillation loss:
        loss = T^2 * KL( log_softmax(student_logits / T) || teacher_probs )
    where teacher_probs were already softened at T in Step 2, and KLDivLoss expects the
    student side as LOG-probabilities. (reduction="batchmean".)
    """
    # TODO: subclass transformers.Trainer; override compute_loss(self, model, inputs,
    #       return_outputs=False):
    #   labels = inputs.pop("probs")               # Nx3 teacher distribution
    #   logits = model(**inputs).logits            # Nx3 student logits
    #   loss = T**2 * KLDivLoss(reduction="batchmean")(
    #              log_softmax(logits / T, dim=-1), labels)
    #   return (loss, outputs) if return_outputs else loss
    ...


def main() -> None:
    """Train the student with KL distillation and save it to STUDENT_DIR."""
    # TODO:
    #   train_ds, val_ds = load_labeled(); model, tok = build_student()
    #   tokenize both; TrainingArguments (bf16 on GPU, load_best_model_at_end, eval each epoch)
    #   DistillTrainer(...).train(); trainer.save_model(STUDENT_DIR); tok.save_pretrained(STUDENT_DIR)
    ...


if __name__ == "__main__":
    main()
