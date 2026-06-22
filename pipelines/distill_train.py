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
    import json

    from datasets import Dataset

    with open(IN, encoding="utf-8") as f:
        ds = Dataset.from_list([json.loads(line) for line in f if line.strip()])

    split = ds.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]


def build_student():
    """DistilBERT for sequence classification, 3 labels, + its tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL, num_labels=len(LABELS)
    )
    return model, tokenizer


def tokenize(ds, tokenizer):
    """Tokenize text; keep `probs` as the soft-label target column."""
    return ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=128),
        batched=True,
        remove_columns=["text"],
    )


def build_distill_collator(tokenizer):
    """Pad token fields and stack the teacher probabilities as a float tensor."""
    import torch
    from transformers import DataCollatorWithPadding

    pad = DataCollatorWithPadding(tokenizer)

    def collate(features):
        probs = torch.tensor([feature["probs"] for feature in features], dtype=torch.float32)
        token_features = [
            {k: v for k, v in feature.items() if k != "probs"}
            for feature in features
        ]
        batch = pad(token_features)
        batch["probs"] = probs
        return batch

    return collate


from transformers import Trainer


class DistillTrainer(Trainer):
    """HF Trainer subclass with a KL-distillation compute_loss.

    Classic Hinton distillation loss:
        loss = T^2 * KL( log_softmax(student_logits / T) || teacher_probs )
    where teacher_probs were already softened at T in Step 2, and KLDivLoss expects the
    student side as LOG-probabilities. (reduction="batchmean".)
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch.nn.functional as F

        labels = inputs.pop("probs")
        outputs = model(**inputs)
        loss = (TEMPERATURE**2) * F.kl_div(
            F.log_softmax(outputs.logits / TEMPERATURE, dim=-1),
            labels,
            reduction="batchmean",
        )
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        import torch

        with torch.no_grad():
            loss = self.compute_loss(model, dict(inputs))
        return loss.detach(), None, None


def main() -> None:
    """Train the student with KL distillation and save it to STUDENT_DIR."""
    import torch
    from transformers import TrainingArguments

    train_ds, val_ds = load_labeled()
    model, tokenizer = build_student()
    train_ds = tokenize(train_ds, tokenizer)
    val_ds = tokenize(val_ds, tokenizer)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    args = TrainingArguments(
        output_dir=STUDENT_DIR,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=5e-5,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=1,
    )
    trainer = DistillTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=build_distill_collator(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(STUDENT_DIR)
    tokenizer.save_pretrained(STUDENT_DIR)


if __name__ == "__main__":
    main()
