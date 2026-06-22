"""M5 Piece 5, Step 2 — teacher soft-labels the headlines (verbalizer + temperature).

For each headline: build the training prompt -> ONE forward pass -> take the next-token
logits -> pick the 3 label-token logits (the verbalizer) -> softmax/T -> soft label.
Output: {"text", "probs":[p_bullish,p_bearish,p_neutral]} per line.

Runs on the GPU pod (the teacher is 4-bit Qwen-1.5B). Reuse eval_adapter.py for the
model-load + chat-prompt builder. Concepts: docs/m5-distillation-concepts.md (§5–6).

    python -m pipelines.distill_label
"""

# LABELS ORDER IS LOAD-BEARING: the same index order must be used in Step 2 (here),
# Step 3 (training target), and Step 4 (eval). Pick it once.
LABELS = ["bullish", "bearish", "neutral"]
TEMPERATURE = 2.0
BATCH_SIZE = 64
IN = "data/distill/headlines.jsonl"
OUT = "data/distill/labeled.jsonl"


def load_teacher():
    """Load base Qwen-1.5B (4-bit) + the trained LoRA adapter (models/fpb-lora) + tokenizer.

    Reuse eval_adapter.py's loading (PeftModel.from_pretrained) so it's identical to how
    the teacher was evaluated. Put model in eval() mode.
    """
    # TODO: return (model, tokenizer)
    ...


def label_token_ids(tokenizer) -> list[int]:
    """Return the 3 first-token IDs for LABELS, as the model actually EMITS them.

    THE make-or-break step (docs/m5-distillation-concepts.md §5):
      - encode each label as it appears after the prompt (usually with a LEADING SPACE,
        e.g. " bullish") and take the FIRST token id;
      - assert all 3 ids are DISTINCT (print them once to eyeball);
      - return them in LABELS order.
    """
    # TODO
    ...


def soft_label_batch(model, tokenizer, texts: list[str], class_ids: list[int]):
    """One forward pass over a batch of headlines -> Nx3 soft-label tensor (on CPU).

    Steps:
      - build chat prompts (same as training; apply_chat_template(add_generation_prompt=True))
      - tokenize + pad (left or right consistently), build attention_mask, move to device
      - with torch.no_grad(): logits = model(**inputs).logits
      - last = logits[:, -1, :]                       # next-token distribution (~150k vocab)
      - class_logits = last[:, class_ids]             # pick the 3  -> Nx3
      - probs = softmax(class_logits / TEMPERATURE, dim=-1)
      - return probs.float().cpu()
    NOTE: with left-padding, [:, -1, :] is the real last token for every row; with
    right-padding you'd index each row's true last position instead. Be deliberate.
    """
    # TODO
    ...


def main() -> None:
    """Soft-label every headline in IN (batched) and write {text, probs} to OUT."""
    # TODO:
    #   model, tokenizer = load_teacher(); class_ids = label_token_ids(tokenizer)
    #   read IN (jsonl) -> list of texts
    #   loop in BATCH_SIZE chunks -> soft_label_batch -> collect rows {"text":t,"probs":[...]}
    #   write OUT as jsonl; print count
    ...


if __name__ == "__main__":
    main()
