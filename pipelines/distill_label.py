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
    from pipelines.eval_adapter import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def label_token_ids(tokenizer) -> tuple[str, list[int]]:
    """Return the shared emitted prefix plus 3 label-token IDs for LABELS.

    THE make-or-break step (docs/m5-distillation-concepts.md §5):
      - ask the chat template how each assistant answer is actually tokenized;
      - if all labels share an emitted prefix (e.g. newline), score after that prefix;
      - assert all 3 ids are DISTINCT (print them once to eyeball);
      - return them in LABELS order.
    """
    from pipelines.instruction_format import INSTRUCTION

    messages = [
        {"role": "system", "content": "You are a financial sentiment classifier."},
        {"role": "user", "content": INSTRUCTION + "Acme reported stable revenue."},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    continuations = []
    for label in LABELS:
        full_ids = tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": label}],
            add_generation_prompt=False,
            tokenize=True,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Chat template assistant answer does not share generation prefix")
        continuation = full_ids[len(prompt_ids) :]
        if not continuation:
            raise ValueError(f"No continuation tokens found for label {label!r}")
        continuations.append(continuation)

    prefix_len = 0
    while all(len(ids) > prefix_len for ids in continuations):
        token_id = continuations[0][prefix_len]
        if any(ids[prefix_len] != token_id for ids in continuations[1:]):
            break
        prefix_len += 1

    ids = []
    for label, continuation in zip(LABELS, continuations):
        if len(continuation) <= prefix_len:
            raise ValueError(f"Label {label!r} has no class token after shared prefix")
        ids.append(continuation[prefix_len])

    if len(set(ids)) != len(ids):
        details = {
            label: tokenizer.convert_ids_to_tokens(cont)
            for label, cont in zip(LABELS, continuations)
        }
        raise ValueError(f"Label token ids must be distinct after shared prefix: {details}")

    prefix_ids = continuations[0][:prefix_len]
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    print(
        "verbalizer="
        f"prefix={prefix_text!r} "
        f"label_token_ids={dict(zip(LABELS, ids))} "
        f"tokens={dict(zip(LABELS, tokenizer.convert_ids_to_tokens(ids)))}"
    )
    return prefix_text, ids


def soft_label_batch(
    model, tokenizer, texts: list[str], verbalizer_prefix: str, class_ids: list[int]
):
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
    import torch

    from pipelines.instruction_format import INSTRUCTION

    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a financial sentiment classifier."},
                {"role": "user", "content": INSTRUCTION + text},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        + verbalizer_prefix
        for text in texts
    ]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        last = model(**inputs).logits[:, -1, :]
        class_ids_tensor = torch.as_tensor(class_ids, device=last.device)
        class_logits = last.index_select(dim=-1, index=class_ids_tensor)
        probs = torch.softmax(class_logits / TEMPERATURE, dim=-1)
    return probs.float().cpu()


def main() -> None:
    """Soft-label every headline in IN (batched) and write {text, probs} to OUT."""
    import json
    from pathlib import Path

    model, tokenizer = load_teacher()
    verbalizer_prefix, class_ids = label_token_ids(tokenizer)

    with Path(IN).open(encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f if line.strip()]

    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = lambda x, **_: x

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    with out.open("w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(texts), BATCH_SIZE), total=total_batches):
            batch_texts = texts[start:start + BATCH_SIZE]
            batch_probs = soft_label_batch(
                model, tokenizer, batch_texts, verbalizer_prefix, class_ids
            ).tolist()
            for text, probs in zip(batch_texts, batch_probs):
                f.write(json.dumps({"text": text, "probs": probs}) + "\n")
            count += len(batch_texts)

    print(f"wrote {count} labeled headlines -> {OUT}")


if __name__ == "__main__":
    main()
