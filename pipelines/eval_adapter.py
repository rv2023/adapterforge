"""M5 Piece 1 — evaluate the fine-tuned LoRA adapter on the SEALED test set.

Produces the number that matters: macro-F1 on the same frozen 465-row test set
M2 used, compared against the locked baseline bar (0.6885). Run on Colab in the
SAME session as training (adapter on disk, base cached -> reloads in ~1 min).

The eval prompt MUST match the training prompt: feed [system, user] with the
assistant turn left for the model to generate. Greedy decoding, parse the word.
"""

import json

import torch
from peft import PeftModel
from sklearn.metrics import f1_score  # reuse M2's metric — same macro-F1
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "models/fpb-lora"
USE_4BIT = True
TEST_FILE = "data/instruction/test.jsonl"
BASELINE_F1 = 0.6885
LABELS = ["bullish", "bearish", "neutral"]


def load_model_and_tokenizer():
    """Load the 4-bit base, then attach the trained adapter on top."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_kwargs = {}
    if USE_4BIT:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    return model, tokenizer


def predict_one(model, tokenizer, messages) -> str:
    """Given [system, user] messages, generate the model's label word.

    `messages` is the chat WITHOUT the assistant turn.
    """
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    prompt_len = ids.shape[-1]
    out = model.generate(ids, max_new_tokens=5, do_sample=False)
    text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip().lower()
    for label in LABELS:
        if text == label or text.startswith(label):
            return label
    return "unknown"


def main():
    model, tokenizer = load_model_and_tokenizer()
    golds = []
    preds = []

    with torch.inference_mode(), open(TEST_FILE, encoding="utf-8") as f:
        for line in f:
            messages = json.loads(line)["messages"]
            golds.append(messages[-1]["content"].strip().lower())
            preds.append(predict_one(model, tokenizer, messages[:-1]))

    macro_f1 = f1_score(golds, preds, average="macro", labels=LABELS)
    print(f"macro_f1={macro_f1:.4f} baseline={BASELINE_F1:.4f} beat_bar={macro_f1 > BASELINE_F1}")


if __name__ == "__main__":
    main()
