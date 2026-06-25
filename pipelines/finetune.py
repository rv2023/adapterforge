"""M5 Piece 1 — QLoRA fine-tune of Qwen on the chat-format PhraseBank data.

Same code runs three places; only the config constants change:
  - local CPU smoke test:  small model, 4-bit OFF, MAX_STEPS=4
  - Colab T4 real run:     Qwen-1.5B, 4-bit ON, full epochs
  - M7 EKS job:            same as Colab, scheduled by Kubernetes
"""

import os

import datasets
import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# --- config: dev (laptop CPU smoke) vs real (GPU). Select with AF_MODE=real. ---
# Closes the long-open tech debt: the real config now lives in git, not ephemeral edits.
AF_MODE = os.getenv("AF_MODE", "dev")
if AF_MODE == "real":
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    USE_4BIT = True
    MAX_STEPS = -1          # -1 -> train by epochs, not a step cap
    BATCH_SIZE = 16
    NUM_EPOCHS = 3
else:
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    USE_4BIT = False
    MAX_STEPS = 2
    BATCH_SIZE = 1
    NUM_EPOCHS = 1
# Task-specific paths, env-overridable so the SAME SFT pipeline trains other adapters
# (e.g. the M8 summarizer: AF_DATA_DIR=data/instruction_summ AF_ADAPTER_DIR=models/fpb-summarizer).
DATA_DIR = os.getenv("AF_DATA_DIR", "data/instruction")
ADAPTER_DIR = os.getenv("AF_ADAPTER_DIR", "models/fpb-lora")


def load_datasets():
    """Load the three JSONL splits as a HF DatasetDict."""
    return datasets.load_dataset(
        "json",
        data_files={
            "train": f"{DATA_DIR}/train.jsonl",
            "val": f"{DATA_DIR}/val.jsonl",
        },
    )


def build_model_and_tokenizer():
    """Ingredient 1 (the 'Q') + tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_kwargs = {}
    if USE_4BIT:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["device_map"] = {"": 0}   # load 4-bit straight onto the GPU
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def build_lora_config():
    """Ingredient 2 (the 'LoRA'): which layers get adapters, and how big."""
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def build_trainer(model, tokenizer, lora_config, ds):
    """Ingredients 4 + 5: SFTConfig (knobs) + SFTTrainer (loop)."""
    cfg = SFTConfig(
        output_dir=ADAPTER_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_steps=MAX_STEPS,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=1,
        bf16=USE_4BIT,
        report_to="none",
    )
    return SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )


def main():
    ds = load_datasets()
    model, tokenizer = build_model_and_tokenizer()
    trainer = build_trainer(model, tokenizer, build_lora_config(), ds)
    trainer.train()
    trainer.save_model(ADAPTER_DIR)
    adapter_size_bytes = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, files in os.walk(ADAPTER_DIR)
        for name in files
    )
    print(
        f"Adapter saved to {ADAPTER_DIR} "
        f"({adapter_size_bytes} bytes, {adapter_size_bytes / 1024 / 1024:.2f} MiB)"
    )



if __name__ == "__main__":
    main()
