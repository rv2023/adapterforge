"""M5 Piece 3 — data-parallel QLoRA fine-tune via Ray Train (1-GPU vs 2-GPU scaling).

Goal: run the Qwen-1.5B QLoRA fine-tune across N GPUs with Ray Train, measure throughput
(samples/sec) at num_workers=1 and num_workers=2, and see how close 2 GPUs get to a true
2× (the gap = NCCL sync overhead, explained by the interconnect / nccl-tests bandwidth).

Run it twice on the 2-GPU pod:
    python -m pipelines.ray_finetune 1     # single GPU baseline
    python -m pipelines.ray_finetune 2     # data-parallel across 2 GPUs

Concepts + design: docs/m5-interconnect-notes.md, docs/m5-floating-point-primer.md.
Reuses the LoRA config from pipelines.finetune.

NOTE (the finicky bit we flagged): bitsandbytes 4-bit + multi-GPU DDP can fight on device
placement. If num_workers=2 errors on placement, fall back to plain LoRA bf16 (USE_4BIT=False)
— the scaling/NCCL lesson is identical either way.
"""

import sys
from pathlib import Path

import ray
from ray.train import ScalingConfig

import datasets
import torch
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from ray.train.torch import TorchTrainer
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer

from pipelines.finetune import build_lora_config

# --- held constant across the 1-GPU and 2-GPU runs ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
USE_4BIT = True            # real QLoRA recipe; flip to False if 4-bit+DDP fights us
# ABSOLUTE path — Ray changes each worker's CWD to its own session dir, so a relative
# path would resolve under /tmp/ray/... and not find the data. Anchor to the repo via __file__.
DATA_DIR = str(Path(__file__).resolve().parents[1] / "data" / "instruction")
PER_DEVICE_BATCH = 8       # per-GPU batch — HELD CONSTANT so 2 GPUs = 2x data/step (weak scaling)
MAX_STEPS = 60             # short burst; we want throughput, not a model


def build_model_and_tokenizer():
    """Load Qwen for QLoRA on THIS worker's GPU.

    Mirrors finetune.build_model_and_tokenizer but pinned to the real 1.5B + 4-bit config.
    Ray Train sets CUDA_VISIBLE_DEVICES so each worker sees only its own GPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_kwargs = {}
    if USE_4BIT:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def train_func(config):
    """Runs on EACH Ray worker (one per GPU). This is the per-GPU training job.

    Ray Train has already: spawned this worker, pinned it to a GPU, and set up the
    torch process group (so DDP/NCCL all-reduce works across workers).
    """
    ds = datasets.load_dataset("json", data_files={"train": f"{DATA_DIR}/train.jsonl"})
    model, tokenizer = build_model_and_tokenizer()
    sft_config = SFTConfig(
        output_dir="/tmp/ray-qlora",
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=1,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        peft_config=build_lora_config(),
        processing_class=tokenizer,
        callbacks=[RayTrainReportCallback()],
    )
    trainer = prepare_trainer(trainer)
    out = trainer.train()
    ray.train.report(out.metrics)


def main(num_workers: int):
    """Launch the data-parallel run across `num_workers` GPUs and report throughput.

    Args:
        num_workers: 1 (single-GPU baseline) or 2 (data-parallel).
    """
    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=True),
    )
    result = trainer.fit()
    throughput = result.metrics.get("train_samples_per_second")
    print(f"num_workers={num_workers} train_samples_per_second={throughput}")


if __name__ == "__main__":
    # num_workers from argv: `python -m pipelines.ray_finetune 1`  (default 1)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(n)
