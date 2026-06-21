"""M5 Piece 2 — bf16 efficiency experiment (the JD "≥5% step-time" bullet).

Goal: MEASURE training step time, not produce a model. We run 4 short throwaway
bursts and compare bf16 vs fp32 step time in two conditions:

              fp32      bf16
  4-bit OFF   A1        A2     ← clean isolated bf16 effect (JD wording)
  4-bit ON    B1        B2     ← apples-to-apples with the real QLoRA recipe

Each run = 10 warmup steps (discarded) + 50 measured steps, model discarded.
Everything is logged to MLflow; the headline numbers are the % step-time gains
(bf16 vs fp32) for each row.

Design + reasoning: docs/m5-floating-point-primer.md (Part 3).
Mirrors the building blocks in pipelines/finetune.py but PARAMETERIZED per run.
"""

import gc
import time
import sys
from pathlib import Path
from statistics import median
from dataclasses import dataclass

import datasets
import mlflow
import torch
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipelines.finetune import build_lora_config

# --- experiment constants (held constant across all 4 runs) ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = "data/instruction"
BATCH_SIZE = 8
N_WARMUP = 10          # discarded — CUDA/cuDNN/allocator warmup
N_MEASURED = 50        # timed steady-state steps
MLFLOW_EXPERIMENT = "m5-efficiency"


@dataclass
class RunConfig:
    """One cell of the 2x2 grid.

    Attributes:
        label: human name for the run, e.g. "A2_no4bit_bf16".
        use_4bit: whether the base is quantized (the A-row vs B-row toggle).
        precision: "fp32" or "bf16" (the column toggle).
        num_workers: DataLoader worker processes.
    """
    label: str
    use_4bit: bool
    precision: str
    num_workers: int = 0


# the 5 runs, in execution order A1, A2, B1, B2, C (C = dataloader-workers lever vs A2)
RUNS = [
    RunConfig(label="A1_no4bit_fp32", use_4bit=False, precision="fp32"),
    RunConfig(label="A2_no4bit_bf16", use_4bit=False, precision="bf16"),
    RunConfig(label="B1_4bit_fp32", use_4bit=True, precision="fp32"),
    RunConfig(label="B2_4bit_bf16", use_4bit=True, precision="bf16"),
    RunConfig(label="C_no4bit_bf16_workers4", use_4bit=False, precision="bf16", num_workers=4),
]


class StepTimer(TrainerCallback):
    """The stopwatch: record steady-state per-step wall time.

    HF calls on_step_end after each optimizer step. GPU work is async, so we must
    synchronize before reading the clock or we'd time queue-submission, not compute.
    We keep one anchor timestamp at the end of warmup, then measure the next
    N_MEASURED step-end gaps.
    """

    def __init__(self):
        self.timestamps = []
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.step_count += 1
        if self.step_count >= N_WARMUP:
            self.timestamps.append(time.perf_counter())

    def median_step_time(self) -> float:
        """Median seconds/step over the measured window (warmup dropped).

        Returns:
            Median of N_MEASURED per-step deltas after N_WARMUP warmup steps.
        """
        deltas = [
            end - start
            for start, end in zip(self.timestamps, self.timestamps[1:])
        ]
        return median(deltas)


def build_model_and_tokenizer(cfg: RunConfig):
    """Load Qwen at the precision/quantization this run requires.

    Maps the run's knobs to the loader:
      - 4-bit ON:  BitsAndBytesConfig(load_in_4bit=True,
                     bnb_4bit_compute_dtype = bf16 if precision=="bf16" else fp32),
                   then prepare_model_for_kbit_training(model)
      - 4-bit OFF: from_pretrained(torch_dtype = bf16 if precision=="bf16" else fp32)
    """
    dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_kwargs = {}
    if cfg.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def build_trainer(model, tokenizer, ds, cfg: RunConfig, timer: StepTimer):
    """SFTTrainer wired for a 60-step throwaway timing run.

    Key SFTConfig knobs:
      - max_steps = N_WARMUP + N_MEASURED
      - per_device_train_batch_size = BATCH_SIZE, gradient_accumulation_steps = 1
      - bf16 = (cfg.precision == "bf16")   # autocast for adapter/activations
      - report_to="none"  (we log to MLflow ourselves, not via the Trainer)
      - callbacks=[timer]
    Reuse the LoRA config from finetune.build_lora_config().
    """
    sft_config = SFTConfig(
        output_dir=f"/tmp/adapterforge-efficiency-{cfg.label}",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_steps=N_WARMUP + N_MEASURED,
        num_train_epochs=1,
        logging_steps=1,
        bf16=(cfg.precision == "bf16"),
        dataloader_num_workers=cfg.num_workers,
        report_to="none",
    )
    return SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        peft_config=build_lora_config(),
        processing_class=tokenizer,
        callbacks=[timer],
    )


def run_one(cfg: RunConfig, ds) -> dict:
    """Execute one run and return its measured metrics.

    Returns a dict like:
        {"median_step_time_s": float, "samples_per_sec": float, "peak_vram_gb": float}
    """
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    timer = StepTimer()
    model = tokenizer = trainer = None
    try:
        model, tokenizer = build_model_and_tokenizer(cfg)
        trainer = build_trainer(model, tokenizer, ds, cfg, timer)
        trainer.train()

        median_step_time_s = timer.median_step_time()
        peak_vram_gb = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available()
            else 0.0
        )
        return {
            "median_step_time_s": median_step_time_s,
            "samples_per_sec": BATCH_SIZE / median_step_time_s,
            "peak_vram_gb": peak_vram_gb,
        }
    finally:
        del trainer, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Run the 2x2 grid, log each run to MLflow, print the two % gains.

    Steps:
      - load the dataset once (datasets.load_dataset on train.jsonl)
      - mlflow.set_experiment(MLFLOW_EXPERIMENT)
      - for each RunConfig: start an MLflow run, log params (precision, use_4bit,
        batch_size, n_warmup, n_measured, model), call run_one(), log metrics
      - after all 4: gain_A = pct_faster(A1, A2), gain_B = pct_faster(B1, B2); print both
    """
    ds = datasets.load_dataset("json", data_files={"train": f"{DATA_DIR}/train.jsonl"})
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = {}
    for cfg in RUNS:
        with mlflow.start_run(run_name=cfg.label):
            mlflow.log_params(
                {
                    "precision": cfg.precision,
                    "use_4bit": cfg.use_4bit,
                    "batch_size": BATCH_SIZE,
                    "n_warmup": N_WARMUP,
                    "n_measured": N_MEASURED,
                    "model": MODEL_NAME,
                }
            )
            metrics = run_one(cfg, ds)
            mlflow.log_metrics(metrics)
            results[cfg.label] = metrics

    def pct_faster(fp32_label: str, bf16_label: str) -> float:
        fp32_step_time = results[fp32_label]["median_step_time_s"]
        bf16_step_time = results[bf16_label]["median_step_time_s"]
        return (fp32_step_time - bf16_step_time) / fp32_step_time * 100

    gain_A = pct_faster("A1_no4bit_fp32", "A2_no4bit_bf16")
    gain_B = pct_faster("B1_4bit_fp32", "B2_4bit_bf16")
    gain_dl = pct_faster("A2_no4bit_bf16", "C_no4bit_bf16_workers4")
    print(f"bf16 step-time gain without 4-bit: {gain_A:.2f}%")
    print(f"bf16 step-time gain with 4-bit: {gain_B:.2f}%")
    print(f"dataloader-workers step-time gain (bf16, no 4-bit): {gain_dl:.2f}%")


if __name__ == "__main__":
    main()
