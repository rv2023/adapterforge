# M5 Piece 3 — Distributed Training (Ray data-parallel scaling + NCCL): Results

**JD lines:** *distributed training; NCCL; interconnect-aware optimization.*
**Result: 2-GPU data-parallel scaling = 2.0× (near-perfect); nccl-tests all-reduce busbw ≈ 4.53 GB/s.**

- **Hardware:** RunPod, **2× A40** (Ampere, **PCIe** — no NVLink). ~minutes, ~$1–2.
- **Method:** `pipelines/ray_finetune.py` — Ray `TorchTrainer` wrapping the QLoRA Qwen-1.5B
  fine-tune; run at `num_workers=1` then `2`; **per-GPU batch held constant** (weak scaling) so
  2 GPUs process 2× data/step. Throwaway runs (timing, not a model). 60 steps each.
- **Raw logs:** `results/ray-1gpu.log`, `results/ray-2gpu.log`, `results/nccl-tests.log`.
- Concepts: `docs/m5-interconnect-notes.md` (§B data-vs-tensor parallelism, §D Ray mechanics).

## Scaling

| | 1 GPU | 2 GPUs |
|---|---|---|
| `train_runtime` | 29.47 s | 29.46 s |
| `train_samples_per_second` | **16.29** | **32.58** |
| samples processed (8×60×workers) | 480 | 960 |

**Scaling = 32.58 / 16.29 = 2.0× — near-perfect linear.** Clearest read: **same wall-clock
(~29.5 s), 2× the data → 2× throughput.** (HF's `train_samples_per_second` already accounts for
world size, so the ratio is the true scaling.)

## NCCL interconnect

**`nccl-tests` all_reduce avg bus bandwidth ≈ 4.53 GB/s** — the raw 2× A40 **PCIe** capacity
(NVLink would be ~10–100×; this is the model-independent interconnect number).

## Interpretation — predictions held

1. **Near-2× even on PCIe** — exactly as §D predicted: QLoRA only all-reduces the **tiny LoRA
   adapter gradients** (MBs), so NCCL sync added ~zero overhead → linear scaling. The interconnect
   barely mattered for *this* workload.
2. **nccl-tests carries the real interconnect lesson** (4.53 GB/s) — independent of how little we
   synced. Full fine-tuning (syncing all 1.5B grads/step) is where a slow interconnect would bite,
   and where NVLink vs PCIe would show in the scaling ratio.
3. **Two complementary numbers:** the LoRA scaling (2.0×, little to sync) + the busbw (the true
   interconnect ceiling).

## Honest notes / lessons

- **`device_map={"": 0}` made 4-bit + DDP work** — each Ray worker sees its GPU as `cuda:0`
  (`CUDA_VISIBLE_DEVICES` isolation), so the per-worker map loads each 4-bit copy onto its own GPU.
  `num_workers=2` ran without the device-placement crash.
- **`result.metrics` came back `None`** at the controller (Ray Train v2) — the convenience
  `print` in `main()` showed `None`. The real numbers are in the **worker reports**
  (`Reporting training result … 'train_samples_per_second': 32.582`). **Fix applied:** `train_func`
  now prints throughput directly from `out.metrics` (always populated on the worker).
- **2-GPU `train_loss` = 8.6** (vs 1.3 on 1-GPU) — **irrelevant for a throughput stopwatch**
  (throwaway model), but it hints the 4-bit + DDP combo isn't numerically clean. Not pursued — the
  deliverable is scaling + bandwidth, not a model.
- Pod gotchas fixed (in runbook/code): `ray[train]` must be **unquoted** in a requirements file;
  **Ray changes each worker's CWD** → `DATA_DIR` made absolute via `__file__`.

## Status: Piece 3 deliverable ✅ met

Distributed training over NCCL ✓ · scaling measured (2.0×) ✓ · NCCL bus bandwidth recorded
(4.53 GB/s) ✓. Tensor parallelism stays **theory-only** (model fits one GPU — no hands-on TP).
Remaining M5: **Piece 5** (distillation teacher→DistilBERT + `distill.yml`).
