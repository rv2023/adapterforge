# M5 Piece 2 — bf16 Efficiency Experiment: Results

**JD line:** *"5%+ training step-time reduction, documented."* (Validation spec row E.)
**Result: bf16 cut step time ~41.6% (no 4-bit) / ~37.6% (4-bit) — ~8× past the ≥5% bar.**

- **Hardware:** RunPod, 1× **A40** (Ampere), CUDA-12.8 driver. ~minutes, ~$1 of compute.
- **Method:** `pipelines/efficiency_experiment.py` — 60-step throwaway runs (10 warmup
  discarded + 50 measured), batch 8, Qwen2.5-1.5B, same LoRA config; `StepTimer` callback
  (`cuda.synchronize()` + `perf_counter()` at `on_step_end`, median of the 50 steady-state
  steps). Only one knob changes per comparison.
- **Raw log:** `results/m5-efficiency.log`.
- Design + concepts: `docs/m5-floating-point-primer.md` (Parts 1–3).

## Runs (Trainer full-run, 60 steps incl. warmup)

| Run | config | runtime | samples/s |
|---|---|---|---|
| A1 | no-4bit, **fp32** | 40.92 s | 11.7 |
| A2 | no-4bit, **bf16** | 24.56 s | 19.6 |
| B1 | 4-bit, **fp32** | 44.81 s | 10.7 |
| B2 | 4-bit, **bf16** | 28.10 s | 17.1 |
| C  | no-4bit, bf16, **workers=4** | 23.55 s | 20.4 |

## Headline gains (StepTimer median, the measured deliverable)

| Comparison | gain |
|---|---|
| **bf16 vs fp32, no 4-bit** (A2 vs A1) | **41.6%** ✅ |
| **bf16 vs fp32, with 4-bit** (B2 vs B1) | **37.6%** ✅ |
| **dataloader workers 4 vs 0** (C vs A2) | **3.92%** |

The careful StepTimer median (41.6%) matches the Trainer's crude full-run number (~40%) →
the warmup-discard + synchronize + median methodology is sound.

## Interpretation

1. **Why ~40%, not ~5%?** On Ampere, **fp32 matmuls don't use Tensor Cores; bf16 does.** So
   bf16 is a ~1.7× speedup, not a marginal trim. The JD's "5%" is a conservative floor; real
   bf16-on-Ampere gains are far larger. Comfortably exceeded.
2. **4-bit gain (37.6%) < no-4bit gain (41.6%)** — as predicted: the per-layer dequant
   overhead eats a little of bf16's advantage.
3. **Dataloader workers: ~3.92%** — small, as predicted. The 1.5B model is mostly
   compute-bound, so prefetching only filled a small data stall (it *was* nonzero → there was
   a minor stall worth removing, but the GPU is the real bottleneck).
4. **"4-bit = memory, bf16 = speed" — proven in the data.** Same precision, 4-bit ON vs OFF:
   bf16 24.56 s (off) vs 28.10 s (on) → **4-bit is ~14% *slower***. 4-bit buys memory (fit a
   big model on a small GPU) but costs step time. Independent knobs; you pick per constraint.

## Honest notes / lessons

- **peak_vram_gb was logged only to the pod-local MLflow store**, which vanished on teardown
  (no remote server stood up yet). So the memory numbers weren't retained — which is exactly
  why the standing decision is `MLFLOW_TRACKING_URI` → a durable remote server (Tier 3).
  `results/m5-efficiency.log` (committed) is the retained record for this run.
- Pod setup gotchas hit + fixed (folded into `scripts/runpod_efficiency.sh` +
  `docs/runpod-workflow.md`): debian-managed packages block `pip` uninstall; a blanket
  `--ignore-installed` clobbered the CUDA torch with a cu13 wheel the A40 driver was too old
  for; and `data/` is gitignored so it must be shipped to the pod (not regenerated).
- Status: **JD efficiency bullet met and documented.** (If a hosted MLflow server is later
  stood up, re-run to capture `peak_vram_gb` durably and screenshot the compare view.)
