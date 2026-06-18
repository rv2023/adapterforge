# M0 Session Notes — GPU Foundations (COMPLETE)

Session: 2026-06-18. Owner: Karthik. Tutor-mode learning. Ran on Colab Tesla T4 (free).
**M0 status: COMPLETE — all 5 drills done.** Next: M5 (LLM fine-tuning + distillation).

## Why M0

The GPU/training foundations M5 assumes: read any GPU error, explain the CUDA stack
unaided, size any model to any GPU on paper. Karthik was new to all of these.

## Drills (all done, on a Colab T4 = ~15 GB VRAM)

| Drill | What he saw |
|---|---|
| M0.1 | CUDA stack (hardware→driver→toolkit→PyTorch→code); `nvidia-smi` reads bottom 2 layers. T4, driver 580, **nvidia-smi "CUDA 13.0" = driver max**, `torch.version.cuda = 12.8` = what PyTorch uses (12.8 ≤ 13.0). |
| M0.2 | Tensor→GPU; 4000² fp32 ≈ 62 MiB (elements×4). Timings: **CPU fp32 1187ms, GPU fp32 43ms (~27×), GPU fp16 12ms (~3.6×)**. GPU=parallel cores; fp16 faster (less bytes + Tensor Cores) AND half VRAM. `torch.cuda.synchronize()` needed (GPU is async). |
| M0.3 | Checkpoint = weights only. 25M-param Linear → **95 MB file** = params×4. Training VRAM **440 MiB ≈ 4.6× the file** (weights + gradients + Adam states + activations). `memory_allocated` (tensors) < `nvidia-smi` (adds ~100 MiB CUDA context). |
| M0.4 | Save **model + optimizer + epoch** → resume continues smoothly (loss kept descending). Restoring `opt.state_dict()` keeps Adam's momentum/variance → no kink. Matters for spot/preemptible GPUs in M5. |
| M0.5 | VRAM math (see below). |

## Core concepts taught (he was new to these)

- **Training loop in 5 beats:** weights (the learned numbers = params) → forward pass
  produces **activations** (scratch work) → **loss** (how wrong) → backward pass produces
  **gradients** (one nudge-direction per weight) → **optimizer** updates weights.
- Worked a toy by hand: 1 param `w`, `pred=w·x`, `x=2 y=10`, lr=0.1 → w: 1→4.2→4.84→4.97→5,
  loss 64→2.56→0.10→0 (the valley descent with real numbers).
- **Adam** = optimizer that also keeps **momentum + variance per weight** (= optimizer
  state, 2 extra numbers/param). This is what `opt.state_dict()` held in M0.4.
- **Precision = bytes per ONE number:** fp32=4, fp16/bf16=2, int8=1, int4=0.5.
- **"16 bytes/param" is NOT a precision** — it's the *stack of numbers* training keeps per
  weight: weight(fp16 2) + grad(fp16 2) + Adam momentum(fp32 4) + variance(fp32 4) +
  fp32 master(4) = 16. Storage/inference keeps just the weight (1 number).

## The VRAM formulas (M0 finish line — he can now size on paper)

```
storage / inference =  params × precision        (weights only)
full fine-tune      ≈  params × 16 bytes          (the 5-number stack, Adam mixed)
QLoRA               =  base params × 0.5 (int4, FROZEN → no grad/optim) + tiny adapter
```
Worked on the 15 GB T4:
| Scenario | VRAM | Fits 15 GB? |
|---|---|---|
| Qwen-1.5B fp16 inference | 1.5B×2 = **3 GB** | ✅ |
| Qwen-1.5B full fine-tune | 1.5B×16 = **24 GB** | ❌ |
| Qwen-1.5B QLoRA (int4 base) | 1.5B×0.5 + adapter ≈ **1–4 GB** | ✅ |
| 7B full fine-tune | 7B×16 = **112 GB** | ❌ (needs many GPUs) |

**Punchline (why M5 uses QLoRA):** full-FT Qwen-1.5B = 24 GB won't fit a cheap GPU;
QLoRA (4-bit frozen base + tiny adapter) ≈ 1–4 GB fits easily. Proven by arithmetic.

## Also done this session (housekeeping, uncommitted unless noted)

- **CI fix:** `adapter-sdk/tests/test_hf_adapter.py` now MOCKS `load_dataset`
  (monkeypatch where it's *used*) → unit tests run offline, fast (71s→3s); the 3
  network-dependent failures are gone. (Lesson: unit tests mock I/O; integration tests
  hit the real thing but never gate CI.)
- **M4 dedup refactor:** `register_model_with_dossier(model, test_df)` in
  register_baseline.py is the single source of truth; dag.py + loop.py call it.
- **New:** `requirements.txt`, real `make` targets (mlflow/dagster/lineage/control-plane/
  serving/drift/loop/register), `docs/USAGE.md` (scope + activity→command + walkthrough),
  gitignored runtime artifacts (audit.jsonl, predictions.jsonl, drift_report.html).

## Next session — M5 (LLM Fine-Tuning, Distillation, Distributed)

- ⚠️ **Costs money** (first time): RunPod ~$0.40/hr, one 2-GPU session, budget ~$10–12.
  Per cost guardrails, confirm $/hr before renting anything. Develop on a tiny model
  locally/Colab first, then the real run on rented GPU.
- Build: PhraseBank→instruction format → **QLoRA fine-tune Qwen-1.5B** (HF Trainer+PEFT)
  → register via control plane (same M3 gate); bf16 efficiency experiment (≥5% step-time,
  documented in MLflow); Ray Train + NCCL 2-GPU run + nccl-tests bus bandwidth; interconnect
  theory; distill LLM teacher → DistilBERT student → register; `distill.yml` workflow.
- The LLM adapter enters production through the SAME M3 gate, served by the SAME serving
  layer, monitored by the SAME M4 drift loop. The teacher/student pair feeds M8's router.

## Open threads
- Commit this session's work (CI test fix, dedup, requirements.txt, Makefile, USAGE.md,
  gitignore, m0/m4 notes). M1 SDK README still pending (rule 5). Marquez Docker may still
  be running (`make lineage-down`). Registry has demo cruft; production = v1.
