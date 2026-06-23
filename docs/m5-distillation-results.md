# M5 Piece 5 — Distillation: Results

**Goal:** distill the QLoRA Qwen-1.5B teacher into a cheap DistilBERT student.
**Result: student macro-F1 = 0.7556** — beats the sklearn baseline (0.6885) by ~7 pts;
~9 pts under the teacher (0.8477), so it **misses** the "within 2–3 pts of teacher" bar.

| Model | Architecture | macro-F1 (frozen FPB test) |
|---|---|---|
| baseline | TF-IDF + LogReg | 0.6885 |
| **student** | **DistilBERT SEQ_CLS, ~66M** | **0.7556** |
| teacher | QLoRA Qwen2.5-1.5B | 0.8477 |

## Method

- **Data:** 23,279 unlabeled Alpha Vantage headlines (Step 1).
- **Teacher soft-labels** (Step 2): verbalizer — one forward pass → next-token logits → pick the
  `bullish`/`bearish`/`neutral` token logits (ids `bull`/`bear`/`neutral`, prefix `''`) → softmax/T
  (T=2). Validated 20/20 against the teacher's actual generation before labeling all 23k.
- **Student** (Step 3): DistilBERT `SEQ_CLS` + **KL-divergence** to the teacher's soft labels
  (`T² · KL`), 3 epochs, `load_best_model_at_end` on eval_loss. Runs on RunPod A40.
- **Eval** (Step 4): macro-F1 on the **frozen FPB test set** (same sealed exam as baseline/teacher).
- **Registered** (Step 5) as `fpb-student` — registry now holds 3 heterogeneous models.

## Interpretation — why 0.7556, not within 2–3 pts

**Most likely: domain shift.** We distilled on **Alpha Vantage news headlines** but test on **FPB**.
- The teacher was trained on FPB, so its labels on AV news are noisier (AV is out-of-domain for it).
- The student learns AV-domain patterns that don't fully transfer to the FPB exam.

This is the honest cost of choosing **live AV data** over **in-domain FPB text** for distillation. It
*works* (beats baseline, ~20× cheaper, real heterogeneous model) but caps below the teacher.

**Improvement paths (not done):** distill on in-domain FPB train text; more epochs / tune T; bigger
student; more/cleaner labels.

## The durability incident (the real MLOps lesson)

Mid-run we discovered the **original 1.5B teacher did not exist** — every adapter on the laptop and
in MLflow was **0.5B** (the Colab teacher was ephemeral; the laptop copy got overwritten by a 0.5B
dev run). We **retrained** the 1.5B teacher on the pod (`AF_MODE=real`, finetune.py — the real config
is now in git). Then the **pod crashed**, wiping the `/tmp` venv (and risking the retrained teacher).
Fixes applied + lessons:
- `finetune.py` `AF_MODE` switch → real config lives in git (was ephemeral-only).
- `device_map={"":0}` for 4-bit load on a single GPU.
- Robust verbalizer (single-render divergence — the cross-render prefix check broke on BPE seams).
- **The fix that matters: durable model storage.** `models/` is gitignored + GPU is ephemeral →
  models must go to **DVC/S3** (TODO), not "live on the laptop." Also: put the venv on the
  persistent `/workspace`, and pull models off the pod the moment they're produced.

## Status

Piece 5 deliverable **met in spirit** (a working, registered, ~20×-cheaper distilled student that
beats the baseline) — though below the aggressive teacher-gap bar, with a documented cause + path.

**DECISION (2026-06-22): ACCEPT 0.7556 and move on** — the domain-shift finding *is* the learning;
not re-distilling in-domain. **M5 is complete** (Pieces 1–3, 5; tensor parallelism theory-only).
Durable storage done (teacher + student in DVC/S3). Next milestone: **M6 (vLLM serving)**.
