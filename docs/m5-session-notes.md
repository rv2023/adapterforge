# M5 Session Notes — LLM Fine-Tuning (Piece 1: DONE — registered + promoted)

Sessions: 2026-06-18/19/21. Owner: Karthik. Tutor-mode learning.
**M5 Piece 1: FULLY DONE** — QLoRA adapter beats the baseline AND is governed production
(macro-F1 **0.8477** vs **0.6885**, registered `fpb-sentiment` v14, promoted via M3 gate).
**M5 Piece 2: DESIGNED + CODED, not yet run** — bf16 efficiency experiment
(`pipelines/efficiency_experiment.py`) ready; GPU work moved to **RunPod (Colab dropped)`;
remaining = add the dataloader run, then rent the pod. Next: Pieces 3 (Ray/NCCL), 5 (distill).
Session 4 (2026-06-21) below covers the number-format deep dive + Piece-2 design + RunPod switch.

## Session 4 (2026-06-21) — Piece 2 design + number-format deep dive + RunPod switch

Goal: set up the **bf16 efficiency experiment** (JD "≥5% step-time"). Most of the session
was a from-scratch teaching of how numbers are stored (Karthik asked to go all the way to
the metal), then the experiment design, then standardizing GPU work on RunPod.

**Teaching delivered (captured durably, not just in chat):**
- `docs/m5-floating-point-primer.md` — **the big one**, three parts:
  - Part 1: bits → 2^N values; sign/exponent/mantissa as scientific notation (exponent =
    range/scale, mantissa = precision); the `×0.301` base-2↔base-10 bridge; fp32 range
    (~10^±38) + precision (~7 digits) derived; the **ExMy** naming; fp32/bf16/fp16/fp8
    (E4M3/E5M2) table; worked **8.1** forward+backward, **8.2** contrast, 8.1 across all
    formats; why training picks bf16 (gradients need *range*, not digits).
  - Part 2: 4-bit quantization = **buckets** (T-shirt-size menu; "bucket #6" = 6th menu
    item; lossy like JPEG not ZIP); menu values come from the data's range (NF4 packs near
    zero); how-many-buckets = 2^bits; why uncompress (labels aren't numbers; only a sliver
    at a time); the three coexisting precisions in QLoRA.
  - Part 3: the two Qwen sizes are different models (dev 0.5B vs real 1.5B); frozen base +
    trainable adapter (textbook + sticky-notes); what a "layer" is + per-step uncompress;
    Piece 1 (quality/F1) vs Piece 2 (speed, throwaway); **4-bit = memory opt, bf16 = speed
    opt, independent**; the LOCKED experiment design.
- `docs/m5-interconnect-notes.md` — STUB for Piece 3 (RoCE/IB hierarchy, GPUDirect RDMA,
  NCCL knobs, tensor- vs data-parallelism, the honest single-node-only gap). Numbers TBD on
  the 2-GPU pod.

**Piece 2 experiment built (`pipelines/efficiency_experiment.py`) — Karthik wrote it, reviewed:**
- 2×2 grid + 1 extra: fp32 vs bf16 × 4-bit OFF/ON (rows A/B), plus a **dataloader-workers**
  pair (compare A2 `num_workers=0` vs a new `num_workers=4` run) — the 3rd JD lever.
- `StepTimer(TrainerCallback)`: `cuda.synchronize()` then `perf_counter()` at `on_step_end`;
  anchors at end of warmup (append once `step_count >= N_WARMUP`) → 50 clean steady-state
  deltas; median. **Throwaway runs** (no save), 10 warmup + 50 measured, batch 8.
  Metrics → `median_step_time_s`, `samples_per_sec`, `peak_vram_gb`.
- Review fixes Karthik made: memory cleanup between runs (`del`+`gc.collect()`+`empty_cache()`,
  ordered so the previous model is freed before the next loads — avoids 2 models at once);
  off-by-one in the timer; dead code removed; `import gc`.
- **Honest expectations documented:** B (4-bit on) may show <5% (dequant overhead masks it);
  dataloader pair likely ~0% (1.5B is compute-bound, not data-starved) — both are *findings*.

**Decisions locked this session:**
- **GPU runs on RunPod; Colab dropped.** 3-tier operating model written: `docs/operating-model.md`
  + summarized in CLAUDE.md. Tier 1 local/CPU (free, ~70%), Tier 2 RunPod (per-min, M5/M6
    learning), Tier 3 own EKS cluster (per-node-hr, M7/M8 platform work). Cost guardrail on 2 & 3.
- **MLflow logs via `MLFLOW_TRACKING_URI` → remote/cloud server** (operator mindset: prod
  tracking lives in cloud infra). No code change (mlflow reads the env var). Honest caveat:
  that server is a Tier-3 component not yet stood up → `results/m5-efficiency.log` is the
  interim safety net.
- **Kubeflow stays an M8 stretch** (Dagster primary) — confirmed, no change.
- RunPod plumbing (boilerplate): `requirements-gpu.txt` (no torch — keep template's CUDA
  build; bitsandbytes here), `scripts/runpod_efficiency.sh` (CUDA gate + auto torch-wheel
  fallback + data regen + tee to log), `docs/runpod-workflow.md` (rent→run→**teardown**, cost).

**Concepts Karthik now owns (this session):** floating point end-to-end (forward+backward,
all formats); why bf16 (range over precision for gradients); 4-bit = lossy bucketing, not a
float; the three QLoRA precisions and why fp32-vs-bf16 needs 4-bit OFF to be clean; "4-bit =
memory, bf16 = speed" are independent knobs (resolves the "but prod uses 4-bit" worry — we
measure both A and B); honest step-time measurement (warmup, sync, median, hold-constant);
dataloader/num_workers (prefetch; helps only if GPU is data-starved); where the MLOps
lifecycle runs (GPU only for train + LLM inference; ~70% is CPU); the 3-tier compute model.

**Next:** add the one `num_workers=4` run (skeleton given) → review → **rent the RunPod pod**
(confirm $/hr ~$0.40, Ampere+ ≥24 GB, set MLFLOW_TRACKING_URI) → run `scripts/runpod_efficiency.sh`
→ record the % gains → teardown. Then Piece 3 (Ray/NCCL), Piece 5 (distillation).

## Session 3 (2026-06-19) — register + promote the adapter (Piece 1 close)

Goal: govern the LLM like any production model — register it with a dossier, push it
through the gated `/promote`. All laptop work, no GPU.

**What got built / changed:**
- `models/fpb-lora/eval_metrics.json` = `{"test_f1": 0.8477, "n_test": 465}` — the F1 is now
  a persisted artifact next to the adapter, not a number remembered off ephemeral Colab.
- `eval_adapter.py` — `main()` now RETURNS the f1 and writes `eval_metrics.json` (full precision,
  no rounding — it's a stored measurement) so future evals auto-emit it. The v14 json was
  hand-written before this change (no GPU here to re-run the eval).
- `register_baseline.py` refactored (option A — parametrize, less duplication):
  - `register_model_with_dossier(test_df, test_f1, log_and_register)` — model-agnostic core.
    Computes eval_hash + commit, opens the run, calls the caller's log+register callback,
    stamps the 4 dossier tags. (Signature changed from `(model, test_df)`.)
  - `register_sklearn(model, test_df)` — thin wrapper: passes `evaluate(...)` + a sklearn
    log lambda. Existing callers (`register_baseline`, `loop.py:46`, `dag.py:49`) updated to it.
  - `register_adapter(adapter_dir, test_df)` — reads eval_metrics.json, **n_test cross-check**
    (explicit `raise ValueError`, not bare assert), logs the adapter dir as artifacts, registers
    via `MlflowClient.create_model_version`.
- Cleaned `models/fpb-lora/`: removed the redundant download zip, its `:Zone.Identifier`
  sidecar, and `checkpoint-2/` (training scratch). Kept adapter + tokenizer + chat_template.

**Trust-boundary design (the key idea this session):** split *who produces what*. Colab (GPU,
ephemeral) produces only the unreproducible measurement (`test_f1`). The laptop (durable
system-of-record) recomputes the integrity-critical, deterministic value (`eval_set_hash`)
fresh from `split_data` — never trusts a hash off the remote box. So a stale/wrong Colab file
can't sneak a wrong exam past the gate; `n_test==465` is the cheap cross-check.

**MLflow 3.x gotcha (hit + fixed):** `mlflow.register_model("runs:/<run>/adapter", NAME)` raises
`Unable to find a logged_model with artifact_path adapter` — 3.x expects a first-class *logged
model* (MLmodel descriptor + flavor), which `log_artifacts` does NOT create. Fix: lower-level
`MlflowClient().create_model_version(name, source=get_artifact_uri("adapter"), run_id=...)` —
the "register externally-produced artifacts as a version" path. No serving wrapper needed (that's M6).

**Verified:**
- v14 dossier: test_f1 0.8477, eval_set_hash == EXPECTED_HASH (bit-for-bit), schema v1, commit 00e8af4.
- Gate prediction (all 4 pass) → `/promote` returned 200 `{"promoted":"14"}`. Production flipped
  1→14. Audit line `"decision":"promoted","previous_production":1,"approved_by":"karthik"`.
- **Break-it (rule 4), predicted then confirmed:** ran `loop.py` against the LLM production model
  → `MlflowException: No such artifact: 'MLmodel'` at loop.py:30. The sklearn loader dies before
  even reaching `named_steps["tfidf"]`. NOT a bug — the system surfacing that serving + drift are
  model-aware and assume sklearn. Making them model-aware = M6 (vLLM/LoRA load) + M8 (router/drift).
  Break is runtime-only (prod alias in gitignored mlflow.db); committed code is clean.

**Concepts reinforced:** the gate reads only tags (never loads the model / recomputes F1) → the
**hash gate is the real teeth against the *realistic* failure (comparing models scored on different
exams)**, NOT against a deliberately fabricated F1 (for that you'd recompute the score in the control
plane, or lock down who can write tags). Recompute invariants (hash), trust measurements (F1) but
anchor them to a verified exam. Stateless compute (Colab) vs stateful governance (registry/audit) —
why training runs on throwaway GPU but promotion must run where the durable system-of-record lives;
M7 makes both real cluster services.

**Decision:** keep v14 as production (it's the real Piece-1 deliverable — "the LLM IS production"),
unlike M4 which reverted its *fake* 0.75 demo model. loop.py/serving stay sklearn-only until M6.

## What M5 Piece 1 is

Replace the M2 sklearn stand-in with a real fine-tuned LLM. Take Qwen2.5-1.5B, QLoRA
fine-tune it on Financial PhraseBank to classify bullish/bearish/neutral, and prove it
beats the locked sklearn bar on the *same frozen test set*. QLoRA = quantize the frozen
base to 4-bit (the "Q", shrinks 6 GB→~0.75 GB) + train tiny high-precision LoRA adapters
(the "LoRA", kills the gradient/optimizer memory) → a 1.5B fine-tune fits on a free GPU.

## Sub-steps (this session)

| Step | What |
|---|---|
| 1 | Instruction-format data → chat-messages JSONL (`pipelines/instruction_format.py`) |
| 2 | QLoRA training script, TRL SFTTrainer + PEFT (`pipelines/finetune.py`); CPU smoke test |
| 3 | Real run on free Colab T4 (4-bit Qwen-1.5B, batch 16, 3 epochs, ~60 min) |
| 4 | Generative-classifier eval vs 0.6885 (`pipelines/eval_adapter.py`) → 0.8477 |

## What got built

- `pipelines/instruction_format.py` — reshapes PhraseBank (text,label) → chat-messages
  JSONL under `data/instruction/`. **Reuses M2 `load_data`/`split_data` (seed=42)** so the
  test split is bitwise-identical to the 0.6885 bar; verified row-for-row. Bare label word
  as the assistant turn (eval string-matches it; all learning pressure on the one decision).
- `pipelines/finetune.py` — QLoRA fine-tuner. Ingredients: 4-bit base (BitsAndBytesConfig
  nf4 + bf16 compute) + `prepare_model_for_kbit_training`; `LoraConfig` r=16, alpha=32,
  dropout=0.05, all 7 target_modules; HF datasets loader; `SFTConfig` + TRL `SFTTrainer`
  (auto chat-template + loss-masking); train; save adapter. Dev/real switch via constants
  `MODEL_NAME`/`USE_4BIT`/`MAX_STEPS`.
- `pipelines/eval_adapter.py` — generative classifier eval: `PeftModel.from_pretrained`
  (adapter on frozen base) + `apply_chat_template(add_generation_prompt=True)` + greedy
  `generate(max_new_tokens=5, do_sample=False)` + parse word → sklearn macro-F1 vs 0.6885.
- `requirements.txt` — added M5 block (torch/transformers/peft/trl/accelerate;
  bitsandbytes Colab-only). `.gitignore` — added `/models/` (adapters/checkpoints/optimizer).

## Verified (the result)

- **CPU smoke test** (Qwen2.5-0.5B, 4-bit off, 2 steps): loss fell 5.12→3.52, adapter
  saved. Proved all 7 ingredients assemble for $0. Saw `optimizer.pt` ≈ 2× adapter size
  on disk = the Adam two-moments bucket from the VRAM math, made real.
- **Real Colab run:** loaded Qwen-1.5B in 4-bit (338 shards), tokenized 2170/465 clean,
  408 steps over 3 epochs. Mild overfit at epoch 3 (eval_loss 1.079→1.113, token-acc flat)
  — margin so large it didn't matter; did NOT re-run at 2 epochs.
- **Eval:** macro-F1 **0.8477** vs **0.6885** → **beats by ~16 pts (+23% relative)** on the
  identical sealed test set neither model trained on. Core deliverable met.

## Concepts Karthik now owns

- QLoRA's three layers: (1) full fine-tune memory = weights + gradients + optimizer +
  activations (~24 GB for 1.5B, optimizer alone ~2× weights); (2) LoRA freezes the base,
  trains tiny side-matrices → gradient/optimizer memory becomes a rounding error, adapter
  is MBs; (3) the "Q" quantizes the frozen base to 4-bit; adapter stays bf16 (precision
  follows training — quantize what you freeze, keep precise what you learn).
- The 7-ingredient fine-tune anatomy; what peft / trl / accelerate each do; SFTTrainer's
  chat-templating + loss-masking; **SEQ_CLS vs CAUSAL_LM** (we chose generative).
- LoRA knobs: r (capacity), alpha (strength, conv 2r), dropout; target_modules = the
  q/k/v/o attention + gate/up/down MLP linear layers; broad ("all-linear") vs minimal.
- Training mechanics: step = 1 batch = 1 fwd + 1 bwd + 1 update; epoch = full pass (136
  steps); 3 epochs = 408 steps; effective batch = batch × grad-accum; eval pauses training
  for a still snapshot; eval batch size is a speed dial, not a quality dial.
- Overfitting read: train loss ↓ while eval loss ↑ = memorizing trivia, not the rule
  (same data, early passes teach the rule, late passes teach the noise).
- **Fair comparison = same frozen test set + LLM trained only on train split** (the M3
  hash-the-eval-set principle). Big margin = robust win, overfit cost negligible here.

## Open threads / next

1. **Next session: register the adapter via M3 control plane (dossier test_f1=0.8477,
   eval_set_hash) + push through gated `/promote`** — governs the LLM like any prod model.
2. Cleanups (not blocking): refactor `finetune.py` dev/real switch to an env var (AF_MODE)
   so the real config is reproducible from git; fix the size-print to report only
   `adapter_model.safetensors` (the 156/163 MiB number is os.walk over checkpoints).
3. Trained adapter lives only on the laptop (downloaded from ephemeral Colab); `models/`
   is gitignored — decide a real home (DVC/S3) for trained adapters.
4. For Piece 5 (distillation teacher), consider `load_best_model_at_end=True` — teacher
   quality propagates to the DistilBERT student, so the best epoch matters more there.
5. Remaining M5: Piece 2 (bf16 efficiency, the JD "5%"), Piece 3 (Ray Train + NCCL on a
   paid 2-GPU RunPod pod — confirm $/hr first), Piece 5 (distillation + distill.yml).
