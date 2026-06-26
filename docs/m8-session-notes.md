# M8 — Capstone: Session Notes

Running log of M8. Concepts + design: `docs/m8-concepts.md`.

## Session 1 — 2026-06-25 (kickoff + router design)

**Context set:** M8 = close the self-healing loop (drift→retrain→promote→**router
reroutes**, zero downtime) + cost-aware multi-model serving. Two-plane architecture
completed (serving plane ↔ retraining plane, meet at registry+gate).

**Router design LOCKED** (concepts §1–6):
- Backend abstraction = named callables `(text)->(output, confidence)`, transport-agnostic
  (in-process/mock local → HTTP/KServe prod) → routing testable free, no GPU.
- Routing table: summarize→summarizer (stub); classify+accurate→llm_sentiment;
  classify+cheap→student; classify+**escalate**→student then escalate to LLM if
  confidence<THRESHOLD (default tier; a **model cascade**).
- Tiers = cost↔accuracy spectrum: cheap / **escalate** (default) / accurate.
- Renamed `auto`→`escalate` (self-documenting) at Karthik's call.
- Decisions: task/tier in body; student must emit `softmax().max()` confidence (flip M6
  `confidence=None`); THRESHOLD~0.70 (tune via escalation-rate-vs-accuracy on frozen eval);
  cost=deployment concern (GPU slices) not router code; summarizer stubbed.
- File layout `serving/router/`: backends.py / routing.py (tutor-protected, Karthik writes
  route()) / app.py (thin FastAPI). Unit-testable with mock backends.

**Carry-over debts to clear in M8:** model-aware retraining (loop.py retrains sklearn →
can't pass gate vs LLM); drift sensor off sklearn TF-IDF; sklearn retired-not-deleted.

**Build plan (free-first):** router (free) → 2nd adapter (GPU) → debts (free) → hot-swap+
KServe (paid) → CRD/operator stretch → polish.

**Next:** scaffold `serving/router/` (route() + escalate cascade as Karthik's TODOs);
build + unit-test routing logic locally (mock LLM, real student on CPU).

**Loose ends:** git push (whole M6+M7+M8 stack local); deferred-backlog.md tracks pending
M6/M7 items; AWS fully destroyed ($0).

## Session 2 — 2026-06-25 (router built + tested; drift sensor decoupled)

**Router DONE + tested** (`serving/router/`): `routing.route()` (summarize/accurate/cheap/
escalate-cascade, ValueError on unknown task/tier), `backends.py` (mock LLM, real student,
summarizer stub), thin FastAPI `app.py`. **`serving/app.py` distilbert predictor now emits
softmax confidence** (cascade enabler; was None). **9 routing unit tests** (mock backends,
all paths, 0.09s) in `serving/tests/test_routing.py`; `make test` now includes serving.

**Debt 2 — drift sensor decoupled (model-agnostic):** `drift.reference_analyzer_vocab()`
builds (analyzer, vocab) from TRAIN text via CountVectorizer — no served-model load.
`detect_drift()` + `loop.drift_detected()` switched to it. **Verified:** `detect_drift()`
→ psi=16.7153, DRIFT yes (matches M4 ~16.7), now model-agnostic. **REMAINING:**
`detect_drift_evidently()` still loads sklearn@production (line ~100) — apply the same
`analyzer, vocab = reference_analyzer_vocab()` swap to fully clear Debt 2.

**Drift learnings (concepts §8):** drift = a DATA property not a model property → per
input-stream/TASK, not per model (sentiment LLM+student share one reference; summarizer
gets its own ECTSum reference → parametrize `reference_analyzer_vocab(task)` later).
**Data drift** (P(X), unsupervised, OOV/PSI — what we built) vs **concept drift** (P(y|X),
same input/different correct label, needs delayed-label accuracy — named, not simulated).

**Next:** finish detect_drift_evidently swap → **Debt 1 (model-aware retraining)** =
design-heavy: `loop.retrain_and_register()` must produce an **LLM adapter** (dispatch the
GPU QLoRA pipeline keyed on model_kind) instead of inline sklearn. Then 2nd adapter (GPU),
hot-swap/KServe.

## Session 3 — 2026-06-25 (Debt 1: model-aware retraining DONE)

`loop.py` is now model-aware (dispatch verified via mocks; ruff clean):
- `production_model_kind()` reads `model_kind` from control-plane `/production`.
- `RETRAIN_BY_KIND = {sklearn: retrain_sklearn, lora_adapter: retrain_lora}`;
  `retrain_and_register()` dispatches by production kind (RuntimeError on unknown).
- `retrain_lora()` chains the M5 pipeline: instruction_format → **set AF_MODE=real BEFORE
  importing finetune** (finetune reads it at module load) → finetune → eval_adapter →
  `register_adapter("models/fpb-lora", test_df)`. GPU-bound → runs on a GPU runner
  (prod: retrain.yml); wiring done, real run deferred.
- `retrain_sklearn()` kept as legacy (never wins the gate vs the LLM).
- run_loop() unchanged. **Debt 1 architecture complete**; the loop is no longer a dead
  sklearn no-op.

**Decision (2026-06-25): SKIP the kopf operator / AdapterDeployment CRD** → deferred-
backlog. Rationale: in this project it's not a functional need — the router + M8 hot-swap +
vLLM multi-LoRA already cover multi-adapter serving + rerouting; the operator would only
re-express that as a CRD. Its value here is purely the JD "operators/CRDs" checkbox +
resume signal → optional stretch, not core. Core remaining = the GPU finale (2nd adapter +
KServe). Also did: hygiene cleanup (prompt-contract dedup, dead-code, ruff-clean repo),
README skeleton (Karthik filled prose; 3 built-vs-designed tweaks pending), MLflow UI moved
to :5555 (was clashing with Marquez API :5000).

**Debt 2 FULLY CLOSED** (2026-06-25): `detect_drift_evidently` swapped to
`reference_analyzer_vocab()` too → 0 `sklearn.load_model` in drift.py; both functions run
model-agnostic (psi 16.7; Evidently report written). Drift is now entirely decoupled from
the served model.

## Session 4 — 2026-06-25 (2nd adapter / summarization — prep + data verified)

Scaffolded the summarization pipeline (concepts §9); reuses M5 QLoRA heavily.
- `pipelines/summarize_format.py` — ECTSum (GitHub repo zip) → chat-messages JSONL.
  **load_ectsum VERIFIED**: train 1681 / val 249 / test 495 (ECTSum's real sizes), sample
  transcript→bullet-summary correct. `python -m pipelines.summarize_format` writes
  data/instruction_summ/*.jsonl.
- `pipelines/finetune.py` — DATA_DIR/ADAPTER_DIR env-overridable (AF_DATA_DIR/AF_ADAPTER_DIR)
  → same SFT trains the summarizer (data/instruction_summ → models/fpb-summarizer).
- `pipelines/eval_summarizer.py` — ROUGE-L vs base zero-shot (base via
  model.disable_adapter()); writes eval_metrics.json. Needs `pip install rouge-score`; GPU.
- DS format taught (concepts §9): raw ECTSum = (transcript.txt, summary.txt) pairs →
  chat JSONL where transcript=prompt, summary=SFT target. Same shape as sentiment → finetune reused.

**Run (GPU session):** `python -m pipelines.summarize_format` → `AF_MODE=real
AF_DATA_DIR=data/instruction_summ AF_ADAPTER_DIR=models/fpb-summarizer python -m
pipelines.finetune` → `python -m pipelines.eval_summarizer`.
**⚠️ Open design:** register/promote `fpb-summarizer` needs a **task-aware gate** (current
gate is sentiment-pinned: EXPECTED_HASH + F1). Then wire router `build_summarizer`.

**M8 remaining:** 2nd LoRA adapter (summarization, ECTSum — GPU) · hot-swap + KServe
(cluster/GPU) · CRD/operator stretch (free, kind) · polish (arch diagram, README JD-map,
demo). Free/local M8 core (router + both debts) is DONE.

## Session 5 — 2026-06-25 (task-aware promotion gate IMPLEMENTED)

`control-plane/app.py` now task-aware (was sentiment-pinned + ignored the `{name}` path
param — latent bug). **`GATE_CONFIG`** keyed by model name: per-model `expected_hash`,
`expected_schema`, `margin`, `floor`, `metric_label`. `promote(name,…)` looks up
`GATE_CONFIG[name]` (404 if unknown) and threads `name` through get_dossier/
get_production_version/set_alias → each model gated vs its OWN exam + OWN incumbent
(F1-vs-F1, ROUGE-vs-ROUGE never cross). Summarizer hash deferred via
`os.getenv("FPB_SUMMARIZER_EXPECTED_HASH")`, **fail-closed** if unset. ruff clean, parses.
**Last sentiment-pinned piece closed** — gate + drift sensor + retraining all task/model-aware.

**Remaining for the summarizer to go live:** write `register_summarizer` (mirror
register_student → register under `fpb-summarizer`, score on ECTSum test → its hash) → set
`FPB_SUMMARIZER_EXPECTED_HASH` → GPU run (format→finetune→eval→register→promote) → wire
`backends.build_summarizer` to the real adapter.

## Session 6 — 2026-06-26 (register_summarizer done + summarizer GPU run)

**register_summarizer implemented + reviewed** (commit 73eae73, ruff clean): blends
register_adapter (LoRA artifact logging, model_kind=lora_adapter) + register_student (NEW
name → create_registered_model first, swallowing only RESOURCE_ALREADY_EXISTS; MODEL_NAME
monkeypatch w/ finally restore). Hash reproducibility hinges on `summarize_format` sorting
filenames (it does) → printed `eval_set_hash` == the gate's tag.

**Phase B — RunPod L4 (~$0.45/hr) summarizer training DONE.** Env-hell fixes (worth keeping):
- Pod came with a working `torch 2.8.0+cu128` (driver CUDA 12.8). `pip install -r
  requirements-gpu.txt` (unpinned transformers 5.12.1) **dragged in torch 2.12.1+cu130** →
  `cuda False` ("driver too old") + `torchvision::nms does not exist` (torch≠torchvision).
- **Fix:** reinstall the matching CUDA stack pinned to the driver:
  `pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
  --index-url https://download.pytorch.org/whl/cu128` → `cuda True`, transformers 5.12.1
  imports clean (alignment fixed the nms crash — no need to downgrade transformers this time).
- Lesson: on a RunPod PyTorch template, **never let an unpinned install replace torch**;
  pin torch+vision+audio to the cu build matching the driver. Also `debian blinker` has no
  RECORD → `--ignore-installed blinker` (but NOT with `--index-url`, which lacks blinker).
- `summarize_format` → 1681/249/495 ✅. **Trained** Qwen2.5-1.5B 4-bit, BATCH_SIZE lowered
  16→4 (long ECTSum transcripts OOM risk), 3 epochs / 1263 steps / ~68 min, loss ~2.0,
  adapter saved (**398 MiB — unexpectedly large**; backlog: likely embed/lm_head saved).
- **Eval bottleneck observed:** `eval_summarizer` = 990 *sequential* greedy generations
  (495 × adapter+base), GPU util ~25%, ~80 min — literally the M6 "naive" path. Backlog:
  HF batched generate (util ~70%+, ~10-15 min); vLLM `--enable-lora` optional/heavier.

**Concept captured — "CPU 100% / GPU 25% / top says idle": how to read utilization.**
All three readings were consistent + healthy, just measured against different denominators:
- **`ps` per-process ~100%** = % of **one core**. The eval is **single-threaded** (a
  sequential generate loop) → it pegs exactly one core, no more.
- **`top` %Cpu(s) ~94% idle** = averaged across **all** vCPUs. Shared RunPod host had ~100
  vCPUs → one busy core ≈ 1% of the box. Same fact, host-wide denominator. (`load avg ~6`
  agrees: ~6 runnable threads on a ~100-vCPU box = near-idle.)
- **`nvidia-smi` 25%** = GPU busy only 25% of wall-clock. Cause = **batch size 1**: text
  gen is **autoregressive** (token N+1 needs N → strictly serial, can't parallelize one
  sequence), so the GPU gets a teaspoon of work per token, finishes in µs, then **waits on
  the CPU** to launch the next → starved, idle 75% of the time.
- **Bottleneck = the serial one-at-a-time round-trip**, not raw CPU or GPU compute. CPU core
  saturated by Python/launch overhead; GPU starved by tiny work units. (~158k round-trips =
  495×2×~160 tokens.)
- **Adding cores does NOT help**: one thread runs on one core at a time, and an
  autoregressive sequence can't be split across cores. The fix is **GPU parallelism via
  batching** (32 prompts/launch → GPU fills up + CPU launches amortized → both numbers
  improve, ~5–30×). This is the M6 naive-vs-continuous-batching lesson, live.
- **vCPU ≠ physical core**: 1 physical core w/ SMT = 2 vCPUs; `nproc`/`top` count vCPUs.
  This pod: `nproc`=**128** vCPUs (~64 physical cores). Process ≈100% = ~1 vCPU of 128 →
  1/128 ≈ 0.8% busy → `top` ~99% idle. All readings consistent.
- **Proof it's single-threaded**: `ps %CPU` *can* exceed 100% for multithreaded procs
  (8 cores → 800%). Ours pegged ~100% (not 200/800) = hard evidence it uses exactly one
  core. Why it can't use the other 127: the Python gen loop holds the **GIL** (one bytecode
  thread), and the math is on the **GPU** so extra CPU threads have nothing to do — the idle
  cores aren't blocked, there's just **no parallel work** to give them.
- **`torch.compile` red herring**: transformers 5.x spawned ~32 `_inductor/compile_worker`
  procs at startup (compiled once, ~4s CPU total, then idle) — NOT recompiling per-shape
  (verified via worker `time`≈0 over 90 min). Not the bottleneck.

**⚠️ Before pod teardown:** `models/` is gitignored + register runs on the laptop (mlflow.db
is there) → must `tar` + download `models/fpb-summarizer` first (M5 lesson: teacher was lost).

**Eval RESULT + summarizer PROMOTED TO PRODUCTION (governance proof #2).**
- `eval_summarizer` → **ROUGE-L adapter=0.1330 vs base zero-shot=0.1015, beat_bar=True** ✅
  (modest absolute — transcripts truncated ~1024 tok + 3 epochs — but the JD bar was
  "beat base zero-shot", met). `eval_metrics.json` written (test_f1=0.1330, n_test=495).
- Pod→laptop transfer: `scp -P <port>` got **Connection refused** (RunPod pod had no direct
  TCP SSH exposed) → used the working path; lesson: prefer `runpodctl send/receive` or the
  Jupyter file browser over direct scp on RunPod.
- Cleaned `checkpoint-*` (398 MB → ~47 MB), tar'd, downloaded, **terminated pod** (~$0.90 total).
- `register_summarizer` → **fpb-summarizer v1** registered; printed
  `eval_set_hash=f55821bc05e3f0bd9d06a56827c8961508a5e844b05e76a4eee2375f762fe94f`.
- Promoted via the gate: `FPB_SUMMARIZER_EXPECTED_HASH=f55821… uvicorn app:app` →
  `POST /models/fpb-summarizer/promote {version:1, approved_by:karthik}` → **promoted**.
  `/production` confirms dossier: v1 · test_f1 0.1330 · hash f55821… · schema v1 ·
  commit 73eae73 · model_kind lora_adapter.
- **What this proves:** a SECOND, different task went train→eval→register→**gated promote**
  with ZERO hand-editing of the gate — task-aware `GATE_CONFIG` scored it ROUGE-vs-ROUGE
  against its OWN ECTSum exam, and the pinned hash matched the candidate tag bit-for-bit
  (the whole point of pinning: no promotion on a non-canonical/leaked test set).

**Phase D DESIGN LOCKED — wire the real summarizer backend (Option A, Karthik's call).**
- Decision: serve the REAL adapter (not a mock like `build_llm_sentiment`). Slow on laptop
  CPU but genuinely closes the loop; prod swaps to vLLM HTTP later (router unchanged).
- Shape: add `build_summary_predictor(name, version)` in `serving/app.py` (sits beside
  `build_lora_predictor`/`build_distilbert_predictor`; NOT in `PREDICTOR_BUILDERS` — it's a
  task-specific generate loader, not a `model_kind` dispatch). It reuses
  `eval_adapter.load_model_and_tokenizer` (load) + `eval_summarizer.generate_summary`
  (generate), returns `predict_fn(text) -> (summary, None)`.
- Router `backends.build_summarizer` just delegates → `build_summary_predictor("fpb-summarizer","1")`
  (mirrors how `build_student` delegates to `serving.app`).
- **Critical gotcha:** use `summarize_format.build_chat_messages` (the SUMMARIZER prompt
  the adapter trained on), NOT `instruction_format` (sentiment) — a drifted prompt silently
  tanks output. `confidence=None` (generation has no softmax max; cascade is classify-only).
- **Status: Karthik to implement the two bodies (tutor rule); skeleton+TODOs handed over.**
