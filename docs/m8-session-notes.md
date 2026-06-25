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
