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

**Debt 2 FULLY CLOSED** (2026-06-25): `detect_drift_evidently` swapped to
`reference_analyzer_vocab()` too → 0 `sklearn.load_model` in drift.py; both functions run
model-agnostic (psi 16.7; Evidently report written). Drift is now entirely decoupled from
the served model.

**M8 remaining:** 2nd LoRA adapter (summarization, ECTSum — GPU) · hot-swap + KServe
(cluster/GPU) · CRD/operator stretch (free, kind) · polish (arch diagram, README JD-map,
demo). Free/local M8 core (router + both debts) is DONE.
