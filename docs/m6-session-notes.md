# M6 — Serving Frameworks + GPU Sharing: Session Notes

Running log of M6 work. Concepts live in `docs/m6-serving-concepts.md`; results will
land in `docs/m6-*-results.md` per piece.

## Plan (decided 2026-06-22 kickoff)

Piece order:
0. **Model-aware serving fix** — local, free. Foundation for the benchmark.
1. **vLLM + benchmark** — RunPod single GPU. p50/p95/p99 + throughput vs naïve FastAPI
   (same model, two stacks); watch KV-cache VRAM grow with concurrency.
2. **Triton or KServe** — one model through it + half-page "when would I pick each" note.
3. **A100 MIG lab** — FULL hands-on (decided): enable MIG → 2 isolated instances
   (LLM + student) → disable → time-slicing crash demo. ~$1.8/hr A100.
4. **DCGM → Prometheus** — graph GPU mem / SM util during the benchmark.

## Session 1 — 2026-06-22 (kickoff + design)

**Done:**
- Oriented on M6; taught the full concept stack and saved it to
  `docs/m6-serving-concepts.md`:
  - serving basics; why the current FastAPI is naïve for LLMs
  - autoregression; prefill vs decode (decode wastes the GPU); KV-cache
  - vLLM continuous batching (vs static); PagedAttention (OS virtual-memory borrow)
  - GPU sharing: time-slicing / MPS / MIG + the three-way tradeoff
  - the two-layer model: sharing (many models → one card) vs vLLM (many requests →
    one model); they stack
  - dispatch (M6) vs router (M8)
- **Design locked for Piece 0 (model-aware serving fix):**
  - stamp a `model_kind` registry tag (`lora_adapter` / `distilbert`)
  - backfill the tag onto existing v14 (and the student version)
  - serving reads `model_kind` from the control-plane `/production` response →
    dict-dispatch to (loader, predictor); download artifacts from registry then reuse
    the existing `eval_adapter` / `eval_student` loaders
  - **sklearn retired, NOT deleted**; build dispatch for LLM + student only
- **M8 carry-over written** into `docs/PROJECT_PLAN.md` (two-plane architecture;
  retraining is sklearn-bound and can't pass the gate vs the LLM; drift sensor
  piggybacks on TF-IDF vocab).

**Decisions:** MIG depth = full hands-on. Start order = Piece 0 first (free, local,
prerequisite for the benchmark).

**Next:** implement Piece 0 step 1 — add the `model_kind` tag at registration +
backfill v14/student. Then serving dispatch.

## Session 2 — 2026-06-22 (Piece 0, Step 1 DONE)

- `model_kind` tag wired into `register_model_with_dossier` as a **required** param
  (Karthik's call — gate/dispatch-trusted tags shouldn't be silently omittable).
  Bonus he added: `MODEL_KINDS` frozenset + a registration-time validation guard
  (invalid kind fails at register time, not at serving load).
- Wrappers pass their kind: `register_sklearn`→`sklearn`, `register_adapter`→
  `lora_adapter`, `register_student`→`distilbert`. loop.py/dag.py covered (go through
  `register_sklearn`).
- `pipelines/backfill_model_kind.py` (idempotent) ran + **verified in registry**:
  `fpb-sentiment` v14 = `lora_adapter`, `fpb-student` v2 = `distilbert`. Old sklearn
  fpb-sentiment v1–v13 left untagged (retired, never served).
- Open design Q for Step 2: what should serving dispatch do if `model_kind` is missing?

**Next:** Piece 0, Step 2 — serving dispatch (`serving/app.py`): read `model_kind`
from `/production`, dict-dispatch to (loader, predictor); download artifacts from
registry then reuse `eval_adapter` / `eval_student` loaders.

**Open / deferred:** loop.py model-aware retraining + drift sensor → M8. MPS hands-on
optional. M1 SDK README + RoCE/IB explainer still open (rule 5).
