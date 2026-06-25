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
