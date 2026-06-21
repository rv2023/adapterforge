# Kellogg Asset-Management Practicum — Vision (v3, built ON AdapterForge)

**Status:** future vision (post-M8). Captured 2026-06-21. This is a **product built on the
AdapterForge platform**, not a replacement for the core. The platform (SDK → control plane →
drift/trigger → registry → routing → serving → observability) is the **reusable engine**; the
practicum adds new signals + a fusion/verdict layer + report generation on top.

**Two goals, one engine:**
- *Lenovo MLOps JD* = the **engine** (AdapterForge core, M1–M8).
- *Kellogg practicum* = a **multi-signal asset-intelligence product** on that engine (this doc).
- Story: *"I built a model-agnostic MLOps platform, then deployed a real multi-signal
  asset-intelligence product on it."*

## The vision (the flow)

```
many INPUT STREAMS (continuous + batch)
   → per-stream ADAPTERS tag + rate a signal (Bull/Bear/Neutral + value) → backend store
   → CONSOLIDATION model ties all signals to a specific asset/ETF → per-asset table
   → VERDICT model (learns signal weights) → final verdict (Bull/Bear/Neutral) + TARGET PRICE
   → DASHBOARD
   → REPORT model generates daily/weekly/quarterly briefs, triggered when signals change
   → ANALYST LLM Q&A: free-form questions → RAG over signals/verdicts/reports/logs → grounded answer
```

## Input streams (each = an Adapter, M1 SDK pattern)

| Stream | Cadence |
|---|---|
| Twitter/X tweets | continuous |
| News headlines | continuous |
| Smart money / flows | continuous-ish |
| Insider activity (Form 4) | event/batch |
| Political activity (e.g. congressional trades, policy) | event/batch |
| Macro-economic releases | scheduled/batch |
| Company financials (balance sheet, fundamentals) | quarterly/batch |
| Quarterly filings + earnings calls | quarterly/batch |
| (others as discovered) | — |

The SDK is built for exactly this — "standardized ingestion across diverse sources," continuous
*and* batch, with Pandera validation + versioning per source.

## Component mapping → reuse / new / hard

| Practicum component | AdapterForge mapping | Reuse / New / Hard |
|---|---|---|
| The input streams | one **Adapter** each (M1) | **reuse pattern**; many **new adapters + real data feeds** |
| Per-stream signal rating (Bull/Bear/Neutral + value) | **per-signal classifier models** (generalize the sentiment model) | **new models**, same train/registry/promote pattern |
| Consolidate signals → tie to asset/ETF | **fusion layer keyed by asset** + entity→asset linking (the v2 **NER** piece) | **new** (fusion + entity linking) |
| Verdict: direction **+ target price** ("trained on weights") | **meta-model / ensemble** over per-signal scores | **new + research-grade** ⚠️ |
| Dashboard | **serving + observability** (M3 / M7) | **reuse** |
| Daily/weekly report generation | **generative report writer** (generalize earnings summarizer; overlaps v2 **RAG**) | **new** (overlaps v2-B) |
| **Analyst LLM Q&A** (free-form questions, grounded on signals/verdicts/reports/logs) | **RAG assistant** = vector DB + retrieval + **generative** LLM (this IS v2-B, folded into the practicum) | **new + major** ⚠️ |
| "when signals change" trigger | the **drift/event → trigger chain** (M4) | **reuse** |

## The verdict target: directional + target price (chosen) — with guardrails

Target price is **quant alpha** — hard, easily overfit, and prone to **look-ahead/leakage bias**.
It's only worth anything if it's **falsifiable**, so the methodology is non-negotiable:
- **De-risk in order:** first prove the **directional verdict** backtests > a naive baseline, *then*
  add the target-price head on top. (Don't chase price before direction works.)
- **Backtesting framework is a first-class deliverable** (as much work as the models): point-in-time
  data only, **no look-ahead**, walk-forward/out-of-sample splits, transaction-cost awareness,
  evaluate vs **realized returns** (direction: hit-rate/AUC; price: calibrated error + a trading
  P&L sanity check). Without this, the verdict is unfalsifiable.

## Honest risk flags

1. **Target price** — research-grade; treat as a guarded stretch on top of a working directional model.
2. **Evaluation/backtesting** — the make-or-break subsystem; leakage discipline decides credibility.
3. **Data acquisition** — insider/smart-money/fundamentals/filings often need **paid/licensed feeds**;
   the SDK eases *ingesting*, not *obtaining*.

## Phased plan (after the AdapterForge core, M1–M8)

- **Phase 0** — reuse the finished platform (M1–M8: SDK, control plane, drift/trigger, serving, obs).
- **Phase 1** — 2–3 high-value adapters first (news ✓ already; add tweets + fundamentals).
- **Phase 2** — per-signal classifiers + **entity→asset linking** (v2-A NER).
- **Phase 3** — **fusion/consolidation** per asset + **directional verdict** + the **backtest framework**.
- **Phase 4** — **target-price** head on top, under the backtest's leakage controls.
- **Phase 5** — **report generation** (generative).
- **Phase 6** — **Analyst LLM Q&A** (RAG): vector DB over signals/verdicts/reports/logs +
  retrieval + a **generative** LLM (NOT the classifier) → grounded answers. This is **v2-B RAG**,
  folded in as the practicum's analyst-facing layer. Eval = retrieval relevance + answer
  faithfulness/groundedness (cite the signals it used). Heaviest phase; do last.

**Sequencing rule:** finish the core platform first — it's the engine the practicum runs on. Build
the practicum as a parallel/after track, signal-by-signal, eval-first.
