# M4 Session Notes — Lineage, Drift, and the Automated Loop (IN PROGRESS)

Session: 2026-06-17. Owner: Karthik. Tutor-mode learning.
**M4 status: ~60% done.** Drift (hand-rolled + Evidently), prediction logging, and
Dagster orchestration all working. Remaining: Marquez lineage + the trigger chain.

## What M4 is

Turns the manual M1–M3 pieces into a **self-driving loop**: detect when the live
model goes stale (drift), retrain, re-evaluate, and re-promote through the M3 gate —
zero human action. SRE analogy: drift = health check, trigger chain = auto-remediation
runbook, M3 gate = admission control that stops the auto-fix making things worse.

## Sub-steps

| Step | Status | What |
|---|---|---|
| M4.1 | ✅ | Drift detection: OOV signal + hand-rolled PSI + KS + regime fixture |
| M4.2 | ✅ | Evidently drift report (HTML + JSON verdict) |
| M4.3 | ✅ | Prediction logging (served model → JSONL) |
| M4.4a | ✅ | Dagster DAG (3 assets: ingest → train → register) |
| M4.4b | ⬜ | OpenLineage → Marquez lineage (Docker) — NEXT |
| M4.5 | ⬜ | Trigger chain: drift → webhook → GH Actions retrain → promote gate → auto-promote |

## What got built this session

- `pipelines/drift.py` — `oov_rate(headline, analyzer, vocab)` (OOV = fraction of a
  headline's tokens not in the production model's TF-IDF vocab; tokenize with
  `vectorizer.build_analyzer()`, NOT `.split()`). Hand-rolled `psi(ref, cur, bins=10)`
  (np.histogram → fractions → epsilon-clamp → PSI formula). `detect_drift()` =
  PSI + KS (scipy `ks_2samp`); `detect_drift_evidently()` = Evidently `DataDriftPreset`
  → `docs/drift_report.html` + dict verdict. Loads prod model via
  `models:/fpb-sentiment@production`.
- `pipelines/regime_headlines.csv` — ~40 hawkish/QT/crypto headlines (quoted CSV) to
  simulate a regime change.
- `serving/app.py` — added `log_prediction()` → `serving/predictions.jsonl`
  (text, label, confidence, model_version, ts) on every `/predict`.
- `pipelines/dag.py` — Dagster: assets `financial_phrasebank` → `baseline_model` →
  `registered_model` (each wraps M1/M2/M3 code). Run: `dagster dev -f pipelines/dag.py`
  (UI :3000), Materialize all.

## Key results / numbers

- Drift on regime batch: mean OOV ref 0.084 vs current 0.477; **PSI 16.7** (>>0.2),
  KS p 7.6e-34 → DRIFT yes. Evidently auto-picked KS, p-value matched the hand-roll
  exactly. Evidently dataset verdict: `DriftedColumnsCount` share 1.0 ≥ 0.5 →
  the field M4.5's loop reads: `result["metrics"][0]["value"]["count"] > 0`.
- Disagreement demo (case 1): tiny 0.015 shift + 8000 samples → PSI 0.026 (no drift)
  but KS p 4.5e-18 (significant) → why the trigger uses PSI magnitude, not KS p.

## Concepts Karthik now owns

- Drift = compare REFERENCE (training-time) vs CURRENT (live) distributions of a
  signal; threshold → alert.
- Signal vs test: OOV = the signal (a leading INPUT indicator); PSI/KS = tests on it.
  OOV is leading (cause), confidence/class-mix are lagging (effects).
- PSI = magnitude; KS = max-gap + significance (sample-size sensitive). When they
  disagree: KS-significant-but-PSI-low = trivial-but-real at scale (trust PSI);
  PSI-high-but-KS-not = too-small sample (distrust). Trigger on PSI > 0.2.
- Other metrics: Wasserstein, JSD, KL (PSI ≈ binned symmetric KL), chi-square,
  embedding/domain-classifier drift. Combination = signals × tests → aggregate.
- OOV tokenization must match TF-IDF (`build_analyzer()`); epsilon in PSI avoids log0.
- Prediction logging = monitoring data; synchronous-per-request adds latency →
  production uses async/queue/log-pipeline (observability shouldn't slow the system).
- Dagster: asset-based orchestration; deps via param names; enforces return type
  annotations at runtime (caught int-vs-str). `dagster dev -f`.
- OpenLineage data model (for next session): Job, Run, Dataset; events wire
  dataset→job→dataset; Marquez draws the graph.

## Resume next session (M4.4b + M4.5)

1. **Marquez lineage:** stand up Marquez in Docker (multi-container: Postgres + API
   :5000 + web). PORT CONFLICTS to manage — Marquez web wants :3000 (Dagster) and API
   wants :5000 (MLflow UI); remap (e.g. web → :3001). Then emit OpenLineage events for
   the pipeline (manual `openlineage-python` client is the robust path) → view graph.
2. **M4.5 trigger chain:** drift verdict → webhook → GitHub Actions `retrain.yml` →
   Dagster retrain → control-plane `/promote` gate → auto-promote only if better.
   The M3 promote API is the gate this loop calls.

## Open threads

- Commit M4 work (Karthik's commits): `pipelines/drift.py`, `regime_headlines.csv`,
  `pipelines/dag.py`, `serving/app.py` changes. `.gitignore`: add
  `docs/drift_report.html`, `serving/predictions.jsonl`.
- Registry now has dummy v2 (bad hash) + v3/v4 (Dagster re-runs) — clean up or leave.
- M1 SDK README still pending (rule 5).
- pip installs this session: scikit-learn (earlier), mlflow, fastapi/uvicorn,
  requests, matplotlib, evidently (0.7.21), dagster + dagster-webserver, scipy (via sklearn).
