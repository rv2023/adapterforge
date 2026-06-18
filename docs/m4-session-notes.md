# M4 Session Notes — Lineage, Drift, and the Automated Loop (COMPLETE)

Sessions: 2026-06-17. Owner: Karthik. Tutor-mode learning.
**M4 status: COMPLETE — all 5 sub-steps done.** Next: M0 GPU drills, then M5.

## What M4 is

Turns the manual M1–M3 pieces into a **self-driving loop**: detect when the live
model goes stale (drift), retrain, re-evaluate, and re-promote through the M3 gate —
zero human action. SRE analogy: drift = health check, trigger chain = auto-remediation
runbook, M3 gate = admission control that stops the auto-fix making things worse.

## Sub-steps (all done)

| Step | What |
|---|---|
| M4.1 | Drift: OOV signal + hand-rolled PSI + KS + regime fixture |
| M4.2 | Evidently `DataDriftPreset` (HTML report + JSON verdict) |
| M4.3 | Prediction logging (served model → JSONL) |
| M4.4 | Dagster DAG (3 assets) + OpenLineage→Marquez lineage graph |
| M4.5 | `loop.py` (drift→retrain→gate→auto-promote) + `retrain.yml` CI artifact |

## What got built

- `pipelines/drift.py` — `oov_rate` (tokenize via `build_analyzer()`), hand-rolled `psi`,
  `detect_drift` (PSI+KS scipy), `detect_drift_evidently` (→ `docs/drift_report.html`).
- `pipelines/regime_headlines.csv` — ~40 hawkish/QT/crypto headlines (quoted CSV).
- `serving/app.py` — `log_prediction` → `serving/predictions.jsonl` per `/predict`.
- `pipelines/dag.py` — Dagster assets financial_phrasebank→baseline_model→registered_model.
  (`sys.path.insert(pipelines)` for imports; Dagster enforces return type annotations.)
- `docker-compose.marquez.yml` — Marquez stack (db 6543, api 5000, web 3001). Web needs
  `MARQUEZ_HOST=marquez-api` (server-side proxy) + `WEB_PORT=3000`.
- `pipelines/lineage.py` — emits OpenLineage events (Job/Run/Dataset) → Marquez graph.
- `pipelines/loop.py` — the loop: `drift_detected()` → `retrain_and_register()` →
  `request_promotion()` (POST control-plane /promote). Robust version lookup via
  `search_model_versions` + max (info.registered_model_version was flaky/None).
- `.github/workflows/retrain.yml` — CI artifact: `repository_dispatch` webhook →
  runs `loop.py`. Fully live in M7 (remote MLflow + control plane via secrets).

## Verified (the demo)

- Drift on regime batch: mean OOV ref 0.084 vs current 0.477; **PSI 16.7**, KS p 7.6e-34
  → DRIFT yes. Evidently auto-picked KS, matched the hand-roll.
- PSI/KS disagreement (case 1): tiny 0.015 shift + 8000 samples → PSI 0.026 (no) but
  KS p 4.5e-18 (yes) → why the trigger uses PSI magnitude, not KS p.
- Marquez lineage graph renders: huggingface.flare-fpb → ingest → silver → train →
  model.baseline → register → model.fpb-sentiment.
- **Loop, both paths:** reject path (loop.py retrain not better → gate 409, prod safe);
  accept path (simulated-better v with correct hash, F1 0.75 → gate 200 promoted →
  production flipped → first `"promoted"` audit line with previous_production=1). Then
  production reverted to the real v1 baseline for a clean state.

## Concepts Karthik now owns

- Drift = ref vs current distributions; signal (OOV, leading INPUT indicator) vs test
  (PSI magnitude / KS significance); disagreement cases; other metrics (Wasserstein/
  JSD/KL/chi2/embedding). Tokenize like TF-IDF; epsilon in PSI.
- Prediction-logging tradeoff (sync vs async). Dagster assets + runtime type checks.
- Marquez = lineage store/viewer (receives OpenLineage events, builds the graph);
  OpenLineage = the standard; Dagster = the executor. Linked by matching dataset names.
- The loop: drift → retrain → gate → auto-promote ONLY if better; webhook =
  repository_dispatch; one body (loop.py) runs both locally and in CI.
- **Train data changes on retrain; the EVAL set stays frozen → hash unchanged → fair
  comparison. Updating the exam is a rare, deliberate, governed event (bump
  EXPECTED_HASH + re-baseline), which the hash gate forces you to do consciously.**
- Use case: live news feed (Alpha Vantage adapter) → /predict → logged → drift loop.
  LLM (M5) enters through the SAME gate, served by the SAME endpoint. 2 LLM adapters
  (on one Qwen base) + 1 DistilBERT student + baseline; M8 router picks cheap vs LLM.

## Open threads / next

- Commit M4 work + `.gitignore` (drift_report.html, predictions.jsonl, mlflow.db,
  mlruns/, data/, audit.jsonl). Registry has demo cruft (dummy v2 bad-hash, v3–v11 from
  reruns); production = v1 (clean). Marquez stack still running in Docker (stop with
  `docker compose -f docker-compose.marquez.yml down` when not needed).
- M1 SDK README still pending (rule 5).
- **Next milestone: M0 GPU drills, then M5** (LLM fine-tuning + distillation).
