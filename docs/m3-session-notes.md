# M3 Session Notes — MLOps Control Plane (COMPLETE)

Session: 2026-06-16. Owner: Karthik. Tutor-mode learning.
**M3 status: all 5 sub-steps DONE.** Next milestone: M4 (lineage, drift, automated loop).

## What M3 is

A **control plane** for models — the brain that governs which model is allowed to
be Production, separate from the **data plane** (MLflow holds models; serving answers
predictions). Turns "promote" from a human clicking a UI into a governed API with
policy gates + an audit trail. Direct payoff of M2's frozen-eval-set discipline.

## What got built

- `pipelines/register_baseline.py` — registers the M2 baseline into the MLflow Model
  Registry as `fpb-sentiment` v1 **with a dossier** (tags): `test_f1`, `eval_set_hash`
  (SHA-256 of the frozen test set via `to_csv(index=False)`), `schema_version`,
  `code_commit` (git rev-parse). Reuses baseline.py.
- `control-plane/app.py` — FastAPI service:
  - `POST /models/{name}/promote` — runs **4 gates**, then aliases the version
    `production` (MLflow 3.x uses aliases, not stages):
    1. **approved_by** present (else 400)
    2. **eval_set_hash == EXPECTED_HASH** constant (the one true exam) — else 409
    3. **schema_version == EXPECTED_SCHEMA** — else 409
    4. **F1 gate**: incumbent exists → `cand_f1 ≥ prod_f1 + MARGIN`; empty throne →
       `cand_f1 ≥ MIN_F1_FLOOR`. Fail-closed; a rejection changes nothing.
  - `write_audit` + `reject` helpers → append-only `control-plane/audit.jsonl`
    (every decision: promoted/rejected, who, why, ts) — logs BOTH accepts & rejects.
  - `GET /models/{name}/production` — what's live + dossier (serving calls this).
  - `GET /models/{name}/lineage/{version}` — commit + schema + hash (provenance).
- `serving/app.py` — FastAPI data plane. At startup ASKS the control plane
  `/production`, loads that version (`models:/fpb-sentiment/{version}`), serves
  `POST /predict {text}` → `{label, confidence, model_version}`. Brain decides,
  serving obeys.

## Run commands (two services, two ports)

```
uvicorn app:app --app-dir control-plane --reload          # control plane :8000
uvicorn app:app --app-dir serving --port 8001 --reload    # serving      :8001 (needs CP up)
```

## Verified

- First promotion (empty throne) → v1 promoted via MIN_F1_FLOOR. Production -> v1.
- Re-promote v1 → 409 (can't beat itself + margin). Reject.
- approved_by="" → 400. Reject.
- Serving loaded v1 from the control plane; `/predict` returns labels + confidence.
  (All 3 test sentences returned `neutral` — NOT an M3 bug: it's a 0.69-F1 baseline
  hedging to the majority class on out-of-distribution input. Model quality = M5.)
- **Break-it (rule 4):** registered v2 with `test_f1=0.80` but a WRONG eval_set_hash
  → promote → **REJECTED** by the hash gate despite the higher F1. Production stayed
  v1; rejection logged. Proves the hash gate blocks unverified-exam scores.

## Concepts Karthik now owns

- control plane vs data plane (his SRE/k8s background — brain vs muscle).
- log vs register vs promote (3 distinct MLflow actions; M2 only logged).
- the **dossier**: metadata a model must carry so gates can read it later.
- **hashing for integrity**: SHA-256 fingerprint of the frozen test set; pinning to a
  known-good constant is stronger than incumbent-comparison (proves "the one true
  exam", works on first promotion too). Alternatives: dataset version IDs (DVC hash),
  row-count+schema (weak), store-the-whole-set (wasteful), seed+recipe (fragile).
- empty-throne first-promotion handling (floor vs incumbent+margin).
- fail-closed governance; audit both accepts and rejects (failed-login analogy).
- FastAPI basics (decorated functions = endpoints, Pydantic request models, /docs);
  tags are strings → cast before comparing; 400 (malformed) vs 409 (policy conflict).

## Close-out / next session

- Commit M3 (Karthik's commits): `pipelines/register_baseline.py`, `control-plane/`,
  `serving/`, `docs/m3-session-notes.md`. Add `.gitignore` for `mlflow.db`, `mlruns/`,
  `data/`, `control-plane/audit.jsonl`, `docs/overfit_curve.png`.
- Cleanup: the dummy v2 (fake F1 / wrong hash) is still in the registry — delete it or
  leave as a demo artifact.
- Still pending from M1: SDK README in Karthik's own words (rule 5).
- **M4** = wire M1–M3 into a Dagster DAG + OpenLineage→Marquez; prediction logging;
  Evidently drift (PSI/KS); regime-change drift sim; trigger chain: drift → webhook →
  GH Actions retrain → control-plane gate → auto-promote. The promote API built here
  is the gate that M4's automated loop calls.
