# AdapterForge — Usage Guide (through M4)

## What AdapterForge is (scope)

AdapterForge is a **Fixed-Income Market Intelligence MLOps platform**. Through M4,
it lets you:

1. **Ingest** financial text through a validated Data Adapter SDK into a medallion
   data lake (raw → validated/silver).
2. **Train & track** a sentiment baseline (TF-IDF + Logistic Regression), with every
   run recorded in MLflow.
3. **Govern promotion**: a control-plane API that only lets a model become Production
   if it passes policy gates (F1 margin, frozen eval-set hash, schema, approver) — with
   an immutable audit trail.
4. **Serve** the current Production model behind an inference API that asks the control
   plane what's live.
5. **Self-heal**: detect data drift, automatically retrain, and re-promote through the
   gate (only if better) — with full data lineage in Marquez.

**In scope now: M1–M4 (the platform).** Not yet built: M5 (LLM fine-tuning +
distillation), M6 (GPU serving), M7 (Kubernetes + observability), M8 (cost-aware router).
Each step below has a `make` shortcut; the raw command is shown too.

## Quick reference — activity → command

| I want to… | Command |
|---|---|
| Set up the project (venv + deps) | `make install` |
| Land the training data (PhraseBank) | `python -c "from adapter_sdk.adapters.hf import HFDatasetAdapter; HFDatasetAdapter().run()"` |
| Run the SDK tests | `make test` |
| Train the baseline + log to MLflow | `python pipelines/baseline.py` |
| Browse/compare runs in MLflow | `make mlflow` (→ :5555) |
| See overfitting on a curve | `python pipelines/overfit.py` |
| Register a model with its dossier | `make register` |
| Run the governance API | `make control-plane` (→ :8000) |
| Promote a model (governed) | `curl -X POST .../models/fpb-sentiment/promote …` |
| Serve predictions | `make serving` (→ :8001) |
| Get a prediction | `curl -X POST .../predict …` |
| Check for data drift | `make drift` |
| Orchestrate ingest→train→register | `make dagster` (→ :3000) |
| View data lineage | `make lineage` + `python pipelines/lineage.py` (→ :3001) |
| Run the self-healing loop | `make loop` |
| Stop the lineage stack | `make lineage-down` |

The sections below walk each of these in order, with context.

---

## Prerequisites

- **Python 3.11+** (3.13 used here)
- **Docker** (only for the Marquez lineage UI in M4)
- **`.env`** at the repo root for the live-news adapter (optional for the core flow):
  ```
  ALPHA_VANTAGE_API_KEY=your_key_here
  ```
- (Optional) AWS credentials if you use the DVC/S3 remote.

## Ports (so nothing collides)

| Port | Service |
|------|---------|
| 8000 | control plane (governance API) |
| 8001 | serving (inference API) |
| 5555 | MLflow UI |
| 5000 | Marquez API |
| 3000 | Dagster UI |
| 3001 | Marquez web UI |
| 6543 | Marquez Postgres |

---

## 0. Setup (once)

```bash
make install            # creates .venv, installs adapter-sdk[dev] + requirements.txt
source .venv/bin/activate
```
Manual equivalent:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e "adapter-sdk[dev]" && pip install -r requirements.txt
```

---

## 1. M1 — land data via the SDK

Run the HuggingFace adapter to fetch Financial PhraseBank, validate it against
schema v1, and land it in the **silver** zone:
```bash
python -c "from adapter_sdk.adapters.hf import HFDatasetAdapter; HFDatasetAdapter().run()"
ls data/validated/financial_phrasebank.parquet     # should exist
```
Run the SDK test suite:
```bash
make test
```

---

## 2. M2 — baseline + experiment tracking

Train the TF-IDF + LogReg baseline and log the run to MLflow:
```bash
python pipelines/baseline.py
```
Compare runs in the MLflow UI:
```bash
make mlflow            # http://localhost:5555   (stop with Ctrl+C)
```
See overfitting on a curve (writes `docs/overfit_curve.png`):
```bash
python pipelines/overfit.py
```
**Outcome:** the bar = macro-F1 ≈ **0.6885** on the frozen test set (C=10).

---

## 3. M3 — register, govern, serve

**3a. Register** the baseline as `fpb-sentiment` v1 with its dossier (test_f1,
eval_set_hash, schema_version, code_commit):
```bash
make register          # = python pipelines/register_baseline.py
```

**3b. Start the control plane** (leave it running — use a separate terminal):
```bash
make control-plane     # http://localhost:8000
```

**3c. Promote v1** through the gate (the first promotion onto an empty throne):
```bash
curl -X POST http://127.0.0.1:8000/models/fpb-sentiment/promote \
  -H "Content-Type: application/json" \
  -d '{"version":"1","approved_by":"your-name"}'
```
Inspect governance state:
```bash
curl http://127.0.0.1:8000/models/fpb-sentiment/production      # what's live + dossier
curl http://127.0.0.1:8000/models/fpb-sentiment/lineage/1       # provenance
cat control-plane/audit.jsonl                                   # the audit trail
```

**3d. Start serving** (separate terminal — it asks the control plane what's in
production and loads it):
```bash
make serving           # http://localhost:8001
```
Get a prediction:
```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Profits surged after the company raised full-year guidance"}'
```

---

## 4. M4 — drift, lineage, the self-healing loop

**4a. Drift detection** (OOV signal → PSI + KS, then an Evidently HTML report):
```bash
make drift             # prints PSI/KS verdict; writes docs/drift_report.html
```

**4b. Orchestration** with Dagster (the ingest → train → register DAG):
```bash
make dagster           # http://localhost:3000 → "Materialize all"
```

**4c. Lineage** with Marquez (needs Docker):
```bash
make lineage                       # Docker up: web :3001, api :5000
python pipelines/lineage.py        # emit OpenLineage events
# open http://localhost:3001 → namespace "adapterforge" → see the graph
make lineage-down                  # stop Marquez when done
```

**4d. The self-healing loop** (control plane must be up on :8000, production set):
```bash
make loop              # drift → retrain → register → gate → auto-promote IF better
```
With the unchanged baseline the gate **rejects** the retrain (not better than the
incumbent + margin) — production stays safe. That is the loop working: it only
promotes an improvement.

---

## The full demo, in order (the showpiece)

```bash
# terminal A
make control-plane

# terminal B
source .venv/bin/activate
make register
curl -X POST http://127.0.0.1:8000/models/fpb-sentiment/promote \
  -H "Content-Type: application/json" -d '{"version":"1","approved_by":"demo"}'
make drift             # → DRIFT: yes (PSI 16.7)
make loop              # → drift detected → retrain → gate decision (auto, no human)
```
Result: drift is detected, a candidate is retrained and registered, and the
governance gate decides whether to promote it — **end to end, zero human action.**

---

## Teardown

```bash
make lineage-down      # stop Marquez containers
deactivate             # leave the venv
```
Local state lives in `mlflow.db` / `mlruns/` (gitignored). Runtime logs
(`control-plane/audit.jsonl`, `serving/predictions.jsonl`, `docs/drift_report.html`)
are generated and gitignored.
