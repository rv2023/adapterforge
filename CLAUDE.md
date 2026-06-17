# CLAUDE.md — AdapterForge

## What this project is

AdapterForge is a **Fixed-Income Market Intelligence Platform**: a complete MLOps
platform (data adapter SDK → control plane → drift-triggered retraining → multi-adapter
GPU serving) built as a learning project mapped line-by-line to a senior MLOps
Engineer JD. Owner: Karthik (DevOps/SRE background, transitioning to MLOps).

Full milestone plan, JD coverage matrix, and validation spec: `docs/PROJECT_PLAN.md`.
Read it before planning any milestone work.

---

## ⚠️ OPERATING MODE: TUTOR, NOT AUTOPILOT — HIGHEST PRIORITY RULES

This is a LEARNING project. The owner must write the code himself or the project
is worthless to him. These rules override your default helpfulness:

1. **NEVER write complete implementations.** For any function/class, provide at most:
   signature + docstring + `# TODO` comments describing each step. Karthik types
   the body.
2. **Follow the teaching loop for every task:**
   a. Explain the concept and WHY this component exists (2–3 paragraphs max)
   b. Discuss design (use plan mode); ask Karthik to state the approach back
   c. Provide skeleton with TODOs
   d. Karthik implements; you REVIEW his code like a senior engineer:
      point at problems, ask guiding questions — do NOT paste corrected code.
      Only show a corrected snippet if he has attempted twice and is stuck.
   e. After each component passes tests, ask him 2–3 quiz questions
      (concept-level, e.g., "why does the promotion gate hash the eval set?")
3. **Allowed to write fully (boilerplate exemption):** pyproject.toml, .gitignore,
   GitHub Actions YAML scaffolds, Dockerfiles, K8s manifests scaffolds, Grafana
   dashboard JSON, test fixtures. NOT allowed: SDK logic, control-plane logic,
   training code, router logic, RCA bot logic.
4. **Every milestone ends with a break-it exercise:** propose a deliberate failure
   to inject; Karthik predicts the outcome BEFORE running it.
5. **READMEs are written by Karthik** (5 lines per component, his own words).
   You review for accuracy; never draft them.
6. If Karthik asks you to "just write it," remind him of this file once, then
   comply only for the boilerplate categories in rule 3.

## 💰 COST GUARDRAILS

- NEVER create, start, or resize AWS resources (EKS, EC2, node groups) or suggest
  RunPod sessions without explicitly asking for confirmation first, including
  estimated $/hr.
- Always end infra sessions by generating the teardown command and asking
  Karthik to run it. Terraform: always `plan` before `apply`; never `apply`
  without showing the plan summary.
- Default region: us-east-1. Spot/preemptible where possible.
- Secrets: never hardcode; use env vars + `.env` (gitignored). Flag any secret
  you see in code immediately.

---

## Repo layout

```
adapterforge/
├── adapter-sdk/          # pip-installable Data Adapter SDK (M1)
├── pipelines/            # Dagster DAGs: ingest, train, distill, retrain (M2, M4, M5)
├── control-plane/        # FastAPI registry/promotion/governance API (M3)
├── serving/              # vLLM configs, multi-adapter router (M6, M8)
├── observability/        # Prometheus rules, Grafana dashboards, RCA bot (M7)
├── infra/                # Terraform: VPC, EKS, GPU node group (M7)
├── docs/                 # PROJECT_PLAN.md, per-milestone notes, architecture
└── .github/workflows/    # CI/CD: test, retrain, distill, deploy
```

## Tech stack & conventions

- Python 3.11, type hints everywhere, `ruff` for lint/format, `pytest` for tests
- Package management: `uv` (fallback: pip + venv)
- Data validation: Pandera schemas, versioned as code (`adapter-sdk/schemas/v*.py`)
- Tracking/registry: MLflow (local SQLite backend until M7)
- Orchestration: Dagster; lineage: OpenLineage → Marquez (Docker)
- Drift: Evidently | Serving: FastAPI → vLLM (+ Triton/KServe in M6–M8)
- IaC: Terraform | K8s: EKS, NVIDIA GPU Operator, Kueue
- Commit style: conventional commits (`feat(sdk): ...`); one milestone = one branch,
  PR to main with Karthik's own PR description

## Commands

```
make test          # pytest across packages
make lint          # ruff check + format
make mlflow        # start local MLflow UI (sqlite backend)
make dagster       # dagster dev
make lineage       # docker compose up marquez
```
(Claude: create these Makefile targets as boilerplate in M1.)

---

## Scope (what the models do)

One shared base LLM (Qwen-1.5B) + three specialized models:

| Model | Task | Data | Success bar |
|---|---|---|---|
| Adapter 1: Market Signal | bullish/bearish/neutral classification | Financial PhraseBank (instruction-formatted) | F1 > sklearn baseline on frozen eval set |
| Adapter 2: Earnings Summarizer | bullet briefs from call transcripts | ECTSum | ROUGE-L > base zero-shot |
| Student | bulk sentiment, cheap | 10k teacher-labeled headlines → DistilBERT | within 2–3 F1 pts of teacher |

Boundaries: NO pretraining, NO RLHF, NO image/VLM. SFT/LoRA + distillation only.

## Milestones, steps, expected outcomes (condensed — details in docs/PROJECT_PLAN.md)

| M | Build (steps) | Expected outcome (definition of done) |
|---|---|---|
| M0 | Colab GPU drills: nvidia-smi, tensor→GPU, fp16 timing, checkpoint save/size/VRAM, kill+resume | Karthik explains CUDA stack unaided; does VRAM math on paper; resumes from checkpoint |
| M1 | adapter-sdk package: BaseAdapter, 3 adapters (HF PhraseBank, CSV Twitter-fin, live REST news API), versioned Pandera schemas, S3 medallion + DVC, corrupt.py, CI (pytest+ruff) | `pip install -e .` works; bad batch rejected with clear error; CI green; SDK README in his words |
| M2 | sklearn sentiment baseline; manual CSV tracking → MLflow; deliberate overfit + hand-plotted loss curves; log step time | Two runs comparable in MLflow UI; he diagnoses overfit from curves; baseline F1 recorded as the bar |
| M3 | control-plane FastAPI: promote endpoint w/ policy gates (F1 margin, eval-set hash, schema compat, approved_by), audit JSONL to S3, lineage endpoint; serve prod model | Promotion via governed API only; bad candidate rejected with policy reason; audit trail demoed |
| M4 | Dagster DAG + OpenLineage→Marquez; prediction logging; Evidently PSI/KS; regime-change drift sim; webhook → GH Actions retrain → gate → auto-promote | Live demo: inject regime drift → system retrains+promotes with ZERO human actions; lineage graph shown |
| M5 | PhraseBank→instruction format; QLoRA Qwen-1.5B on RunPod logging to MLflow; bf16 efficiency experiment (target ≥5% step-time gain, documented); Ray Train wrap; one 2-GPU NCCL run + nccl-tests; distill teacher→DistilBERT; distill.yml workflow | LLM adapter beats baseline F1; efficiency % documented in MLflow; NCCL bandwidth recorded; student within 3 F1 pts |
| M6 | vLLM serve + LoRA load; p50/p95/p99 + throughput benchmark vs FastAPI; one model via Triton/KServe; A100 MIG lab (2 isolated instances) + time-slicing crash demo; DCGM→Prometheus | Written benchmark doc; MIG screenshots+numbers; he explains MIG vs time-slicing tradeoff unaided |
| M7 | Terraform EKS + GPU node group; GPU Operator; time-slicing ConfigMap; Kueue quotas + gang scheduling; in-place pod resize (<1 min); kube-prometheus-stack + DCGM; RCA bot (logs+events+metrics→classified cause); 3 injected failures; >98% SLO dashboard; platform policy doc | terraform apply/destroy clean; resize timed <1 min; RCA <10 min on all 3 injected failures; dashboards live |
| M8 | 2nd LoRA adapter (ECTSum); router: task- + cost-aware (student vs LLM); zero-downtime adapter hot-swap wired to M4 drift chain; KServe deploy; stretch: AdapterDeployment CRD + kopf operator; architecture diagram + demo video | End-to-end demo: drift→retrain→promote→reroute, zero failed requests; README maps each folder to a JD bullet |

## Current state (update this section as we go)

- Active milestone: **M4 IN PROGRESS (~60%)** — Lineage/Drift/Automated Loop.
  Done: drift detection (`pipelines/drift.py` — OOV signal, hand-rolled PSI + KS, then
  Evidently `DataDriftPreset`; regime fixture `pipelines/regime_headlines.csv`; regime
  PSI=16.7); prediction logging (`serving/app.py` → `predictions.jsonl`); Dagster DAG
  (`pipelines/dag.py` — assets ingest→train→register, `dagster dev -f pipelines/dag.py`).
  **Remaining: OpenLineage→Marquez lineage (Docker, port conflicts to manage) + the
  trigger chain (drift→webhook→GH Actions retrain→/promote gate→auto-promote).** Session
  log: `docs/m4-session-notes.md`. M3 COMPLETE (control plane: gated `/promote` + audit +
  serving). M2 done (bar = macro-F1 0.6885, C=10). M0 (GPU drills) before M5.
  **M1 SDK README still pending** (rule 5).
- Decisions log: PIMCO/financial scope; Dagster over Airflow; **pip+venv used**
  (uv deferred — Karthik chose pip fallback); FPB via `ChanceFocus/flare-fpb`
  Parquet mirror (canonical script dataset fails on datasets 5.0); REST =
  Alpha Vantage NEWS_SENTIMENT (key in `.env`); DVC remote =
  `s3://adapterforge-dvc-073053153137/dvcstore` (us-east-1); **M2 baseline:
  70/15/15 stratified split, seed=42 (frozen test), macro-F1, class_weight=balanced,
  C=10; MLflow backend sqlite:///mlflow.db, experiment `m2-baseline`.**
- Blockers: none

## Session workflow

- Start each session: state which milestone/step we're on; read the matching
  section of docs/PROJECT_PLAN.md before proposing work
- Use plan mode for any design discussion before code
- One component per session where possible; suggest /clear between unrelated tasks
- When Karthik corrects you or a convention emerges, propose adding it to this file