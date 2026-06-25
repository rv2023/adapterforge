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

## Compute tiers (where work runs) — see docs/operating-model.md

Develop free, execute paid, always tear down. Never run on a GPU what a CPU can do.
- **Tier 1 — Local (laptop CPU, $0):** all CPU work + writing/smoke-testing GPU code
  before it ships up (M1–M4 + plumbing, ~70%).
- **Tier 2 — RunPod (rented GPU, per-minute):** GPU *learning* runs — QLoRA fine-tune,
  bf16 efficiency, Ray/NCCL, vLLM benchmark, MIG lab (M5, M6). **Colab dropped.**
- **Tier 3 — Own cluster (EKS GPU via Terraform, per-node-hour):** platform-level GPU
  work — K8s scheduling, GPU Operator, Kueue, observability, RCA, routing (M7, M8).
- Cost guardrail applies to Tiers 2 **and** 3: confirm $/hr first, tear down after.

---

## Scope (what the models do)

One shared base LLM (Qwen-1.5B) + three specialized models:

| Model | Task | Data | Success bar |
|---|---|---|---|
| Adapter 1: Market Signal | bullish/bearish/neutral classification | Financial PhraseBank (instruction-formatted) | F1 > sklearn baseline on frozen eval set |
| Adapter 2: Earnings Summarizer | bullet briefs from call transcripts | ECTSum | ROUGE-L > base zero-shot |
| Student | bulk sentiment, cheap | 10k teacher-labeled headlines → DistilBERT | within 2–3 F1 pts of teacher |

Boundaries: NO pretraining, NO RLHF, NO image/VLM. SFT/LoRA + distillation only.
(NER + RAG analyst assistant deferred to **v2** — captured in PROJECT_PLAN Addendum 4;
build only after M8 so the core stays mapped to the MLOps-platform JD.)

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

- Active milestone: **M8 (Capstone — multi-adapter routing).** Kickoff pending.
  Deferred/pending M6+M7 tasks consolidated in **`docs/deferred-backlog.md`** (MIG/
  time-slicing [need GPU], in-place resize [scaffolded, run on kind], RCA bot [designed,
  build on kind], SLO dashboard, M6 write-up personalization, M1 README). AWS fully
  destroyed ($0). **M7 CORE DONE** (Terraform EKS + GPU Operator + DCGM→Prometheus +
  Kueue, all on real EKS); rest = backlog. Prior M7 detail:
- (M7) **K8s GPU Platform.** Kickoff 2026-06-25:
  K8s=solid→assume it; free-first (kind) strategy. **Terraform EKS+GPU module authored
  (`infra/`)** — community modules, system + GPU node groups, GPU **desired=0** cost lever,
  prebaked-driver GPU AMI (`AL2023_x86_64_NVIDIA`) + Operator `driver.enabled=false`
  (revised from base-AMI — Operator can't build drivers on Amazon Linux; see concepts §7),
  single NAT, EKS 1.31, IRSA. **DONE this milestone so far:** cluster applied; GPU node
  scaled via `aws eks update-nodegroup-config` (module ignore_changes on desired_size);
  GPU Operator provisioned (device-plugin/DCGM/MIG-mgr; **DCGM→Prometheus = Piece 4 ✅**);
  **Kueue ✅** (quota+gang demoed, `k8s/m7-kueue/`). Addons codified in **separate**
  `infra/addons/` module (own state; helm/k8s providers via exec → avoids cluster+chart
  same-state chicken-egg): gpu-operator, kube-prometheus-stack (ephemeral), kueue.
  **DEFERRED:** in-place resize + **RCA bot** (designed, concepts §9) + SLO dashboard =
  non-GPU → do free on **kind** next; **MIG (P3)** needs A100/H100 (g5/A10G can't MIG;
  AWS A100 $$+quota) → separate session if ever; time-slicing needs a GPU session.
  **Cost:** ~$0.20/hr base + ~$1/hr per GPU; `terraform destroy` addons then infra between
  sessions (code in git, re-apply ~15 min). Concepts: `docs/m7-concepts.md`. Log:
  `docs/m7-session-notes.md`. **M6 COMPLETE** (P0-2 done; P3/P4 → M7; write-ups drafted).
  Historical M6 detail:
- (M6) Active milestone was **M6 (Serving + GPU sharing).** Kickoff done
  (2026-06-22): concepts taught + saved (`docs/m6-serving-concepts.md` — vLLM
  autoregression/prefill-decode/KV-cache, continuous batching, PagedAttention,
  time-slicing/MPS/MIG + the two-layer "sharing vs vLLM" model, dispatch-vs-router).
  Piece order decided: **(0) model-aware serving fix [local, free] → (1) vLLM+benchmark
  on RunPod → (2) Triton/KServe + selection note → (3) FULL hands-on A100 MIG lab →
  (4) DCGM→Prometheus**. Design locked for the model-aware fix: stamp a `model_kind`
  registry tag (`lora_adapter`/`distilbert`; sklearn **retired not deleted**), serving
  reads it from the control-plane `/production` response and dict-dispatches to the
  right (loader, predictor). M8 carry-over (two-plane architecture; retraining is
  sklearn-bound + can't pass the gate vs the LLM; drift sensor piggybacks on TF-IDF)
  written into `docs/PROJECT_PLAN.md` M8 section. Progress log: `docs/m6-session-notes.md`.
  **M5 COMPLETE** (see memory/m5-progress.md). Historical M5 detail below:
- (M5 history) Steps done: (1) instruction-format
  (`pipelines/instruction_format.py` → `data/instruction/{train,val,test}.jsonl`, chat-messages
  format, reuses M2 `split_data` so test set is bitwise-identical to the 0.6885 frozen set);
  (2) QLoRA training script (`pipelines/finetune.py`, TRL `SFTTrainer` + PEFT, dev/real switch via
  `MODEL_NAME`/`USE_4BIT`/`MAX_STEPS`) — CPU smoke test PASSED (loss fell, 34 MB adapter saved).
  LLM deps added to `requirements.txt` (torch CPU/transformers/peft/trl/accelerate; bitsandbytes Colab-only).
  (3) **REAL Colab T4 run DONE** (4-bit Qwen-1.5B, batch 16, 3 epochs, ~60 min) — mild overfit at
  epoch 3 (eval_loss 1.079→1.113) but fine. (4) **Eval (`pipelines/eval_adapter.py`, generative
  classifier: PeftModel + chat-template + greedy generate + parse word + macro-F1) DONE → macro-F1
  = 0.8477, BEATS the 0.6885 baseline by ~16 pts.** Core Piece-1 deliverable met. (5) **REGISTER +
  PROMOTE DONE — Piece 1 fully closed.** Adapter registered as `fpb-sentiment` **v14** with a real
  dossier (test_f1=0.8477; eval_set_hash recomputed fresh on laptop, matches the gate's pinned
  EXPECTED_HASH bit-for-bit; schema v1; commit 00e8af4), then **promoted through the gated /promote
  (approved_by=karthik) → LLM is now production** (audit line: previous_production=1, the sklearn
  baseline it dethroned). Code: refactored `register_baseline.py` into model-agnostic
  `register_model_with_dossier(test_df, test_f1, log_and_register)` + thin `register_sklearn` wrapper
  (callers in loop.py/dag.py updated) + new `register_adapter` (reads `models/fpb-lora/eval_metrics.json`,
  n_test cross-check, logs adapter via `log_artifacts` + `MlflowClient.create_model_version` — MLflow 3.x
  `register_model("runs:/…")` needs a *logged model*, raw artifacts don't qualify). `eval_adapter.py`
  `main()` now returns the f1 + writes `eval_metrics.json` so future evals auto-emit it (the v14 json
  was hand-written before this change). **Known next domino (M6, not a bug):** loop.py:30 + serving/app.py
  use `mlflow.sklearn.load_model` → fail against the LLM production model (`No such artifact: 'MLmodel'`,
  confirmed); making serving/drift model-aware is M6/M8. Break is runtime-only (prod alias in gitignored
  mlflow.db); committed code is clean. **Next: Piece 2** (bf16 efficiency experiment, the JD "5%"),
  Piece 3 (Ray/NCCL on RunPod), Piece 5 (distillation).
  M0–M4 COMPLETE. M4 (Lineage/Drift/Automated Loop) all done & demoed.
  Drift (`pipelines/drift.py` — OOV + hand-rolled PSI/KS + Evidently; regime fixture;
  PSI=16.7); prediction logging (`serving/app.py` → `predictions.jsonl`); Dagster DAG
  (`pipelines/dag.py`); OpenLineage→Marquez lineage (`docker-compose.marquez.yml` +
  `pipelines/lineage.py`, graph renders); the self-healing loop (`pipelines/loop.py` —
  drift→retrain→/promote gate→auto-promote; BOTH paths proven: reject-if-not-better and
  accept-on-better) + CI artifact `.github/workflows/retrain.yml` (repository_dispatch
  webhook, live in M7). Registration is centralized in `register_model_with_dossier`
  (pipelines/register_baseline.py), reused by dag.py + loop.py (deduped). Usage guide:
  `docs/USAGE.md`. Session log: `docs/m4-session-notes.md`. **M0 COMPLETE** (GPU
  foundations done on Colab T4 — CUDA stack, VRAM math, training concepts from scratch;
  `docs/m0-session-notes.md`). **Next: M5** (QLoRA fine-tune Qwen-1.5B + distillation;
  costs money — RunPod ~$0.40/hr, confirm $/hr first). M3 COMPLETE (gated `/promote` + audit +
  serving). M2 done (bar = macro-F1 0.6885, C=10). **M1 SDK README still pending** (rule 5).
  Cleanup pending: registry demo cruft (dummy v2 bad-hash + v3–v13 reruns; **production=v14, the LLM**);
  Marquez Docker stack may still be running. Trained adapter (`models/fpb-lora/`) lives only on the
  laptop + as the v14 MLflow artifact; `models/` gitignored — still need a real home (DVC/S3).
- Decisions log: PIMCO/financial scope; Dagster over Airflow; **pip+venv used**
  (uv deferred — Karthik chose pip fallback); FPB via `ChanceFocus/flare-fpb`
  Parquet mirror (canonical script dataset fails on datasets 5.0); REST =
  Alpha Vantage NEWS_SENTIMENT (key in `.env`); DVC remote =
  `s3://adapterforge-dvc-073053153137/dvcstore` (us-east-1); **M2 baseline:
  70/15/15 stratified split, seed=42 (frozen test), macro-F1, class_weight=balanced,
  C=10; MLflow backend sqlite:///mlflow.db, experiment `m2-baseline`.**
- M5 decisions: **GPU runs on RunPod, Colab dropped** (3-tier model, docs/operating-model.md);
  **GPU MLflow logs via `MLFLOW_TRACKING_URI` → remote/cloud tracking server** (prereq: that
  server is a Tier-3 cloud-infra component; `results/m5-efficiency.log` is the interim safety
  net); Piece-2 efficiency experiment levers = **fp32 vs bf16 × 4-bit on/off + a dataloader-
  workers pair** (`pipelines/efficiency_experiment.py`); Kubeflow stays an **M8 stretch**
  (Dagster primary).
- Blockers: none

## Session workflow

- Start each session: state which milestone/step we're on; read the matching
  section of docs/PROJECT_PLAN.md before proposing work
- Use plan mode for any design discussion before code
- One component per session where possible; suggest /clear between unrelated tasks
- When Karthik corrects you or a convention emerges, propose adding it to this file