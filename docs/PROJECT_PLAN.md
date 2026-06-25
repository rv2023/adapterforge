# Project Plan: "AdapterForge" — A Multi-Adapter LLM MLOps Platform

**What this is:** One continuous project, built across 9 milestones (~16 weeks), whose
finished state implements the Lenovo MLOps Engineer JD line by line. Every milestone
produces a working, demoable artifact in a single monorepo.

**Hardware reality:** No local NVIDIA GPU required, ever.
- Laptop (CPU): SDK code, Dagster, MLflow, FastAPI, tests, CI — ~70% of the project
- Google Colab / Kaggle (FREE GPU): all of Milestone 0, quick GPU experiments
- RunPod (rented per session): fine-tuning, MIG, serving benchmarks (~$15–25 total)
- AWS: S3 throughout (cents); EKS GPU node in short bursts for M7–M8 (~$20–30 total)

**The monorepo (create this in Milestone 1):**
```
adapterforge/
├── adapter-sdk/          # Data Adapter SDK — pip-installable package
├── pipelines/            # Dagster DAGs: ingest, train, distill, retrain
├── control-plane/        # Registry/promotion/governance API (wraps MLflow)
├── serving/              # Multi-adapter router, vLLM configs
├── observability/        # Prometheus rules, Grafana dashboards, RCA automation
├── infra/                # Terraform (EKS, GPU nodes), K8s manifests
└── .github/workflows/    # CI/CD: tests, retrain, distill, deploy
```

**Anti-vibe-coding rules:** manual version before the tool; type every line;
break something on purpose per milestone; 5-line README per component in your own words.

---

## M0 — GPU Foundations (Week 0, parallel with M1)
**Runs on:** Google Colab or Kaggle (free GPU). Laptop for notes only.

Build/do:
- `nvidia-smi` literacy: driver version, CUDA version, VRAM, SM utilization
- Tensor to GPU, time matmul CPU vs GPU, mixed precision (fp32 vs fp16) timing
- Checkpoint lifecycle: save state_dict → check disk size → load → measure
  `torch.cuda.memory_allocated()` → reproduce the "100MB file → ~2GB VRAM" effect
- Kill a training loop mid-run, resume from checkpoint
- VRAM math drills: weights = params × bytes/precision; 7B fp16 ≈ 14GB; LoRA ≈ tens of MB

**Expected outcome:** You can read any GPU error message, explain the
CUDA stack (hardware → driver → toolkit → PyTorch → code) without notes, and
size any model to any GPU on paper. *JD line touched: NVIDIA ecosystem familiarity (CUDA).*

---

## M1 — Data Adapter SDK v0.1 (Weeks 1–2)
**Runs on:** Laptop + S3.

Build:
- `adapter-sdk`: a real pip-installable package (`pip install -e .`) with:
  - `BaseAdapter` abstract class: `read() → validate() → version() → land()`
  - `HFDatasetAdapter` (Financial PhraseBank), `CSVAdapter` (Twitter Financial
    News topics), `RestAPIAdapter` (live financial news API, e.g., Marketaux/Finnhub
    free tier) — three sources, one real and live,
    one interface = "standardized ingestion across diverse sources"
  - Pandera schemas as versioned code (`schemas/v1.py`, `v2.py`); every landed
    dataset is stamped with schema version + SDK version in S3 metadata
  - Lands to S3 medallion zones (raw/validated/features), DVC-tracked
- `corrupt.py` chaos tool; prove the SDK rejects bad batches
- GitHub Actions CI: pytest + ruff on every push (your first CI/CD pipeline)

**Expected outcome:** A versioned, tested, installable SDK any pipeline imports —
not scripts. You can explain why an SDK beats per-team scripts (the "100% adoption"
problem). *JD lines: Data Adapter SDK; built/extended SDKs; CI/CD foundations.*

---

## M2 — Training + Experiment Tracking (Weeks 3–4)
**Runs on:** Laptop.

Build:
- sklearn baseline (TF-IDF + LogisticRegression) for financial sentiment
  (bullish/bearish/neutral) on Financial PhraseBank via the SDK's feature output
- 3 runs tracked manually in CSV → then MLflow (params/metrics/artifacts/model)
- Overfit on purpose (50 epochs, 500 rows); plot train-vs-val loss yourself
- Start the habit that pays off in M5: log **step time** and **samples/sec**
  as first-class metrics in every run

**Expected outcome:** You can diagnose a training run from its curves and compare
any two runs in MLflow. Baseline model exists for the registry.
*JD lines: MLflow; production ML pipeline foundations.*

---

## M3 — MLOps Control Plane v0.1 (Week 5)
**Runs on:** Laptop.

Build:
- `control-plane/`: a FastAPI service that wraps MLflow Registry and OWNS policy:
  - `POST /models/{name}/promote` — enforces governance gates before promotion:
    eval F1 ≥ production + margin, schema version compatible, eval set hash matches,
    `approved_by` field present
  - Audit log: every promotion/demotion appended to an immutable JSONL in S3
  - `GET /models/{name}/lineage` — which data version + code commit produced it
- Serve the production model: `mlflow models serve` → FastAPI inference endpoint
  that asks the control plane "what's in Production?" at startup

**Expected outcome:** Promotion is no longer a human clicking a UI — it's a governed
API with policy + audit trail. This is the difference between "using MLflow" and
"architecting a control plane," and you can articulate it.
*JD lines: Architect/implement MLOps Control Plane — registry, versioning, promotion, governance.*

---

## M4 — Lineage, Drift, and the Automated Loop (Weeks 6–7)
**Runs on:** Laptop + S3.

Build:
- Wire M1–M3 into a Dagster DAG; emit **OpenLineage** events (run Marquez locally
  in Docker) — every dataset/model now traceable: source → schema version → run → model
- Prediction logging to S3 JSONL; confidence-distribution tracking
- Evidently drift jobs (PSI, KS-test); **regime-change drift simulation:** train
  on one market-period slice, then flood the endpoint with headlines from a
  different regime (hawkish/QT/crypto vocabulary) until PSI > 0.2 — real
  distribution shift, not synthetic corruption
- **The trigger chain:** drift alert → webhook → GitHub Actions
  `retrain.yml` → Dagster retrain run → control-plane promotion gate →
  auto-promote only if better. Fully hands-off; you only watch.

**Expected outcome:** A demo you can run live in an interview: inject drift,
watch the system detect, retrain, evaluate, and promote with zero human action —
with full lineage of what happened. *JD lines: data lineage (OpenLineage); automated
drift detection; real-time retraining triggers; CI/CD for retraining; Evidently.*

---

## M5 — LLM Fine-Tuning, Distillation, Distributed (Weeks 8–9)
**Runs on:** Develop on laptop/Colab with a tiny model → execute on RunPod
(4090/A10 ~$0.40/hr; one 2-GPU session; budget ~$10–12).

Build:
- Financial PhraseBank → instruction format ("Classify the market sentiment of
  this statement: ..." → bullish/bearish/neutral); **QLoRA fine-tune Qwen-1.5B**
  (HF Trainer + PEFT),
  remote-logging to your MLflow; adapter checkpoint registered via control plane
- **Efficiency experiment (the "5%+ step time" JD bullet):** baseline fp32 run vs
  bf16 + gradient accumulation vs tuned dataloader workers — measure step time in
  MLflow, document the % gain like a platform engineer would
- Wrap in **Ray Train** TorchTrainer; rent a 2-GPU pod once, run data-parallel,
  watch NCCL initialize; run `nccl-tests` all_reduce and read the bus bandwidth
- **RoCE/InfiniBand awareness module (theory + one benchmark):** why allreduce
  bandwidth gates multi-node scaling; NVLink vs PCIe vs RoCEv2 vs InfiniBand;
  what GPUDirect RDMA skips; NCCL env knobs (NCCL_IB_DISABLE, NCCL_SOCKET_IFNAME)
- **Distillation pipeline:** fine-tuned LLM as teacher labels 10k unlabeled
  headlines → train a DistilBERT student → register it. Now the registry holds
  **heterogeneous models** (1.5B LLM adapter + 66M student) — the raw material
  for hardware-aware routing in M8. Add `distill.yml` GitHub Actions workflow.

**Expected outcome:** You've run distributed training over NCCL, measured and
achieved a documented step-time reduction, and own a teacher/student model pair.
You can whiteboard why interconnect bandwidth dictates multi-node design.
*JD lines: distributed training, interconnect-aware optimization, NCCL,
distillation CI/CD, 5% efficiency gains.*

---

## M6 — Serving Frameworks + GPU Sharing (Weeks 10–11)
**Runs on:** RunPod (single GPU sessions; one A100 session for MIG ~$1.8/hr × 2–3 hrs).

Build:
- Serve the fine-tuned model with **vLLM** (+ LoRA adapter loading); benchmark
  p50/p95/p99 + throughput vs your M3 FastAPI endpoint; observe KV-cache VRAM
  growth as you raise concurrency
- One model through **Triton** (or KServe runtime) so you've touched JD-named tools
- **GPU sharing lab on a rented A100:** enable MIG, create 2 isolated instances,
  run LLM serving in one + student model in the other; then disable MIG and
  demo time-slicing (no memory isolation — crash it on purpose to see why)
- Run **DCGM exporter**, scrape with a local Prometheus, graph GPU memory/SM
  utilization during the benchmark

**Expected outcome:** A written benchmark comparing serving stacks, and hands-on
MIG/time-slicing evidence (screenshots + numbers) — the exact "GPU
virtualization/sharing" preferred qualification.
*JD lines: Triton/KServe; MIG/time-slicing/dynamic partitioning; GPU memory observability.*

---

## M7 — Kubernetes GPU Platform + Observability + RCA (Weeks 12–13)
**Runs on:** EKS via Terraform, one g5.xlarge GPU node created/destroyed per
session (~$1.3/hr; budget ~$20). kind/minikube locally for free practice first.

Build:
- **Terraform** module: VPC + EKS + GPU node group (the IaC JD line; commit it)
- **NVIDIA GPU Operator**: watch it install driver/toolkit/device-plugin/DCGM;
  deploy a pod requesting `nvidia.com/gpu: 1`; read DRA docs and map
  DeviceClass/ResourceClaim/ResourceSlice to what the device plugin did
- Time-slicing via device-plugin ConfigMap: 2 pods, 1 GPU
- **Kueue**: ClusterQueue with quota; submit 3 jobs, watch queueing/admission;
  gang-schedule a 2-worker job
- **Agility demo:** in-place pod vertical resize (K8s InPlacePodVerticalScaling) —
  resize a running workspace pod's CPU/memory and time it (**target <1 min, no restart**)
- **Observability stack:** kube-prometheus-stack + DCGM exporter + your
  pipeline/serving metrics; Grafana dashboards: GPU memory, latency, success rate
- **Automated RCA:** Alertmanager webhook → a Python "RCA bot" that, on failure,
  pulls pod logs + events + DCGM metrics + last MLflow run, classifies the failure
  (OOM / data validation / NCCL timeout / node pressure) and posts a structured
  report. Inject 3 failure types; measure time-to-classified-cause (**target <10 min**)
- Define the **>98% pipeline success-rate SLO** and put it on a dashboard

> **Carried in from M6 (2026-06-25):** the **MIG lab** (Piece 3) + **DCGM→Prometheus**
> (Piece 4) were deferred here — RunPod refused host-level MIG control ("Insufficient
> Permissions"), which the GPU Operator on our own EKS cluster gives us. Do the full MIG
> hands-on (enable MIG, 2 isolated instances LLM+student, OOM-isolation demo) + the
> time-slicing crash demo here, with DCGM per-instance metrics. Local scaffolds to reuse:
> `observability/m6-dcgm/` (DCGM+Prometheus compose) and `docs/runpod-m6-mig-dcgm.md`.

**Expected outcome:** A Terraform-provisioned GPU Kubernetes platform with
queueing, sharing, dashboards, alerting, sub-minute resizing, and automated RCA
that you've tested against injected failures — with numbers to quote.
*JD lines: IaC; GPU Operator/scheduling; in-place resizing; observability;
<10 min RCA; >98% success rate; <1 min resource resizing; Prometheus/Grafana.*

---

## M8 — Capstone Integration: Multi-Adapter Dynamic Routing (Weeks 14–16)
**Runs on:** EKS GPU node or RunPod.

Build:
- Train a **second LoRA adapter** (summarization) → registry now holds:
  base LLM + 2 task adapters + distilled student
- **Router service** in front of vLLM multi-LoRA: routes by task header;
  **hardware/cost-aware rule:** classification traffic → cheap student on a
  small GPU slice; generation traffic → LLM adapter on the big slice.
  Heterogeneous models + hardware-aware inference, demonstrated
- **Zero-downtime adapter swap:** drift on task A → automated retrain (M4 chain)
  → new adapter version promoted → router hot-swaps, requests never fail —
  this is "real-time triggers for retraining **or routing changes**"
- Deploy serving through **KServe** on the M7 cluster
- Stretch (advanced K8s JD line): a minimal `AdapterDeployment` CRD + operator
  (Python/kopf, ~150 lines): apply a YAML naming model+version → operator pulls
  from registry and updates the router. Custom resources + operator, demonstrated
- Polish: architecture diagram, README mapping each component to a JD bullet,
  5-minute recorded demo of the drift→retrain→promote→reroute loop

**Expected outcome:** The portfolio project. Every conversation-starter in the
Lenovo interview points at a folder in this repo.
*JD lines: multi-adapter serving with dynamic routing; heterogeneous models;
hardware-aware inference; K8s operators/CRDs; fully automated pipelines.*

### M8 carry-over context (decided during M6 kickoff, 2026-06-22)

**The two-plane architecture (the M8 target shape).** The platform has two
independent pipelines that meet at exactly one handoff — the registry + promotion
gate (the control plane):

```
SERVING plane  (always on)          RETRAINING plane  (event-driven)
  prod traffic → prod model            drift detected → retrain → gate → promote
  ← model-aware dispatch (M6)          ← loop.py / M4 automation, made model-aware (M8)
```

The serving plane must not know how models are made; the retraining plane must not
know how they're served. They couple only through the registry version + the gate.
M6 builds the **serving** half (model-aware dispatch). M8 builds the **retraining**
half end-to-end (drift → retrain → promote → reroute, zero failed requests).

**Debts M6 deliberately leaves for M8 (do NOT fix earlier):**

1. **Retraining is sklearn-bound and effectively dead.** `pipelines/loop.py`
   `retrain_and_register()` retrains a **sklearn LogReg** (~0.6885). Production is now
   the **LLM (0.8477)**, and promotion gate #1 requires *candidate F1 > production F1
   + margin* — so a retrained sklearn candidate is **rejected every time**. The
   self-healing loop cannot promote anything past the LLM until "retrain" means
   **retrain the LLM (QLoRA)**, not sklearn. M8 must make the retraining plane
   model-aware (retrain whatever kind production is; GPU-bound).

2. **The drift *sensor* piggybacks on sklearn.** `loop.py` `drift_detected()` reads
   the sklearn TF-IDF vocabulary (`model.named_steps["tfidf"]`) as its OOV detector.
   When sklearn is fully retired, the drift sensor needs replacing (text-stats /
   embedding-based), independent of any served model.

3. **sklearn is retired, NOT deleted.** The LogReg version stays registered in MLflow
   and its artifacts stay in DVC/registry (kept for lineage/history). M6 simply does
   not build a serving path for it. The live model kinds going forward are
   `lora_adapter` (LLM, current prod) and `distilbert` (student, M8 router cheap path
   + MIG lab second tenant).

**M6's model-aware groundwork that M8 reuses:** the `model_kind` registry tag
(`sklearn` / `lora_adapter` / `distilbert`) and the serving dispatch keyed on it are
the same primitives the M8 **cost-aware router** needs (route by kind: classification
→ cheap student slice, generation → LLM slice). See `docs/m6-serving-concepts.md`.

---

## JD Coverage Matrix

| JD requirement | Where built | Evidence artifact |
|---|---|---|
| MLOps Control Plane (registry/versioning/promotion/governance) | M3 | control-plane API + audit log |
| Data Adapter SDK, standardized ingestion, versioning | M1 | pip package, 3 adapters, schema versions |
| CI/CD: distillation, drift-triggered retraining, multi-adapter deploy | M4, M5, M8 | GitHub Actions workflows |
| Data lineage (OpenLineage), drift detection, real-time triggers | M4 | Marquez lineage graph, trigger chain demo |
| Multi-adapter serving, dynamic routing, heterogeneous, hardware-aware | M8 | router service + routing rules |
| In-place container resizing (<1 min) | M7 | timed resize demo |
| GPU memory observability | M6, M7 | DCGM + Grafana dashboards |
| Dynamic partitioning / time-slicing / MIG | M6, M7 | A100 MIG lab, K8s time-slicing config |
| Failure analysis / automated RCA <10 min | M7 | RCA bot + injected-failure timings |
| >98% success rate observability | M7 | SLO dashboard |
| 5%+ training step-time reduction | M5 | MLflow efficiency experiment |
| RoCEv2/InfiniBand awareness, NCCL | M5 | nccl-tests results + written explainer |
| Terraform IaC + GPU configs | M7 | infra/ Terraform module |
| MLflow, Evidently, Dagster (orchestrator), KServe/Triton/vLLM | M2–M8 | throughout repo |
| K8s advanced: operators, CRDs, GPU scheduling | M7, M8 | Kueue setup + AdapterDeployment operator |
| Expert Python for infra tooling | everywhere | SDK, control plane, RCA bot |

## Honest gaps (know these for interviews)
- **True multi-node RoCE/InfiniBand:** you'll have theory + single-node NCCL
  benchmarks, not a physical IB fabric. Say exactly that — "awareness" is what
  the JD asks for, and most candidates can't even explain GPUDirect RDMA.
- **Cluster scale / org adoption ("100%"):** you're one person; frame it as
  "I built the mechanism that makes adoption enforceable (SDK + CI gates)."
- **5+ years production MLOps:** your DevOps/SRE years + this platform is the
  bridge story — same reliability discipline, new workload.

## Budget + time summary
| Phase | Weeks | GPU cost |
|---|---|---|
| M0 | 0 (parallel) | $0 (Colab/Kaggle) |
| M1–M4 | 1–7 | $0 (laptop + S3 cents) |
| M5 | 8–9 | ~$10–12 (RunPod) |
| M6 | 10–11 | ~$8–10 (RunPod incl. A100) |
| M7 | 12–13 | ~$20 (burst EKS) |
| M8 | 14–16 | ~$10–15 |
| **Total** | **~16 weeks** | **~$50–60** |

---

# ADDENDUM 1 — Project Scope: What the LLM Is Trained to Achieve

**Product narrative:** AdapterForge is a **Fixed-Income Market Intelligence
Platform** for a fictional asset manager (PIMCO-inspired). A continuous stream of
financial news and earnings-call content feeds portfolio teams. One shared base
LLM (Qwen-1.5B), three specialized models:

| Model | Task | Training data | Method | Success bar |
|---|---|---|---|---|
| Adapter 1: Market Signal | Classify statement → bullish / bearish / neutral (+ topic tags: rates, credit, macro) | Financial PhraseBank (~4.8k, expert-labeled), instruction-formatted | LoRA SFT (M5) | F1 > M2 sklearn baseline on frozen eval set |
| Adapter 2: Earnings-Call Summarizer | Bullet-point PM briefs from call transcripts | ECTSum (public earnings-call transcript/summary pairs) | LoRA SFT (M8) | ROUGE-L > base model zero-shot |
| Model 3: Distilled Student | Bulk sentiment on the live headline stream at low cost | 10k headlines labeled by Adapter 1 (teacher) | Distillation → DistilBERT (M5) | Within 2–3 F1 pts of teacher at ~20× less compute |

**Why this scope works for the platform story:**
- Router (M8) has a real economic reason: the high-volume headline stream →
  student on a small GPU slice; earnings-call summarization and premium analysis →
  LLM adapters on the large slice (hardware/cost-aware routing). Not running a
  1.5B model on every news tick is exactly how finance shops think about inference cost.
- Drift has the most honest story of any domain: **market regime change.**
  Sentiment vocabulary shifts (hawkish/dovish, QT, new asset classes) degrade
  confidence → PSI fires → automated retrain → adapter hot-swap. This is the
  actual phenomenon quant/NLP teams fight in production.
- The live RestAPIAdapter means real, messy, batch-arriving data — the SDK and
  validation layer earn their keep against genuine upstream surprises.
- Heterogeneous models (1.5B LLM + 66M student) are required by the JD.

**Explicit scope boundaries:**
- NO pretraining from scratch (GPU-years; done by Qwen team). We do the SFT/LoRA
  stage — the industry meaning of "training" in MLOps contexts.
- NO RLHF (out of scope for a platform project).
- NO diffusion / text-to-image / VLM (bootcamp taxonomy noted; platform is
  model-agnostic by design — text-only proves it at 1/10th the GPU cost).

---

# ADDENDUM 2 — Bootcamp Coverage Patches

| Topic from bootcamp | Patch |
|---|---|
| Topology-aware placement | M7: theory module (comm cost hierarchy: NVLink > intra-node PCIe > intra-rack > RoCE/IB inter-node) + Kueue placement exercise |
| Tensor parallelism | M5: theory module — splitting *within* layers when a model exceeds one GPU's VRAM; contrast with data parallelism (hands-on); whiteboard-level mastery is the bar |
| Platform & business layer | M7: write a 1-page platform policy doc (train/serve separation, premium vs sandbox, SLA tiers, cost control, multi-tenant fairness, on-prem/cloud/edge tradeoffs); implement two policies in Kueue: quotas = team fairness, PriorityClasses = premium vs sandbox |
| Kubeflow Pipelines | M8 stretch restored: re-implement the Dagster DAG in Kubeflow Pipelines (one weekend) — learn orchestrator-generic vs tool-specific |
| TGI / TorchServe / DeepSpeed | Deliberately selection-criteria-only: deep on vLLM (LLM serving) + Triton/KServe (production serving); write a half-page "when would I pick each" note in M6 |
| Gen-AI taxonomy (text-to-image, VLM) | Out of scope by design; see Addendum 1 boundaries |

---

# ADDENDUM 3 — Complete Validation Parameter Specification

Every parameter the platform validates, by layer. Each row: metric, threshold,
where enforced, milestone.

## A. Data validation (enforced by adapter-sdk Pandera schemas — M1)
| Parameter | Threshold / rule | Action on fail |
|---|---|---|
| Schema: columns + dtypes | exact match to schemas/vN.py | reject batch, quarantine to s3 raw-failed/ |
| Null fraction per column | text: 0%; metadata: <1% | reject batch |
| Label vocabulary | sentiment ∈ {bullish, bearish, neutral}; topic ∈ approved tag set | reject batch |
| Duplicate rows | <0.5% of batch | dedupe + warn |
| Text length | 10 ≤ chars ≤ 2000 | drop rows + warn if >2% dropped |
| Batch volume | within ±50% of trailing batch mean | alert (possible upstream break) |
| Class balance | no class <10% of batch | warn (drift early signal) |
| Freshness | batch timestamp within expected window | alert |
| Encoding | valid UTF-8 | reject rows |

## B. Model validation — promotion gate (enforced by control plane — M3)
| Parameter | Threshold / rule |
|---|---|
| Macro F1 (frozen eval set) | ≥ production model + 1.0 pt margin |
| Per-class F1 | no class regresses >2 pts vs production |
| Precision / recall / confusion matrix | logged + attached to promotion audit record |
| Eval-set hash | must match registered hash (no eval-set drift) |
| Invalid-output rate (LLM) | <1% generations outside label vocabulary |
| ROUGE-1/2/L (summarizer) | ROUGE-L ≥ base zero-shot + margin; length 2–3 sentences ≥95% compliance |
| Teacher-agreement (student) | ≥90% agreement on holdout |
| Latency budget | p95 single-request inference within tier SLA on target hardware |
| VRAM fit | checkpoint + runtime footprint ≤ target GPU slice (paper math from M0) |
| Lineage completeness | data version + code commit + schema version all present |
| Approval field | approved_by present (governance) |

## C. Serving validation (M3 FastAPI → M6 vLLM → M8 router)
| Parameter | Healthy | Alert |
|---|---|---|
| Latency p50 / p95 / p99 | within tier SLA | p95 breach 5 min |
| Throughput (req/s, tokens/s) | ≥ load-test baseline | sustained drop >20% |
| Error rate | <0.5% | >1% over 5 min |
| Timeout rate | <0.1% | any sustained |
| KV-cache utilization (vLLM) | <90% | >95% (request shedding risk) |
| Queue depth | near zero | growing trend |
| Adapter swap downtime | 0 failed requests during hot-swap | any |

## D. Drift validation (Evidently — M4)
| Parameter | Threshold | Action |
|---|---|---|
| Input PSI (per feature / text stats) | <0.1 ok; 0.1–0.2 watch | >0.2 → trigger retrain webhook |
| KS-test p-value | >0.05 | <0.05 on key features → investigate |
| Prediction class distribution | within ±10 pts of training mix | breach → investigate + possible retrain |
| Confidence mean + entropy | stable vs training reference | sustained confidence drop → early drift alarm |
| Delayed-label accuracy | ≥ promotion-time accuracy − 2 pts | breach → retrain |
| Inference volume | within expected band | anomaly → upstream check |

## E. GPU / training validation (DCGM + MLflow — M5, M6, M7)
| Parameter | Healthy | Notes |
|---|---|---|
| VRAM utilization | <90% during serving; high during training is fine | OOM = failure class #1 for RCA bot |
| SM utilization | high during training (>80%) | low = dataloader bottleneck |
| Step time / samples-per-sec | tracked per run in MLflow | regression >5% blocks "efficiency" claims; this is the JD's 5% metric |
| NCCL all-reduce bus bandwidth | recorded from nccl-tests | interconnect-awareness evidence |
| GPU temp / power | within card spec | sustained throttling → placement issue |
| ECC errors | 0 | any → node failure analysis |
| OOM count | 0 in production | RCA bot classification target |

## F. Pipeline / platform SLOs (Prometheus + Grafana — M7)
| Parameter | Target |
|---|---|
| Pipeline run success rate | >98% (JD target) |
| Time-to-classified-RCA on failure | <10 min (JD target, tested via 3 injected failure types) |
| Workspace/job resource resize | <1 min, no restart (JD target) |
| End-to-end drift→promote loop | fully automated, zero human actions |
| Run duration | no >20% regression vs trailing mean |

---

# ADDENDUM 4 — v2 / Future Extensions (post-M8, OUT of core scope)

Deliberately **deferred** so the core stays mapped to the **MLOps-platform JD** (serving,
scheduling, GPU, drift, CI/CD, routing). Build these only **after M8**, as a separate phase —
they add product value + GenAI-app skills but shift the project's identity. Captured here so the
ideas aren't lost (decision 2026-06-21: keep plan, park these as v2).

## V2-A — Entity tagging (NER) — *moderate*
- **What:** a token-classification model that extracts entities from a headline — companies,
  tickers, instruments, bond/curve references ("10Y Treasury", "AAPL", "IG credit").
- **Why it fits:** becomes a **4th heterogeneous model** → richer routing; the dashboard can show
  per-entity sentiment ("Treasuries: bearish, Apple: bullish") instead of just headline-level.
- **Needs:** NER-labeled data (or an NER teacher to distill from), a token-classification head
  (not SEQ_CLS), entity-level F1 eval. ~1 week.
- **Routing:** the firehose path would run student-sentiment **+** NER together to tag the stream.

## V2-B — RAG analyst assistant — *major; changes project identity*
- **What:** free-form analyst Q&A. Analyst asks a question → **retrieve** relevant headlines /
  earnings summaries / student-tagged logs from a **vector DB** → a **generative** LLM answers,
  grounded on the retrieved context.
- **Why it's a different project:** our fine-tuned LLM is a *classifier/summarizer*, not a
  generator/QA model. RAG needs: an **embedding model**, a **vector store**, a **retrieval
  pipeline**, a **generative instruct LLM**, prompt/context assembly, and **RAG eval** (retrieval
  relevance + answer faithfulness — hard to measure). This moves from *MLOps-platform engineer*
  toward *GenAI-application builder* (a different JD). Weeks of work + more GPU cost.
- **Reuses the platform:** the M1–M8 control plane / registry / drift / serving / observability
  all still apply — RAG components would be *registered and served through the same platform*,
  which is the honest tie-back (the platform is model-agnostic by design).

**Sequencing:** finish Piece 5 → M6 → M7 → M8 first. Then V2-A (NER, fits routing), then V2-B
(RAG, as an explicit new phase). Update the JD framing if pursuing V2-B (it's a GenAI-app story).

---

# ADDENDUM 5 — v3 / Kellogg Asset-Management Practicum (built ON the platform)

A larger vision: a **multi-signal asset-intelligence product** — many input streams (tweets,
insider/political activity, macro, news, smart money, fundamentals, filings) → per-signal
Bull/Bear/Neutral ratings → consolidation per asset/ETF → a verdict model (direction + target
price) → dashboard + auto-generated reports. It is built **on** the AdapterForge platform (the
reusable engine), not instead of it. Two goals coexist: the **MLOps JD = the engine**; the
**practicum = a product on the engine**.

**Full plan, component mapping (reuse/new/hard), the target-price guardrails (backtesting +
leakage discipline), risk flags, and the phased rollout: see `docs/practicum-vision.md`.**

**Sequencing:** core platform (M1–M8) first — it's what the practicum runs on. Then build the
practicum eval-first, signal-by-signal. (Decision 2026-06-21: verdict target = directional **+**
target price, with a non-negotiable backtest framework.)