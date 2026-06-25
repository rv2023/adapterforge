<!-- README skeleton — Karthik fills the prose + the JD-bullet column (rule 5: your words).
     The folder list + "what it is" are factual (describe the repo); the JD mapping, the
     one-liner, quickstart, and the demo narrative are YOURS to write. -->

# AdapterForge

AdapterForge is a fixed-income market intelligence MLOps platform that turns adapter-driven data ingestion into trained, evaluated, promoted, and served models. Its pitch is a self-healing production loop: detect drift, retrain, gate, promote, and reroute traffic without a human in the critical path.

**Architecture:** see [docs/architecture.md](docs/architecture.md) (full diagram) ·
**Plan + JD matrix:** [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) ·
**Usage:** [docs/USAGE.md](docs/USAGE.md)

## Repository map → JD coverage

| Folder | What it is | JD bullet it maps to |
|---|---|---|
| `adapter-sdk/` | pip-installable Data Adapter SDK — 3 adapters (HF/CSV/REST), versioned Pandera schemas, medallion + DVC, CI | Standardized data ingestion SDK with versioned schemas and repeatable dataset landing. |
| `pipelines/` | training/eval/distill, drift, the self-healing loop (MLflow, Dagster, QLoRA, Ray, distillation, model-aware retrain) | Production ML pipelines: MLflow tracking, Dagster orchestration, drift detection, retraining triggers, distillation, and distributed training. |
| `control-plane/` | FastAPI registry/promotion API — gated `/promote`, audit JSONL, `/lineage` | MLOps control plane for model registry, promotion policy, governance, auditability, and lineage lookup. |
| `serving/` | model-aware serving, naïve-vs-vLLM benchmark, Triton/ONNX, the M8 task+cost **router** (escalate cascade) | Multi-adapter serving with heterogeneous models, dynamic routing, hardware-aware dispatch, and vLLM/Triton coverage. |
| `observability/` | DCGM → Prometheus stack; (RCA bot — designed) | GPU memory observability, service SLOs, Prometheus/Grafana dashboards, and automated failure analysis. |
| `infra/` | Terraform EKS + GPU node group + a separate `addons/` module (GPU Operator, kube-prometheus-stack, Kueue) | Terraform IaC for a GPU Kubernetes platform with operators, scheduling add-ons, and production observability. |
| `k8s/` | Kueue quota/gang demo, in-place pod resize | Advanced Kubernetes GPU scheduling: Kueue quotas, gang admission, and sub-minute in-place resource resizing. |
| `.github/workflows/` | CI: test/lint, drift-triggered retrain, distill | CI/CD for tests, retraining, distillation, and multi-adapter deployment paths. |
| `docs/` | per-milestone concepts + session notes + the architecture diagram | Interview-ready evidence: architecture, JD coverage, benchmark notes, and milestone decisions. |

## Quickstart

```bash
make install
make test
```

Start MLflow to compare training runs:
```bash
make mlflow
```
Open http://localhost:5555.

Start the lineage stack, emit the demo OpenLineage events, then inspect the graph:
```bash
make lineage
python pipelines/lineage.py
```
Open the Marquez UI at http://localhost:3001. When finished, stop it with:
```bash
make lineage-down
```

For the full runbook, see [docs/USAGE.md](docs/USAGE.md).

## The showpiece demo

Start the live demo with the serving stack healthy, then inject drift into the incoming
traffic and step back. AdapterForge detects the quality drop, kicks off retraining,
re-runs the evaluation gates, promotes the passing adapter, and refreshes the router
without manual approval or failed requests. The useful part is watching the whole loop
close on its own: drift in, new adapter out, traffic rerouted while the service stays up.
