<!-- README skeleton — Karthik fills the prose + the JD-bullet column (rule 5: your words).
     The folder list + "what it is" are factual (describe the repo); the JD mapping, the
     one-liner, quickstart, and the demo narrative are YOURS to write. -->

# AdapterForge

<!-- TODO(Karthik): 1–2 sentences, your words — what this is + the one-line pitch
     (a Fixed-Income Market Intelligence MLOps platform, mapped to the MLOps Engineer JD;
     the showpiece = the self-healing drift→retrain→promote→reroute loop). -->

**Architecture:** see [docs/architecture.md](docs/architecture.md) (full diagram) ·
**Plan + JD matrix:** [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) ·
**Usage:** [docs/USAGE.md](docs/USAGE.md)

## Repository map → JD coverage

<!-- TODO(Karthik): fill the "JD bullet" column in your words. Cross-ref the JD Coverage
     Matrix in docs/PROJECT_PLAN.md. The pitch: every folder points at a JD requirement. -->

| Folder | What it is | JD bullet it maps to |
|---|---|---|
| `adapter-sdk/` | pip-installable Data Adapter SDK — 3 adapters (HF/CSV/REST), versioned Pandera schemas, medallion + DVC, CI | _TODO_ |
| `pipelines/` | training/eval/distill, drift, the self-healing loop (MLflow, Dagster, QLoRA, Ray, distillation, model-aware retrain) | _TODO_ |
| `control-plane/` | FastAPI registry/promotion API — gated `/promote`, audit JSONL, `/lineage` | _TODO_ |
| `serving/` | model-aware serving, naïve-vs-vLLM benchmark, Triton/ONNX, the M8 task+cost **router** (escalate cascade) | _TODO_ |
| `observability/` | DCGM → Prometheus stack; (RCA bot — designed) | _TODO_ |
| `infra/` | Terraform EKS + GPU node group + a separate `addons/` module (GPU Operator, kube-prometheus-stack, Kueue) | _TODO_ |
| `k8s/` | Kueue quota/gang demo, in-place pod resize | _TODO_ |
| `.github/workflows/` | CI: test/lint, drift-triggered retrain, distill | _TODO_ |
| `docs/` | per-milestone concepts + session notes + the architecture diagram | _TODO_ |

## Quickstart

<!-- TODO(Karthik): the few commands to run it — make test / make mlflow / make lineage,
     and where to look. (See docs/USAGE.md.) Your words. -->

## The showpiece demo

<!-- TODO(Karthik): describe the live demo in your words — inject drift, walk away, watch
     it retrain → re-gate → promote → router reroutes, zero humans, zero failed requests. -->
