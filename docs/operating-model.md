# Compute Operating Model (where work runs)

The standing rule for the rest of AdapterForge. Three tiers; develop free, execute
paid, always tear down.

| Tier | Where | What runs there | Milestones | Cost |
|---|---|---|---|---|
| **1. Local** | Laptop (CPU) | All CPU work **+ writing/smoke-testing GPU code** before it goes up | M1–M4 + all plumbing (~70%) | **$0** |
| **2. RunPod** | Rented GPU, per-session | GPU *learning* runs: QLoRA fine-tune, bf16 efficiency, Ray/NCCL 2-GPU, vLLM benchmark, A100 MIG lab | M5, M6 | **per-minute** (~$1–25 total) |
| **3. Own cluster** | EKS GPU nodes via Terraform | **Platform-level** GPU work: K8s scheduling, GPU Operator, Kueue, MIG, observability, RCA, multi-adapter routing, KServe | M7, M8 | **per-node-hour** (burst, ~$20–30) |

## The pattern (every tier)

**Develop on Tier 1 (free) → execute on the paid tier → tear down.** Never run on a
GPU what a CPU can do. GPUs are only for **training** and **LLM inference**; everything
else (SDK, validation, MLflow, registry, drift, Dagster, lineage, RCA, dashboards, CI,
serving the small student model) is CPU.

## Tier 3 is not a re-run of Tier 2

You do **not** redo M5/M6 experiments on the cluster. Tier 3 is the next layer up: instead
of *running one job* (SSH + script), you run the **platform that schedules many jobs**
(containerize + submit to a queue). Same training code; what's new is queuing, fair-share
quotas, GPU sharing, cluster-wide observability, auto-RCA, hot-swap routing. Single-box
experiments (e.g. the A100 MIG lab) can stay on RunPod even in Tier 3 — cheaper than
holding a cluster node.

## Cost guardrail (CLAUDE.md) applies to Tiers 2 AND 3

Both cost real money. Claude never starts a pod or `terraform apply`s a node without an
explicit $/hr confirmation. Always tear down:
- **RunPod:** Stop + Terminate in the console (see [runpod-workflow.md](runpod-workflow.md)).
- **EKS:** `terraform destroy` (always `plan` before `apply`).

> Note: Tier 3 here is **cloud** (rented EKS nodes), not hardware you own. A physical
> on-prem GPU flips to a utilization game (you've paid already → keep it busy / scale to
> zero) — out of scope for this project.
