# M7 — Kubernetes GPU Platform: Concepts

Conceptual reference for M7 (the platform milestone: provision a GPU EKS cluster as
code, then scheduling / sharing / observability / RCA on top). Built up over the M7
kickoff session. Results/runbooks live elsewhere; this is the *why*.

---

## 0. What M7 is + the free-first strategy

You've been *renting* GPUs (RunPod); M7 is where you **build the platform that runs
them**. Pieces: Terraform EKS + GPU node group · NVIDIA GPU Operator · MIG + time-slicing
(carried from M6) · **Kueue** (quota + gang scheduling) · in-place pod resize (<1 min) ·
kube-prometheus-stack + DCGM · **RCA bot** (<10 min) · >98% SLO dashboard.

**Free-first (operating model):** most of it is practiced **free on local `kind`/
`minikube`** — Kueue, in-place resize, kube-prometheus-stack + Grafana + SLO, the RCA
bot logic. Only the GPU-specific bits + the EKS provisioning need **paid AWS**:
GPU Operator, MIG/time-slicing, DCGM-on-GPU. Pattern: build/rehearse free → one tight
paid EKS+GPU session → `destroy`.

---

## 1. The Terraform EKS module (`infra/`) — design decisions

| Decision | Choice | Why |
|---|---|---|
| Modules | community `terraform-aws-modules/{vpc,eks}` | standard, far less boilerplate than raw resources |
| Node groups | system (t3.medium, always on) + **GPU (g5.xlarge)** | keep system pods off the expensive node |
| **GPU cost lever** | GPU node group **`desired_size=0`** | **$0 GPU between sessions**; scale to 1 only when testing |
| GPU AMI | **base AL2023** (not prebaked GPU AMI) | so you *watch* the GPU Operator install driver/toolkit/device-plugin/DCGM |
| GPU pricing | on-demand | spot reclaim mid-MIG-lab is maddening for short labs |
| Sharing | `single_nat_gateway=true` | one NAT, not one-per-AZ (lab cost) |
| State | local (move to S3 later) | simplest; clean apply/destroy per session |
| GPU taint | `nvidia.com/gpu=true:NoSchedule` | only GPU workloads land there; the Operator tolerates it |

Plan verified: **61 to add, 0 change, 0 destroy**, 1 NAT, GPU `desired=0`. Cost of
applying with GPU=0 ≈ **~$0.20/hr** (control plane ~$0.10 + t3.medium + NAT); GPU adds
~$1/hr only when scaled. **`apply` only after plan review + explicit $/hr confirm; always
`destroy` at session end.**

---

## 2. Kueue — quota-aware batch queueing + gang scheduling

### Why vanilla K8s isn't enough
The default `kube-scheduler` places **one pod at a time, greedily**. For batch GPU jobs
that breaks two ways:
1. **No quota / no queue** — submit 100 jobs, K8s tries to schedule all pods; the
   ones that don't fit sit `Pending`. No fairness, no "team A gets 8 GPUs."
2. **No gang scheduling → deadlock** — a 4-worker distributed job: scheduler places 2
   pods (2 GPUs), leaves 2 `Pending`. The 2 running workers **block forever** at the
   NCCL rendezvous waiting for peers, **holding GPUs idle**. Multiple such jobs → cluster
   deadlock + wasted GPU $.

Kueue is a **queueing/admission layer *above* the scheduler** that fixes both.

### Object model
```
ResourceFlavor   "what kind of resource" (a10g GPUs; spot vs on-demand) -> maps to nodes via labels/taints
ClusterQueue     "the quota pool" (cluster-SCOPED, i.e. across namespaces) + policies (borrow/preempt/order)
LocalQueue       namespace-scoped pointer to a ClusterQueue  <- users submit jobs here
Workload         Kueue's internal record of a Job's TOTAL ask  <- the unit that queues + gets admitted
Cohort           group of ClusterQueues that lend each other idle quota
```
("Cluster" in ClusterQueue = cluster-*scoped*, NOT multi-cluster.)

### Admission flow (the key move)
```
1. submit Job labelled  kueue.x-k8s.io/queue-name: <localqueue>
2. Kueue webhook creates it SUSPENDED (spec.suspend=true) -> NO pods yet
3. Kueue makes a Workload (total ask, e.g. 2 GPUs)
4. ClusterQueue QUEUES it; when the FULL ask fits free quota -> ADMIT (suspend=false)
5. kube-scheduler now places the pods (they fit; Kueue reserved the quota)
6. job done -> quota released -> next Workload admitted
```
Kueue gates the **whole job** before any pod is created.

### Gang scheduling falls out of this
Admission is **all-or-nothing on the whole Workload** → a 4-worker job starts only when
all 4 workers' quota is free, so all start together. **No partial placement → no NCCL
deadlock.** This is the safety net the M5 2-GPU run wanted.

### Elastic quota
- **Nominal quota** guaranteed per ClusterQueue.
- **Borrowing** within a cohort (idle queue lends to busy sibling; reclaimed on demand).
- **Preemption** (higher-priority / quota-reclaiming Workload evicts borrowed/low-prio).
- **Queue strategy** StrictFIFO vs BestEffortFIFO.
→ high utilization (borrow) *and* guarantees (reclaim) — what raw requests/limits can't do.

### Single-cluster vs MultiKueue
Default Kueue = one cluster (across its namespaces). **MultiKueue** = a separate feature:
a **manager** cluster dispatches Workloads to multiple **worker** clusters. M7 is
single-cluster; MultiKueue is a "fleet" thing to name, not build here.

### Mental model
**Slurm for Kubernetes** — quota-aware batch queue that admits whole jobs (gang) onto
scarce GPUs, with borrowing + preemption so the hardware stays *fair* and *full*.
M7 deliverable: a ClusterQueue with GPU quota → submit 3 jobs, watch queue→admit →
gang-schedule a 2-worker job. Kueue = scheduler for the **training plane**.

---

## 3. Kueue vs KEDA (different halves)

| | **Kueue** | **KEDA** |
|---|---|---|
| Workload | batch **Jobs** (train/distill/retrain) | long-running **services** (serving) |
| Question | *should this job start?* (admit/queue) | *how many replicas?* (scale) |
| Killer feature | **gang scheduling**, quota, preemption | **scale-to-zero**, event triggers (queue depth, Kafka, Prometheus, cron) |
| Lifecycle | run-to-completion | always on |

One-liner: **Kueue decides *which jobs run*; KEDA decides *how big a running service is*.**
In this project the serving-elasticity (KEDA-like) job is done by **KServe's** autoscaler
(§5), so KEDA itself isn't needed.

---

## 4. NCCL — how multiple GPUs talk during training

**NCCL** = NVIDIA Collective Communications Library ("nickel"). Used in M5 (the 2-GPU run
+ `nccl-tests`, ~4.53 GB/s busbw). In **data-parallel** training each GPU holds a model
copy + processes a different batch slice; every step the gradients must be **averaged
across all GPUs** (so copies stay in sync) — that's an **all-reduce**, performed by NCCL
over the interconnect (NVLink intra-node, InfiniBand/RoCE inter-node) via ring/tree algos.
Collectives: all-reduce (the big one), broadcast, all-gather, reduce-scatter.

**Only matters with 2+ GPUs:**
| Layout | NCCL path |
|---|---|
| 1 GPU | none (nothing to talk to) |
| multi-GPU, one node | intra-node NVLink/PCIe |
| multi-node | inter-node network (interconnect matters) |

**Tie to gang scheduling:** the deadlock is a **multi-*pod*** problem (workers across
nodes). Started pods block at the NCCL rendezvous if peers never schedule → gang
scheduling admits all pods together so the NCCL group forms. (A single pod asking for 2
GPUs is already atomic — no gang issue.)

---

## 5. Namespaces — why, and do we need them

Namespaces = K8s logical partitioning for sharing one cluster safely: per-namespace
**RBAC**, **ResourceQuota** (+ Kueue **LocalQueue**), network policies, organization,
name scoping. They matter for **multi-tenant** clusters (many teams/envs).
**For this solo project: not strictly needed** — one namespace works, or a couple for
tidiness (`serving`/`training`/`monitoring`). The team-a/team-b framing was illustrating
*why Kueue is designed around namespaces*, not a requirement.

---

## 6. KServe — the Kubernetes serving layer (M8, but clarified here)

**"Serverless inference for K8s."** A **serving orchestration layer** (not an engine):
you declare an **`InferenceService`** CR (point at a model `storageUri` + a
ServingRuntime) and KServe builds the Deployment/Service/autoscaler/routing and pulls the
model. It **wraps** engines (vLLM/Triton/TorchServe), doesn't replace them.

What it gives vs raw Deployment+Service: **autoscaling incl. scale-to-zero** (idle → 0
pods → $0), **canary / traffic-splitting**, model pulling from storage, pluggable
**ServingRuntimes**, and the **Open Inference Protocol = the KServe v2 API** (the same
`/v2/models/.../infer` your `triton_client.py` already speaks). Built on Knative
(autoscaling) + Istio (networking); newer "raw deployment" mode can skip Knative.

Relations: **engine vs orchestrator** — vLLM/Triton run a model fast; KServe deploys/
scales/versions/routes them on K8s. vs Kueue/KEDA — KServe is the serving-specific layer
whose autoscaler covers the scale-to-zero/traffic-elasticity job.

**Why M8 uses it:** deploy router + vLLM multi-LoRA + student via KServe → scale-to-zero
for rare adapters, and **canary/traffic-split = the zero-downtime adapter hot-swap**
(drift → new version promoted → shift traffic gradually, zero failed requests).

One-liner: *declare an `InferenceService`; KServe runs your model on an engine like
Triton/vLLM with autoscaling, scale-to-zero, and canary, speaking the v2 protocol.*
