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
| GPU AMI | **REVISED → `AL2023_x86_64_NVIDIA`** (prebaked driver) + Operator `driver.enabled=false` — see §7 | base-AL2023 + Operator-driver fails on Amazon Linux; prebaked driver is the reliable EKS path |
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

---

## 7. Provisioning a GPU node: the software stack + GPU Operator vs AMI vs device-plugin

### The stack a GPU node needs
For a pod to *use* a GPU on K8s, the node needs, bottom to top:

| Layer | Does what | Missing → |
|---|---|---|
| 1. NVIDIA **driver** | kernel software so the OS talks to the GPU | GPU is a brick |
| 2. container **runtime/toolkit** | lets *containers* reach the GPU | pod can't see GPU |
| 3. **device plugin** | tells K8s "node has N GPUs" (exposes `nvidia.com/gpu`) | scheduler blind to GPUs |
| 4. **DCGM exporter** | GPU metrics → Prometheus | no Piece 4 |
| 5. **MIG-manager** | creates/manages MIG partitions | no Piece 3 |

1–2 = make the GPU work on the node; 3 = make K8s aware; 4–5 = monitor + slice.

### Two suppliers
- **The AMI** (node disk image): AWS's **GPU-optimized AMI** ships layers **1–2 prebaked**
  (driver + runtime). This is the AWS-documented "default" path.
- **The GPU Operator** (NVIDIA): installs/manages **all 5** and keeps them updated; can
  skip layers — `driver.enabled=false` = "AMI already did the driver."
- **Bare device plugin** (a DaemonSet): only layer 3; assumes AMI gave 1–2; **no DCGM/MIG**.

### Three options (who supplies what)
| Layer | Opt 1: AMI + device-plugin | **Opt 2: GPU-AMI + Operator (driver off)** | Opt 3: Ubuntu + Operator (all) |
|---|---|---|---|
| driver / runtime | AMI | **AMI** | Operator |
| device-plugin | you | Operator | Operator |
| DCGM / MIG | ❌ none | ✅ Operator | ✅ Operator |
| effort | simplest | reliable, medium | most (CUSTOM AMI) |

- **Opt 1** = simplest, pure GPU scheduling, but **no DCGM/MIG** → doesn't satisfy M7.
- **Opt 2 (CHOSEN)** = AWS GPU AMI supplies the driver (reliable, tested), Operator supplies
  device-plugin + **DCGM** + **MIG** — the typical EKS pattern *and* everything M7 needs.
- **Opt 3** = Operator manages the driver too → needs **Ubuntu** (the Operator's
  driver-container supports Ubuntu/RHEL, **not Amazon Linux**) via an EKS **CUSTOM** AMI
  (look up Canonical AMI + launch-template bootstrap) → more Terraform. The **on-prem /
  large-fleet** pattern (driver-as-container = fleet-wide driver lifecycle).

### Industry practice
- **Managed cloud (EKS/GKE/AKS):** prebaked-driver node image + Operator/device-plugin for
  the rest — i.e. **Opt 2**. Teams don't compile kernel modules on cloud nodes.
- **On-prem / bare-metal / big fleets:** Operator manages the **full** stack incl. drivers
  (Ubuntu/RHEL) for centralized driver lifecycle/upgrades — i.e. Opt 3.
- Interview line: *"On EKS I used the prebaked-driver AMI with the Operator managing
  device-plugin/DCGM/MIG; on-prem I'd let the Operator manage drivers too, for fleet-wide
  driver lifecycle."*

### Decision (2026-06-25): **Opt 2**
GPU node `ami_type = AL2023_x86_64_NVIDIA` (driver+runtime prebaked) + GPU Operator
`--set driver.enabled=false` (device-plugin + DCGM + MIG-manager + NFD). Reliable +
representative + covers Pieces 3 & 4.

---

## 8. ResourceFlavor & LocalQueue (the two Kueue objects that confuse)

**ResourceFlavor — "which *kind* of resource."** Distinguishes varieties of the same
resource ("1 GPU" — A100 or A10G? spot or on-demand? x86 or arm?). Quota is defined
**per-flavor**, and the flavor **maps to nodes** via `nodeLabels` (Kueue adds a
`nodeSelector` on admission so pods land on the right nodes). Empty `nodeLabels` = matches
any node (our CPU demo's `default-flavor`). Analogy: a **SKU** of a resource — the
ClusterQueue budgets "$ per SKU," the flavor says which shelf (nodes) it's on.

**LocalQueue — the namespaced submission handle.** A namespace-scoped pointer to a
ClusterQueue; users submit Jobs to it via the label `kueue.x-k8s.io/queue-name`. The
indirection exists for **multi-tenancy + RBAC**: ClusterQueue is **cluster-scoped /
admin-owned** (the quota pool); LocalQueue is **namespaced / team-owned** (teams interact
only in their namespace but draw from the shared pool). Analogy: **ClusterQueue = the
shared bank vault** (the real quota); **LocalQueue = your team's branch account** that
draws from it.

The full chain:
```
Job (label queue-name) → LocalQueue (namespaced, who submits) → ClusterQueue (shared quota)
                                                                   → ResourceFlavor (which kind + which nodes)
```

---

## 9. RCA bot — automated root-cause analysis (Piece, deferred to a later session)

The platform's automated first-responder: on failure, **gather evidence → classify cause
→ report**, fast (JD: time-to-classified-cause **<10 min**). Observability (Piece 4) says
*something's wrong*; the RCA bot says *what's wrong and why*.

**Flow:** Prometheus alert → Alertmanager → webhook → bot → **COLLECT** (K8s pod
status/exit-code + events + logs; Prometheus/DCGM metrics around failure; last MLflow run)
→ **CLASSIFY** (signatures → cause) → **REPORT** (structured {cause, evidence, fix,
elapsed} → stdout/file, Slack optional).

**Cause taxonomy (signatures):**
| Cause | Signature |
|---|---|
| `oom` | container `lastState.terminated.reason==OOMKilled` / exit 137; DCGM VRAM ≈ max |
| `data_validation` | log regex: Pandera/`SchemaError`/"validation failed" |
| `nccl_timeout` | log regex: `NCCL` + `timeout`/`Watchdog` |
| `node_pressure` | events: `MemoryPressure`/`NodeNotReady`/`Evicted` |
| `unknown` | fallback (don't force a wrong label) |

**Design (decided, build deferred):** module layout `observability/rca/` —
`collector.py` (k8s/prometheus/mlflow), `classifier.py` (rule-based core; LLM-assist
optional), `report.py`, `app.py` (thin FastAPI `/alert` webhook), `cli.py` (run on a
`--namespace/--pod` for local testing). Build **CLI/library first** (testable against a
real failed pod via kubeconfig), then wrap in FastAPI for the Alertmanager demo. Mirrors
the serving design (logic + thin web wrapper). **Tutor-protected component → Karthik
writes the logic.** Tested via **3 injected failures** (OOM, data-validation [reuse M1
`corrupt.py`], NCCL timeout) measuring <10 min. **No GPU needed.**

---

## 10. In-place pod resize (the agility piece) + VPA

**JD line:** in-place container resizing **<1 min, no restart**.

**The mechanism (`InPlacePodVerticalScaling`):** normally changing a pod's CPU/memory
recreates the pod (a restart — disruptive for a workspace/serving pod with warm state).
In-place resize lets the **kubelet adjust a running container's cgroup limits live**, no
restart. Each container declares a `resizePolicy` per resource:
- `NotRequired` → apply in place (non-disruptive)
- `RestartContainer` → must restart the container to apply

Resize via the **resize subresource**:
`kubectl patch pod <p> --subresource resize --patch '{...resources...}'`.
**Proof the demo needs:** resources change, **`restartCount` stays the same** (no restart),
and the patch takes **well under 1 min**.

**Feature gate:** alpha in k8s 1.31 (off by default), on-by-default in 1.33 → so we do it
on **kind** (where we can enable the gate; EKS can't easily flip apiserver/kubelet gates).

**memory + `RestartContainer` nuance (the quiz):** *shrinking* memory live is unsafe — a
process may already be holding pages it can't be forced to release, so memory often uses
`RestartContainer` to apply a decrease cleanly. CPU is a soft/throttleable limit → resizes
live fine.

### In-place resize (mechanism) vs VPA (controller)
| | In-place resize | VPA (Vertical Pod Autoscaler) |
|---|---|---|
| What | core K8s **mechanism** to change a running pod's resources | add-on **controller** that *decides* sizes from observed usage |
| Role | the **hands** (applies, no restart) | the **brain** (recommends/sets) |
| Installed here | yes (feature gate) | **no** — we resize manually |

They **compose**: historically VPA applied recommendations by **evict+recreate** (restart);
now it can apply them **in-place** (no restart) using this mechanism. The autoscaler map:
**HPA** = more replicas (horizontal) · **KEDA** = event-driven horizontal (scale-to-zero) ·
**VPA** = bigger pod (vertical) · **in-place resize** = the *apply* mechanism for vertical
changes without a restart. For the deliverable we resize **manually** to show the
mechanism + agility; VPA is the optional automation layer (out of scope, name it in
interview).

### Steps (kind) — manifests in `k8s/m7-resize/`
1. `kind create cluster --name resize --config k8s/m7-resize/kind-config.yaml` (gate on)
2. `kubectl apply -f k8s/m7-resize/resize-pod.yaml` (resizePolicy NotRequired cpu+mem)
3. record before: `.spec.containers[0].resources` + `.status.containerStatuses[0].restartCount` (0)
4. `time kubectl patch pod resize-demo --subresource resize --patch '{"spec":{"containers":[{"name":"app","resources":{"requests":{"cpu":"500m","memory":"256Mi"},"limits":{"cpu":"1","memory":"512Mi"}}}]}}'`
5. verify: resources changed, **restartCount still 0**, time < 1 min
6. `kind delete cluster --name resize`
