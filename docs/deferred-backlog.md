# Deferred / Pending backlog

Tasks consciously deferred (concepts understood or designed; hands-on/build pending).
Tracked here so they're not lost. None block the JD-mapping core; pick up when convenient.

## M6 — Serving + GPU sharing
| Item | State | To finish |
|---|---|---|
| **MIG lab (Piece 3)** | concept KNOWN; not run | needs **A100/H100** (g5/A10G can't MIG). RunPod blocks host MIG; AWS A100 = $$ + quota. A dedicated MIG-capable GPU session if ever. |
| **Time-slicing crash demo** | concept KNOWN | works on A10G → a short GPU session (RunPod or EKS GPU node) |
| **"Why vLLM wins" write-up** | Claude-drafted | Karthik rewrite in own voice (`docs/m6-benchmark-results.md`) — interview prep |
| **Serving selection note** | Claude-drafted | Karthik fill "when I'd pick each" + one-liner (`docs/m6-serving-selection-note.md`) |
| Benchmark VRAM-under-load graph | not captured | optional; superseded by M7 DCGM→Prometheus |

## M7 — K8s GPU platform
| Item | State | To finish |
|---|---|---|
| **In-place pod resize** | concept KNOWN; scaffolded `k8s/m7-resize/` | run on **kind** (free, ~5 min): patch `--subresource resize`, verify restartCount unchanged |
| **RCA bot** | DESIGNED (concepts §9) | build `observability/rca/` (collector/classifier/report/cli) — Python, free on kind; test vs 3 injected failures (<10 min) |
| **SLO dashboard** | concept KNOWN | Grafana panel + PromQL for >98% pipeline success-rate (on kube-prometheus-stack, free on kind) |
| MIG + time-slicing (carried from M6) | concept KNOWN | needs GPU (see M6 row) |

## M8 — Capstone
| Item | State | To finish |
|---|---|---|
| **2nd LoRA adapter** (summarization, ECTSum) | not started | one GPU session (RunPod) → fills the `summarizer` router backend → real task routing |
| **Hot-swap + KServe deploy** | router/hot-swap logic exists; KServe deploy not done | paid EKS+GPU session — canary/traffic-shift = zero-downtime |
| **`AdapterDeployment` CRD + kopf operator** | **skipped (stretch)** | free (kind) — *not a functional need* (router+hot-swap+vLLM-multiLoRA already cover serving); purely the JD "operators/CRDs" box + resume signal. Build only if you want that checkbox. |
| README JD-bullet tweaks (#1–3 built-vs-designed) | drafted | tighten "deployment paths" / mark RCA+SLO "designed" / resize "scaffolded" (rule 5) |
| Threshold-tuning experiment (escalate cascade) | not done | escalation-rate vs accuracy on the frozen eval set → pick THRESHOLD |

## Long-open (rule 5 — Karthik's words)
| Item | State |
|---|---|
| **M1 SDK README** | pending (5 lines/component, his words) |
| RoCE / InfiniBand one-paragraph explainer | pending |

## How to resume any of these
- kind pieces (resize, RCA bot, SLO): `kind create cluster` → free, no AWS.
- EKS pieces: `cd infra && terraform apply` (~15 min, ~$0.20/hr) → `cd addons && terraform apply` → work → `destroy` both. Cluster code is all in git.
- GPU/MIG: a dedicated rented-GPU session (confirm $/hr first).
