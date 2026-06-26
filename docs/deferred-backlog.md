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
| **2nd LoRA adapter** (summarization, ECTSum) | **TRAINED + PROMOTED to production** (ROUGE-L 0.1330 > base 0.1015; fpb-summarizer v1, hash f55821…) | DONE through the gate. Remaining: wire `backends.build_summarizer` (Option A — real loader, below). |
| **Wire `build_summarizer` (Option A)** | DESIGN locked; Karthik to implement (tutor) | add `build_summary_predictor(name,version)` in `serving/app.py` (reuse `eval_adapter.load_model_and_tokenizer` + `eval_summarizer.generate_summary`, return `(summary, None)`, use `summarize_format.build_chat_messages`); router `build_summarizer` → delegates to it for `fpb-summarizer` v1. Slow on CPU; prod → vLLM HTTP. |
| **Batch the summarizer eval** | not done (current eval = 990 *sequential* greedy gens, ~80 min, GPU util ~25%) | `eval_summarizer` generates one transcript at a time — the M6 "naive" path. Fix: **HF batched generate** (left-pad tokenizer, 16–32 prompts/batch) → util ~70%+, ~10–15 min. vLLM (`--enable-lora`+`LoRARequest`) optional/heavier (env setup + 2 passes for the `disable_adapter` base). Good JD talking point: naive-vs-batched measured on our own eval. |
| ~~Summarizer adapter size (398 MiB)~~ **RESOLVED** | adapter is a normal **36 MiB** (392 LoRA tensors = 28 layers × 7 modules × A/B, r=16; no embed/lm_head). The 398 MiB was the *run dir* = two `checkpoint-*` resume snapshots (118 MB each: adapter copy + optimizer.pt + scheduler). Fix applied: `rm -rf models/fpb-summarizer/checkpoint-*` before register (so `log_artifacts` stays lean). |
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
