# AdapterForge — Full Platform Architecture

Data comes in validated and versioned ① → models get trained and tracked ② →
only governed, gated promotions reach production ③ → a router serves the right
model per request ④ → and when live data drifts, the system retrains, re-gates,
and re-deploys itself with zero humans ⑤ — all on a GPU Kubernetes platform
that's watched and self-diagnosing ⑥.

The **🔁 loop in ⑤ is the showpiece** — the thing you demo live: inject drift,
walk away, watch it heal. Everything else exists to make that loop *safe* (the
gates), *traceable* (lineage), *affordable* (the router + GPU sharing), and
*observable* (the dashboards + RCA bot).

## Full platform

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  ① DATA LAYER — get clean, validated, versioned data                  [M1]   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║   HF: flare-fpb ─┐                                                             ║
║   (PhraseBank)   │                                                             ║
║   CSV: Twitter ──┼─▶  ADAPTER SDK  ─▶ validate(Pandera schema v1)             ║
║   REST: AlphaV ──┘   (BaseAdapter)        │   bad batch ─▶ ❌ rejected         ║
║   (live news)                             ▼                                    ║
║                            MEDALLION ZONES (bronze→silver→gold) on S3          ║
║                            + DVC version stamp  🔖  (data is reproducible)     ║
║                                           │                                    ║
╚═══════════════════════════════════════════┼════════════════════════════════════╝
                                            ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  ② TRAINING LAYER — make models, track every run                  [M2,M5]    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║   split 70/15/15  ──▶  🔒 frozen TEST set  (the shared bar for ALL models)    ║
║         │                                                                      ║
║         ├─▶ BASELINE  TF-IDF + LogReg            [M2]  ── F1 = the bar ⭐      ║
║         ├─▶ LLM       Qwen-1.5B + QLoRA adapter  [M5]  ── must BEAT bar        ║
║         │             • Adapter 1: Market Signal (classify)                    ║
║         │             • Adapter 2: Earnings Summarizer (generate)              ║
║         └─▶ STUDENT   DistilBERT, distilled from LLM teacher [M5] (cheap)      ║
║                       │                                                        ║
║         every run logs: params · F1 · step_time · samples/sec                 ║
║                       ▼                                                        ║
║                  MLflow  (experiments + model registry)                       ║
╚═══════════════════════════════════════════┼════════════════════════════════════╝
                                            ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  ③ CONTROL PLANE — governance owns promotion, not a human click   [M3]       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   FastAPI service wrapping MLflow registry:                                    ║
║     POST /promote  ──▶  GATES:  F1 ≥ prod + margin  ·  eval-set hash matches   ║
║                                 ·  schema compatible ·  approved_by present    ║
║                         pass ✅ ─▶ Production    fail ❌ ─▶ rejected w/ reason  ║
║     GET /lineage   ──▶  which data version + code commit made this model       ║
║     audit log      ──▶  every promote/demote appended to immutable JSONL (S3)  ║
╚═══════════════════════════════════════════┼════════════════════════════════════╝
                                            ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  ④ SERVING LAYER — route requests to the right model              [M6,M8]    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                          ┌──────────────────────────┐                          ║
║   live request ─────────▶│  ROUTER (task + cost) [M8]│                         ║
║                          └───────┬───────────┬──────┘                          ║
║              cheap/bulk ◀────────┘           └────────▶ hard/generative         ║
║              STUDENT (DistilBERT)            vLLM + LoRA adapters [M6]          ║
║                                              (hot-swap, zero-downtime)         ║
║                                              one model via Triton/KServe       ║
╚═══════════════════════════════════════════┼════════════════════════════════════╝
                                            ▼
                                   ┌─────────────────┐
                                   │  LIVE TRAFFIC   │
                                   │  + prediction   │
                                   │  logs ─▶ S3     │
                                   └────────┬────────┘
                                            ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  ⑤ THE AUTOMATED LOOP — drift → retrain → promote → reroute   [M4]  🔁        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║   prediction logs ─▶ Evidently drift (PSI / KS-test)                           ║
║                              │  PSI > 0.2 ?                                     ║
║                              ▼ yes                                             ║
║        webhook ─▶ GitHub Actions retrain.yml ─▶ Dagster retrain run            ║
║                              │                                                 ║
║                              ▼                                                 ║
║          control-plane promotion GATE ③ ─▶ auto-promote ONLY if better        ║
║                              │                                                 ║
║                              ▼                                                 ║
║          serving ④ hot-swaps the new adapter ──▶ (loop closes, 0 humans)      ║
║                                                                                ║
║   Dagster orchestrates ①→⑤ · OpenLineage events ─▶ Marquez (full lineage)     ║
╚════════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  ⑥ FOUNDATION — runs underneath everything above                  [M7]       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   INFRA (IaC):  Terraform ─▶ VPC + EKS + GPU node group                       ║
║                 NVIDIA GPU Operator · MIG / time-slicing · Kueue quotas        ║
║                 in-place pod resize (<1 min)                                   ║
║   OBSERVABILITY: DCGM ─▶ Prometheus ─▶ Grafana (>98% SLO dashboards)           ║
║                 RCA BOT: logs + events + metrics ─▶ classified root cause      ║
╚════════════════════════════════════════════════════════════════════════════════╝

   (M0 = GPU fundamentals drills — prerequisite skills, done before M5)
```

## Milestone → layer map

| Milestone | Builds | Layer |
|---|---|---|
| M0 | GPU fundamentals drills (CUDA, VRAM math, checkpoints) | prerequisite |
| M1 | Adapter SDK, schemas, medallion + DVC, CI | ① Data |
| M2 | sklearn baseline + MLflow tracking (the bar ⭐) | ② Training |
| M3 | Control-plane FastAPI: promotion gates, audit, lineage | ③ Control plane |
| M4 | Dagster DAG, OpenLineage→Marquez, Evidently drift, the loop 🔁 | ⑤ Loop |
| M5 | QLoRA Qwen-1.5B, efficiency exp, NCCL, distillation → student | ② Training |
| M6 | vLLM + LoRA, Triton/KServe, MIG/time-slicing, DCGM | ④ Serving / ⑥ |
| M7 | Terraform EKS + GPU, GPU Operator, Kueue, Grafana SLO, RCA bot | ⑥ Foundation |
| M8 | 2nd adapter, task+cost router, hot-swap, end-to-end demo | ④ Serving |
