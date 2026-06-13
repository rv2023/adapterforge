# M2 — Training + Experiment Tracking: Flow

The job of M2 is to produce **one honest number — "the bar"** — that every later
model must beat on the *same* frozen data. Two views below: how it's built, and
why that one number matters for the rest of the project.

## M2 build flow — from raw data to "the bar"

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1 — LAND DATA  (run your existing M1 HF adapter)               │
│                                                                       │
│   HuggingFace: flare-fpb ──▶ HF Adapter ──▶ validate(schema v1)      │
│   (Financial PhraseBank)                         │                    │
│                                                  ▼                    │
│                                    data/  text │ label               │
│                                    ~4.8k rows  (bullish/bearish/      │
│                                                 neutral)              │
└──────────────────────────────────────┬──────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2 — SPLIT ONCE  (in pipelines/, the heart of M2)              │
│                                                                       │
│        ~4.8k labeled sentences                                       │
│                  │                                                    │
│      ┌───────────┼────────────────────┐                              │
│      ▼           ▼                     ▼                              │
│  ┌────────┐  ┌──────────┐      ┌──────────────┐                      │
│  │ TRAIN  │  │   VAL    │      │     TEST     │  ◀── 🔒 FROZEN VAULT │
│  │  ~70%  │  │   ~15%   │      │     ~15%     │     never trained on │
│  │ ~3.4k  │  │  ~700    │      │    ~700      │     opened ONCE      │
│  └───┬────┘  └────┬─────┘      └──────┬───────┘                      │
│      │            │                   │                              │
│   "textbook"  "practice exam"    "sealed final exam"                 │
└──────┼────────────┼───────────────────┼─────────────────────────────┘
       │            │                   │
       ▼            ▼                   │
┌──────────────────────────────┐       │
│  STEP 2b — TRAIN & TUNE       │       │
│                               │       │
│   TF-IDF  ──▶  LogisticReg    │       │
│  (text→counts) (classifier)   │       │
│        │            ▲         │       │
│   learns from    tune knob    │       │
│   TRAIN          C on VAL ─────┘       │   (pick best C using VAL only)
│        │                      │        │
└────────┼──────────────────────┘        │
         │                                │
         │   final model + best C         │
         └───────────────┬────────────────┘
                         ▼
              ┌──────────────────────┐
              │  SCORE ON TEST ONCE  │
              │   ─────────────────  │
              │   F1 = 0.XX          │ ◀══ ⭐ THE BAR
              └──────────┬───────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEPS 3–5 — TRACK & DIAGNOSE  (do this for all 3 runs)             │
│                                                                       │
│   run params + metrics ──▶  hand CSV   ──▶   MLflow                  │
│   (C, F1, step_time,        (feel the      (compare 2 runs           │
│    samples/sec)              pain)          in the UI)               │
│                                                                       │
│   Overfit run (50 passes, 500 rows) ──▶ plot train vs val curve     │
│                                          └─▶ learn to READ overfit   │
└─────────────────────────────────────────────────────────────────────┘
```

## The payoff — why this one number matters for the rest of the project

```
                          ⭐ THE BAR (F1 from M2's frozen TEST set)
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                         ▼
        ┌───────────┐           ┌────────────┐           ┌──────────────┐
        │   M3      │           │    M5      │           │     M8       │
        │ promotion │           │ Qwen-1.5B  │           │ cost-aware   │
        │  gate     │           │ LoRA LLM   │           │   router     │
        │           │           │            │           │              │
        │ hashes the│           │ scored on  │           │ "is the LLM  │
        │ SAME test │           │ the SAME   │           │  worth it?   │
        │ set 🔒 to │           │ frozen test│           │  → route     │
        │ prove fair│           │ → must     │           │  cheap vs    │
        │ compares  │           │ BEAT bar   │           │  LLM"        │
        └───────────┘           └────────────┘           └──────────────┘

   Same task • Same frozen data 🔒 • Same metric (F1)  →  comparison is FAIR
```

**The thing to internalize:** everything downstream pivots on that one frozen TEST
set. That's why M2 carves it off cleanly and never touches it — the 🔒 vault is the
anchor the whole platform's "is this model actually better?" question hangs from.

## Glossary (concepts behind the diagram)

- **Train / Val / Test split** — learn from train, tune knobs on val, report the
  final honest number on the sealed test set (the frozen eval set).
- **Frozen eval set** — the TEST slice; never trained or tuned on, opened once.
  M3's promotion gate hashes it to prove two models were compared fairly.
- **F1** — metric that rewards a model only if it's good at *every* class
  (precision × recall balanced), so a lazy "always predict neutral" model can't
  hide behind accuracy on an imbalanced dataset.
- **Baseline** — TF-IDF + Logistic Regression (a *classifier*, despite the name).
  Same task/data/metric as the M5 LLM, so the comparison is apples-to-apples.
- **step_time / samples_sec** — logged from day one; the habit that powers M5's
  efficiency experiment.
