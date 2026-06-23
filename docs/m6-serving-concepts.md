# M6 — Serving Frameworks + GPU Sharing: Concepts

Conceptual reference built up over the M6 kickoff session (orientation + Q&A).
M6 is the most "SRE/DevOps" milestone: it's serving, latency, throughput, load,
and GPU-sharing — existing instincts applied to models. Results/numbers live in
separate `m6-*-results.md` docs (added per piece). This is the *why*.

---

## 0. What M6 is, in one breath

Make serving **fast** (vLLM), **prove** it (p50/p95/p99 + throughput benchmark vs the
naïve FastAPI), **share** the GPU (MIG vs time-slicing), and **watch** it
(DCGM → Prometheus). Plus quietly fix the standing bug: serving still assumes a
sklearn model (`serving/app.py` hard-codes `mlflow.sklearn.load_model`).

The four deliverables (from PROJECT_PLAN.md M6):

| # | Deliverable | Runs on | JD line |
|---|---|---|---|
| 1 | vLLM serve + LoRA load; benchmark p50/p95/p99 + throughput vs FastAPI; watch KV-cache VRAM grow with concurrency | RunPod single GPU | LLM serving |
| 2 | One model through Triton or KServe + half-page "when would I pick each" note | RunPod / local | Triton/KServe |
| 3 | A100 MIG lab: enable MIG → 2 isolated instances (LLM + student) → disable → time-slicing crash demo | RunPod A100 (~$1.8/hr) | MIG / time-slicing |
| 4 | DCGM exporter → Prometheus, graph GPU mem / SM util during the benchmark | RunPod + local | GPU observability |

---

## 1. What "serving" means

A trained model is just a file on disk. **Serving** = putting it behind an HTTP
endpoint so something can send input and get a prediction back. Already built:
`serving/app.py` (FastAPI): request `{"text": "..."}` → model runs → response
`{"label": "bullish"}`. Same shape as any web service: request hits a handler,
handler does work, returns a response.

---

## 2. Why the current server is "naïve"

The FastAPI handler does **one request at a time, start to finish**. For LogReg
(~1 ms/prediction) that's fine. For an **LLM** it's not, because an LLM generates
its answer **one token at a time, in a loop** — and the naïve server makes everyone
else wait until the current request is *completely* finished before anyone else can
use the GPU.

Analogy: a checkout line where the cashier scans one item, then **stands and waits**
for you to bag it before scanning the next, instead of scanning the next while you
bag. The GPU is actually idle a lot during one request; the naïve server won't let
anyone fill that idle time. Under load, p99 explodes.

---

## 3. How an LLM actually generates (autoregression)

An LLM predicts **one token at a time**, feeding each prediction back in to predict
the next. Prompt *"Stocks rallied today. Sentiment:"* → answer *"bullish"*:

```
Step 1: read the whole prompt        → predict "bull"
Step 2: read prompt + "bull"         → predict "ish"
Step 3: read prompt + "bullish"      → predict <end>
```

Two facts fall out, and vLLM exploits both:

**Fact 1 — two very different phases:**
- **Prefill** (Step 1): reads the entire prompt at once → lots of parallel math →
  GPU **saturated**, efficient.
- **Decode** (Steps 2,3,…): one new token per step → tiny amount of math for a big
  GPU → GPU mostly **idle**, bottlenecked on overhead, not compute.

  *Crux: decode wastes the GPU.* One request decoding alone might use ~5% of the
  card; the other ~95% sits idle. That idle time is the opportunity.

**Fact 2 — it must remember everything so far:** each step needs the full context
(prompt + all tokens generated so far). Recomputing from scratch every step would be
insane, so the model **caches** intermediate results for every token already seen.
That's the **KV-cache** (key/value cache). It grows by one token's worth of data
every decode step and lives in **VRAM**. This is the whole second half of vLLM.

---

## 4. vLLM Trick 1 — Continuous batching (the speed win)

If one request uses ~5% of the GPU per decode step, you could decode ~20 requests at
once in about the same time. That's **batching** — stack many requests into one GPU
operation. The naïve way (**static batching**) is "wait for 8 requests, run all 8
together until *all 8* finish, then take the next 8." Problem: requests finish at
different times.

```
Static batch of 4:
  req A wants 3 tokens   → done at step 3, then SITS IDLE waiting
  req B wants 50 tokens
  req C wants 5 tokens   → done at step 5, then SITS IDLE waiting
  req D wants 50 tokens
                          ↑ whole batch held hostage by the slowest (B, D)
```

A and C finished early but their GPU seats are wasted until B finishes, and a new
request E that arrives mid-batch must wait for the entire batch to clear. Bad p99.

**Continuous batching** re-decides the batch **every single decode step**, at the
per-token level instead of per-request:

```
Step N:   batch = [A, B, C, D]
A finishes → step N+1: batch = [B, C, D, E]   ← E slotted in immediately, A's seat reused
C finishes → step N+2: batch = [B, D, E, F]   ← F slotted in
```

The moment a request finishes, its seat is freed and the next waiting request takes
it mid-flight — no waiting for the batch to clear. GPU stays packed every step. This
is the single biggest reason vLLM beats naïve FastAPI: **no idle seats, no
head-of-line blocking.**

Analogy: static batching is a fixed-size elevator that won't move until full and
won't open until everyone reaches their floor. Continuous batching is an escalator —
people step on and off continuously, it never stops.

---

## 5. vLLM Trick 2 — PagedAttention (the memory win)

Continuous batching needs many KV-caches in VRAM at once (Fact 2), so **VRAM becomes
the bottleneck** — how many requests you can batch is limited by how many KV-caches
fit.

Naïve waste: you don't know the answer length in advance, so naïve systems
**reserve the maximum** up front ("might be 2,000 tokens → reserve 2,000"). If the
answer is 6 tokens, you reserved 2,000 and used 6 — 99.7% wasted. Across every
concurrent request you can barely fit a handful. This is the **fragmentation problem
an OS solves with virtual memory** — vLLM borrows that solution.

**PagedAttention:** chop the KV-cache into small fixed-size **blocks** (pages), e.g.
16 tokens each. Don't pre-reserve. Hand a request a new block only when it fills the
previous one, on demand. A 6-token answer uses **one** block. Blocks need not be
contiguous in VRAM — a lookup table maps "request A's logical token 30" → "physical
block 7," exactly like an OS page table maps virtual → physical RAM.

Payoff: near-zero wasted VRAM → far more requests fit at once → continuous batching
has more seats to fill → higher throughput. The two tricks reinforce each other.
(Because allocation is on-demand, KV-cache VRAM **grows visibly as concurrency
rises** — that's the live graph captured in deliverable 4.)

**One-paragraph summary:** An LLM generates token-by-token, leaving the GPU mostly
idle during decode and forcing a growing per-request KV-cache in VRAM. vLLM fills
the idle GPU by re-packing many requests into every decode step (continuous batching,
no waiting for slow requests) and fits far more of them in VRAM by allocating
KV-cache in small on-demand pages instead of giant pre-reservations (PagedAttention).
Same GPU, many times the throughput, much better tail latency.

---

## 6. The benchmark design question (apples-to-apples)

The plan says "benchmark vLLM vs your M3 FastAPI endpoint," but the FastAPI currently
serves a **LogReg**. Comparing vLLM-serving-the-1.5B-LLM against FastAPI-serving-LogReg
measures **model size**, not **serving stack** — useless. The honest comparison is
**same model (the Qwen LLM), two stacks**: vLLM vs a naïve FastAPI that does HF
`transformers.generate()` in the request handler. *That* isolates vLLM's batching win.
(Open design decision for Piece 1.)

---

## 7. GPU sharing: one card, two models

By default one GPU = one tenant. On a $1.8/hr A100 that's wasteful (the 66M student
would use a sliver and leave the rest idle). Two ways to share one card:

### Way 1 — Time-slicing (taking turns)

Exactly CPU time-sharing: the GPU runs model A's work for a slice, **context-switches**
to B, back to A, fast enough that both *appear* concurrent.

```
GPU timeline:  [A][B][A][B][A][B][A]...   ← one engine, taking turns
```

**Critical: time-slicing gives NO isolation.**
- **No memory isolation** — both models share the *same* VRAM pool with no limits. If
  A grabs more VRAM than is free it isn't throttled, it **OOM-crashes** — and can
  starve B so **B crashes too**, though B did nothing wrong. Noisy-neighbor, fatal.
- **No fault isolation** — a crash/bad kernel in A can take down the GPU context, B
  with it.
- **No performance guarantee** — if A floods the GPU, B's turns get starved;
  unpredictable latency.

Analogy: two services on one host **with no cgroups, no memory limits** — works great
until one misbehaves, then it takes the neighbor down.

### Way 2 — MIG (Multi-Instance GPU, physical partitioning)

Opposite philosophy: physically carve the silicon into smaller, fully isolated
mini-GPUs. An A100 splits into up to 7 instances, each with its **own dedicated**
compute cores, **own dedicated** VRAM, and **own** memory bandwidth, walled off in
hardware.

```
A100 → 2 MIG instances:
  ┌─────────────┬─────────────┐
  │  MIG 0 LLM  │ MIG 1 student│
  │  own VRAM   │  own VRAM   │   ← hard walls, can't cross
  │  own cores  │  own cores  │
  └─────────────┴─────────────┘
```

- **Hard memory isolation** — MIG 0 cannot touch MIG 1's VRAM. If the LLM OOMs, it
  OOMs **only itself**; the student keeps running.
- **Fault isolation** — a crash in one instance doesn't affect the other.
- **Guaranteed performance** — dedicated cores, no noisy-neighbor starvation.

Analogy: each service gets its **own VM / dedicated cores with hard memory limits** —
true multi-tenancy, blast radius contained.

### The tradeoff (the JD-relevant judgment)

| | Time-slicing | MIG |
|---|---|---|
| Isolation | None (shared everything) | Hard (compute + memory walled off) |
| Blast radius | One OOM kills neighbors | Contained to one instance |
| Utilization | Can be higher (flexible sharing) | Fixed partitions — half-used instance wasted |
| Granularity | Any GPU supports it | A100/H100 only; fixed instance sizes |
| Use when | Trusted, bursty, dev/test; pack tightly, accept risk | Multi-tenant prod; one tenant must never harm another |

One-liner to say unprompted: *"Time-slicing maximizes utilization but gives no
isolation — one tenant can OOM and crash the others. MIG sacrifices some flexibility
for hardware-enforced isolation, so it's the right call for production multi-tenancy
where blast radius matters."*

Deliverable 3 = do **both** on a rented A100: feel MIG's isolation, then feel
time-slicing's lack of it (the crash). That contrast *is* the deliverable.

### Way 3 — MPS (Multi-Process Service)

Start from time-slicing's weakness: with plain time-slicing only **one** process's
work runs **at any instant** — they take turns (`[A][B][A][B]`). Even if A uses only
5% of the cores during its turn, B still can't run *simultaneously*; it waits. The
GPU is busy "all the time" but each turn under-fills it.

**MPS lets multiple processes' work run on the GPU truly at the same time —
spatially, side by side.** A GPU has thousands of cores grouped into **SMs**
(streaming multiprocessors). MPS runs a **daemon** that merges several processes'
CUDA work into one shared GPU context, so kernels from **different processes** occupy
**different SMs at the same instant**:

```
Time-slicing:  instant t → only A's kernel on the GPU      (B waits its turn)
MPS:           instant t → A's kernel on SMs 0–40
                           B's kernel on SMs 41–80          ← both at once, spatial
```

So MPS converts "take turns in time" into "split the cores in space." If A and B each
half-fill the GPU, MPS lets them genuinely overlap → much better utilization. You can
even set a **per-process SM cap** ("client A ≤ 50% of SMs") — a soft compute quota.

**Catch: MPS still gives NO memory isolation.** All processes share the same VRAM
pool, no hard wall — A can still OOM and starve B, and a fault can still hurt
neighbors (newer GPUs add some error containment, but not MIG-grade). MPS improves
**throughput**, not **safety**.

### The three-way map

| | Time-slicing | MPS | MIG |
|---|---|---|---|
| How they share | Take turns in **time** | Run together in **space** (split SMs) | Physically **partition** the card |
| Concurrency | One process at an instant | Many processes at an instant | Many, fully separate |
| Memory isolation | None | **None** (shared VRAM) | **Hard** (own VRAM each) |
| Fault isolation | None | Weak | Hard |
| Compute quota | No | **Yes** (% SM cap) | Yes (fixed partition) |
| Needs special HW | No | No | A100/H100 only |
| Best for | Trusted bursty/dev | Trusted, want max utilization + overlap | Untrusted multi-tenant prod |

Mental shortcut: **time-slicing = turns. MPS = share the cores, no walls. MIG =
walls.** The progression goes loosest → tightest isolation.

---

## 7b. How GPU-sharing relates to vLLM — two layers that STACK

Key insight: GPU-sharing (MIG/time-slicing/MPS) and vLLM are at **two completely
different layers**. They don't compete — they stack.

```
┌─────────────────────────────────────────────┐
│ LAYER 2 — inside ONE process / one GPU slice  │
│   vLLM: how to use the GPU I was given well —  │
│   continuous batching, PagedAttention          │
│   (scheduling many REQUESTS within one model)  │
├─────────────────────────────────────────────┤
│ LAYER 1 — across PROCESSES on the physical card│
│   MIG / time-slicing / MPS: how to divide the  │
│   GPU among different PROCESSES / MODELS        │
└─────────────────────────────────────────────┘
```

- **MIG / time-slicing / MPS** answer: *"I have one physical GPU and several different
  processes (the LLM **and** the student). How do they share the card?"* — an
  **inter-process / infrastructure** question. The GPU operator / Kubernetes decides
  this (that's M7).
- **vLLM** answers: *"Within the one GPU (or MIG slice) I've been handed, how do I
  serve many incoming **requests** to the **same** model as fast as possible?"* — an
  **intra-process / application** question. vLLM doesn't know or care whether it sits
  on a whole A100 or a MIG slice — it just sees "a GPU" and maximizes throughput on it.

They compose. Concrete M6/M8 setup:

```
A100 ──MIG──> MIG instance 0 ──> process running vLLM serving the LLM adapter
        └────> MIG instance 1 ──> process running the DistilBERT student
```

MIG cuts the card (Layer 1); vLLM runs *inside* one slice and batches requests there
(Layer 2). Both at once.

**Subtle point worth keeping:** for **many requests to the *same* model**, vLLM's
continuous batching is *already* the best sharing mechanism — those requests live in
one process, so you don't need MIG/MPS/time-slicing for them; vLLM packs them. You
reach for MIG/MPS/time-slicing only when you have **different models / processes**
wanting the same card (LLM + student) — exactly the M6 sharing lab and the M8
cost-aware router (cheap classification → student slice, generation → LLM slice).

One line: **vLLM handles many requests to one model; MIG/MPS/time-slicing handle many
models to one card.** Different problems, same goal (keep the expensive GPU full),
solved at different layers.

---

## 7c. Model-aware serving (the standing-bug fix, M6 Piece 0)

**The bug:** `serving/app.py` hard-codes `mlflow.sklearn.load_model(...)` — correct in
M3 when production was the LogReg baseline, but production is now the **LLM adapter**
(v14), which is not an sklearn model → load fails (`No such artifact: 'MLmodel'`).
Same hard-coding in `pipelines/loop.py` (the retrain loop).

**Why "model-aware":** the registry now holds heterogeneous kinds, each with a
different load call AND predict path:

| Registered | Kind | Load | Predict |
|---|---|---|---|
| baseline | sklearn LogReg | `mlflow.sklearn.load_model` | `model.predict([text])` |
| `fpb-sentiment` v14 | Qwen + LoRA | PEFT + tokenizer (reuse `eval_adapter.load_model_and_tokenizer` / `predict_one`) | tokenize → `generate` → parse word |
| `fpb-student` | DistilBERT | HF transformers (reuse `eval_student` load/`predict`) | tokenize → forward → argmax |

The server currently hard-codes ONE path (sklearn). Model-aware = it reads what kind
the production version is and picks the right loader + predictor automatically. SRE
analogy: a server welded to one backend driver now has three backends; it must read a
"type" tag and dispatch — like a reverse proxy routing by `Content-Type` instead of
assuming every upstream speaks the same protocol.

**The gap that made it possible:** the dossier tags (`test_f1`, `eval_set_hash`,
`schema_version`, `code_commit`) had **no "kind" tag**. Fix = stamp a **`model_kind`**
tag at registration (`sklearn` / `lora_adapter` / `distilbert`), backfill it onto the
existing v14. Nice surprise: the control plane's `/production` endpoint already returns
*all* dossier tags, so the server receives `model_kind` for free in the response it
already reads — no new API call.

**The decision flow (it's a dict lookup, not an if/elif ladder):**

```
GET /production → {"version": 14, "model_kind": "lora_adapter", ...}
                            │ read the model_kind string
                            ▼
        dispatch table:  { "lora_adapter": (load_lora,      predict_lora),
                           "distilbert":   (load_distilbert, predict_distilbert) }
                            │ key lookup
                            ▼
        call the loader ONCE at startup → use the predict fn per request
```

Registry artifacts vs local dirs: the eval-script loaders read from a local `models/`
dir, but the server loads from the **registry**. So the server first downloads the
version's artifacts (`mlflow.artifacts.download_artifacts`) then points the existing
loader at that download.

**Scope (decided 2026-06-22):** build dispatch for **`lora_adapter` + `distilbert`**
only. **sklearn is retired, not deleted** — its version stays registered + artifacts
stay in DVC (lineage/history); we just write no loader for it. The student (66M) runs
on CPU, so it doubles as the free local smoke test.

### Dispatch (M6) vs Router (M8) — two different "deciders"

Easy to conflate; they're different layers:

| | M6 serving dispatch | M8 cost-aware router |
|---|---|---|
| Decides | *how* to load the ONE prod model | *which* model gets the request |
| Keyed on | `model_kind` tag | task header + cost rule |
| Choices | one model, N code paths | N models |
| When | now (Piece 0) | M8 |

In M6 there is **no model router** — exactly one version is `production`; the server
just reads `model_kind` to pick a *loader*. The thing that picks student-vs-LLM is the
M8 router (classification → cheap student slice; generation → LLM slice). They share
one primitive — knowing each model's kind — which is why the `model_kind` tag is
stamped now: M6 uses it to pick a loader, M8's router reuses it to pick a GPU slice.

### Two-plane architecture (the M8 destination; full note in PROJECT_PLAN.md)

```
SERVING plane (always on)        RETRAINING plane (event-driven)
  prod traffic → prod model        drift → retrain → gate → promote
  ← model-aware dispatch (M6)      ← loop.py, made model-aware (M8)
```

They couple at exactly one handoff: the registry version + the promotion gate. M6
builds the serving half. M8 builds the retraining half (and must make "retrain" mean
retrain the **LLM**, because a retrained sklearn candidate at ~0.6885 can never pass
gate #1 against the LLM at 0.8477).

---

## 8. DCGM → Prometheus (deliverable 4, plain)

**DCGM** is NVIDIA's exporter that publishes GPU metrics (VRAM used, SM/core
utilization, temp, power) in **Prometheus** format — the same monitoring stack
already known. Scrape it with a local Prometheus and graph GPU usage *during* the
benchmark (e.g. watch KV-cache VRAM grow with concurrency). Pure SRE observability,
pointed at a GPU.

---

## 9. Where M6 runs & cost

Most of M6 (the server, load-test harness, Prometheus config) is written **locally
for free**. GPU runs happen on **RunPod**: a single GPU for the vLLM benchmark, and
one **A100** session for the MIG lab (~$1.8/hr, a couple hours). Same drill as M5:
confirm cost, run, tear down.
