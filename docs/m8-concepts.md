# M8 — Capstone (Multi-Adapter Dynamic Routing): Concepts

The milestone where M1–M7 become ONE self-healing, cost-aware serving platform.
This is the *why* + the router design. Build log: `docs/m8-session-notes.md`.

---

## 0. What M8 is — closing the loop

The capstone demo (the interview centerpiece):
> **inject drift → auto-retrain → gated promote → router reroutes live traffic to the new
> model → zero failed requests.** Hands-off.

M4 built drift→retrain→gate→promote. M8 adds the **last link — the router that reroutes
traffic on promotion** — and makes it work with the **LLM** (not the sklearn baseline).

**Two-plane architecture (now completed):**
```
SERVING plane (always on)            RETRAINING plane (event-driven)
  request → ROUTER → model             drift → retrain → gate → promote
        ▲                                            │
        └──────── hot-swap on promote ◄──────────────┘   (router shifts traffic, zero downtime)
```
They meet at the registry + promotion gate (the control plane). M8 wires the hot-swap arrow.

---

## 1. The router — the centerpiece

A FastAPI service in front of the models. **Routing logic is decoupled from how backends
are reached** (the key to testing free):

```
POST /predict {text, task, tier}
   → route(task, tier) picks a BACKEND key
   → BACKENDS[key](text) -> (output, confidence)
   → (escalate tier only) maybe call a 2nd backend
   → return {output, served_by, escalated?}
```

**A backend = a named callable `text -> (output, confidence)`.** Locally: `student` runs
in-process on CPU (real), `llm_sentiment` is a **mock** (canned output) → routing testable
**without a GPU**. In prod: the same slots point at HTTP clients to vLLM / student / KServe.
**Routing logic never changes — only what the backends *are*.** Reuses the `model_kind`
registry tag + the heterogeneous registry + M6 model-aware serving + vLLM multi-LoRA.

### The routing table
```
task=summarize             → summarizer (LLM + summ adapter)   [stub until 2nd adapter trained]
task=classify, accurate    → llm_sentiment                      (LLM always — best accuracy)
task=classify, cheap       → student                            (student only — no safety net)
task=classify, escalate    → student → (conf < THRESHOLD) → llm_sentiment   [DEFAULT; a model cascade]
```

### How a request is routed (worked examples)
- `{summarize}` → summarizer → brief.
- `{classify, accurate}` → llm_sentiment → ("bullish"). 1 call, big model.
- `{classify, cheap}` → student → ("bullish", 0.94). 1 call, never escalates.
- `{classify, escalate}` easy → student (conf 0.94 ≥ 0.70) → keep it (cheap win).
- `{classify, escalate}` hard → student (conf 0.48 < 0.70) → **escalate** → llm_sentiment.
  Response carries `served_by` + `escalated` so you can *measure* cheap-vs-escalated rate.

---

## 2. The tiers — a cost↔accuracy spectrum
```
cheap ──────────────── escalate ──────────────── accurate
student only        student→LLM if unsure        LLM always
cheapest, lowest     balanced (DEFAULT)           priciest, best
```
| tier | behavior | use when |
|---|---|---|
| `cheap` | student only, never escalates (accept lower accuracy) | low-stakes/high-volume; guaranteed cost+latency floor |
| `escalate` | **cascade** — cheap when easy, LLM when hard | **default** — most savings *with* an accuracy safety net |
| `accurate` | LLM always | high-stakes, accuracy non-negotiable |

`cheap` deliberately trades accuracy for cost (no net); the moment accuracy matters →
`escalate` buys it back by escalating only uncertain cases. (Renamed from "auto" →
`escalate` = self-documenting; the *pattern* is formally a **model cascade**.) `cheap`
still earns its place: `escalate` costs an extra LLM call + bimodal latency on hard cases,
so truly low-value traffic may prefer `cheap`'s predictable single call.

---

## 3. Cascade mechanics (the `escalate` tier)
The student isn't uniformly worse — it's worse on **hard** cases. Most inputs are easy;
the student nails those cheaply. **Confidence is the proxy for "hard case":** low
`softmax(logits).max()` ≈ where the student likely errs → escalate *those* to the LLM.
Net: ~majority handled cheaply by the student + LLM accuracy on the uncertain minority.

- **Requires the student to emit confidence** — flip M6's `confidence=None` →
  `softmax(logits).max()`.
- **THRESHOLD** (~0.70) tunes cost vs accuracy: higher → escalate more (accurate, pricey);
  lower → cheaper, less accurate. Tune via an escalation-rate-vs-accuracy experiment on the
  frozen eval set.
- **Caveat:** confidence is imperfect — a model can be *confidently wrong* (calibration).
  Cascade reduces, doesn't eliminate, student errors.

---

## 4. Dispatch (M6) vs Router (M8)
| | M6 dispatch | M8 router |
|---|---|---|
| decides | *how* to load the ONE prod model | *which* model gets the request |
| keyed on | `model_kind` tag | task + tier (+ confidence for escalate) |
| choices | one model, N code paths | N models |

M6 previewed it; M8 builds the real router. Same `model_kind` primitive underneath.

---

## 5. Carry-over debts M8 must clear (or the loop is fake)
1. **Retraining model-aware** — `loop.py` retrains *sklearn* (~0.69) which **can't pass the
   gate** vs the LLM (0.85) → loop is currently dead. M8: "retrain" must produce an **LLM
   adapter**.
2. **Drift sensor** — stop piggybacking on sklearn's TF-IDF vocab (model-agnostic signal).
3. **sklearn retired, not deleted** (kept for lineage, never served).

---

## 6. Design decisions + file layout
- backends = named callables `(text)->(output, confidence)` — transport-agnostic
  (in-process/mock local → HTTP/KServe prod)
- task/tier in the request body; default `tier="escalate"`
- student emits `softmax` confidence (enable the M6 toggle)
- `THRESHOLD` config constant ~0.70 (tune later)
- cost-aware = picks the *backend*; the GPU-slice placement (student small / LLM big) is a
  **deployment** concern (KServe/MIG), not router code
- summarizer stubbed until the 2nd adapter exists

`serving/router/`: `backends.py` (registry + in-process/mock wiring) · `routing.py`
(`route()` decision — tutor-protected, Karthik writes) · `app.py` (thin FastAPI `/predict`).
Routing is **unit-testable** with mock backends (no models needed).

---

## 7. Build plan (free-first)
1. **Router** (free): `route()` + backends + mock LLM + real student (CPU) + escalate
   cascade. Unit-test the routing decisions. ← start here
2. **2nd LoRA adapter** (summarization, ECTSum) — one GPU session (RunPod) → fill the
   summarizer backend.
3. **Carry-over debts** (free Python): model-aware retraining + drift sensor refactor.
4. **Hot-swap + KServe** on the cluster (paid EKS+GPU) — canary/traffic-shift = zero-downtime.
5. Stretch: `AdapterDeployment` CRD + kopf operator (free, kind).
6. Polish: arch diagram, README JD-map, demo video.

---

## 8. Drift in the platform — data vs concept, per-task, model-agnostic

**Drift is a property of the DATA, not the model.** OOV/PSI measures whether the incoming
text distribution shifted from training — independent of which model serves it. So the
M8 fix (debt 2) decouples the drift reference from the served model: build it from the
**training corpus**, not the production model's internals (`reference_analyzer_vocab()`).
Pulling it from the sklearn model's tfidf was a fragile shortcut that breaks once
production is the LLM.

**Per input-stream / task, NOT per model.** Models sharing the same input + task share ONE
drift check:
| Task | Input | Reference | Covers |
|---|---|---|---|
| sentiment | headlines | FPB train text (`reference_analyzer_vocab()`) | **both** the LLM adapter + the student |
| summarization | earnings transcripts | ECTSum train (its OWN reference) | the summarizer adapter |
→ refinement for later: **parametrize `reference_analyzer_vocab(task)`** by task when the
summarizer lands. On drift, retrain the **production model for that task**; the **student**
is refreshed downstream (re-distill from the updated teacher). The router doesn't change.

**Two kinds of drift:**
| | Data / input drift | Concept drift |
|---|---|---|
| What changes | **P(X)** — inputs look different (new vocab/regime) | **P(y\|X)** — same input, **different correct label** |
| Example | new hawkish/QT/crypto vocabulary floods in | "Fed raises rates" flips bearish→bullish across regimes |
| Detect | **unsupervised** (OOV/PSI on X) ← what we built | needs **true labels** — accuracy decay on delayed labels |
| Visible from X alone? | yes | **no** (X looks identical; only labels reveal it) |

The OOV sensor catches **data drift** (free, unsupervised). **Concept drift** needs a
feedback loop — **delayed-label accuracy** (PLAN: "delayed-label accuracy ≥ promotion − 2
pts → retrain"); a weak unsupervised proxy is prediction-distribution/confidence shift.
Named for completeness; the feedback loop isn't simulated yet.

One-liner: *data drift = inputs look different (catch from X, unsupervised); concept drift
= same inputs now mean something different (catch only via labels / accuracy decay).*
