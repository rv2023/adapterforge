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

---

## 9. The 2nd adapter — summarization (ECTSum)

**Why:** the router can only route by *task* if there's more than one task. Sentiment is
task 1 (LLM + student); the summarizer is **task 2**, which makes the router's `summarizer`
backend real and demonstrates "multi-adapter serving" (the JD line).

**Reuse:** summarization needs the SAME machinery as the sentiment LLM (QLoRA SFT on
chat-messages JSONL), so `finetune.py` is reused unchanged — `DATA_DIR`/`ADAPTER_DIR` are
now env-overridable (`AF_DATA_DIR`/`AF_ADAPTER_DIR`). Only the data-prep + eval differ.

**Dataset (ECTSum):** pulled from the GitHub repo zip (rajdeep345/ECTSum). Raw form =
file pairs per split: `data/final/{train,val,test}/ects/<id>.txt` (transcript) +
`gt_summaries/<id>.txt` (bullet summary). Sizes: **train 1681 / val 249 / test 495**
(verified). Summaries anonymize the company as `compname`.

**Conversion (`summarize_format.py`):** each (transcript, summary) pair →
`{"messages":[{system: "You are an earnings-call summarizer."}, {user: INSTRUCTION+transcript},
{assistant: summary}]}` → `data/instruction_summ/{train,val,test}.jsonl`. The **assistant
turn is the SFT target**. Same shape as the sentiment instruction data (→ finetune reused).

**Eval (`eval_summarizer.py`):** metric = **ROUGE-L** (overlap of generated vs reference
summary), not F1. Bar = ROUGE-L(adapter) > ROUGE-L(**base zero-shot**); the base is gotten
elegantly via `model.disable_adapter()` on the same loaded PeftModel. Writes
`eval_metrics.json` (stores ROUGE-L in `test_f1` = "the gate's score"). Needs `rouge-score`.

**Pipeline:** `summarize_format` (data ✅ verified) → `finetune` (AF_DATA_DIR/AF_ADAPTER_DIR,
GPU) → `eval_summarizer` (GPU) → register as `fpb-summarizer` + wire router `build_summarizer`.

**✅ Task-aware promotion gate (implemented).** The gate was sentiment-pinned (global
`EXPECTED_HASH`/`EXPECTED_SCHEMA`/`MODEL_NAME`, and it *ignored* the `{name}` path param —
a latent bug). Now `control-plane/app.py` has a per-model **`GATE_CONFIG`** keyed by name:
each model carries its own `expected_hash`, `expected_schema`, `margin`, `floor`, and a
`metric_label` ("F1" / "ROUGE-L") for readable rejections. `promote(name, …)` looks up
`GATE_CONFIG[name]` (404 if unknown) and threads `name` through `get_dossier`/
`get_production_version`/`set_registered_model_alias` — so **each model is gated against its
OWN frozen exam + its OWN incumbent** (F1-vs-F1, ROUGE-vs-ROUGE never cross). The summarizer
hash is deferred via **`os.getenv("FPB_SUMMARIZER_EXPECTED_HASH")`** and **fail-closed** if
unset (reject "expected hash not configured"). To promote `fpb-summarizer`: register it
(→ ECTSum test-set hash), then set that env var (or hardcode) before starting the control
plane. This was the **last sentiment-pinned piece** — gate, drift sensor, and retraining
are all now task/model-aware.

---

## 10. Hot-swap + KServe — zero-downtime serving (the M8 paid stretch)

Both answer one question: after a promotion, how does serving start using the new model
**without dropping a request?** Hot-swap = the *mechanism*; KServe = the *platform* that
provides it the production way.

**Hot-swap (the mechanism).** Naïve way: restart the serving process so it reloads the new
production model → a downtime gap where requests fail. Hot-swap = replace the model a
**running** server uses while it keeps serving (change the tyre without stopping the car):
load the new adapter alongside the old, route new requests to it, drain in-flight on the
old, then unload the old → **zero failed requests, no restart**. Clean with **vLLM
multi-LoRA**: vLLM holds several LoRA adapters in one running server and add/removes them at
runtime, so "swap" = load new adapter → flip routing → drop old. Our M8 router already does
a *simple* version (point a backend slot at the new version); hot-swap makes that flip
seamless under live traffic.

**KServe (the platform).** A Kubernetes-native model-serving framework. Declare an
**`InferenceService`** CRD ("serve this model") and KServe runs the pods + networking +
rollout. What it gives you (serving versions of familiar SRE primitives):
| Feature | SRE analogy |
|---|---|
| **Canary / traffic splitting** | 10% to new model version, 90% old, ramp to 100% — a canary deploy for models |
| **Autoscaling + scale-to-zero** | no traffic → 0 pods → $0; request → spins up (Knative/HPA-like) |
| **Standard inference protocol + GPU scheduling** | uniform `/predict` + pods land on GPU nodes |
KServe's **canary traffic-shift IS the production-grade zero-downtime reroute** — instead of
our router flipping an in-process pointer, the cluster shifts traffic gradually and rolls
back automatically if the new version errors.

**How they fit the capstone.** Demo goal: drift → retrain → promote → serving reroutes,
zero failed requests. Hot-swap = the switch; KServe = doing it the "real company" way
(canary, autoscale, rollback). **Honest scope:** the router already demonstrates rerouting
on promotion, so these are NOT functionally required for the core demo — they're the
**production-grade upgrade + JD checkboxes** (KServe / canary / zero-downtime). That's why
it's the **paid stretch**: deploying needs a live EKS cluster + GPU nodes ($). Everything
else left in M8 is free write-ups.
