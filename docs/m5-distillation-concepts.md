# M5 Piece 5 — Distillation Concepts

Concepts behind distilling the QLoRA Qwen-1.5B **teacher** into a small DistilBERT **student**.

## 1. What + why

The teacher (QLoRA Qwen-1.5B, F1 0.8477) is **accurate but expensive** (1.5B, needs a GPU,
slow). Running it on *every* headline is wasteful. **Distillation:** the big teacher **labels a
pile of unlabeled headlines**, and a **tiny student** (DistilBERT ~66M, ~23× smaller, CPU-able)
trains to **imitate** those labels → nearly as good, ~20× cheaper. Result: a **heterogeneous
registry** (1.5B LLM + 66M student) = the raw material for **M8's cost-aware router** (bulk →
student, premium → LLM). Bar: student within **2–3 F1 pts** of the teacher on the frozen test set.

## 2. Data flow + train/test split

- **Train on:** unlabeled headlines (Alpha Vantage) + the **teacher's labels** (no ground truth).
- **Tested on:** the **FPB frozen test set** with **real gold labels** (same sealed exam as the
  0.6885 baseline + 0.8477 teacher → comparable).
- The student learns from the teacher's predictions on fresh data; we check it against the real exam.

## 3. SEQ_CLS vs CAUSAL_LM (the model-type switch)

- Teacher (Qwen) = **CAUSAL_LM**: *generates* the label word.
- Student (DistilBERT) = **SEQ_CLS**: an **encoder + classification head** that outputs a
  probability over the 3 classes **directly** — no text generation. Trained with cross-entropy /
  KL on class labels (like the M2 sklearn baseline, but a neural net).

## 4. Hard vs soft labels

- **Hard** = the teacher's single predicted class → `[1,0,0]`. All-or-nothing.
- **Soft** = the teacher's full distribution → `[0.84, 0.04, 0.11]`. The runner-up probabilities are
  the **"dark knowledge"** — they tell the student *how confident / how close the call was*, which
  teaches the teacher's "feel" far better than a bare label. **We chose soft** (KL distillation).

## 5. The verbalizer — extracting a soft label from a *generative* teacher

The teacher generates words, not probabilities. So per headline:

1. **Build the training prompt** (system + user instruction), `apply_chat_template(add_generation_prompt=True)`.
2. **One forward pass** (no generation). The model outputs **logits = a score for every token in
   its ~150k vocab** at the next position (`logits[:, -1, :]`). Higher logit = more likely next token.
3. **Verbalizer = pick out just the 3 logits** at the token IDs for `bullish`/`bearish`/`neutral`
   (ignore the other ~150k). e.g. `[8.0, 5.0, 6.0]`.
4. **Softmax with temperature T** over those 3 → probabilities summing to 1:
   `softmax([8,5,6]/T)`. T>1 **softens** (more dark knowledge); T≈2 standard. → `[0.60,0.13,0.27]`.

That 3-number distribution **is the soft label**. Do it for all 23k headlines; the student trains
to **match** these via KL-divergence.

**The one thing to get exactly right — the label token IDs:** use the token the model *actually
emits* for each word (usually with a **leading space**, e.g. `" bullish"`), take its **first**
token, and **verify all 3 IDs are distinct** (print them once). If two collide, the soft label is
garbage. This is where the verbalizer usually goes wrong.

## 6. "Why not just prompt the model to output the probabilities?"

You can — it'll generate numbers — but they're **made-up text** ("what numbers usually go here"),
**not** its true internal confidence. LLMs verbalize confidence badly (round, overconfident,
uncalibrated). The **logits ARE the real distribution**. Plus practical wins for the logit path:

| | Prompted probabilities | Logit verbalizer (chosen) |
|---|---|---|
| Source of numbers | model **narrating a guess** | model's **actual internal** scores |
| Calibration | poor / invented | the real distribution |
| Reliability | malformed JSON, may not sum to 1 | deterministic, clean, normalized |
| Cost | autoregressive generation (many tokens) | **one forward pass** |
| Fit to our teacher | off-distribution (it was fine-tuned to emit **one word**) | on-distribution (logit at that one-word slot) |

Prompting-for-probabilities is what you do when you only have **API access** (no logits, e.g. GPT-4
as labeler). We have the model **in memory**, so we take the **real logits** — strictly better.
(The "just ask for the answer" instinct *is* valid for the simpler **hard-label** fallback.)

## 7. The Piece-5 build (6 steps)

| # | Step | File | Runs on | Status |
|---|---|---|---|---|
| 1 | Pull ~N unlabeled headlines (Alpha Vantage) | `collect_headlines.py` | laptop | ✅ **23,279** |
| 2 | Teacher **soft-labels** them (verbalizer + T) | `distill_label.py` | GPU pod | ⬜ |
| 3 | Train **DistilBERT SEQ_CLS** w/ KL loss | `distill_train.py` | GPU pod | ⬜ |
| 4 | Eval student on the **frozen test set** vs 0.8477 | reuse eval pattern | laptop/pod | ⬜ |
| 5 | **Register** the student (heterogeneous registry) | reuse `register_model_with_dossier` | laptop | ⬜ |
| 6 | `distill.yml` GitHub Actions workflow | boilerplate scaffold | CI | ⬜ |
