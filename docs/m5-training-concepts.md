# M5 Training Concepts — forward/backward, validation, train/val/test

General ML training concepts that came up while building M5. Not M5-specific, but captured here.

## 1. The full training step vs a forward-only pass

A **training step** is three things; a forward-only pass is just the first:

```
forward pass  →  backward pass (gradients)  →  [all-reduce if multi-GPU]  →  weight update
   ✅                    ✅                            ✅                          ✅
```

- **Forward pass** — run data through the model → prediction + **loss**.
- **Backward pass** — from the loss, compute **gradients** (how to nudge each weight). *Training only.*
- **All-reduce** — in data-parallel multi-GPU, average gradients across GPUs (NCCL). *Lives in the
  backward/update — that's why Piece 3 must run full steps, not forward-only, to measure it.*
- **Weight update** — optimizer nudges the weights. *Training only.*

| | forward | backward | update |
|---|---|---|---|
| Training step | ✅ | ✅ | ✅ |
| Validation/eval | ✅ | ❌ | ❌ |

## 2. Validation — what + why

**What:** run the model on a **held-out set** (data it did NOT train on), **forward pass only**
(compute loss/metrics), and **do not learn from it** (no backward, no update). A *score*, not a
learning step.

**Why (the forward pass is the means; honest evaluation is the end):**
1. **Generalization** — training loss only shows fit to data already seen (could be memorizing);
   validation shows performance on unseen data = did it learn the **rule** or just memorize?
2. **Catch overfitting** — train loss ↓ while **val loss ↑** = memorizing trivia, not the pattern.
   Only visible by comparing the two. (Piece 1 saw this: eval_loss 1.079→1.113 at epoch 3.)
3. **Model selection** — which checkpoint, when to stop, which hyperparameters: validation is the
   unbiased signal for "is this version better?"

**Exam analogy:** training data = practice problems you study *with answers*; validation =
**mock exams** you take repeatedly while studying and adjust strategy from.

## 3. Why a TEST set too (not just validation)

**You "use up" the validation set by making decisions with it.** Every choice based on validation
(early stop, hyperparams, pick checkpoint) **indirectly fits** to it → its score becomes
**optimistic/biased** → no longer an honest estimate of real-world performance.

The **test set** is locked in a vault, **touched exactly once at the very end**, after all
decisions are frozen. Because no choices used it, its score is an **unbiased** estimate of
performance on truly new data.

| Set | Used for | How often | Stays honest? |
|---|---|---|---|
| Train | model learns | every step | — |
| Validation | tune/select **during** dev | repeatedly | **gets biased** |
| Test | one **final** unbiased score | **once, at the end** | **honest** |

**Exam analogy (finished):** test = the **real final exam**, taken once, never seen, never tuned
to → the honest verdict on whether you actually learned.

**Core rule:** *you can't honestly measure on data you used to make decisions.* Validation gets
contaminated by your tuning; test stays clean by being touched once.

**In AdapterForge:** both the sklearn baseline (**0.6885**) and the LLM (**0.8477**) were scored
on the **frozen test set** — the sealed, identical exam. Validation was used *during* training
(overfit detection). The **M3 promotion gate hashes the eval set** so every model is scored on
the **bit-identical sealed exam** — no eval-set drift, fair apples-to-apples. The test set is
sacred *because* its honesty depends on never being used for decisions.

## 4. "Training as a stopwatch" — Piece 2 vs Piece 3

Both run **real training** but **throw the model away** — the goal is timing, not a model.
They differ only in the variable measured:

| | Piece 2 (efficiency) | Piece 3 (distributed) |
|---|---|---|
| Variable | **precision** (fp32 vs bf16) | **# of GPUs** (1 vs 2, data-parallel) |
| Tool | plain SFTTrainer, 1 GPU | **Ray Train** TorchTrainer, 1→2 GPUs |
| Metric | step time | throughput (samples/sec) → scaling ratio |
| Validation run? | no | no |
| Model kept? | no (throwaway) | no (throwaway) |

Neither runs validation — they're pure training steps timed as a measurement. (Validation/test
matter for *quality* judgments like Piece 1's F1, not for *speed* measurements.)
