# M2 Session Notes — Training + Experiment Tracking

Session date: 2026-06-11. Owner: Karthik. Tutor-mode learning session.

## Where we are

M2 steps 1–3 **done**, step 4 (MLflow) **just started**, step 5 (overfit) pending.

| Step | Status | What |
|---|---|---|
| 1 | ✅ | Landed Financial PhraseBank via the HF adapter into the silver zone |
| 2 | ✅ | Baseline built; produced the bar |
| 3 | ✅ | Hand-tracked runs in `pipelines/runs.csv`; locked the bar |
| 4 | 🔄 | MLflow — concept explained, `pip install mlflow` next, skeleton not yet written |
| 5 | ⬜ | Deliberate overfit (50 passes, 500 rows) + plot train-vs-val curve |

## The headline result

**THE BAR = macro-F1 `0.6885` on the frozen TEST set, with `C=10`.**
This is the number every future model in the project (M5 LLM, M5 student) must beat
on the same sealed test set.

## What got built

- `pipelines/baseline.py` — load silver parquet → split 70/15/15 (stratified, seeded)
  → TF-IDF + LogReg pipeline → score. `build_model(c)` takes `C`; `main` times the
  fit (`perf_counter`), computes samples/sec, currently scores `C=10` on TEST and
  prints the LOCKED BAR.
- `pipelines/runs.csv` — hand-typed tuning log (validation scores):

  | C | val_macro_f1 | step_time_s | samples/sec |
  |---|---|---|---|
  | 0.01 | 0.5113 | 0.30 | 7211 |
  | 0.1 | 0.6062 | 1.42 | 1531 |
  | 1.0 | 0.6685 | 3.89 | 559 |
  | 10.0 | 0.6714 | 7.92 | 274 |
  | (100 earlier) | 0.6528 | — | — |

- `docs/m2-flow.md`, `docs/architecture.md` — diagrams (created this session).

## Concepts covered (the part that matters)

- **Why a baseline exists:** sets the bar so "the LLM is good" has a number to beat.
  A cheap model that ties the LLM is a *valuable* finding (→ M8 cost-aware router).
- **Three piles (train/val/test):** train = learn, val = tune knobs, test = sealed
  vault opened once. Decide on **validation**, confirm once on **test**. Tuning on
  test inflates the bar into a number that lies about future performance.
- **Why F1 not accuracy:** data is neutral-heavy (imbalanced); accuracy lets a
  lazy "always neutral" model hide. Macro-F1 forces good performance on every class.
  Chance level for 3 classes ≈ 0.33.
- **TF-IDF → LogReg:** TF-IDF turns text into meaningful numbers (down-weights
  filler words via IDF); LogReg is a *classifier* (despite the name) that learns a
  weight per word. Same task/data/metric as the M5 LLM → comparison is fair.
- **`class_weight="balanced"`** (fixes imbalance, kept on) vs **`C`** (regularization
  dial — how big the word-weights may grow; low=underfit/cautious, high=overfit/
  memorizes flukes). Change one knob at a time.
- **The C sweep is the underfit→sweet-spot→overfit curve.** Karthik generated it:
  0.01→0.51, 0.1→0.61, 1→0.67, 10→0.671(best), 100→0.653(dropping). C=10 wins on
  validation but only by ~0.003 over C=1 at 2× the train time — a real cost/accuracy
  call (foreshadows M8).
- **seed vs 70/15/15:** sizes vs *which rows*. Seed = "jumble then slice", frozen
  so the test set is identical every run. This is what M3's promotion gate hashes.
- **solver / converge / max_iter / C** (valley analogy): solver = ball rolling
  downhill to best weights; converge = reached bottom; max_iter = step budget
  (default 100, we use 1000); C shapes the valley — high C = flat plain = many
  steps = slow (explains the 26× timing spread).
- **step_time / samples_sec:** speed metrics logged from day one; habit pays off in
  M5's "≥5% step-time gain" experiment. samples/sec is the size-independent rate.

## Resume point (next session)

1. `pip install mlflow` (may already be done — check).
2. Answer the warm-up: sort `C`, `val_f1`, `step_time_sec`, `samples_per_sec`,
   `random_seed` into **params** (settings chosen) vs **metrics** (results measured).
   Expected: params = C, random_seed; metrics = val_f1, step_time_sec, samples_per_sec.
3. Get the MLflow skeleton for `main`: wrap the run in `mlflow.start_run()`, log the
   param(s), log the metrics, log the model artifact; local SQLite backend.
4. Run twice, open the MLflow UI, compare two runs side by side (the "aha").
5. Then Step 5: deliberate overfit + hand-plotted train-vs-val curve.

## Open threads

- M1 SDK README in Karthik's own words (rule 5) — still pending from M1.
- Untracked docs not yet committed: `docs/m2-flow.md`, `docs/architecture.md`,
  `docs/m2-session-notes.md`, plus `pipelines/`. Commits are Karthik's to make.
