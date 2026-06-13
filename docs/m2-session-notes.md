# M2 Session Notes — Training + Experiment Tracking (COMPLETE)

Sessions: 2026-06-11 → 2026-06-13. Owner: Karthik. Tutor-mode learning.
**M2 status: all 5 steps DONE.** Next milestone: M3 (control plane).

## Headline result

**THE BAR = macro-F1 `0.6885` on the frozen TEST set, `C=10`.**
Every future model (M5 LLM, M5 student) must beat this on the same sealed test set.

## Steps

| Step | Status | What |
|---|---|---|
| 1 | ✅ | Landed Financial PhraseBank via HF adapter → silver zone |
| 2 | ✅ | Baseline: stratified seeded 70/15/15 split, TF-IDF + LogReg, locked the bar |
| 3 | ✅ | Hand-tracked C-sweep in `pipelines/runs.csv`; locked bar on test once |
| 4 | ✅ | Re-logged runs to MLflow (sqlite backend); compared in UI — matched CSV exactly |
| 5 | ✅ | Deliberate overfit (SGDClassifier, 500 rows, 50 epochs); plotted + read curves |

## What got built

- `pipelines/baseline.py` — load silver parquet → `split_data` (stratified, seeded
  70/15/15) → `build_model(c)` (TF-IDF + LogReg pipeline) → `evaluate` (macro-F1).
  `main` logs to MLflow: params (C, random_seed), metrics (val_f1, step_time_sec,
  samples_per_sec), and the model artifact. Backend = `sqlite:///mlflow.db`,
  experiment `m2-baseline`.
- `pipelines/runs.csv` — hand-typed validation tuning log (the deliberate "pain").
- `pipelines/overfit.py` — Step 5: tiny train (500 rows) + 50 epochs via
  `SGDClassifier(loss="log_loss")` + `partial_fit`; logs train/val log-loss per
  epoch; saves `docs/overfit_curve.png`; prints the sweet-spot (min val loss) epoch.
- `mlflow.db`, `mlruns/` — MLflow store (gitignore candidates).
- Diagrams: `docs/m2-flow.md`, `docs/architecture.md`.

## Session-1 concepts (baseline + tracking)

- Why a baseline: sets the bar so "the LLM is good" has a number to beat. A cheap
  model that ties the LLM is a *valuable* finding (→ M8 cost-aware router).
- Three piles: train=learn, val=tune, test=sealed vault opened once. Decide on
  validation, confirm once on test. Tuning on test inflates the bar into a lie.
- F1 not accuracy (neutral-heavy imbalance; chance ≈ 0.33 for 3 classes).
- TF-IDF → LogReg: TF-IDF = text→numbers (IDF down-weights filler); LogReg is a
  *classifier* despite the name. Same task/data/metric as M5 LLM → fair comparison.
- `class_weight="balanced"` (fixes imbalance, kept on) vs `C` (regularization).
- The C sweep = underfit → sweet-spot → overfit curve (Karthik generated it):
  0.01→0.51, 0.1→0.61, 1→0.668, 10→0.671(best), 100→0.653. C=10 wins on validation
  but only ~0.003 over C=1 at 2× train time — a real cost/accuracy call (→ M8).
- seed vs 70/15/15 = "which rows" vs "how big"; seed = jumble-then-slice, frozen.
- MLflow building blocks: set_tracking_uri (sqlite, needed for the M3 registry),
  set_experiment, start_run, log_param/log_metric/log_model. Re-logged runs matched
  the hand CSV exactly → the "aha" (seed makes it reproducible).

## Session-2 concepts (overfitting, deep ML mechanics)

- solver / converge / max_iter / C (valley analogy): solver = ball rolling to best
  weights; converge = reached bottom; max_iter = step budget (cap, stops early if
  converged); C shapes the valley (high C = flat plain = many steps = slow → explains
  the 26× timing spread across the C sweep).
- epoch vs C vs iteration: C = a separate experiment; epoch = one pass over train
  (one downhill step); the curve = train/val loss measured after each epoch.
- Overfitting is a SHAPE, not a number: train loss ↓ forever while val loss bottoms
  (sweet spot) then ↑. The widening train-val GAP = overfitting in motion.
- Forced it with SGDClassifier + partial_fit (run epochs by hand to measure between
  steps; LogReg one-shot fit can't show a curve). Tiny data + many epochs = overfit.
- Two knobs that both control overfit: **C/alpha** (freedom — how big weights may
  grow) and **epochs** (time). They interact: strong reg can hold back epoch-driven
  overfit. Demonstrated live: l2 default → val flat ~0.78; `penalty=None` → val
  CLIMBS to ~1.10. The regularized model GENERALIZES BETTER (0.78 < 1.10) — that's
  *why* we regularize.
- alpha double-duty in SGD: with `penalty=None` it leaves the objective but still
  sets the learning rate under `learning_rate="optimal"` (η ≈ 1/(α(t+t₀))).
- The math: minimize `E(w) = (1/n)Σ L(yᵢ,f(xᵢ)) + α·R(w)`; l2 → R=½‖w‖², None → R=0;
  update `w ← w − η(∇L + α∇R)`.
- Unifying picture: the model IS `y = mx + c` with thousands of x's (TF-IDF values),
  m = word-weights (what fit learns), c = bias, wrapped in softmax for 3 classes.
  Capital-C is NOT in the prediction — it lives in the *training objective*, a budget
  on how big the m's may grow. lowercase c = intercept (in prediction); capital C =
  regularization (in training). Different worlds, unlucky letters.

## Two quiz answers (for the record)

1. Val loss bottoms at epoch 1 then climbs → fix = **early stopping** (stop training
   at the sweet spot / lowest val loss). Meet it again in M5.
2. More data resists overfitting because the model can't memorize its way through
   thousands of varied rows the way it memorizes 500 — there's too much variety to
   fake, so it's forced to learn generalizable patterns instead.

## To close the milestone / next session (M3)

- Commit the work (Karthik's commits): `pipelines/` (baseline.py, overfit.py,
  runs.csv) + `docs/*.md`. Suggest gitignoring `mlflow.db`, `mlruns/`, `data/`,
  `docs/overfit_curve.png`.
- Still pending from M1: SDK README in Karthik's own words (rule 5).
- M3 = control-plane FastAPI: promote endpoint with policy gates (F1 margin,
  eval-set hash, schema compat, approved_by), audit JSONL to S3, lineage endpoint.
  The MLflow sqlite backend chosen here is what M3's registry reads.
