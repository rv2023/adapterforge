# M6 — Serving Frameworks + GPU Sharing: Session Notes

Running log of M6 work. Concepts live in `docs/m6-serving-concepts.md`; results will
land in `docs/m6-*-results.md` per piece.

## Plan (decided 2026-06-22 kickoff)

Piece order:
0. **Model-aware serving fix** — local, free. Foundation for the benchmark.
1. **vLLM + benchmark** — RunPod single GPU. p50/p95/p99 + throughput vs naïve FastAPI
   (same model, two stacks); watch KV-cache VRAM grow with concurrency.
2. **Triton or KServe** — one model through it + half-page "when would I pick each" note.
3. **A100 MIG lab** — FULL hands-on (decided): enable MIG → 2 isolated instances
   (LLM + student) → disable → time-slicing crash demo. ~$1.8/hr A100.
4. **DCGM → Prometheus** — graph GPU mem / SM util during the benchmark.

## Session 1 — 2026-06-22 (kickoff + design)

**Done:**
- Oriented on M6; taught the full concept stack and saved it to
  `docs/m6-serving-concepts.md`:
  - serving basics; why the current FastAPI is naïve for LLMs
  - autoregression; prefill vs decode (decode wastes the GPU); KV-cache
  - vLLM continuous batching (vs static); PagedAttention (OS virtual-memory borrow)
  - GPU sharing: time-slicing / MPS / MIG + the three-way tradeoff
  - the two-layer model: sharing (many models → one card) vs vLLM (many requests →
    one model); they stack
  - dispatch (M6) vs router (M8)
- **Design locked for Piece 0 (model-aware serving fix):**
  - stamp a `model_kind` registry tag (`lora_adapter` / `distilbert`)
  - backfill the tag onto existing v14 (and the student version)
  - serving reads `model_kind` from the control-plane `/production` response →
    dict-dispatch to (loader, predictor); download artifacts from registry then reuse
    the existing `eval_adapter` / `eval_student` loaders
  - **sklearn retired, NOT deleted**; build dispatch for LLM + student only
- **M8 carry-over written** into `docs/PROJECT_PLAN.md` (two-plane architecture;
  retraining is sklearn-bound and can't pass the gate vs the LLM; drift sensor
  piggybacks on TF-IDF vocab).

**Decisions:** MIG depth = full hands-on. Start order = Piece 0 first (free, local,
prerequisite for the benchmark).

**Next:** implement Piece 0 step 1 — add the `model_kind` tag at registration +
backfill v14/student. Then serving dispatch.

## Session 2 — 2026-06-22 (Piece 0, Step 1 DONE)

- `model_kind` tag wired into `register_model_with_dossier` as a **required** param
  (Karthik's call — gate/dispatch-trusted tags shouldn't be silently omittable).
  Bonus he added: `MODEL_KINDS` frozenset + a registration-time validation guard
  (invalid kind fails at register time, not at serving load).
- Wrappers pass their kind: `register_sklearn`→`sklearn`, `register_adapter`→
  `lora_adapter`, `register_student`→`distilbert`. loop.py/dag.py covered (go through
  `register_sklearn`).
- `pipelines/backfill_model_kind.py` (idempotent) ran + **verified in registry**:
  `fpb-sentiment` v14 = `lora_adapter`, `fpb-student` v2 = `distilbert`. Old sklearn
  fpb-sentiment v1–v13 left untagged (retired, never served).
- Open design Q for Step 2: what should serving dispatch do if `model_kind` is missing?

**Next:** Piece 0, Step 2 — serving dispatch (`serving/app.py`): read `model_kind`
from `/production`, dict-dispatch to (loader, predictor); download artifacts from
registry then reuse `eval_adapter` / `eval_student` loaders.

## Session 3 — 2026-06-22 (Piece 0, Step 2 DONE — model-aware serving)

`serving/app.py` is now model-aware:
- `PREDICTOR_BUILDERS` dict dispatches on `model_kind` → per-kind builder returning a
  `predict_fn(text) -> (label, confidence)` closure. `lora_adapter` + `distilbert`
  (sklearn intentionally absent — retired).
- Builders reuse the eval scripts: `eval_adapter.load_model_and_tokenizer` (now takes
  `adapter_dir`) + `predict_one` (LLM, prompt reuses `instruction_format.INSTRUCTION`);
  `eval_student.predict` (student). Artifacts pulled from registry via
  `download_version_artifacts(name, version)` → `mlflow.artifacts.download_artifacts`.
- **Decisions:** confidence = `None` for both kinds for now (LLM has no clean prob;
  distilbert could add softmax later for drift); missing/unknown `model_kind` →
  `RuntimeError` (fail loud at startup, names the bad kind).
- **Refactor (testability):** production load moved from module-import time into a
  FastAPI `lifespan` handler → stashed on `app.state`, endpoint reads
  `request.app.state`. Module now imports cleanly with no control plane / no GPU.
- **Bug caught by the smoke test:** `download_version_artifacts` hardcoded
  `MODEL_NAME` → a version number is ambiguous across registered models (LLM under
  `fpb-sentiment`, student under `fpb-student`). Fixed by threading `name` through
  download + both builders + `load_production_model` (passes `MODEL_NAME`; production
  always lives under fpb-sentiment). Latent in the real flow, but matters for the M8
  router which loads `fpb-student` by name.
- **Verified:** `build_distilbert_predictor("fpb-student","2")` on CPU →
  bullish/bearish/neutral correct. ruff clean. LLM path code-complete, GPU-only (will
  run for real in Piece 1).

**Next:** Piece 0 done. **Piece 1 — vLLM + benchmark on RunPod** (confirm GPU $/hr
first): vLLM serve the LoRA adapter, load-test harness, p50/p95/p99 + throughput vs
this naïve FastAPI (same model, two stacks), watch KV-cache VRAM grow with concurrency.

## Session 4 — 2026-06-23 (Piece 1 kickoff — benchmark harness)

**Decisions locked:** RTX 4090 (~$0.40/hr) for the benchmark (A100 only later for the
MIG lab — only A100/H100 do MIG); **bf16 on both stacks** (isolate the serving engine,
not precision); **standard sweep** (concurrency 1/4/8/16/32). Build sequence:
(1) harness [local] → (2) naïve bench server bf16 [local write] → (3) vLLM launch
[pod] → (4) rent 4090, run both, capture VRAM [pod] → (5) write-up.

**Concepts taught + saved** to `docs/m6-serving-concepts.md` §10: the load-test
harness (ThreadPoolExecutor, percentiles, persist-numbers-off-the-ephemeral-pod);
temperature & greedy decoding (T=0 = greedy = matches naïve `do_sample=False`; ties
back to distillation's T=2 softening); two-servers-two-APIs (naïve `/predict` vs vLLM
`/v1/chat/completions`) and why a separate `vllm_sender` is required + the
apples-to-apples prompt-matching trap.

**Harness (`pipelines/benchmark_serving.py`) — IN PROGRESS (Karthik writing):**
- DONE + verified locally: `summarize` (p50/p95/p99 + req/s), `run_level`
  (ThreadPoolExecutor concurrency — overlap confirmed: 16×0.1s → c=1 ~1.6s, c=8 ~0.2s),
  `load_prompts` (reuse frozen test set, strip INSTRUCTION), `naive_sender`, argparse
  + `make_send` closures (good: default-arg binding avoids late-binding bug). ruff clean.
- **TODO (open):** (a) `vllm_sender` must switch to `/v1/chat/completions` with the
  same `[system, user(INSTRUCTION+text)]` messages — current code uses `/v1/completions`
  with a bare statement (breaks apples-to-apples + chat-trained model). (b) finish
  `main` — print the table + **persist rows to json/md** (don't lose them with the pod).
  (c) nit: import `instruction_format.INSTRUCTION` instead of the hand-copied prefix
  (instruction_format now lazy-imports baseline, so it's cheap).

**Next:** finish harness TODOs → build naïve bench server (bf16) → vLLM launch script →
confirm 4090 spend + teardown → run.

## Session 5 — 2026-06-23 (Piece 1 RUN — benchmark done on RunPod L4)

GPU: **L4** (4090 was out of capacity; any Ampere+ works, gap is the same). bf16 both
stacks, sweep 1/4/8/16/32, 200 req/level.

**Env saga (RunPod image was cu130 torch on a CUDA-12.4 driver = broken OOB):**
- pip pulled torch+cu130 → `cuda.is_available()=False` (driver too old). Fix: pinned
  **vllm==0.6.6** → torch 2.5.1+**cu124** (matches driver) → cuda True.
- venv never isolated (image sets `PYTHONPATH` to system dist-packages) + `venv` hung →
  **abandoned venv, used system Python.**
- system transformers too new (5.12) → eagerly imported torchaudio (cu130) →
  `libcudart.so.13`. Fix: `pip install transformers==4.46.3` + `pip uninstall torchaudio`.
- `numpy 2.5` vs vllm's `<2` → `pip install "numpy<2" "scipy<1.13"`.
- `HF_HUB_ENABLE_HF_TRANSFER=1` set but pkg missing → `export HF_HUB_ENABLE_HF_TRANSFER=0`.
- vLLM chat: transformers ≥4.44 ships no default chat template → added
  `--chat-template models/fpb-lora/chat_template.jinja` (now folded into runpod_vllm.sh).

**RESULTS (L4, results/benchmark_serving_{naive,vllm}.{json,md}):**

| conc | naïve p99 ms | vLLM p99 ms | naïve req/s | vLLM req/s |
|---|---|---|---|---|
| 1 | 163 | 115 | 8.0 | 10.6 |
| 4 | 1222 | 216 | 6.4 | 25.6 |
| 8 | 1987 | 214 | 5.3 | 48.3 |
| 16 | 4039 | 268 | 5.2 | 90.3 |
| 32 | **10667** | **302** | **4.7** | **128.2** |

At c=32: vLLM ~**27× throughput**, ~**35× lower p99**. Shapes: naïve throughput DROPS +
p99 explodes 65×; vLLM throughput CLIMBS + p99 ~flat. = continuous batching + PagedAttention.

**TODO (Karthik):** peak-VRAM numbers (naïve vs vLLM @ c=32) for the write-up;
`docs/m6-benchmark-results.md` (his words). Confirm pod terminated.

**Next piece:** Piece 4 (DCGM→Prometheus) or Piece 2 (Triton/KServe + selection note);
Piece 3 (A100 MIG lab) is the big one.

## Session 6 — 2026-06-23 (Piece 2 kickoff — Triton/ONNX concepts)

VRAM for Piece 1 marked **not captured** in `m6-benchmark-results.md` (deferred to
Piece 4 DCGM). "Why vLLM wins" paragraph still Karthik's TODO.

**Concepts taught + saved** (`docs/m6-serving-concepts.md` §11): Triton = generalist
serving server (any framework) vs vLLM the LLM specialist; Triton **dynamic batching**
(request-boundary) vs vLLM **continuous batching** (per-token). **ONNX** = portable
framework-agnostic model file (graph + weights; PDF/bytecode/Docker-image analogy;
export via torch.onnx/optimum → ONNX Runtime → Triton ONNX backend; verify output
matches PyTorch). **Why ONNX fits the student but NOT the LLM:** static graph (one
forward) vs autoregressive loop + stateful growing KV-cache + dynamic shapes → LLMs
need vLLM or TensorRT-LLM-in-Triton instead. Right-tool-per-model-type table →
foreshadows M8 cost-aware router.

**Piece 2 plan (decided):** tool = **Triton** (KServe = M8); model = **DistilBERT
student**; backend = **ONNX (decision A)**; where = **local Docker CPU**
(`nvcr.io/nvidia/tritonserver`), **$0**. Deliverables: student served through Triton +
one inference request; half-page selection note (vLLM/Triton/KServe/TGI/TorchServe/
DeepSpeed), Karthik's words.

**Next:** teach Triton model-repository layout → ONNX-export skeleton + config.pbtxt →
run Triton in Docker → inference request → selection note.

## Session 7 — 2026-06-23/24 (Piece 2 BUILD — student served through Triton)

**DONE — DistilBERT student served through Triton (local Docker, CPU, $0):**
- `pipelines/export_student_onnx.py` — PyTorch student → ONNX (LogitsOnly wrapper for
  tensor-in/out; dynamic batch+seq; verify allclose vs PyTorch). Export blob
  (`model.onnx` + `model.onnx.data`, 256 MB) gitignored.
- `serving/triton/model_repository/distilbert-student/{config.pbtxt,1/model.onnx}` —
  onnxruntime backend, dynamic_batching, KIND_CPU.
- `serving/triton/triton_client.py` — tokenizes client-side, KServe v2 infer request,
  argmax→label. **Verified:** "record profits"→bullish, "shares plunged"→bearish;
  logits match the PyTorch export bit-for-bit. Triton `distilbert-student | 1 | READY`.

**Three ONNX-export gotchas hit + solved (now in concepts §11e):**
1. dynamo exporter ignores `dynamic_axes` → `dynamo=False` (classic).
2. 1-row dummy freezes OUTPUT batch dim to [1,3] → trace with a 2-row dummy → [-1,3]
   (Triton needs -1 for batching; max_batch_size>0 is a promise the model takes batches).
3. SDPA attention doesn't trace cleanly (graph diverges, allclose fails) →
   `attn_implementation="eager"`. (Verify caught a real wrong graph — fixed export, did
   NOT loosen atol.)

**TODO (Karthik, rule 5):** the half-page selection note (vLLM vs Triton vs KServe vs
TGI/TorchServe/DeepSpeed — when to pick each). Then Piece 2 closed.

**Next pieces:** Piece 4 (DCGM→Prometheus), Piece 3 (A100 MIG lab — the big one).

**Open / deferred:** loop.py model-aware retraining + drift sensor → M8. MPS hands-on
optional. M1 SDK README + RoCE/IB explainer still open (rule 5).
