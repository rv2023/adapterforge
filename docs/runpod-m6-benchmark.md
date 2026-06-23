# RunPod runbook — M6 Piece 1: vLLM vs naïve FastAPI benchmark

Goal: same model, two serving stacks, measured under load. Decisions: **RTX 4090
(~$0.40/hr)**, **bf16 both sides**, **standard sweep** (concurrency 1/4/8/16/32).
Estimated session: ~1.5–2 hrs ⇒ **~$1**. Confirm $/hr in the RunPod UI before you
start; run the teardown at the end.

> ⚠️ The pod is **ephemeral**. The benchmark writes `results/benchmark_serving_*.{json,md}`
> — **pull those off the pod before teardown** or they're gone (M5-teacher lesson).

---

## 0. Create the pod
- RunPod → GPU → **RTX 4090** (24 GB). Spot/community if available (cheaper).
- A PyTorch base template is fine. Put the workspace on the **persistent volume**
  (`/workspace`) so a crash doesn't wipe it.
- **Confirm the $/hr shown before clicking deploy.**

## 1. Repo + adapter
```bash
cd /workspace
git clone <your repo>  adapterforge   # or pull if already cloned
cd adapterforge
# pull the trained adapter from DVC/S3 (needs your AWS creds in env):
dvc pull models/fpb-lora.dvc
ls models/fpb-lora            # expect adapter_config.json + adapter_model.safetensors
```

## 2. Environment
The riskiest step (same family of pain as M5's torch). vLLM ships its own torch, so
install it into a clean venv and let it pull compatible deps; add what the naïve
server needs (peft/accelerate) on top.
```bash
python -m venv /workspace/m6-venv
source /workspace/m6-venv/bin/activate
pip install -U pip
pip install vllm                       # brings a compatible torch + transformers
pip install peft accelerate fastapi uvicorn requests pandas pyarrow scikit-learn mlflow
python -c "import torch, vllm, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```
If `cuda` is False or vLLM import fails: check the pod's CUDA vs the torch vLLM
pulled (M5 notes: driver-vs-toolkit mismatch). Worst case use the official
`vllm/vllm-openai` Docker image for the vLLM half and our venv for the naïve half.

## 3. Sanity: data + a single request each
```bash
ls data/instruction/test.jsonl        # the prompt source (ship via git or regenerate)
```

### Naïve server (port 8001)
```bash
ADAPTER_DIR=models/fpb-lora uvicorn serving.bench_naive:app --port 8001 &
# wait for "Application startup complete", then:
curl -s localhost:8001/predict -H 'content-type: application/json' \
  -d '{"text":"the company posted record profits"}'      # expect {"label":"bullish"}
```

### vLLM (port 8000)
```bash
ADAPTER_DIR=models/fpb-lora bash scripts/runpod_vllm.sh &
# wait for the vLLM startup banner, then:
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"fpb",
  "messages":[{"role":"system","content":"You are a financial sentiment classifier."},
              {"role":"user","content":"Classify the financial sentiment of the following statement as exactly one of bullish, bearish, or neutral.\n\nStatement: the company posted record profits"}],
  "max_tokens":5,"temperature":0}'
```

## 4. Run the benchmark (one stack at a time, so they don't share the GPU)
Watch VRAM in a separate shell during each run (the KV-cache growth observation):
```bash
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1
```

```bash
# naïve (vLLM stopped):
python -m pipelines.benchmark_serving --server naive --url http://localhost:8001
# stop naïve, start vLLM, then:
python -m pipelines.benchmark_serving --server vllm --url http://localhost:8000 --model fpb
```
Expect: naïve p99 climbs steeply with concurrency; vLLM stays much flatter + far
higher req/s. Note the **peak VRAM at concurrency 32** for each.

## 5. Pull results OFF the pod (before teardown!)
```bash
# from your laptop:
scp -P <port> root@<host>:/workspace/adapterforge/results/benchmark_serving_*.{json,md} ./results/
# (or: dvc add results/... && dvc push, or git add -f on the pod and push)
```

## 6. Teardown
```bash
# stop the servers
pkill -f "uvicorn serving.bench_naive" || true
pkill -f "vllm serve" || true
```
Then **STOP/TERMINATE the pod in the RunPod UI** (terminate = stops billing; stopping
only pauses compute but volume may still bill). Confirm it's gone.

## 7. Write-up
`docs/m6-benchmark-results.md` (Karthik's words): the two tables side by side, peak
VRAM per stack, and the one-paragraph "why vLLM wins" (continuous batching +
PagedAttention, §4–5 of the concepts doc).
