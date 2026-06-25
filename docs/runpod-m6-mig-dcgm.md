# RunPod runbook — M6 Piece 3 (MIG lab) + Piece 4 (DCGM→Prometheus), one A100 session

Goal: feel **MIG isolation** vs **time-slicing's lack of it**, with **DCGM→Prometheus**
showing per-instance GPU metrics. Concepts: `docs/m6-serving-concepts.md` §7 (MIG/
time-slicing/MPS) + §8/§10 (DCGM). Est: A100 ~$1.8/hr × ~2–3 hr ⇒ **~$4–6**. Confirm
$/hr before deploy; terminate after.

> ⚠️ Capture screenshots/numbers as you go (the deliverable is *evidence*), and pull
> them off before teardown. Pod is ephemeral.

## 0. Create the pod + GATE on MIG being controllable
- RunPod → **A100 80GB** (MIG needs A100/H100). Prefer a dedicated/bare-metal-ish offer.
- **FIRST command — do not proceed until this works:**
  ```bash
  nvidia-smi -mig 1            # enable MIG (may need: -i 0, no running GPU procs)
  nvidia-smi --query-gpu=mig.mode.current --format=csv
  ```
  If this is **permission denied / not supported**, STOP — RunPod is blocking host-level
  MIG control. Terminate the pod; do the MIG hands-on in **M7** (EKS + GPU Operator).
  (Time-slicing + DCGM can still be demoed on any GPU; see §4–5.)

## 1. MIG: create 2 isolated instances
```bash
nvidia-smi mig -lgip                       # list available GPU-instance profiles
# create two instances (+ their compute instances). 3g.40gb x2 fills an 80GB A100:
nvidia-smi mig -cgi 3g.40gb,3g.40gb -C
nvidia-smi -L                              # note the two MIG-<UUID> device IDs
```

## 2. Run a model in EACH instance (isolation in action)
Pin each process to one MIG device via `CUDA_VISIBLE_DEVICES`:
```bash
# MIG 0 — the LLM via vLLM (reuse scripts/runpod_vllm.sh, set the device + a port)
CUDA_VISIBLE_DEVICES=MIG-<uuid-0> PORT=8000 bash scripts/runpod_vllm.sh &
# MIG 1 — the student via Triton (or serving/bench_naive). e.g. Triton:
CUDA_VISIBLE_DEVICES=MIG-<uuid-1> docker run --rm --gpus '"device=MIG-<uuid-1>"' \
  -p 8100:8000 -v "$PWD/serving/triton/model_repository:/models" \
  nvcr.io/nvidia/tritonserver:24.08-py3 tritonserver --model-repository=/models &
```

## 3. DCGM → Prometheus (Piece 4), per-instance metrics
```bash
docker compose -f observability/m6-dcgm/docker-compose.yml up -d   # or run dcgm-exporter on host (see caveat in that file)
curl -s localhost:9400/metrics | grep -E "DCGM_FI_DEV_FB_USED|GPU_UTIL" | head
# Prometheus UI http://localhost:9090 — query DCGM_FI_DEV_FB_USED, DCGM_FI_DEV_GPU_UTIL
```
Drive load at both servers (the Piece-1 harness for the LLM; the triton_client for the
student) and **screenshot the graphs** — you should see the two MIG instances tracked
separately. That's the KV-cache-VRAM-growth graph we skipped in Piece 1, now captured.

## 4. Prove MIG isolation (the money demo)
Make the LLM instance try to over-allocate (e.g. push concurrency / a big batch until it
OOMs). **Expected:** only MIG-0 errors; **MIG-1 (student) keeps serving** — hard wall.
Record it.

## 5. Disable MIG → time-slicing, and crash it on purpose
```bash
# tear down MIG
nvidia-smi mig -dci && nvidia-smi mig -dgi
nvidia-smi -mig 0
```
Now both models share the FULL GPU with no isolation (default time-slicing). Repeat the
over-allocation: **expected:** the OOM/contention takes down the *neighbor* too (or
starves it) — no wall. That contrast (§7) IS the deliverable.

## 6. Pull evidence + teardown
- Save screenshots + the numbers (per-instance VRAM, the isolation vs crash outcomes).
- `nvidia-smi -mig 0` (leave clean), stop containers.
- **Terminate the pod** in the RunPod UI. Confirm billing stopped. 💰

## 7. Write-up (Karthik, rule 5)
`docs/m6-mig-results.md`: MIG screenshots + numbers, the isolation-vs-time-slicing
outcome, and the "MIG vs time-slicing tradeoff in my words" (the JD line: explain it
unaided).
