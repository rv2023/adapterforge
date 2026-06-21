#!/usr/bin/env bash
# M5 Piece 3 — Ray Train data-parallel scaling + nccl-tests, on a 2-GPU RunPod pod.
#
# Usage (from repo root, on a 2-GPU pod):
#   bash scripts/runpod_ray.sh
#
# Prereqs: RunPod "PyTorch" template, GPU count = 2 (Ampere+), repo checked out, and
# data/instruction/train.jsonl shipped to the pod (see docs/runpod-workflow.md).
#
# COST: 2-GPU pod (~2x single). Renting/stopping is manual in the console. TEAR IT DOWN.

set -euo pipefail

# --- 0. venv inheriting the system CUDA torch (same rationale as runpod_efficiency.sh) ---
python -m venv --system-site-packages /tmp/af-venv
# shellcheck disable=SC1091
source /tmp/af-venv/bin/activate

# --- 1. CUDA gate — must see 2 GPUs ---
python -c "import torch; n=torch.cuda.device_count(); \
assert torch.cuda.is_available() and n>=2, f'need 2 GPUs, see {n}'; \
print('GPUs:', n, '|', torch.cuda.get_device_name(0))"

# --- 2. deps (includes ray[train]) ---
pip install -r requirements-gpu.txt

# --- 3. data must be present (NOT regenerable on a fresh pod) ---
if [ ! -f data/instruction/train.jsonl ]; then
  echo "ERROR: data/instruction/train.jsonl missing — ship it first (docs/runpod-workflow.md)." >&2
  exit 1
fi
mkdir -p results

# --- 4. scaling: 1-GPU baseline, then 2-GPU data-parallel ---
echo "=== Ray Train: 1 GPU ===" | tee results/ray-1gpu.log
python -m pipelines.ray_finetune 1 2>&1 | tee -a results/ray-1gpu.log
echo "=== Ray Train: 2 GPUs ===" | tee results/ray-2gpu.log
python -m pipelines.ray_finetune 2 2>&1 | tee -a results/ray-2gpu.log

# --- 5. nccl-tests all_reduce bus bandwidth (best-effort; don't lose the ray results) ---
# Needs the CUDA toolkit (nvcc at /usr/local/cuda) + NCCL. On a plain runtime image the
# build may fail; a "devel" PyTorch template has nvcc. If NCCL headers aren't found, pass
# NCCL_HOME=<path to libnccl>. We don't let a failure here abort the script.
set +e
(
  git clone --depth 1 https://github.com/NVIDIA/nccl-tests /tmp/nccl-tests
  cd /tmp/nccl-tests
  make -j CUDA_HOME=/usr/local/cuda
  # -g 2 = 2 GPUs; sweeps sizes 8B..256MB. Read the "busbw" column (GB/s).
  ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 2
) 2>&1 | tee results/nccl-tests.log
set -e

echo
echo "Done. Compare throughput across results/ray-1gpu.log and results/ray-2gpu.log"
echo "(scaling = 2gpu_samples_per_sec / 1gpu_samples_per_sec; ideal 2.0)."
echo "nccl-tests busbw (GB/s) is in results/nccl-tests.log — explains the gap from 2.0."
echo "Pull the results/*.log files off the pod, then TEAR THE POD DOWN."
