#!/usr/bin/env bash
# M6 Piece 1 — launch vLLM serving the Qwen base + the `fpb` LoRA adapter, in bf16.
#
# Apples-to-apples with the naïve server: SAME base, SAME adapter, SAME precision
# (bf16). The only difference is the serving engine (vLLM = continuous batching +
# PagedAttention). Requests address the adapter via  "model": "fpb".
#
#   ADAPTER_DIR=models/fpb-lora bash scripts/runpod_vllm.sh
#
# Then, from another shell:
#   python -m pipelines.benchmark_serving --server vllm --url http://localhost:8000 --model fpb
set -euo pipefail

BASE="Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR="${ADAPTER_DIR:-models/fpb-lora}"
PORT="${PORT:-8000}"

# adapter_config.json: r=16 -> vLLM's default --max-lora-rank (16) is exactly enough.
vllm serve "$BASE" \
  --enable-lora \
  --lora-modules "fpb=${ADAPTER_DIR}" \
  --dtype bfloat16 \
  --max-lora-rank 16 \
  --gpu-memory-utilization 0.9 \
  --port "$PORT"
