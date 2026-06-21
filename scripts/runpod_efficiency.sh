#!/usr/bin/env bash
# M5 Piece 2 — run the bf16 efficiency experiment on a RunPod GPU pod.
#
# Usage (from the repo root, on the pod):
#   bash scripts/runpod_efficiency.sh
#
# Prereqs: a RunPod "PyTorch" template (CUDA torch preinstalled) and this repo
# checked out. The script installs the rest of the deps, makes sure the
# instruction data exists, runs all 4 timing runs, and saves the output to a
# committable log under results/.
#
# COST: this only runs ON an already-started pod. Renting/stopping the pod is a
# manual step you do in the RunPod console — see docs/runpod-workflow.md.

set -euo pipefail

# --- 0. ensure a CUDA-enabled torch is present ---
# RunPod "PyTorch" templates ship CUDA torch already. On a bare template torch may be
# missing or a CPU-only wheel. Install the CUDA wheel ONLY if needed, so we never
# clobber a working CUDA build with the default CPU one.
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "CUDA torch not found -> installing CUDA wheel"
  pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

# hard gate: we must be on a CUDA box before spending compute
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU visible'; \
print('GPU:', torch.cuda.get_device_name(0))"

# --- 1. deps ---
# Some RunPod base images ship `blinker` via the OS package manager (no pip RECORD),
# which makes pip refuse to upgrade it ("uninstall-no-record-file"). Install a fresh
# pip-managed copy first so the requirements install doesn't choke on it.
pip install --ignore-installed blinker
pip install -r requirements-gpu.txt

# --- 2. ensure the instruction data exists (regenerate if missing) ---
if [ ! -f data/instruction/train.jsonl ]; then
  echo "instruction data missing -> regenerating with pipelines.instruction_format"
  python -m pipelines.instruction_format
fi

# --- 2b. MLflow destination heads-up ---
# We log to a remote tracking server via MLFLOW_TRACKING_URI (see docs/runpod-workflow.md).
if [ -z "${MLFLOW_TRACKING_URI:-}" ]; then
  echo "WARNING: MLFLOW_TRACKING_URI is not set -> MLflow logs to pod-LOCAL store (lost on teardown)."
  echo "         results/m5-efficiency.log is your interim safety net. Pull it before teardown."
else
  echo "MLflow tracking -> ${MLFLOW_TRACKING_URI}"
fi

# --- 3. run the experiment, tee to a durable log ---
mkdir -p results
LOG="results/m5-efficiency.log"
echo "Running efficiency experiment -> ${LOG}"
python -m pipelines.efficiency_experiment 2>&1 | tee "${LOG}"

echo
echo "Done. The two headline gains are at the bottom of ${LOG}."
echo "Pull this file off the pod (and the mlflow data if you logged remotely)"
echo "BEFORE you tear the pod down. Teardown is manual — see docs/runpod-workflow.md."
