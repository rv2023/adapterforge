#!/usr/bin/env bash
# M5 Piece 2 — run the bf16 efficiency experiment on a RunPod GPU pod.
#
# Usage (from the repo root, on the pod):
#   bash scripts/runpod_efficiency.sh
#
# Prereqs:
#   - A RunPod "PyTorch" template (CUDA torch preinstalled), GPU count = 1, Ampere+ >=24GB.
#   - This repo checked out.
#   - data/instruction/train.jsonl present on the pod (it is gitignored and NOT regenerable
#     on a fresh pod — ship it first; see docs/runpod-workflow.md).
#
# COST: this only runs ON an already-started pod. Renting/stopping is a manual console
# step — see docs/runpod-workflow.md. TEAR THE POD DOWN when done.

set -euo pipefail

# --- 0. venv that INHERITS the system CUDA torch ---
# RunPod base images ship some Python packages via the OS (blinker, cryptography, ...) with
# no pip RECORD, so a plain `pip install` fails ("uninstall-no-record-file"); and a blanket
# --ignore-installed clobbers the template's CUDA torch with a mismatched build (the cu13
# wheel that an A40's CUDA-12.8 driver is too old for). A venv with --system-site-packages
# fixes BOTH: pip installs upgrades INTO the venv (shadowing the un-removable debian ones,
# no uninstall attempted) and inherits the system CUDA torch (we never list/reinstall torch).
python -m venv --system-site-packages /tmp/af-venv
# shellcheck disable=SC1091
source /tmp/af-venv/bin/activate

# --- 1. CUDA gate (must pass before spending compute) ---
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU visible'; \
print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))"

# --- 2. deps (torch deliberately NOT in this file) ---
pip install -r requirements-gpu.txt

# --- 3. data must already be on the pod (NOT regenerable here) ---
# data/ is gitignored and pipelines.instruction_format reads a local validated parquet that
# isn't cloned. Ship data/instruction/train.jsonl first: git force-add+pull, scp, runpodctl,
# or `dvc pull`. See docs/runpod-workflow.md.
if [ ! -f data/instruction/train.jsonl ]; then
  echo "ERROR: data/instruction/train.jsonl missing on the pod." >&2
  echo "       Ship it first (git force-add+pull / scp / runpodctl / dvc pull) —" >&2
  echo "       see docs/runpod-workflow.md." >&2
  exit 1
fi

# --- 3b. MLflow destination heads-up ---
if [ -z "${MLFLOW_TRACKING_URI:-}" ]; then
  echo "WARNING: MLFLOW_TRACKING_URI not set -> MLflow logs to a pod-LOCAL store (lost on teardown)."
  echo "         results/m5-efficiency.log is your interim safety net. Pull it before teardown."
else
  echo "MLflow tracking -> ${MLFLOW_TRACKING_URI}"
fi

# --- 4. run the experiment, tee to a durable log ---
mkdir -p results
LOG="results/m5-efficiency.log"
echo "Running efficiency experiment -> ${LOG}"
python -m pipelines.efficiency_experiment 2>&1 | tee "${LOG}"

echo
echo "Done. The 3 headline gains are at the bottom of ${LOG}."
echo "Pull ${LOG} off the pod (and re-log to durable MLflow if needed) BEFORE teardown."
echo "TEAR THE POD DOWN now — it bills while running."
