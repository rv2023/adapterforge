# RunPod GPU Workflow (replaces Colab for M5+)

From M5 Piece 2 onward, GPU runs happen on **RunPod**, not Colab. Dev/learning still
happens locally on the laptop (CPU) for free; RunPod is only spun up for the actual
GPU measurement/training, then torn down.

> **Cost guardrail (CLAUDE.md):** renting a pod costs money. Claude will never start a
> pod for you. You start/stop it yourself in the RunPod console. Always confirm the
> $/hr first and **tear it down when done** — idle pods keep billing.

## Why RunPod over Colab here

- bf16 needs **Ampere+** Tensor Cores (A10 / A40 / 4090 / A100). Colab's T4 is Turing —
  it *cannot* accelerate bf16, so the efficiency experiment would measure nothing on it.
- RunPod lets us pick an Ampere+ card by the minute and run headless from a script.

## Cost estimate (confirm live before renting)

| GPU | rough $/hr (community/spot) | fits this job? |
|---|---|---|
| RTX A4000 / A5000 | ~$0.20–0.40 | yes (24 GB+ helps the fp32 no-4bit run) |
| RTX 4090 | ~$0.40–0.70 | yes |
| A10 / A40 | ~$0.30–0.80 | yes |

The 4 runs are ~60 steps each → **minutes**. Budget **~$1–3** for the whole session
including setup/debug. Pick a card with **≥24 GB** so the fp32 (no-4bit) run fits at
batch 8.

## One-time per session

1. **Rent** a pod in the RunPod console (you do this): choose a **PyTorch template**
   (CUDA torch preinstalled), an **Ampere+ GPU ≥24 GB**, and Spot/Community for price.
   *Confirm the $/hr shown before you click deploy.*
2. **Get the repo onto the pod** (web terminal or SSH):
   ```bash
   git clone <your-fork-url> adapterforge && cd adapterforge
   # or: git pull, if the pod's volume already has it
   ```
3. **Run the experiment:**
   ```bash
   bash scripts/runpod_efficiency.sh
   ```
   This installs `requirements-gpu.txt`, regenerates `data/instruction/` if absent,
   runs all 4 timing runs, and tees output to `results/m5-efficiency.log`.

## Getting the results back (durability)

The pod is **ephemeral** — its local `mlruns/` / `mlflow.db` vanish on teardown.

**Decision: log to a remote tracking server via `MLFLOW_TRACKING_URI`.** Rationale: as an
operator, prod tracking lives in cloud infra — local is for coding, RunPod for learning,
**cloud infra is where the real project runs**. So the pod logs *straight to* the durable
server; no "pull before teardown" dance.

The code needs **no change** — MLflow reads `MLFLOW_TRACKING_URI` from the environment.
Set it before running:
```bash
export MLFLOW_TRACKING_URI=https://<your-mlflow-server>      # e.g. on the cloud cluster
bash scripts/runpod_efficiency.sh
```

> **Prerequisite (be honest):** this needs a **reachable MLflow tracking server** to exist.
> That server is itself a cloud-infra component (Tier 3) and may not be stood up yet. Until
> it is, the script's `results/m5-efficiency.log` (tee'd output) is the interim safety net —
> pull it down and commit it:
> ```bash
> scp -P <pod-ssh-port> root@<pod-ip>:/workspace/adapterforge/results/m5-efficiency.log ./results/
> ```

## Teardown (do this every time)

In the RunPod console, **Stop** then **Terminate** the pod (Terminate also frees the
volume if you don't need it). Verify the pod no longer appears as running so it stops
billing. There is no Claude command for this — it's a manual console action by design.

## Checklist

- [ ] Confirmed $/hr before renting
- [ ] PyTorch template, Ampere+ GPU ≥24 GB
- [ ] `torch.cuda.is_available()` is True on the pod
- [ ] Ran `scripts/runpod_efficiency.sh`
- [ ] Pulled `results/m5-efficiency.log` down and committed it
- [ ] **Terminated the pod**
