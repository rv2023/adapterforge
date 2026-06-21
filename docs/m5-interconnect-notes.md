# M5 Piece 3 — Interconnect & Parallelism Notes (STUB)

> Placeholder so these JD-line theory deliverables aren't forgotten. The concepts are
> captured below; the **measured numbers** (nccl-tests bandwidth) get filled in when we
> rent the 2-GPU RunPod pod for Piece 3. Tier 2 work (see docs/operating-model.md).

The whole topic is about **one situation: training split across multiple GPUs.** When GPUs
train together, after every step they must **share their learning** (average their gradient
updates). These notes are about the *wires* that sharing travels over, and the *ways* to
split a model across GPUs.

---

## A. RoCE / InfiniBand awareness module (written explainer + 1 benchmark)

**Why it matters:** if the wires between GPUs are slow, a GPU finishes computing and then
**sits idle waiting to sync** → the *connection*, not the GPU, becomes the bottleneck.
Scaling to many GPUs is gated by sync speed. (JD: "interconnect-aware optimization.")

**The comm-cost hierarchy (fastest → slowest):**
- **NVLink** — super-fast direct cable between GPUs **in the same box**.
- **PCIe** — slower internal bus inside a box (the graphics-card slot).
- **InfiniBand (IB) / RoCEv2** — special high-speed **networks between separate machines**.

**Key terms:**
- **GPUDirect RDMA** — lets a GPU send data **straight** to another GPU over the network,
  *skipping* the CPU and system RAM (which the data would otherwise detour through).
  Skipping the detour = lower latency, higher bandwidth.
- **NCCL** — NVIDIA's library that actually performs the GPU-to-GPU collective ops
  (all-reduce, etc.). Tuning knobs (env vars):
  - `NCCL_IB_DISABLE` — turn InfiniBand transport on/off.
  - `NCCL_SOCKET_IFNAME` — pick which network interface NCCL uses.

**Deliverable / TODO (on the 2-GPU pod):**
- [ ] Run `nccl-tests` all_reduce; record **bus bandwidth (GB/s)**: `__________`
- [ ] One-paragraph written explainer of the hierarchy + GPUDirect RDMA in my own words.

---

## B. Data vs tensor parallelism — the two reasons to go multi-GPU

**The misconception to kill:** data parallelism is NOT about VRAM. There are two *unrelated*
reasons to use multiple GPUs (like 4-bit=memory vs bf16=speed earlier):

| Reason | Technique | Model fits on 1 GPU? | Gives you |
|---|---|---|---|
| Model **too big** for one GPU's VRAM | **tensor/model parallelism** (split the *model*) | **No** | makes it **fit** |
| Model **fits**, want it **faster** | **data parallelism** (full copy per GPU) | **Yes** | more **throughput/speed** |

**Data parallelism does the OPPOSITE of saving VRAM:** every GPU holds a *full copy* of the
model, so VRAM **per GPU** stays the same as the 1-GPU case (a bit *more*, for comms buffers)
and **total** VRAM is **N× more**, not less. It buys **speed** — each GPU chews a different
slice of the batch (~2× data/step); NCCL all-reduce just keeps the copies in sync.

| | Data parallelism (we DO this) | Tensor parallelism (we LEARN only) |
|---|---|---|
| Splits the... | **data** (batch across GPUs) | **model** (a layer's matrix across GPUs) |
| Each GPU holds | a **full copy** of the model | **a slice** of the model |
| VRAM effect | same per GPU, N× total (no saving) | less per GPU (the point) |
| When | model **fits** on one GPU | model **too big** for one GPU (70B+) |
| Our case | Qwen-1.5B fits → 2-GPU hands-on (Piece 3) | never needed → theory only |

**Why we rent 2 GPUs in Piece 3 even though Qwen-1.5B fits on 1:** we don't *need* them for
capacity — we rent them purely to **learn + measure** distributed data-parallel training and
the NCCL interconnect (the JD skills). It's a learning exercise, not a requirement.

**Why tensor parallelism stays theory-only:** it's the *VRAM-driven* technique, and our model
never overflows one GPU, so we never *need* it hands-on — just know the contrast.

**One-liner:** *data parallelism replicates the model for SPEED (more VRAM, not less); tensor
parallelism splits the model for CAPACITY. We do the first hands-on because our model fits;
the second stays theory because it never has to.*

**TODO:** [ ] whiteboard note: when does a model force tensor parallelism (VRAM math)?

---

## C. Honest gap — true multi-node RoCE/IB fabric

- **Node** = one physical machine. **Multi-node** = training across **several machines**,
  not just several GPUs in one box.
- A real IB **"fabric"** = the expensive specialized datacenter network wiring many
  machines together. Not accessible to a solo learner (cost + real datacenter hardware).
- **So:** we do **single-node** (2 GPUs in ONE rented box) NCCL benchmarks + theory only.
- **Interview framing (say it plainly):** *"I have the theory + single-node NCCL benchmarks,
  not a physical IB fabric — and most candidates can't even explain GPUDirect RDMA."*
  A deliberately-acknowledged limit, not a failure.

---

## D. Ray Train data-parallel mechanics (how Piece 3 actually works)

`pipelines/ray_finetune.py`. Captures the implementation learnings.

### Who does what — Ray orchestrates, your code loads
- **Ray Train** = orchestration only: spawn N worker processes, pin each to a GPU, wire up the
  torch **process group** (so NCCL all-reduce works). It does NOT touch the model.
- **`train_func`** runs *inside each worker* and does the loading: `from_pretrained` (the 4-bit
  base) + `SFTTrainer(peft_config=...)` (the LoRA adapter).
- So each worker independently loads its **own full 4-bit copy + adapter onto its own GPU** =
  data parallelism (N complete copies, one per GPU).

### Where the 4-bit model comes from + device placement
- Qwen ships **bf16** on disk. The **4-bit model is born at `from_pretrained`** — bitsandbytes
  quantizes bf16 → 4-bit (nf4 buckets) on the fly, straight into a device's VRAM. Never 4-bit
  on disk.
- A **4-bit model can't be `.to()`'d after creation** ("concrete poured in place"). So it must
  load directly onto the target GPU. With no `device_map` it may land on CPU → HF Trainer's
  auto-move-to-GPU then **crashes**.
- Fix: `device_map={"": 0}` in the 4-bit branch. **Per-worker-safe** because Ray sets
  `CUDA_VISIBLE_DEVICES` so each worker sees its GPU as `cuda:0`. (`device_map="auto"` is the
  anti-pattern — it *shards* one model across GPUs = model parallelism, wrong here.)

### How the data is partitioned (you don't tell Ray explicitly)
- Each worker loads the **full** dataset, but HF Trainer — once distributed (via
  `prepare_trainer` + Ray's process group) — **auto-inserts a `DistributedSampler`** that hands
  each **rank** a **disjoint slice** of indices. No manual sharding.
- `per_device_train_batch_size` is **per worker**; held constant → 2 workers = 2× distinct
  samples/step (true **weak scaling**), no overlap.
- Caveat: loading the full dataset on every worker is fine for our 2170 rows; for huge datasets
  use **Ray Data** (`ray.data` + `get_dataset_shard`) to stream a shard per worker. Not needed here.

### What actually gets all-reduced (reading the scaling result)
- Only the **adapter** is trainable (base frozen) → only **tiny LoRA gradients** get all-reduced
  (MBs, not GBs). So NCCL traffic is small → **scaling may be near-2× even on cheap PCIe** — not
  a bug, it's *because* QLoRA syncs so little.
- The interconnect lesson therefore shows up: **strongly in `nccl-tests`** (raw all-reduce
  bandwidth, model-independent) and **weakly in our LoRA run** (little to sync). Full fine-tuning
  (syncing all 1.5B gradients/step) is where the interconnect would really bite.
- So Piece 3 yields **two complementary numbers:** the LoRA scaling ratio (likely ~2×) and the
  nccl-tests busbw GB/s (the true interconnect capacity).
