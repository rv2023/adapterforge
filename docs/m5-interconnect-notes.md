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

## B. Tensor-parallelism theory module (whiteboard-level, no code)

Two ways to use multiple GPUs:

| | Data parallelism (we DO this) | Tensor parallelism (we LEARN only) |
|---|---|---|
| Idea | Each GPU holds a **full copy** of the model; each processes different data; average gradients | Cut a **single layer's matrix** into slices, one slice per GPU |
| When | Model **fits** on one GPU | Model is **too big** for one GPU (e.g. 70B+) |
| Our case | Qwen-1.5B fits → we do this on 2 GPUs | Never needed hands-on → theory only |

**Why theory is enough here:** Qwen-1.5B fits on one GPU, so we never *need* to split a
layer across GPUs. But interviewers ask, so know the contrast: data parallelism splits the
**data**, tensor parallelism splits the **model**.

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
