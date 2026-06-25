# M6 Piece 2 — Serving framework selection note: "when would I pick each?"

Scope (PROJECT_PLAN): deep hands-on on **vLLM** (Piece 1) + **Triton** (Piece 2); the
rest are **selection-criteria-only**. This note = the half-page judgment of when to
reach for which. Facts below are reference; the **"when I'd pick each"** calls are
Karthik's words (rule 5).

## What each tool is (reference)

| Tool | What it is | Sweet spot |
|---|---|---|
| **vLLM** | LLM inference *engine*: continuous batching + PagedAttention, OpenAI-compatible API, multi-LoRA | High-throughput **LLM** serving on one box |
| **Triton** (NVIDIA) | General-purpose multi-framework *server* (ONNX/TensorRT/PyTorch/TF/Python; vLLM + TensorRT-LLM backends); dynamic batching, ensembles, concurrent models, versioning, metrics | A **mixed fleet** of models behind one uniform production server |
| **KServe** | Kubernetes-native serving *layer* (CRD `InferenceService`): autoscaling incl. scale-to-zero, canary rollout, standard runtimes (can wrap Triton/vLLM) | **K8s** shops wanting serverless, standardized deploys (← M8) |
| **TGI** (HuggingFace) | LLM serving server (continuous batching, tensor parallel, quantization) — same niche as vLLM, HF ecosystem | LLM serving when you're HF-centric / want TGI's ops |
| **TorchServe** | General PyTorch model server (`.mar` archives, custom handlers) | Plain PyTorch models; simple, PyTorch-only; not LLM-optimized |
| **DeepSpeed-Inference / MII** (Microsoft) | Large-model inference optimizer: tensor parallelism, kernel injection, ZeRO-Inference for models too big for one GPU | **Very large** models / multi-GPU inference |

Two axes to keep straight: **engine** (vLLM, TGI, DeepSpeed, TensorRT-LLM — how one
model runs fast) vs **server/orchestrator** (Triton, KServe, TorchServe — how models are
hosted, batched, versioned, scaled). They compose (e.g. KServe → Triton → vLLM backend).
And recall the model-type split (concepts §11c): **encoder/classifier → ONNX/Triton**;
**decoder LLM → vLLM / TensorRT-LLM**, not plain ONNX.

## When I'd pick each (Karthik — your words)
<!-- DRAFT (Claude) — Karthik to rewrite in his own voice + be ready to defend each call. -->
- **vLLM** — my default for **LLM serving** (used it in Piece 1): PagedAttention, strong
  multi-LoRA support, OpenAI-compatible API, broad adoption. It's what I'd reach for to
  serve the Qwen+LoRA adapter at high throughput.
- **TGI** — the main vLLM alternative; same idea (continuous batching, tensor parallel,
  quantization). I'd pick it over vLLM mainly in a **HuggingFace-centric** shop already
  standardized on HF tooling; otherwise vLLM.
- **Triton** — when serving a **mixed fleet** of (mostly non-LLM) models — classifiers,
  embeddings, vision — behind **one production server** with dynamic batching, versioning,
  and metrics. That's exactly the DistilBERT student via ONNX in Piece 2. For LLMs inside
  Triton I'd use its **vLLM / TensorRT-LLM backend**, not plain ONNX (§11c).
- **KServe** — when I'm **on Kubernetes** and want **serverless inference** (autoscaling,
  scale-to-zero), **canary rollouts**, and standardized `InferenceService` deploys. It
  **wraps** engines like Triton/vLLM rather than replacing them — which is why it's the
  **M8** choice (we're on EKS by then).
- **DeepSpeed-Inference / MII** — when the **model is too big for one GPU** and I need
  tensor parallelism / ZeRO-Inference across GPUs. Not relevant for a 1.5B model.
- **TorchServe** — probably **wouldn't pick today** for new work: general PyTorch serving
  but not LLM-optimized and has lost momentum; Triton fills the generalist role better.

## One-liner to say in an interview (Karthik — your words)
<!-- DRAFT (Claude) — make it yours. -->
Separate the **engine** (how one model runs fast — vLLM/TGI/TensorRT-LLM/DeepSpeed) from
the **orchestrator** (how models are hosted, scaled, versioned — Triton/KServe/
TorchServe), then match by **model type**: decoder LLMs → vLLM/TensorRT-LLM, encoders/
classifiers → ONNX on Triton — with KServe tying it together on Kubernetes.
