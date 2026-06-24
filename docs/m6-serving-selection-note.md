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
<!-- TODO(Karthik): one or two sentences each, from the project's experience.
     Prompts to address:
     - vLLM vs TGI: both are LLM engines — what would tip you one way? (you used vLLM;
       what did you like? when might TGI win?)
     - Triton: when is the generalist the right call over a per-model engine?
       (mixed fleet, the student via ONNX, dynamic batching, one ops surface)
     - KServe: what does it add that Triton/vLLM don't? (and why it's the M8 choice)
     - DeepSpeed: the one situation it's clearly the answer (model > 1 GPU).
     - TorchServe: would you pick it today? why / why not?
     Tie back to THIS project: LLM → vLLM (Piece 1); student → Triton/ONNX (Piece 2);
     M8 router puts cheap classification on the student slice, generation on the LLM. -->

## One-liner to say in an interview (Karthik — your words)
<!-- TODO(Karthik): a single sentence that captures the engine-vs-orchestrator split and
     the model-type→stack rule, so you can answer "how do you choose a serving stack?"
     unprompted. -->
