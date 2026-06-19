# M5 Session Notes — LLM Fine-Tuning (Piece 1 core: DONE)

Sessions: 2026-06-18/19. Owner: Karthik. Tutor-mode learning.
**M5 Piece 1 status: CORE DONE — QLoRA adapter beats the M2 baseline.**
macro-F1 **0.8477** vs baseline **0.6885** (same sealed test set). Next: register/
promote the adapter, then Pieces 2 (efficiency), 3 (Ray/NCCL), 5 (distillation).

## What M5 Piece 1 is

Replace the M2 sklearn stand-in with a real fine-tuned LLM. Take Qwen2.5-1.5B, QLoRA
fine-tune it on Financial PhraseBank to classify bullish/bearish/neutral, and prove it
beats the locked sklearn bar on the *same frozen test set*. QLoRA = quantize the frozen
base to 4-bit (the "Q", shrinks 6 GB→~0.75 GB) + train tiny high-precision LoRA adapters
(the "LoRA", kills the gradient/optimizer memory) → a 1.5B fine-tune fits on a free GPU.

## Sub-steps (this session)

| Step | What |
|---|---|
| 1 | Instruction-format data → chat-messages JSONL (`pipelines/instruction_format.py`) |
| 2 | QLoRA training script, TRL SFTTrainer + PEFT (`pipelines/finetune.py`); CPU smoke test |
| 3 | Real run on free Colab T4 (4-bit Qwen-1.5B, batch 16, 3 epochs, ~60 min) |
| 4 | Generative-classifier eval vs 0.6885 (`pipelines/eval_adapter.py`) → 0.8477 |

## What got built

- `pipelines/instruction_format.py` — reshapes PhraseBank (text,label) → chat-messages
  JSONL under `data/instruction/`. **Reuses M2 `load_data`/`split_data` (seed=42)** so the
  test split is bitwise-identical to the 0.6885 bar; verified row-for-row. Bare label word
  as the assistant turn (eval string-matches it; all learning pressure on the one decision).
- `pipelines/finetune.py` — QLoRA fine-tuner. Ingredients: 4-bit base (BitsAndBytesConfig
  nf4 + bf16 compute) + `prepare_model_for_kbit_training`; `LoraConfig` r=16, alpha=32,
  dropout=0.05, all 7 target_modules; HF datasets loader; `SFTConfig` + TRL `SFTTrainer`
  (auto chat-template + loss-masking); train; save adapter. Dev/real switch via constants
  `MODEL_NAME`/`USE_4BIT`/`MAX_STEPS`.
- `pipelines/eval_adapter.py` — generative classifier eval: `PeftModel.from_pretrained`
  (adapter on frozen base) + `apply_chat_template(add_generation_prompt=True)` + greedy
  `generate(max_new_tokens=5, do_sample=False)` + parse word → sklearn macro-F1 vs 0.6885.
- `requirements.txt` — added M5 block (torch/transformers/peft/trl/accelerate;
  bitsandbytes Colab-only). `.gitignore` — added `/models/` (adapters/checkpoints/optimizer).

## Verified (the result)

- **CPU smoke test** (Qwen2.5-0.5B, 4-bit off, 2 steps): loss fell 5.12→3.52, adapter
  saved. Proved all 7 ingredients assemble for $0. Saw `optimizer.pt` ≈ 2× adapter size
  on disk = the Adam two-moments bucket from the VRAM math, made real.
- **Real Colab run:** loaded Qwen-1.5B in 4-bit (338 shards), tokenized 2170/465 clean,
  408 steps over 3 epochs. Mild overfit at epoch 3 (eval_loss 1.079→1.113, token-acc flat)
  — margin so large it didn't matter; did NOT re-run at 2 epochs.
- **Eval:** macro-F1 **0.8477** vs **0.6885** → **beats by ~16 pts (+23% relative)** on the
  identical sealed test set neither model trained on. Core deliverable met.

## Concepts Karthik now owns

- QLoRA's three layers: (1) full fine-tune memory = weights + gradients + optimizer +
  activations (~24 GB for 1.5B, optimizer alone ~2× weights); (2) LoRA freezes the base,
  trains tiny side-matrices → gradient/optimizer memory becomes a rounding error, adapter
  is MBs; (3) the "Q" quantizes the frozen base to 4-bit; adapter stays bf16 (precision
  follows training — quantize what you freeze, keep precise what you learn).
- The 7-ingredient fine-tune anatomy; what peft / trl / accelerate each do; SFTTrainer's
  chat-templating + loss-masking; **SEQ_CLS vs CAUSAL_LM** (we chose generative).
- LoRA knobs: r (capacity), alpha (strength, conv 2r), dropout; target_modules = the
  q/k/v/o attention + gate/up/down MLP linear layers; broad ("all-linear") vs minimal.
- Training mechanics: step = 1 batch = 1 fwd + 1 bwd + 1 update; epoch = full pass (136
  steps); 3 epochs = 408 steps; effective batch = batch × grad-accum; eval pauses training
  for a still snapshot; eval batch size is a speed dial, not a quality dial.
- Overfitting read: train loss ↓ while eval loss ↑ = memorizing trivia, not the rule
  (same data, early passes teach the rule, late passes teach the noise).
- **Fair comparison = same frozen test set + LLM trained only on train split** (the M3
  hash-the-eval-set principle). Big margin = robust win, overfit cost negligible here.

## Open threads / next

1. **Next session: register the adapter via M3 control plane (dossier test_f1=0.8477,
   eval_set_hash) + push through gated `/promote`** — governs the LLM like any prod model.
2. Cleanups (not blocking): refactor `finetune.py` dev/real switch to an env var (AF_MODE)
   so the real config is reproducible from git; fix the size-print to report only
   `adapter_model.safetensors` (the 156/163 MiB number is os.walk over checkpoints).
3. Trained adapter lives only on the laptop (downloaded from ephemeral Colab); `models/`
   is gitignored — decide a real home (DVC/S3) for trained adapters.
4. For Piece 5 (distillation teacher), consider `load_best_model_at_end=True` — teacher
   quality propagates to the DistilBERT student, so the best epoch matters more there.
5. Remaining M5: Piece 2 (bf16 efficiency, the JD "5%"), Piece 3 (Ray Train + NCCL on a
   paid 2-GPU RunPod pod — confirm $/hr first), Piece 5 (distillation + distill.yml).
