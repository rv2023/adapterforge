# M5 Primer — Floating Point from Scratch (fp32 / fp16 / bf16 / fp8)

> Context: groundwork for the M5 Piece 2 **bf16 efficiency experiment** (the JD's
> "≥5% step-time gain"). Before measuring fp32 vs bf16 step time, we needed to actually
> understand what these number formats *are*. This is that walkthrough, captured.

---

## 1. Everything is switches (bits)

A chip is billions of tiny switches (transistors). Each is **ON (1)** or **OFF (0)** — one **bit**.

- One switch alone says only two things.
- Line up N switches and read them as a group → they can spell **2^N** different values.
  - 8 switches → 2^8 = **256** values
  - 23 switches → 2^23 ≈ **8.4 million** values

A row of switches counts in **base 2**: place values double right-to-left (… 8, 4, 2, 1).
Example: `101` = (1×4)+(0×2)+(1×1) = 5.

## 2. Storing numbers with decimals = scientific notation

You can't store huge and tiny numbers exactly with a fixed budget, so the computer uses
school **scientific notation**, e.g. `602,000,000 = 6.02 × 10^8`. That splits any number into
three independent pieces:

```
[ sign ]  [ exponent = how big/small (the SCALE) ]  [ mantissa = the digits (the PRECISION) ]
```

- **sign**     → + or − (1 bit)
- **exponent** → the ZOOM/scale dial (range)
- **mantissa** → the FINENESS dial (how many trustworthy digits)

Mental model — a digital caliper: the **mantissa** is the digits on the readout (how fine),
the **exponent** is the units switch (mm vs m vs km, i.e. how big).

## 3. fp32 layout (the standard, 32 bits)

```
 1 switch   │   8 switches    │   23 switches
  sign      │   exponent      │   mantissa
            │  (range/scale)  │  (precision)
```

Two facts that keep tripping people up:

1. **"8 exponent" is NOT 10^8.** The 8 is a *count of switches* → 2^8 = 256 possible
   powers, running roughly **−126 to +127**, and the base is **2** (switches double).
   Biggest scale = `2^127`.
2. **The mantissa is a value between 1 and 2**, written `1.xxxxx` (normalized — exactly one
   digit before the point). The "8.4 million" is *how finely* you can land between 1 and 2,
   NOT a multiplier.

### The base-2 → base-10 bridge

```
2 ≈ 10^0.301        (because log10(2) ≈ 0.301)
```

So convert any power of 2 to base 10 by multiplying the power by 0.301:

- Range:     `2^127  = 10^(127 × 0.301) ≈ 10^38`  → fp32 reaches ~10^±38
- Precision: `2^23   = 10^(23 × 0.301)  ≈ 10^6.9` → ~7 significant **decimal** digits
- Sanity check you already know: `2^10 = 1024 ≈ 10^3` (a "kilo" byte). ✓

> **The mantissa is binary, not decimal.** It stores binary halving-fractions
> (1/2, 1/4, 1/8, … 1/2^23) after the leading `1.`. "~7 decimal digits" is just a
> human-friendly *translation* of how fine 23 binary bits are (via ×0.301), like
> converting km to miles. Same for "10^38" — the chip works in `2^127`.

### Building fp32's biggest / smallest

Every stored number = **(mantissa, between 1 and 2) × 2^(exponent)**.

```
Biggest  = ~2      × 2^127  = 2^128 ≈ 3.4 × 10^38
Smallest = 1.0     × 2^-126 =        ≈ 1.2 × 10^-38   (smallest "normal")
```

## 4. The naming convention: ExMy

Every format is just how you split the switches between the two dials:

- **E** = number of **E**xponent bits (range/zoom)
- **M** = number of **M**antissa bits (precision/fineness)
- + 1 sign bit, always.

| Short | Real name | sign | **E** (range) | **M** (precision) | total |
|---|---|---|---|---|---|
| fp32 | single precision (IEEE 754 binary32) | 1 | 8 | 23 | 32 |
| **bf16** | **bfloat16** ("brain float", from Google Brain) | 1 | 8 | 7 | 16 |
| fp16 | half precision (IEEE 754 binary16) | 1 | 5 | 10 | 16 |
| fp8 **E4M3** | FP8 (OCP/NVIDIA) | 1 | 4 | 3 | 8 |
| fp8 **E5M2** | FP8 (OCP/NVIDIA) | 1 | 5 | 2 | 8 |

- **bf16 = E8M7** → kept fp32's *full range*, slashed precision.
- **fp16 = E5M10** → more precision, but *range cut* to E5 → overflows/underflows.

## 5. What each can actually hold (end-user terms)

| Format | Biggest number | Smallest (≠0) | Trustworthy digits |
|---|---|---|---|
| fp32 | ~3.4 × 10^38 | ~1.2 × 10^-38 | ~7 |
| **bf16** | ~3.4 × 10^38 *(= fp32!)* | ~1.2 × 10^-38 | **~2** |
| fp16 | only **65,504** | ~6 × 10^-5 | ~3 |
| fp8 E4M3 | only **448** | ~0.0019 | ~1 |
| fp8 E5M2 | **57,344** | ~6 × 10^-5 | <1 |

Punchlines:
- **fp16 can't even store 100,000** — anything past 65,504 → infinity → math breaks.
- **bf16 reaches as far as fp32** (10^±38) but only ~2 trustworthy digits.

## 6. Worked example — store 8.1 in fp32 (forward)

Whole part `8 = 1000`. Fraction `0.1` via repeated ×2:

```
0.1 ×2 = 0.2 → 0
0.2 ×2 = 0.4 → 0
0.4 ×2 = 0.8 → 0
0.8 ×2 = 1.6 → 1   (keep .6)
0.6 ×2 = 1.2 → 1   (keep .2)
0.2 ×2 = 0.4 → 0   ← repeats forever: 0011 0011 …
```

So `8.1 = 1000.00011001100110011… (infinite)`. Normalize (shift point 3 left):

```
8.1 = 1.00000011001100110011010…  × 2^3
```

Fill the switches (mantissa keeps 23 bits, rounded):

```
0   10000010   00000011001100110011010
↑      ↑              ↑
sign  exponent      mantissa (infinite tail chopped + rounded)
      = 3+127=130
```

**Actually stored = 8.1000003814697…, NOT exactly 8.1.** The infinite tail got cut → the
~7-digit precision limit, live. (Same reason `0.1 + 0.2 = 0.30000000000000004`.)

## 7. Decode recipe (backward: switches → number)

```
1. sign bit:      0 → +, 1 → −
2. exponent bits: read as binary, SUBTRACT 127  → power of 2
3. mantissa:      prepend "1." → read as binary fraction → significand
4. multiply:      significand × 2^power
```

Backward on 8.1's switches:
```
sign 0 → +
exponent 10000010 = 130 → 130−127 = 3 → ×2^3 = ×8
mantissa 1.00000011001100110011010 → significand ≈ 1.0125000477
1.0125000477 × 8 = 8.1000003815   ✓ round-trip
```

Clean intuition: true 8.1 needs significand `8.1 ÷ 8 = 1.0125` exactly; best 23-bit binary
guess is `1.0125000477…` → overshoots → **8.1 rounds up**.

## 8. 8.1 vs 8.2 (both round, opposite directions)

| You typed | Significand wanted | 23-bit result | Stored | Error |
|---|---|---|---|---|
| 8.1 | 1.0125 | 1.0125000477 | **8.1000003815** | rounds **up** ↑ |
| 8.2 | 1.025  | 1.0249999762 | **8.1999998093** | rounds **down** ↓ |

Both `0.1` and `0.2` are infinite in binary; 23 switches can't hold either exactly. One rounds
up, one down → their errors don't cancel → `0.1 + 0.2 ≠ 0.3`.

## 9. 8.1 stored across every format (precision dying, made visible)

Same start: `8.1 = 1.0000001100110011… × 2^3` (significand wants 1.0125). Only **M** changes:

| Format | M bits | 8.1 stored as | what happened |
|---|---|---|---|
| fp32 | 23 | **8.1000003815** | ~7 digits — essentially exact |
| fp16 | 10 | **8.1015625** | ~3 digits |
| bf16 | 7 | **8.125** | ~2 digits — snapped to ~nearest 1/8 |
| fp8 E4M3 | 3 | **8.0** | steps of 1/8 in significand → grid 8,9,10… → ".1" erased |
| fp8 E5M2 | 2 | **8.0** | steps of 1/4 → grid 8,10,12… → even coarser |

Fewer mantissa bits → coarser grid → the number snaps to the nearest tick. That *is* precision loss.

## 10. Why training defaults to bf16 (the payoff)

- Training constantly handles **very tiny numbers** (gradients — the little weight nudges).
  Those need **wide range** or they underflow to zero and learning stalls. They do NOT need
  many digits.
- **bf16** = wide range (E8, same as fp32) + coarse digits → tiny numbers survive. ✅ ideal.
- **fp16** = narrow range (E5, max 65,504) → tiny/large numbers break → needs "loss scaling"
  rescue hacks. ⚠️
- Both are **half of fp32** → half the bytes to move → **faster… IF the GPU has the hardware.**

### The hardware catch (gates the experiment)

bf16 fast Tensor Cores exist only on **Ampere+ (A100 / A10 / 4090, compute ≥ 8.0)**.
The free **Colab T4 is Turing (7.5) — NO bf16 acceleration**; on a T4 the accelerated
half-precision is **fp16**. So the Piece-2 experiment has a real fork:

| Option | Hardware | A/B precision | Cost | Note |
|---|---|---|---|---|
| A | Free Colab T4 | fp32 vs **fp16** | $0 | honest, but "fp16 story"; JD says bf16 → footnote |
| B | RunPod A10/4090 | fp32 vs **bf16** | ~$0.40/hr (~$2–4) | matches JD/plan verbatim; bf16 actually accelerated |

(Decision still open — confirm $/hr before any RunPod spend per the cost guardrail.)

## 10b. Tensor Cores — the hardware behind the bf16 win

(Added session 4, after the A40 run measured bf16 ~41% faster than fp32 — this explains *why*.)

**Tensor Cores are physical hardware** — transistors NVIDIA etches into the GPU chip. PyTorch
does NOT create them; it *uses* them via the stack (same as M0):

```
Your code (HF Trainer bf16=True)
  ↓ PyTorch
  ↓ cuBLAS / cuDNN   ← NVIDIA libs that dispatch a matmul to Tensor Cores
  ↓ CUDA driver
  ↓ GPU silicon      ← Tensor Cores physically live HERE
```

A GPU has **two** kinds of math units:
- **CUDA cores** — general-purpose, do full fp32, one multiply-add at a time.
- **Tensor Cores** — specialized: multiply a small **matrix tile in one shot**, but only accept
  **low-precision inputs** (bf16/fp16). (Analogy: CUDA core = calculator doing one ×; Tensor
  Core = a machine multiplying a whole grid at once, but only takes "small" numbers.)

Training is mostly matmuls, so *which unit runs them* sets the speed:
- **bf16 matmul** → 16-bit inputs fit Tensor Cores → fast path. ✅
- **fp32 matmul** → 32-bit inputs too big → falls back to CUDA cores → slow path. ⏳

**Two compounding reasons bf16 wins:** (1) Tensor Cores, (2) half the bytes = half the memory
traffic. That's the ~1.7× (≈40%) we measured on the A40.

**Honest asterisk — TF32:** Ampere+ *can* push fp32 through Tensor Cores in a truncated **TF32**
mode (mantissa chopped to ~10 bits, fp32 range kept), enabled via
`torch.backends.cuda.matmul.allow_tf32 = True`. It speeds fp32 up, but bf16 still wins (half the
bytes + full Tensor-Core throughput) — which is why our gap stayed large. So precisely:
*"fp32 runs on CUDA cores or truncated-TF32 Tensor Cores; bf16 runs on full Tensor Cores AND
moves half the data."*

**What to actually use:**

| Situation | Use |
|---|---|
| Training on **Ampere+** (A40/A100/4090) | **bf16** (fast + stable) — the proven default |
| Training on **Turing (T4)** — no bf16 | **fp16** |
| Need **max precision** / debugging numerics | **fp32** (accept the slowdown) |
| Stuck in fp32 on Ampere, want some speed | fp32 **+ TF32 enabled** |

## 11. Honest step-time measurement (for when we run it)

1. **Warmup** — discard the first ~5–10 steps (CUDA kernel compile, cuDNN autotune, allocator
   warmup).
2. **GPU is async** — `torch.cuda.synchronize()` before reading the clock, or you time queue
   submission, not compute.
3. **Median over a fixed window** (~50 steps), not mean — robust to outlier steps.
4. **Normalize to samples/sec** so it stays comparable if batch changes.
5. **Hold everything else constant**: same GPU/session, model, data, batch size, LoRA config,
   step count. Change **one** knob per run.
6. **Confounder**: bf16 frees VRAM — don't also raise batch size, or you can't attribute the
   speedup to precision vs batch.
7. Note: **gradient accumulation ≠ speedup** (it buys larger *effective* batch, not faster
   per-sample compute). **Dataloader workers** only help if GPU is *waiting on CPU*
   (data-bound) — measure, don't assume. So the ≥5% is attributed to **precision**.

---

# PART 2 — 4-bit Quantization & Buckets (the QLoRA base)

> Floating point (Part 1) is how *one* number is stored. This part is how the *whole frozen
> base model* is squeezed to fit a small GPU. This is the "Q" in QLoRA — what
> `load_in_4bit=True` + `nf4` actually do in `finetune.py`.

## 12. The scale problem → why quantize

Qwen2.5-1.5B has **~1.5 billion** weights. Storage depends on the format of each weight:

```
1.5B × 4 bytes (fp32) = ~6 GB    just for weights
1.5B × 2 bytes (bf16) = ~3 GB
1.5B × 0.5 bytes (4-bit) = ~0.75 GB
```

A free T4 has ~15 GB, and activations/gradients/optimizer state pile on top. **4-bit squeezes
the frozen base ~8× so it fits.** It's a *memory* trick, not an accuracy goal.

## 13. 4-bit is NOT a tiny float — it's a lookup table ("buckets")

4 switches = only 2^4 = **16 values** — no room for a useful sign+exponent+mantissa float.
So 4-bit doesn't store mini-floats. It stores a **bucket number** (a row index into a small
menu of allowed values).

**T-shirt-size analogy (the core idea):**
- People's exact chest sizes are all different (37.2", 38.9"…) — too many to record precisely.
- Shops offer a small menu: S=36", M=38", L=40", XL=42".
- Each person is **rounded to the nearest size**; the shop records "**M**", not "37.2"".
- A "bucket" = one menu item. Recording "M" loses the exact inches (lossy).

Number the menu instead of naming it (computers prefer numbers):
```
bucket 0 = 36"   bucket 1 = 38"   bucket 2 = 40"   bucket 3 = 42"
```
**"bucket #6" just means "the 6th item on the menu."** The stored 4 bits ARE the row number.

| T-shirts | Model weights |
|---|---|
| exact chest sizes | exact weight values |
| too many to record | too many to store |
| menu S/M/L/XL | menu of allowed values |
| round to nearest size | round to nearest menu value |
| record "M" not 37.2" | record "bucket #6" not −0.18 |
| "M" only remembers ~38" | "bucket #6" only remembers −0.2 |

## 14. Where do the menu values come from?

From the **actual weights** — find their range, then spread the menu across it (just like a
shop measures its customers before picking sizes).

Example: a group of weights all fall in **−0.6 … +0.6**, with **4 menu slots**, evenly spaced:
```
step = (0.6 − (−0.6)) ÷ 3 gaps = 1.2 ÷ 3 = 0.4
bucket 0 = −0.6   bucket 1 = −0.2   bucket 2 = 0.2   bucket 3 = 0.6
```
That's where −0.6, −0.2, 0.2, 0.6 come from: lowest weight, highest weight, even steps between.
If weights spanned −2…+2, the menu would stretch to −2, −0.67, 0.67, 2. **The menu fits the data.**

**NF4 refinement** (`bnb_4bit_quant_type="nf4"`): most weights bunch near zero (bell curve), so
NF4 places **more buckets near zero** and fewer in the tails → smaller rounding error where most
weights live. Same principle, smarter spacing.

## 15. Storing & reading back one weight (lossy)

```
real weight (bf16):  −0.18
   ↓ round to nearest menu item
stored:              bucket 1  (= 0001, just 4 bits)   ← −0.18 forgotten forever
   ↓ look up bucket 1 in the menu
used in math (bf16): −0.2      ← NOT −0.18; the 0.02 is gone
```

**It is LOSSY — like a JPEG, not a ZIP.** Dequantizing does NOT restore the lost detail; it
just re-inflates the coarse bucket value into a bf16 container so the chip can do math.

## 16. Why uncompress at all (if it's lossy and we need bf16 anyway)?

Two reasons:
1. **The stored 4 bits are a label, not a number.** `activation × "bucket #6"` is meaningless
   (#6 is a pointer). You must look up its value (−0.2) first. The GPU's math units only
   multiply real formats (bf16/fp16/fp32) — there's no "multiply a 4-bit label" instruction.
2. **We only uncompress a sliver at a time.** The whole model *sits in memory* as 4-bit
   (~0.75 GB, always). During compute we dequantize just the current layer's weights to bf16,
   use them, and **throw the bf16 copy away** before the next layer. The full model never
   exists in bf16 at once.

So the win is in **what's resident** (0.75 GB vs 3 GB), not the brief per-layer uncompress.
Phone-photos analogy: 10,000 JPEGs fit on the phone; tapping one decompresses *that one* to
view, then closes it — you never decompress all 10,000 at once.

## 17. How many buckets? → 2^(bits) (same rule as everything else)

The bucket number is just a row of switches, so N bits → 2^N buckets:

| bits | buckets | note |
|---|---|---|
| 1-bit | 2 | |
| 2-bit | 4 | (the T-shirt example) |
| 3-bit | 8 | |
| **4-bit** | **16** | **← QLoRA nf4** |
| 8-bit | 256 | |

**Tradeoff:** more bits → more buckets → finer menu → less rounding error, but more storage.
4-bit (16 buckets) is the QLoRA sweet spot: big model fits a small GPU, and NF4's smart spacing
+ the **trainable full-precision LoRA adapters compensating** keep accuracy fine.

## 18. The three precisions live at once in QLoRA (ties Part 1 + Part 2)

During one forward pass, three number formats coexist:

| Where | Format | Set by (finetune.py) |
|---|---|---|
| Frozen base weights, at rest | **4-bit nf4** (buckets) | `load_in_4bit=True` |
| Base weights, instant before a matmul | **bf16** | `bnb_4bit_compute_dtype=torch.bfloat16` |
| LoRA adapters + activations | **bf16 (autocast)** | `bf16=USE_4BIT` |

**Why this makes the fp32-vs-bf16 experiment messy:** the base is *always* 4-bit underneath and
gets dequantized to bf16 regardless, so a "fp32 run" on top of 4-bit isn't truly fp32. Two/three
knobs interact → can't cleanly attribute a % gain. Hence the "what to hold constant" decision:
- **Option A — plain LoRA, no 4-bit:** flip only fp32↔bf16. One knob, cleanest attribution
  (needs more VRAM → the bigger GPU anyway).
- **Option B — keep 4-bit, vary only `compute_dtype`:** matches the real Piece-1 recipe, fuzzier
  attribution.

---

# PART 3 — QLoRA Mechanics & the Piece-2 Experiment Design

> This part captures the long back-and-forth that turned the abstract concepts above into a
> concrete experiment plan. It records *what we're doing in Piece 2, why, and the mental model
> behind it.*

## 19. The two Qwen sizes are different models (not one tuned into the other)

Fine-tuning **never changes a model's size.** `finetune.py` has two configs:

- **Qwen2.5-0.5B-Instruct** (line 18 default) — the **dev/CPU smoke-test** model (small enough
  for the laptop). This is what's committed.
- **Qwen2.5-1.5B-Instruct** — the **real** model; on Colab you swap `MODEL_NAME` to this and set
  `USE_4BIT=True`.

You do NOT "tune 1.5B down to 0.5B." They're different-sized members of the same family (like a
1.5L vs 0.5L engine). Shrinking a big model into a small one is **distillation** (Piece 5), and
even that trains a *separate* student.

## 20. What QLoRA actually trains: only the adapter (frozen base + sticky notes)

| | Used in the math? | Updated (trained)? | Stored as |
|---|---|---|---|
| **Base model** (Qwen) | Yes (read) | **No — frozen** | 4-bit (real run) |
| **LoRA adapter** | Yes | **Yes — the only thing trained** | full precision |

**Textbook + sticky-notes analogy:** the base is a *printed textbook* — you read it but can't
write in it (frozen), and it's stored shrunk (4-bit) to save shelf space. The LoRA adapter is
*sticky notes* in the margins. A forward pass reads **textbook + sticky notes together**; training
only ever **edits the sticky notes.** That's why the saved adapter is ~34 MB while the base is GBs.

## 21. What a "layer" is, and the per-step uncompress

A model is a **stack of layers** (≈28 floors in Qwen-1.5B). Data flows bottom→top, one floor at a
time. Each floor holds **weight matrices** doing two jobs:
- **Attention** — each word looks at other words for context (`q_proj, k_proj, v_proj, o_proj`).
- **Feed-forward** — each word is processed on its own (`gate_proj, up_proj, down_proj`).

Those `*_proj` names in the LoRA config **are the base layers' matrices**; LoRA bolts a small
trainable matrix beside each.

**Per training step (when 4-bit is ON):**
```
FORWARD (floor 1 → 28): unzip floor's 4-bit base → bf16 → compute(base+adapter) → discard
check how wrong (loss)
BACKWARD (28 → 1):       compute how to nudge ONLY the adapter
UPDATE:                  tweak adapter weights a hair; base untouched
```
This repeats **every step** — all floors re-unzipped each pass. That repeated unzip costs time.

**Compress/uncompress is a pair:** uncompressing only happens *because* the base was compressed
(4-bit). With **4-bit OFF**, weights already sit in usable fp32/bf16 → **no unzip step at all.**

## 22. Piece 1 vs Piece 2 — same action, different question

| | Piece 1 (DONE) | Piece 2 (NOW) |
|---|---|---|
| Question | "Is the model **good**?" | "Is training **fast**?" |
| Measure | **macro-F1** (quality) → 0.8477 | **step time / samples-per-sec** (speed) |
| Keep the model? | **Yes** — registered v14, promoted | **No — throwaway** |
| Compute F1? | Yes | **No — irrelevant** |
| 4-bit | ON (T4 needs it) | see §24 (we run both ON and OFF) |

Piece 1 = a test-drive (is the car good? keep it). Piece 2 = a dynamometer (how fast is the
engine? read the number, don't keep it). Piece 2 runs short **throwaway** training bursts purely
to time the steps.

## 23. The two optimizations are independent (resolves the "but prod uses 4-bit" worry)

| Optimization | Goal | Buys you |
|---|---|---|
| **4-bit** (quantization) | **Memory** — make the model *fit* | 6 GB → 0.75 GB; fits a small GPU |
| **bf16** (precision) | **Speed** — make each step *faster* | half the bytes + fast Tensor Cores |

They solve unrelated problems and toggle independently. Production keeps **4-bit** because the
deploy/train hardware (free T4) is small (a *memory* decision). The experiment measures the
**bf16 speed** win (a *measurement* decision). You **measure** one way and **ship** another —
no contradiction.

## 24. DECISION: the Piece-2 experiment (LOCKED)

**Hardware:** RunPod **A10/4090** (Ampere+), because bf16 Tensor Cores only exist there — the
free T4 (Turing) can't accelerate bf16, so the speedup wouldn't show. *Cost ~$0.40/hr (~$2–4);
confirm exact $/hr + teardown before any spend (cost guardrail).*

**Precision A/B:** **fp32 → bf16.** Note Qwen ships *on disk* in bf16, but `from_pretrained`
without `torch_dtype` **upcasts to fp32** — so fp32 is the *naive default* a beginner trains in
unknowingly (slow). bf16 is the *deliberate* optimization. The experiment quantifies that gain.

**We run BOTH conditions** (each is a short ~50-step throwaway burst; ~minutes, negligible cost):
| Run | 4-bit | Compare | Answers | Apples-to-apples w/ prod? |
|---|---|---|---|---|
| **A** | OFF | true fp32 vs true bf16 | "bf16 effect in isolation" (clean, JD's literal wording) | No |
| **B** | ON | fp32-compute vs bf16-compute | "does bf16 help in **my real** QLoRA recipe?" | **Yes** |

**Why both:** A gives the clean, defensible reference number and matches the JD bullet's wording;
B answers Karthik's apples-to-apples objection (you ship 4-bit, so measure with 4-bit). Reporting
both is the strongest, most honest deliverable.

**Honest expectation:** in **B** the per-layer unzip overhead sits on both runs and may shrink the
gap → B's % may come out **below 5%**. That's a *finding*, not a failure — documenting *why*
(dequant overhead) is exactly the platform-engineering insight the JD's efficiency bullet wants.

**Held constant across all runs** (the "change only one thing" rule): same model (Qwen-1.5B), same
data, same batch size, same step count, same LoRA config. Only the precision (and the 4-bit
toggle, which defines the A vs B pair) changes. Measurement hygiene per §11 (warmup discard,
`cuda.synchronize`, median over a window, samples/sec, no batch-size confound).

**Deliverable:** the % step-time gain(s) logged to MLflow as the documented Piece-2 result.
