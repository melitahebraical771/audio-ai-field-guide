# 🧊 The Complete Guide to LLM Quantization

> From bit-level floating-point math to 1-bit models — a technical reference for understanding *why* and *how* large language models get compressed.

<div align="center">

![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Language](https://img.shields.io/badge/language-English-blue)
![Topic](https://img.shields.io/badge/topic-LLM%20Quantization-orange)
![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)

</div>

---

## 📌 About This Guide

Quantization is one of the most important optimization techniques in LLM deployment — it's the reason a 70-billion-parameter model can go from requiring a rack of $100k GPUs to running on a single consumer graphics card.

This guide is built around **mental models + exact math + fully worked numerical examples**, so instead of memorizing definitions, you walk away with a real intuition for the underlying mechanics:

- 📐 The precise math behind affine (linear) mapping
- ⚖️ Symmetric vs. asymmetric quantization, with a side-by-side worked example
- 🎯 Calibration strategies (MinMax, MSE, Entropy/KL-divergence)
- 🔬 The outlier-feature problem in LLMs and the state-of-the-art fixes (LLM.int8(), SmoothQuant, AWQ, GPTQ)
- 🧮 Floating-point architecture (FP32 → FP16 → BF16 → INT8), decoded bit by bit
- 💾 Exact memory math for deploying large models
- 🚀 The frontier: 4-bit, ternary weights, and 1-bit BitNet models

> ℹ️ **Transparency note:** This guide is an independently written, restructured, and substantially expanded technical reference inspired by the *style and scope* of Maarten Grootendorst's visual quantization guides — it is not a reproduction of his text (the original was behind a paywall at time of writing). All formulas, tables, and worked examples here were written and verified independently. For the original animated, illustration-first treatment, see [his newsletter](https://newsletter.maartengrootendorst.com/).

---

## 📑 Table of Contents

- [1. What Is Quantization, and Why Does It Matter?](#1-what-is-quantization-and-why-does-it-matter)
- [2. Floating-Point Architecture](#2-floating-point-architecture)
  - [2.1 Anatomy of FP16](#21-anatomy-of-fp16)
  - [2.2 Why BF16 Replaced FP16 for Training](#22-why-bf16-replaced-fp16-for-training)
  - [2.3 Worked Example: Decoding π in FP16](#23-worked-example-decoding-π-in-fp16)
- [3. Memory Requirements of Large Models](#3-memory-requirements-of-large-models)
- [4. Linear Mapping: The Math of Uniform Quantization](#4-linear-mapping-the-math-of-uniform-quantization)
- [5. Symmetric vs. Asymmetric Quantization](#5-symmetric-vs-asymmetric-quantization)
  - [5.1 Worked Example: Head-to-Head Comparison](#51-worked-example-head-to-head-comparison)
- [6. Calibration Strategies](#6-calibration-strategies)
- [7. Granularity: Per-Tensor, Per-Channel, Per-Group](#7-granularity-per-tensor-per-channel-per-group)
- [8. Weights vs. Activations](#8-weights-vs-activations)
- [9. PTQ vs. QAT](#9-ptq-vs-qat)
- [10. Static vs. Dynamic Quantization](#10-static-vs-dynamic-quantization)
- [11. The Outlier-Feature Problem in LLMs](#11-the-outlier-feature-problem-in-llms)
  - [11.1 LLM.int8()](#111-llmint8)
  - [11.2 SmoothQuant](#112-smoothquant)
  - [11.3 AWQ](#113-awq)
  - [11.4 GPTQ](#114-gptq)
- [12. The 4-Bit Frontier and the GGUF Format](#12-the-4-bit-frontier-and-the-gguf-format)
- [13. The Final Frontier: 1.58-Bit Models and BitNet](#13-the-final-frontier-158-bit-models-and-bitnet)
- [14. Summary and a Practical Decision Table](#14-summary-and-a-practical-decision-table)
- [15. Further Reading](#15-further-reading)

---

## 1. What Is Quantization, and Why Does It Matter?

**Quantization** is the process of mapping continuous, high-precision values (like `FP32` or `FP16`) into a discrete, lower-precision space (like `INT8` or `INT4`). It targets three costs simultaneously:

| Cost | Effect of quantization |
|---|---|
| 💾 Memory footprint (VRAM/RAM) | Reduced 4–8× depending on target precision |
| 🚌 Memory bandwidth | Directly reduced — less data has to move per operation |
| ⚡ Latency & power draw | Integer arithmetic runs faster and cooler than floating-point on most hardware |

> 💡 **Key insight:** Deep neural networks are remarkably robust to numerical noise. That robustness is *precisely* what makes quantization possible — the goal was never infinite precision, only *sufficient* precision.

---

## 2. Floating-Point Architecture

Before compressing a number, it helps to know what you're compressing. The IEEE 754 standard splits every floating-point number into three fields:

```
┌─────────┬──────────────────┬─────────────────────────────┐
│  Sign   │     Exponent      │          Mantissa            │
│ (± bit) │  (dynamic range)  │      (precision/resolution)  │
└─────────┴──────────────────┴─────────────────────────────┘
```

- **Sign:** whether the number is positive or negative.
- **Exponent:** sets the *dynamic range* — are we in the territory of an electron's mass or a planet's mass?
- **Mantissa:** sets the *precision* — how finely can we resolve values within that range?

### 2.1 Anatomy of FP16

| Field | Bits | Technical role | Practical effect |
|---|:---:|---|---|
| Sign | 1 | Positive/negative | Direction on the number line |
| Exponent | 5 | Dynamic range | Shifts the binary point via powers of 2 |
| Mantissa | 10 | Precision | Spacing between representable values |

Reconstruction formula:

```
value = (-1)^sign × 1.mantissa × 2^(exponent − 15)
```

The **15** is called the **bias** (`2^(5-1) - 1`). It exists because the raw exponent field can only express non-negative integers (0–31), but we also need negative exponents to represent very small numbers. Subtracting the bias gives us that negative range for free.

- Exponent = 0 → true exponent `0 − 15 = -15` (very small numbers)
- Exponent = 30 → true exponent `30 − 15 = +15` (very large numbers)

**The implicit bit:** hardware always assumes a leading `1.` before the mantissa that is never stored — a free bit of precision, effectively giving the 10-bit mantissa 11 bits of resolution.

**Special values:**
- All exponent bits = 1, mantissa = 0 → `Infinity`
- All exponent bits = 1, mantissa ≠ 0 → `NaN`

### 2.2 Why BF16 Replaced FP16 for Training

During training, gradients can shrink extremely fast. With only 5 exponent bits, FP16 underflows to zero for anything smaller than `2^-14` — a real problem for stable training (**underflow**).

Google's fix was **BF16 (Brain Float 16)**:

| Format | Sign | Exponent | Mantissa | Dynamic range | Precision |
|---|:---:|:---:|:---:|---|---|
| FP16 | 1 | 5 | 10 | Narrow (±65,504) | Higher |
| BF16 | 1 | 8 (same as FP32) | 7 | Full FP32 range (±3.4×10³⁸) | Lower |

BF16 trades mantissa precision for FP32's full dynamic range — which is why it's the de facto standard for training modern LLMs: it essentially eliminates overflow/underflow in gradients, at the cost of a slightly coarser representation of any single value.

### 2.3 Worked Example: Decoding π in FP16

Suppose the following bit pattern is stored in memory, meant to approximate π:

```
Sign: 0     Exponent: 10000     Mantissa: 1001001000
```

**Step 1 — Sign bit:**
`(-1)^0 = +1` → positive number.

**Step 2 — Decode the exponent:**
Binary `10000` = decimal `16`. Subtract the bias (15): `16 − 15 = 1` → true exponent `2^1 = 2`.

**Step 3 — Compute the mantissa:**
In `1001001000`, bits 1, 4, and 7 are set:

```
2^-1 = 0.5
2^-4 = 0.0625
2^-7 = 0.0078125
─────────────────
Sum  = 0.5703125
```

**Step 4 — Assemble the final value:**

```
value = 1.5703125 × 2^1 = 3.140625
```

| | Value |
|---|---|
| True π | 3.1415926... |
| FP16 reconstruction | 3.140625 |
| **Quantization error** | **0.000967** |

This small error is a direct consequence of the mantissa's 10-bit resolution ceiling in FP16 — the exact same phenomenon that, at a much larger scale, is what INT8/INT4 quantization is fundamentally fighting against.

---

## 3. Memory Requirements of Large Models

The raw formula for the memory needed to load a model's weights:

```
Memory (bytes) = (number of parameters) × (bits per parameter ÷ 8)
```

For a 70-billion-parameter model (like Llama-3-70B), here's what bit-width reduction buys you:

| Format | Bits/param | Bytes/param | Raw weight memory | Approx. hardware needed |
|---|:---:|:---:|---:|---|
| FP32 | 32 | 4 | 280 GB | Multi-GPU cluster of top-tier accelerators |
| FP16 / BF16 | 16 | 2 | 140 GB | At least 2× A100/H100 (80GB) |
| INT8 | 8 | 1 | 70 GB | A single A100 (80GB) |
| INT4 | 4 | 0.5 | 35 GB | Two consumer RTX 4090s |

> ⚠️ **Important caveat:** this table only covers *static weight* memory. Real inference also needs memory for the **KV cache** (stored attention history) and **activations**. For a model with a raw 35 GB weight footprint at INT4, budget roughly **40–45 GB of VRAM** in practice to avoid out-of-memory crashes during long conversations.

---

## 4. Linear Mapping: The Math of Uniform Quantization

Key insight: model weights never come anywhere near `±3.4×10³⁸`. They typically live in a tight range like `[-2.5, 3.8]`. So there's no need to map the *entire* FP32 range onto our target integer range — we only need to stretch or squeeze the **actual data range** onto the target range (e.g., `[-127, 127]` for INT8). This is called **affine (linear) mapping**.

**Quantization formula:**

```
scale (S)      = (β − α) / (q_max − q_min)
zero_point (Z) = round(q_min − α / S)
x_quantized    = round(x / S + Z)      ← clamped to [q_min, q_max]
```

**Dequantization formula (the inverse):**

```
x_dequantized = S × (x_quantized − Z)
```

Where:

- **α, β:** the real min/max of the actual FP32 data
- **q_min, q_max:** the min/max of the target integer range (e.g., `-128` and `127` for signed INT8)
- **Scale (S):** a floating-point number representing the "step size" in the discrete space
- **Zero-point (Z):** an integer guaranteeing that real-world zero maps *exactly* onto an integer — critical for zero-padding and ReLU-like layers that must represent zero without error

### A visual intuition for bit reduction

Reducing color depth in an image is a useful analogy. Force an image down to 8 colors and file size drops dramatically — but zoom in and the edges look blocky and noisy. Quantization is the search for the mapping that minimizes that visible damage (model accuracy loss) for a given bit budget (color count).

| Format | Bits | Value range | π as represented |
|---|:---:|---|:---:|
| FP32 | 32 | up to `±3.4×10³⁸` | Full precision |
| FP16 | 16 | `[-65504, 65504]` | `3.140625` |
| BF16 | 16 | up to `±3.4×10³⁸` | Slightly less precise than FP16 |
| INT8 | 8 | `[-127, 127]` | `3` (all fractional detail lost) |

---

## 5. Symmetric vs. Asymmetric Quantization

### Asymmetric quantization

The full real-world range `[α, β]` maps onto the full target range (e.g., `[0, 255]` or `[-128, 127]`). Real-world zero doesn't necessarily land on integer zero — `Z ≠ 0`.

- ✅ **Advantage:** every bit of resolution (all 256 INT8 states) is spent on real data — ideal for skewed, one-sided distributions like ReLU outputs, which are always non-negative.
- ⚠️ **Cost:** the `Z` term introduces an extra offset in matrix multiplication that hardware must compensate for — slightly slower.

### Symmetric quantization

The range is assumed centered on zero: `[-max(|x|), +max(|x|)]`. As a result, `Z = 0` always, and the formula simplifies to:

```
x_quantized = round(x / S)
```

- ✅ **Advantage:** with `Z = 0`, no compensation term is needed in matrix multiplication — the fastest possible path on tensor cores.
- ⚠️ **Cost:** if the data is skewed (e.g., a range of 0 to 10), symmetric mapping wastes half the INT8 space (the entire negative half) and effective resolution collapses.

### Decision table

| Property | Asymmetric | Symmetric |
|---|---|---|
| Zero-point value | Variable (`Z ≠ 0`) | Always zero (`Z = 0`) |
| Bit-resolution usage | Maximal, even for skewed data | Can waste half the range on skewed data |
| Hardware speed | Slightly slower (offset overhead) | Fastest possible |
| Best suited for | Activations (typically skewed) | Weights (typically zero-centered, near-normal) |

### 5.1 Worked Example: Head-to-Head Comparison

Suppose an FP32 tensor has minimum `α = -2.0` and maximum `β = 4.0`. We want to quantize `x = 1.5` to signed INT8 (`q_min = -128`, `q_max = 127`).

#### Asymmetric method

```
S = (β − α) / (q_max − q_min) = (4.0 − (−2.0)) / (127 − (−128)) = 6.0 / 255 ≈ 0.02353

Z = round(q_min − α / S) = round(-128 − (−2.0 / 0.02353)) = round(-128 + 85) = -43

x_q = round(1.5 / 0.02353 + (-43)) = round(63.75 − 43) = round(20.75) = 21
```

📌 Result: `1.5` maps to integer **21**.

**Dequantization and error:**

```
x_dequant = S × (x_q − Z) = 0.02353 × (21 − (−43)) = 0.02353 × 64 ≈ 1.50588

error = |1.5 − 1.50588| = 0.00588
```

#### Symmetric method

```
max(|x|) = max(|-2.0|, |1.5|, |4.0|) = 4.0

S = max(|x|) / q_max = 4.0 / 127 ≈ 0.03150

x_q = round(1.5 / 0.03150) = round(47.62) = 48
```

📌 Result: `1.5` maps to integer **48**.

**Dequantization and error:**

```
x_dequant = S × x_q = 0.03150 × 48 ≈ 1.51181

error = |1.5 − 1.51181| = 0.01181
```

#### Analysis

| Method | Stored integer | Reconstructed value | Quantization error |
|---|:---:|:---:|:---:|
| Asymmetric | 21 | 1.50588 | **0.00588** |
| Symmetric | 48 | 1.51181 | **0.01181** (~2× larger) |

**Why the gap?** The data range `[-2.0, 4.0]` is skewed. The symmetric method was forced to virtually extend the range down to `-4.0` just to preserve the centered-on-zero property — and that wasted half the bit resolution on the (unused) negative side. Even so, many hardware accelerators accept this small accuracy hit in exchange for the raw matmul speed gain of dropping the `Z` term entirely.

---

## 6. Calibration Strategies

Model weights are fixed, so extracting their min/max is trivial. But **activations** change with every new input — so how do we pin down a `scale` factor for something that's constantly moving?

The answer: run a small, representative **calibration dataset** (a few hundred diverse samples) through the model to observe real activation behavior. Three main strategies exist for choosing the clipping threshold:

| Strategy | Mechanism | Sensitivity to outliers |
|---|---|---|
| **MinMax** | Simplest approach — takes the observed range at face value | Very high — a single outlier can wreck the resolution of the entire tensor |
| **MSE** | Chooses a clipping threshold that minimizes mean-squared error between the original and quantized tensor | Balanced — trims extreme outliers to protect the bulk of the distribution |
| **Entropy (KL-divergence)** | Picks the threshold that minimizes the KL-divergence between the original and quantized distributions (NVIDIA TensorRT's default) | Balanced, information-theoretically grounded |

---

## 7. Granularity: Per-Tensor, Per-Channel, Per-Group

Quantization can be applied at different granularities within a tensor:

- **Per-tensor (per-layer):** one `(S, Z)` pair for an entire weight matrix. Cheapest in memory, but causes noticeable accuracy loss in large models.
- **Per-channel (per-row):** a separate `(S, Z)` pair per output channel. The modern gold standard — it absorbs cross-channel distribution variance without much overhead.
- **Per-group:** weights are split into small blocks (e.g., 32 or 64 elements) with their own scale each. This is the backbone of aggressive INT4/INT3 formats like **GGUF** and **AWQ**.

```
Per-Tensor:   [■■■■■■■■■■■■■■■■]  ← one S for everything
Per-Channel:  [■■■■][■■■■][■■■■][■■■■]  ← one S per row/channel
Per-Group:    [■■][■■][■■][■■][■■][■■][■■][■■]  ← one S per small block
```

---

## 8. Weights vs. Activations

| | **Weights** | **Activations** |
|---|---|---|
| Nature | Long-term memory of the model; fixed after training | Transient forward-pass outputs; fully dynamic per input |
| Distribution | Usually near-normal, roughly zero-centered | Often skewed and asymmetric (especially post GeLU/SwiGLU) |
| Recommended scheme | Symmetric, per-channel | Asymmetric (to match the skew) |
| Special challenge | — | **Emergent outlier features** in large LLMs |

**A note on bias terms:** biases (`b`) make up a tiny fraction of total parameters but carry outsized mathematical sensitivity. This is why the overwhelming majority of frameworks never quantize biases — they're kept in FP16/FP32 to prevent linear drift across layers.

---

## 9. PTQ vs. QAT

### Post-Training Quantization (PTQ)

Converts an *already-trained* model's weights without any retraining. Fast, cheap, and by far the most common approach in practice.

- ✅ Fast and inexpensive
- ⚠️ Below 8 bits, accuracy can fall off a cliff — a phenomenon known as the **quantization cliff**.

### Quantization-Aware Training (QAT)

The model is exposed to quantization noise *during* training or fine-tuning. **Fake-quantization** layers quantize-then-dequantize values on the forward pass so downstream layers learn to compensate for the resulting noise.

Since `round()` isn't differentiable, backpropagation relies on an approximate gradient trick called the **Straight-Through Estimator (STE)**.

- ✅ Highest achievable accuracy, even below 3 bits
- ⚠️ Computational cost is close to a full training run — expensive

---

## 10. Static vs. Dynamic Quantization

### Dynamic quantization

Weights are converted to INT8 offline in advance, but activation scale factors are computed on the fly at runtime, per tensor or per token.

- ✅ High accuracy — it adapts to real, live data distributions
- ⚠️ Runtime overhead from computing scales on the fly; popular for memory-bound stacks like transformer decoder layers

### Static quantization

`S` and `Z` for *both* weights and activations are computed offline during calibration and baked directly into the model graph.

- ✅ Maximum possible throughput — zero runtime overhead for scale computation
- ⚠️ If production data drifts statistically from the calibration set, output quality degrades

---

## 11. The Outlier-Feature Problem in LLMs

A strange phenomenon appears in large language models (beyond roughly 6.7B parameters): certain activation channels produce values up to **100× larger** than the rest. These outliers carry critical semantic and syntactic information — clipping them damages model quality severely, but keeping them intact wrecks the resolution available to everything else in INT8.

Three key algorithms address this tension:

### 11.1 LLM.int8()

Monitors activations, identifies the specific channels containing outliers, and keeps *only those channels* in FP16 — the remaining 99% of the matrix is converted to INT8. The two partial results (INT8 + FP16) are summed coordinate-wise for the final output.

### 11.2 SmoothQuant

Key insight: quantizing weights is easy; quantizing activations (because of outliers) is hard. SmoothQuant applies a proportional scaling factor to shift the "difficulty" of quantization from activations onto weights:

```
Y = X · W = (X · diag(s)⁻¹) · (diag(s) · W)
```

With this trick, both sides can be uniformly and accurately quantized to INT8.

### 11.3 AWQ (Activation-aware Weight Quantization)

Key insight: not all weights matter equally. AWQ examines activation distributions to identify the specific weights connected to critical channels, and protects them from aggressive quantization (often less than 1% of weights are protected), minimizing final error at INT4.

### 11.4 GPTQ

Uses the inverse of the Hessian matrix to compensate for layer-wise error — as each weight is quantized, the resulting error is redistributed across the weights that haven't been quantized yet, minimizing cumulative error across the whole layer. One of the most widely used methods for reaching INT4 with minimal accuracy loss.

---

## 12. The 4-Bit Frontier and the GGUF Format

### The world of 4-bit

At 4 bits, each parameter can only take one of **16 distinct states** (`2⁴`). This is roughly the boundary between running massive models locally on consumer GPUs (like an RTX 4090) and needing expensive server-grade hardware.

Key innovations that made 4-bit practical:

- **GPTQ:** Hessian-based error compensation (see above)
- **AWQ:** protecting critical weights based on activation importance
- **NF4 (NormalFloat4)** in the **QLoRA** technique: distributes quantization levels according to the normal distribution of the weights (rather than uniformly), squeezing the maximum possible information out of 4 bits.

### GGUF (GPT-Generated Unified Format)

A general-purpose file format developed by the `llama.cpp` ecosystem. Unlike older formats, metadata, tokenizer settings, and quantized weights are all encapsulated inside **a single file**.

Its strategic advantage is an architecture built on direct memory mapping (`mmap`), which lets a user split large models block-by-block between system RAM and GPU VRAM (**VRAM offloading**). Quantization types like `Q4_K_M` or `Q8_0` combine small quantized blocks that strike an excellent balance between speed and accuracy.

---

## 13. The Final Frontier: 1.58-Bit Models and BitNet

### The Era of 1-Bit LLMs: BitNet

A fundamental redesign at the level of hardware linear algebra. Instead of training in floating-point and compressing afterward, the model is convinced from the very start of pre-training that it has no need for weights beyond the bare minimum. In BitNet, weights can only take the values `+1` or `-1`.

The result: the heavy matrix multiplication (`X × W`) that consumes the bulk of a GPU's power and transistor budget collapses into simple addition and subtraction — a multi-fold leap in hardware energy efficiency.

### All Large Language Models Are in 1.58 Bits

An evolved version of BitNet, known as **BitNet b1.58**. By adding a third state (the value `0`) to the binary system, each weight can now take one of three values: `{-1, 0, +1}`.

Mathematically, storing 3 distinct states requires `log₂(3) ≈ 1.58` bits of space — which is where the name comes from.

1.58-bit models match 1-bit systems in speed, energy consumption, and bandwidth, while matching the perplexity and reasoning ability of full-scale FP16 models.

**The magic of zero:** in a pure binary system, the model is forced to assign a weight to every single connection — it has no way to ignore anything. Introducing zero brings **structured sparsity** into the model for free: it can now cleanly filter out noisy or irrelevant channels (feature selection) without needing any extra transistors.

---

## 14. Summary and a Practical Decision Table

Quantization isn't just a file-size trick — it's a paradigm shift in AI engineering, proof that deep neural networks don't need infinite precision to function well. The future of edge deployment and local AI is entirely dependent on the continued evolution of these statistical optimization methods.

### 🧭 Quick decision guide

| Your scenario | Recommended approach |
|---|---|
| Running a model on a single consumer GPU (e.g., RTX 4090) | **GGUF (Q4_K_M)** or **AWQ** |
| Need maximum accuracy at very low bit-width and have compute budget | **QAT** |
| Just want to shrink a model fast, no retraining | **PTQ** with MSE/Entropy calibration |
| Activations have severe outliers (very large models) | **SmoothQuant** or **LLM.int8()** |
| Need maximum throughput in production with stable traffic | **Static quantization** |
| Inputs are highly variable and unpredictable | **Dynamic quantization** |
| Exploring the research frontier / future-proofing architecture | **BitNet / 1.58-bit** |

---

## 15. Further Reading

- 📰 [Maarten Grootendorst's newsletter](https://newsletter.maartengrootendorst.com/) — the original visual, animated treatment of quantization concepts
- 📄 LLM.int8(): *8-bit Matrix Multiplication for Transformers at Scale*
- 📄 SmoothQuant: *Accurate and Efficient Post-Training Quantization for LLMs*
- 📄 AWQ: *Activation-aware Weight Quantization for LLM Compression and Acceleration*
- 📄 GPTQ: *Accurate Post-Training Quantization for Generative Pre-trained Transformers*
- 📄 BitNet b1.58: *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*
- 🛠️ [llama.cpp](https://github.com/ggml-org/llama.cpp) — reference implementation of the GGUF format

---

<div align="center">

**If this guide was useful, ⭐ star the repo — Forks, Issues, and PRs are welcome.**

</div>
