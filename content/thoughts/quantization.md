---
date: '2024-02-05'
description: reducing neural network memory and compute through low-precision data types like int8 and fp16, using calibration and rounding schemes.
id: quantization
modified: 2026-06-05 15:08:06 GMT-04:00
tags:
  - seed
  - ml
title: Quantization
---

See also: [[thoughts/images/htn-openllm.pdf|this talk]] I gave at Hack the North 2023.

Quantization stores weights or activations using fewer representable values. This saves memory and data movement. Packed `int8` weights take one quarter of the space of `fp32` weights, before scales and other metadata. Inference speed also depends on the kernel, its accumulator type, and conversion overhead.

Keep the formats separate:

- `int8` and `int16` store integers. A scale maps them back to the values used by the model.
- [[thoughts/quantization#`fp32 -> fp16`|`fp16`]] and `bfloat16` are 16-bit floating-point formats. Each value has its own exponent.
- `fp8` names a family of floating-point formats with different ranges and precision.

Post-training quantization uses a trained model. Conversion can happen offline or while loading it; the name describes its relation to training.

![[thoughts/images/quantisation-format.webp|from baseten introduction into quantization format]]

## metrics for calibration

Calibration chooses the range represented by the quantizer. A wide range includes outliers and spreads the available levels further apart. A narrow range gives finer steps and clips more values. Min/max, percentiles, mean squared reconstruction error, and KL divergence make different choices about that tradeoff. Use representative inputs, then check the model's task metric. A good histogram match alone does not establish model accuracy. [TensorRT's calibration discussion](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)

### [[thoughts/Kullback-Leibler divergence|KL calibration]]

Collect an activation histogram, try candidate clipping thresholds, and compare each reference distribution $P$ with a quantized approximation $Q$ expanded onto the same bins:

$$
D_{\mathrm{KL}}(P\Vert Q)
= \sum_{j:P_j>0} P_j\log\frac{P_j}{Q_j}.
$$

Choose the candidate with the smallest divergence. Both histograms must be normalized. A bin with $P_j>0$ and $Q_j=0$ makes the divergence infinite, so zero-bin handling matters. The comparison is over activation values, even when the network itself produces something other than probabilities. [PyTorch's calibration discussion](https://pytorch.org/blog/quantization-in-practice/)

## `fp32 -> fp16`

First check the operation's input, output, and accumulator types. Hardware support varies: CPUs with [AVX-512 FP16](https://www.intel.com/content/www/us/en/content-details/669773/intel-avx-512-fp16-instruction-set-for-intel-xeon-processor-based-products-technology-guide.html) can do native half-precision arithmetic. Storing a tensor in `fp16` does not determine every intermediate's precision.

For IEEE binary16:

$$
\begin{aligned}
\text{smallest positive normal} &= 2^{-14} \approx 6.10\times10^{-5},\\
\text{smallest positive subnormal} &= 2^{-24} \approx 5.96\times10^{-8},\\
\text{largest finite value} &= (2-2^{-10})2^{15}=65504.
\end{aligned}
$$

The normal minimum is often mistaken for the smallest representable value. Subnormals fill part of that gap, with fewer significant bits. Some execution paths flush them to zero. [NVIDIA's half-precision format guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)

For a normalization denominator $\sqrt{\sigma^2+\epsilon}$, an $\epsilon=10^{-12}$ rounds to zero if converted directly to `fp16`. A zero variance would then leave a zero denominator. Whether this happens depends on the kernel's arithmetic: for example, [TensorRT normalization defaults to `fp32` accumulation](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html). Check the implementation before changing $\epsilon$. Overflow is another failure mode: a reduction can exceed $65504$ even when each input fits. [TensorRT accuracy considerations](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/accuracy-considerations.html)

## `fp32 -> int8`

An affine quantizer has a positive scale $S$, an integer zero point $Z$, and integer limits $q_{\min},q_{\max}$. Quantize and reconstruct with

$$
\begin{aligned}
x_q &= \operatorname{clip}\!\left(
  \operatorname{round}(x/S)+Z,\ q_{\min},\ q_{\max}
\right),\\
\hat{x} &= S(x_q-Z).
\end{aligned}
$$

$\hat{x}$ approximates $x$. The zero point makes zero exactly representable when $Z$ lies in the integer range. The integer limits belong to the chosen scheme; signed `int8` permits $[-128,127]$, while a symmetric scheme may use $[-127,127]$ with $Z=0$. [Jacob et al.](https://arxiv.org/abs/1712.05877)

For example, choose $S=1/127$, $Z=0$, and $[-127,127]$. Then

$$
x=0.3
\quad\longrightarrow\quad x_q=38
\quad\longrightarrow\quad \hat{x}=\frac{38}{127}\approx0.29921.
$$

Without clipping, nearest rounding gives $|x-\hat{x}|\leq S/2$. A value above $1$ clips to $1$ in this scheme, so that bound no longer applies. A smaller scale improves resolution while reducing the covered range.

For quantized weights and activations, the reconstructed dot product is

$$
\sum_i \hat{w}_i\hat{a}_i
= S_wS_a\sum_i(w_{q,i}-Z_w)(a_{q,i}-Z_a).
$$

Integer kernels commonly use `int32` accumulators for the sum, then rescale the result to the output format. Accumulator width still imposes a limit on the reduction length. The storage type alone does not specify the arithmetic. [Integer inference construction](https://arxiv.org/html/1712.05877v1#S2.SS4)

Per-channel or per-group scales let subsets with different ranges use different step sizes. They add scale metadata, and the kernel must support that layout. [PyTorch's granularity discussion](https://pytorch.org/blog/quantization-in-practice/)

## `mxfp4`

MXFP4 combines 32 four-bit E2M1 elements with one shared eight-bit E8M0 scale. E2M1 has one sign bit, two exponent bits, and one fraction bit. E8M0 encodes a power-of-two scale, with one reserved NaN encoding. The [OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) defines the block format.

For element $i$,

$$
\hat{V}_i=SP_i.
$$

The nonnegative E2M1 values are

$$
\{0,\tfrac12,1,\tfrac32,2,3,4,6\}.
$$

The scale moves those levels together. A large value in a block can therefore force smaller values onto coarse steps or down to zero. Including the shared scale, storage is

$$
\frac{32\cdot4+8}{32}=4.25\ \text{bits per element},
$$

before padding and layout overhead.

[Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537) gives an example conversion based on the largest input exponent. For a finite, nonzero block whose resulting scale is representable,

$$
\begin{aligned}
e_{\mathrm{shared}}
  &=\left\lfloor\log_2\max_i|V_i|\right\rfloor-2,\\
S&=2^{e_{\mathrm{shared}}},\\
P_i&=Q_{\mathrm{E2M1}}(V_i/S).
\end{aligned}
$$

Here $Q_{\mathrm{E2M1}}$ rounds to the element format and clamps finite overflow to $\pm6$. The exponent rule can still clip: a block maximum of $7$ gives $S=1$ and reconstructs as $6$. An all-zero block needs a separate case, such as $S=1$ and all $P_i=0$, because $\log_2 0$ is undefined. The paper presents a conversion recipe; the format allows other recipes.

MX formats apply to both training and inference. A format definition alone gives no guarantee that a particular model trains or retains its accuracy at four bits. The paper evaluates several MX precisions across different workloads.

![[thoughts/images/compute-flow-mxformat.webp]]

Stochastic rounding is a separate choice. Between adjacent levels, rounding probabilities can preserve the input in expectation, provided it is in range. Individual samples still lose information, and clipping still introduces bias. [Gupta et al.](https://arxiv.org/abs/1502.02551)

## quantization time

- **Post-training dynamic quantization** stores quantized weights and selects activation scales from the current inputs at runtime. This adds range-estimation and conversion work.
- **Post-training static quantization** fixes activation scales using representative calibration data before inference. Inputs outside that calibration range can clip. [PyTorch's workflow discussion](https://pytorch.org/blog/quantization-in-practice/)
- **Quantization-aware training** simulates rounding and clipping in the forward pass while keeping trainable weights in floating point. A straight-through gradient estimator usually substitutes for the derivative of rounding. After training, conversion replaces the simulation with the quantized inference operations. [PyTorch's QAT explanation](https://pytorch.org/blog/quantization-aware-training/)

## methods

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes) provides quantized operations and model-loading tools. Check the selected format and backend when using it.

### GPTQ

[GPTQ](https://arxiv.org/abs/2210.17323) quantizes weights after training. For a layer with weights $W$ and calibration inputs $X$, its reconstruction objective is

$$
\min_{\hat{W}}\ \|WX-\hat{W}X\|_F^2,
$$

with $\hat{W}$ constrained to the chosen quantized representation. The calibration inputs reveal which weight errors affect the layer outputs. GPTQ uses approximate second-order information to compensate the remaining weights as it quantizes. Activation precision is a separate choice; GPTQ's weight quantization alone does not make the whole model integer-only.

## floating point

For `fp32` and `fp16`, IEEE 754 binary formats split each scalar into a sign bit, a biased exponent, and a fraction. The fraction is the stored part of the significand.

| format     | total bits | sign bits | exponent bits | fraction bits | bias $B$ |
| ---------- | ---------- | --------- | ------------- | ------------- | -------- |
| `fp32`     | 32         | 1         | 8             | 23            | 127      |
| `fp16`     | 16         | 1         | 5             | 10            | 15       |
| `bfloat16` | 16         | 1         | 8             | 7             | 127      |
| `fp8 e5m2` | 8          | 1         | 5             | 2             | 15       |
| `fp8 e4m3` | 8          | 1         | 4             | 3             | 7        |

Given sign $s$, stored exponent $e$, fraction integer $f$, and fraction width $m$, a normal value decodes as

$$
x=(-1)^s\left(1+\frac{f}{2^m}\right)2^{e-B}.
$$

For subnormals, $e=0$, the implicit leading bit is zero, and

$$
x_{\mathrm{sub}}=(-1)^s\frac{f}{2^m}2^{1-B}.
$$

In `fp32` and `fp16`, an all-ones exponent denotes infinity when the fraction is zero and NaN when it is nonzero. An all-zero exponent and fraction give signed zero. IEEE 754 supports several rounding modes; round-to-nearest, ties-to-even is the default. Overflow handling depends on the rounding mode and conversion operation. [CUDA floating-point guide](https://docs.nvidia.com/cuda/floating-point/index.html)

The FP8 variants need their own special-value rules. E5M2 supports infinities and NaNs. The E4M3 format used here has finite values up to $448$ and NaNs, with no infinities; most all-ones exponent encodings remain finite. [Transformer Engine's FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-1.8/user-guide/examples/fp8_primer.html)

> [!example] decoding `0x40490fdb` (`fp32`)
>
> - bits: `0` | `10000000` | `10010010000111111011011`
> - $s=0$, $e=128$, $f=\texttt{0x490fdb}=4\,788\,187$
> - $x=\left(1+\frac{4\,788\,187}{2^{23}}\right)2^{128-127}\approx3.14159274$
>
> This is the nearest `fp32` value to $\pi$.

`bfloat16` keeps the exponent width of `fp32`, with fewer fraction bits. `fp16` spends more of its bits on the fraction and has a smaller exponent range. The choice depends on whether the computation needs more range or finer spacing. [TensorRT's precision comparison](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/accuracy-considerations.html)
