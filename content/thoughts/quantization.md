---
id: quantization
tags:
  - seed
  - ml
date: "2024-02-05"
modified: 2025-08-10 01:07:13 GMT-04:00
title: Quantization
---

See also: [[thoughts/images/htn-openllm.pdf|this talk]] I gave at Hack the North 2023.

> reduce computational and memory costs of running inference with representing the weight and activations with low-precision data type

- `int16` - [[thoughts/quantization#`fp32` to `fp16`|half precision]]
- `bfloat16`
- `int8`
- `fp8`

> [!note]
> This also applies to post-training quantization, where the methodology is applied after the model has been trained, instead of during load-time.

![[thoughts/images/quantisation-format.webp|from baseten introduction into quantization format]]

## metrics for calibration

the idea is to compare the difference between two probability distribution when scaling, for example from `int16` to `int8`

### [[thoughts/Kullback-Leibler divergence|KL calibration]]

## `fp32` to `fp16`

> Does my operation support `fp16`?

- CPU does support saving `fp16` weights, but computations are done in `fp32`

> Does my operation _sensitive_ to `fp16`?

For example `epsilon` in `LayerNormalization` usually is very small $1e^{-12}$, but smallest value in `fp16` is $\approx 6e^{-5}$, which cause `NaN` issues.

## `fp32` to `int8`

Consider a float `x` in `[a, b]`, such that _affine quantization scheme_:

$$
x = S \cdot (x_q - Z)
$$

where:

- $x_q$ is the quantized `int8` associated with `x`
- $S$ and $Z$ are scaling and zero-point parameters
  - $S$ is the scale, positive `float32`
  - $Z$ is the zero-point, or the `int8` value corresponding to value `0` in `fp32`

Thus quantized value $x_q$ is: $x_q = \text{round}(x / S + Z)$

And `fp32` value outside of `[a, b]` is clipped to closest representable value.

$$
\forall x \in [a, b] \quad x_q = \text{clip}(\text{round}(x/S + Z), \text{round}(a/S + Z), \text{round}(b/S + Z))
$$

https://arxiv.org/abs/1712.05877

## mxfp4

see also: [specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

stands for _microscaling (mx) of 4-bit floating-point (fp4)_

https://arxiv.org/pdf/2310.10537, first proposed in Open Compute Project (OCP), backed by OpenAI, AMD, NVIDIA, Microsoft, Meta.

Developed for training, given that FP4 is "good enough" in inference.

- E2M1: 1 sign bit, 2 exponent bit, and 1 mantissa bit per parameter.
- Block: divided into 32 block_size
- Use a common 8-bit shared scale, best fit all values in a block.
- The value is decoded as:
  $$
  X_i = P_i \times S
  $$
  where $X_i$ is the reconstructed value, $P_i$ is the FP4 quantized value, and $S$ denotes the shared scale.

To preserve gradient integrity:

- Stochastic Rounding: Randomizes rounding direction, ensuring no systematic loss of information during training updates prevents bias and preserves learning progress.
- Random Hadamard Transform
- Group-wise Quantization

```pseudo
\begin{algorithm}
\caption{Convert vector of scalar floats $\{V_i\}_{i=1}^k$ to an MX block $\{X,\{P_i\}_{i=1}^k\}$}
\begin{algorithmic}
\Require $e^{\max}_{\text{elem}}$ \Comment{exponent of the largest normal number in the element data format}
\State $\text{shared\_exp} \gets \left\lfloor \log_2\!\left(\max_i |V_i|\right) \right\rfloor - e^{\max}_{\text{elem}}$
\State $X \gets 2^{\text{shared\_exp}}$
\For{$i = 1$ \textbf{to} $k$}
    \State $P_i \gets \text{quantize\_to\_element\_format}\!\left(\frac{V_i}{X}\right)$ \Comment{clamp to normal-number range}
\EndFor
\State \textbf{return} $X,\ \{P_i\}_{i=1}^{k}$
\end{algorithmic}
\end{algorithm}
```

![[thoughts/images/compute-flow-mxformat.webp]]

## quantization time

- Post-training **dynamic quantization**: range of each activation is computed on the fly at _runtime_
- Post-training **static quantization**: range of each activation is computed _offline_ before _runtime_
  - Observers are put on activations to collect their value
  - certain number of forward passes on calibration datasets
  - range of each computation are computed according to some _calibration technique_
- **Quantization aware training**: range of each activation is computed _during training_
  - `fake_quantize` operations are inserted in the computation graph
  - `fake_quantize` is a no-op during inference, but during training, it simulates the effect of quantization

## methods

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 

### GPTQ

https://arxiv.org/abs/2210.17323
