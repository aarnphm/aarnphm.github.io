---
id: quantization
tags:
  - seed
  - ml
date: "2024-02-05"
title: Quantization
---

See also: [[thoughts/images/htn-openllm.pdf|this talk]] I gave at Hack the North 2023.

> reduce computational and memory costs of running inference with representing the weight and activations with low-precision data type

- `int16` - [[thoughts/quantization#`fp32` to `fp16`|half precision]]
- `bfloat16`
- `int8`

> [!note]
> This also applies to post-training quantization, where the methodology is applied after the model has been trained, instead of during load-time.

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

See also: [paper](https://arxiv.org/abs/1712.05877)

## quantization time

- Post-training **dynamic quantization**: range of each activation is computed on the fly at _runtime_
- Post-training **static quantization**: range of each activation is computed _offline_ before _runtime_
  - Observers are put on activations to collect their value
  - certain number of forward passes on calibration datasets
  - range of each computation are computed according to some _calibration technique_
- **Quantization aware training**: range of each activation is computed _during training_
  - `fake_quantize` operations are inserted in the computation graph
  - `fake_quantize` is a no-op during inference, but during training, it simulates the effect of quantization

## Methods and libraries

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and [GPTQ](https://arxiv.org/abs/2210.17323)
