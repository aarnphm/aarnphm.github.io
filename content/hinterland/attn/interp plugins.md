---
date: '2026-05-26'
description: activation capture and steering constraints for vLLM
id: interp plugins
modified: 2026-06-05 15:08:10 GMT-04:00
tags:
  - ml
  - alignment
title: interp plugins
---

vLLM loads `vllm.general_plugins` through Python entry points. General plugins mainly register custom models through `ModelRegistry`. SAE support therefore belongs in a custom model class with explicit activation sites. The public engine should receive an ordinary model config. See the [vLLM plugin contract](https://github.com/vllm-project/vllm/blob/main/docs/design/plugin_system.md).

## registration

The registration function must be safe to run more than once because vLLM can call it several times in one process.

```toml title="pyproject.toml"
[project.entry-points."vllm.general_plugins"]
register_sae_model = "vllm_sae_plugin:register"
```

```python title="vllm_sae_plugin/__init__.py"
def register() -> None:
  from vllm import ModelRegistry

  if 'SAECausalLM' not in ModelRegistry.get_supported_archs():
    ModelRegistry.register_model(
      'SAECausalLM', 'vllm_sae_plugin.model:SAECausalLM'
    )
```

The model implementation reads the SAE configuration from the model config whose architecture resolves to `SAECausalLM`.

## activation contract

Each checkpoint names one layer and one site:

```python
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TapSpec:
  layer: int
  kind: Literal['attn', 'mlp']
```

The custom model places each attention tap after the output projection and before the residual addition. It places each MLP tap after the down projection and before its residual addition. The checkpoint must be trained on that same tensor. Module names are too weak for this contract because vLLM models can use different layer layouts and output shapes.

At one tap, the data path is

$$
\hat{x}=D\bigl(E(x)+\Delta z\bigr),
$$

Here $E$ encodes the activation, $\Delta z$ applies steering, and $D$ reconstructs the hidden state. Capture mode returns $x$. Reconstruction mode returns $\hat{x}$.

## request state

Storage for sparse features needs explicit request ownership.

- Each live request owns a unique slot and a current length.
- Cache tensors use a slot axis, such as `[max_num_seqs, max_model_len, k]`.
- Scheduler compaction updates the request-to-slot map before the next model step.
- Request release clears the slot and its length.
- Prefix sharing uses explicit block ownership. A position number alone cannot identify cached features across requests.

This prevents two new request IDs from writing to the same leading slice, which the old prototype allowed.

## kernels

Start with a plain PyTorch path: encode, choose the top $k$ features, steer, and decode. Profile that path before adding a fused kernel.

vLLM's [`CustomOp.register_oot`](https://docs.vllm.ai/en/stable/design/custom_op/) replaces an existing vLLM `CustomOp` class. It only affects a class that vLLM already instantiates. If `SAECausalLM` needs a fused SAE path, call that path from the tap code and keep the PyTorch path as the correctness reference.

Set the overhead target to $<5\%$ for median latency and tail latency under the stated batch and sequence length distribution.

## acceptance

- With every tap disabled, logits match the unmodified model at the configured tolerance.
- Capture-only mode leaves logits unchanged.
- Reconstruction and steering preserve shape, dtype, device, and tensor parallel layout.
- Two concurrent requests cannot read or overwrite each other's sparse features.
- Eager mode, compiled mode, and CUDA graphs produce the same outputs.
- Request completion releases all plugin state.
- Metrics report reconstruction error, feature activity, and latency without synchronizing the decode loop on every token.

## deferred

- transcoders across layers
- Matryoshka sparsity selection
- feature blocks for shared prefixes
- online SAE training
- drift aggregation across requests
