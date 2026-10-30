---
date: '2022-11-07'
description: array computation with automatic differentiation, vectorisation and compilation of supported Python functions through XLA.
id: Jax
modified: 2026-10-24 01:39:45 GMT-04:00
tags:
  - ml
title: Jax
---

JAX supplies NumPy-style array operations and transformations of Python functions. `jax.numpy` is the array interface; [[thoughts/Autograd|automatic differentiation]] and compilation operate on functions built from supported operations.

- [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) traces a function and compiles its array computation through [[thoughts/XLA|XLA]]. Python side effects are outside the ordinary pure-function contract.
- [`jax.grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html) returns the gradient of a scalar-valued function with respect to selected arguments.
- [`jax.jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html) returns the function value and a [[thoughts/Vector calculus#Jacobian matrix|Jacobian]]-vector product. For $f:\mathbb{R}^n\to\mathbb{R}^m$, its tangent output is $J_f(x)v$.
- [`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html) maps a function over specified array axes, propagating that batch dimension through the operations.

> [!important] immutability
> Ordinary `jax.Array` values are immutable. An update such as `x.at[i].set(value)` returns a new array value. The compiler can reuse storage when that preserves the program's semantics.

For example, normalise finite logits along the last axis by subtracting `logsumexp`:

```python
import jax
import jax.numpy as jnp


@jax.jit
def normalize_logits(logits):
  return jax.nn.log_softmax(logits, axis=-1)


logp = normalize_logits(jnp.array([1.0, 2.0, 3.0]))
print(jnp.exp(logp).sum())  # approximately 1.0
```

This computes $\log p_i=\ell_i-\log\sum_j\exp(\ell_j)$, with a numerically stable implementation. Clipping individual log probabilities afterward changes their total probability mass, so a noise floor needs a separate normalization policy. Related implementation: [Entropix's `dslider.py`](https://github.com/xjdr-alt/entropix/blob/main/entropix/dslider.py).

## asynchronous dispatch

JAX can return an array handle before the device has finished computing its contents. Python can enqueue further work while the device runs. Reading the value on the host forces a wait; `.block_until_ready()` waits without requiring a host copy. See [asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html).

```python
x = jnp.arange(10, dtype=jnp.float32)
result = jnp.dot(x, x).block_until_ready()
print(result)  # 285.0
```

For a benchmark, warm up the compiled function first, then time a call that waits for its result. Otherwise compilation or dispatch can dominate the measured interval.

## control flow

During ordinary `jit` tracing, array arguments carry shape and dtype information, while their values remain unknown. Python can unroll `for i in range(3)` during tracing. A Python `if x < 3` needs the value of `x` to select a branch, so it fails when `x` is a dynamic scalar argument. See the [control-flow discussion](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-jit).

Use [`lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html) for a scalar predicate that must be evaluated at runtime:

```python
@jax.jit
def piecewise(x):
  return jax.lax.cond(
    x < 3,
    lambda value: 3.0 * value**2,
    lambda value: -4.0 * value,
    x,
  )


print(piecewise(2.0))  # 12.0
print(piecewise(4.0))  # -16.0
```

Both branches are traced and must return compatible structures, shapes and dtypes. Their Python side effects therefore cannot represent runtime branch execution.

Alternatively, mark a hashable Python argument as static when specialization on its value is intended:

```python
from functools import partial


@partial(jax.jit, static_argnums=(0,))
def static_piecewise(x):
  if x < 3:
    return 3.0 * x**2
  return -4.0 * x
```

Distinct static values enter the compilation cache key and can require separate compilations. This is useful for a few configuration choices; a frequently changing data value can cause repeated compilation. The [`jit` guide](https://docs.jax.dev/en/latest/jit-compilation.html) explains this tradeoff.

## buffers

> [!question] How does JAX handle memory buffers?

Array immutability describes the values visible to the program. It leaves the compiler free to reuse buffers internally. At a compiled-call boundary, [buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html) explicitly gives up an input buffer so that it can be reused for an output. A donated input must not be used afterward; suitable output shapes and dtypes are also needed for reuse.

[Flashbax](https://github.com/instadeepai/flashbax) supplies replay-buffer data structures for reinforcement learning, a separate concern from JAX's device-buffer ownership.
