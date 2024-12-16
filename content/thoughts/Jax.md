---
id: Jax
tags:
  - seed
  - ml
date: "2022-11-07"
modified: 2024-12-16 09:56:15 GMT-05:00
title: Jax
---

Numpy + [[thoughts/Autograd|Autograd]]. Use [[thoughts/XLA|XLA]] to compile and run NumPy code on accelerators.

Asynchronous dispatch, for sync use `block_until_ready()`

```python
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(0)
x = random.normal(key, (10,))
jnp.dot(x, x.T).block_until_ready()
```

- notable function:
  - `jit()` for compilation of multiple computations
  - `grad()` for performing transformation (autodiff, [[thoughts/Vector calculus#Jacobian matrix|Jacobian]]-vector product)
  - `vmap()` for auto-vectorisation

> Arrays are **immutable** in Jax

- Treat functions as pure as to compiled with [[thoughts/XLA]]

```python title="entropix/dslider.py"
from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

@jax.jit
def kl_divergence(logp: jnp.ndarray, logq: jnp.ndarray) -> jnp.ndarray:
  """Compute KL divergence between two log probability distributions."""
  p = jnp.exp(logp)
  return jnp.sum(jnp.where(p > 0, p * (logp - logq), 0.0), axis=-1)


@jax.jit
def ent_varent(logp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Compute entropy and varentropy from log probabilities."""
  p = jnp.exp(logp)
  ent = -jnp.sum(p * logp, axis=-1)
  diff = logp + ent[..., None]
  varent = jnp.sum(p * diff**2, axis=-1)
  return ent, varent


@jax.jit
def normalize_logits(logits: jnp.ndarray, noise_floor: float) -> jnp.ndarray:
  """Normalize logits to log probabilities with noise floor truncation."""
  shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
  normalized = shifted - jax.nn.logsumexp(shifted + EPS, axis=-1, keepdims=True)
  # noise floor calculated for bfloat16
  return jnp.where(normalized < noise_floor, jnp.log(EPS), normalized)
```

_references: [github](https://github.com/xjdr-alt/entropix/blob/main/entropix/dslider.py)_

## control flow

see also [link](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-jit)

The following works:

```python /jax.jit/
@jax.jit
def f(x):
  for i in range(3): x = 2 * x
  return x

print(f(3))

@jax.jit
def g(x):
  y = 0.
  for i in range(x.shape[0]): y = y + x[i]
  return y

print(g(jnp.array([1., 2., 3.])))
```

> [!warning]- doesn't work
>
> ```python {2-4}
> @jax.jit
> def fail(x):
>   if x < 3: return 3. * x ** 2
>   else    : return -4 * x
>
> fail(2)
> ```

Reasoning: `jit` traces code on `ShapedArray` abstraction, where each abstract value represents the set of all array values with a fixed shape and dtype

> [!important]+ type coercion tradeoff
>
> If we trace a Python function on a `ShapedArray((), jnp.float32)` that isn’t committed to a specific concrete value,
> when we hit a line like if `x < 3`, the expression x < 3 evaluates to an abstract `ShapedArray((), jnp.bool_)` that represents the set `{True, False}`.

Fix: you can use `static_argnums` to specify which argument should be treated as static

```python
@jit(static_argnums=(0,))
def f(x):
  if x < 3:
    return 3. * x ** 2
  else:
    return -4 * x
```

## buffers

> [!question] How does JAX handle memory buffers?

[fast replay buffers](https://github.com/instadeepai/flashbax)
