---
id: Jax and autograd
tags:
  - seed
  - ml
title: autograd and Jax
date: 2022-11-07
---

### Jax

- Numpy + [[dump/Autograd|Autograd]]. Use [[dump/XLA|XLA]] to compile and run NumPy code on accelerators.
- Asynchronous dispatch, for sync use `block_until_ready()`

```python
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(0)
x = random.normal(key, (10,))
jnp.dot(x, x.T).block_until_ready()
```

- notable function:
  - `jit()` for compilation of multiple computations
  - `grad()` for performing transformation (autodiff, Jacobian-vector product)
  - `vmap()` for auto-vectorisation

> Arrays are **immutable** in Jax

- Treat functions as pure as to compiled with [[XLA]]

```python
import jax.numpy as jnp

from jax import jit


@jit
def diff(a, b, w=0):
    return jnp.dot(a, b) + w
```
