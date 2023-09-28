---
title: "Autodiff and Jax"
date: 2022-08-21T21:47:43-07:00
tags:
  - seed
  - machinelearning
---

### Jax?

- Numpy with [[dump/Autograd]] builtin. Use [[dump/XLA]] to compile and run NumPy code on
  accelerators.
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
	return jnp.dot(a,b) + w
```
