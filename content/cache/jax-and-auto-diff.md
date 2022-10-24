---
title: "Autodiff and Jax"
date: 2022-08-21T21:47:43-07:00
tags:
  - seed
  - machinelearning
---

In this article, we will build a basic word search model using Flax and Jax. Throughout the building process, we will explore what autodiff is and how Jax enables researcher to
write better and more efficient ML application. Source code can be found [_here_](https://github.com/aarnphm/semantic-jax)

### What is Jax?

- Numpy with [[Autograd]] builtin. Use [[XLA]] to compile and run NumPy code on accelerators.
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

- Centred around the idea of pure function.