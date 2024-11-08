---
id: Jax
tags:
  - seed
  - ml
date: "2022-11-07"
modified: "2024-11-07"
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
  - `grad()` for performing transformation (autodiff, Jacobian-vector product)
  - `vmap()` for auto-vectorisation

> Arrays are **immutable** in Jax

- Treat functions as pure as to compiled with [[thoughts/XLA]]

```python
import jax.numpy as jnp

from jax import jit


@jit
def diff(a, b, w=0):
    return jnp.dot(a, b) + w
```

## control flow

see also [link](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-jit)

The following works:

```python
@jit
def f(x):
  for i in range(3): x = 2 * x
  return x

print(f(3))

@jit
def g(x):
  y = 0.
  for i in range(x.shape[0]): y = y + x[i]
  return y

print(g(jnp.array([1., 2., 3.])))
```

> [!warning]- doesn't work
>
> ```python
> @jit
> def fail(x):
>   if x < 3: return 3. * x ** 2
>   else:       return -4 * x
>
> # This will fail!
> fail(2)
> ```

Reasoning: `jit` traces code on `ShapedArray` abstraction, where each abstract value represents the set of all array values with a fixed shape and dtype

> [!important]- type coercion tradeoff
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
