---
date: '2021-10-10'
description: primal values, tangents and adjoints in automatic differentiation.
id: Autograd
modified: 2026-06-05 15:08:27 GMT-04:00
seealso:
  - '[[thoughts/XLA|XLA]]'
tags:
  - seed
  - ml
title: Autograd
---

Given

$$
f(x)=e^{2x}-x^3,
\qquad f'(x)=2e^{2x}-3x^2,
$$

writing down the derivative is manual differentiation. Symbolic differentiation constructs a derivative expression; finite differences estimate it from nearby function evaluations. Automatic differentiation applies the chain rule to the program's operations and evaluates the derivatives at the supplied input. Floating-point rounding still applies. [Baydin et al., sections 2–3](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf) explain these distinctions.

## forward mode

The **primal** is the ordinary value $v_i$. Its **tangent** $\dot v_i$ is the derivative along a chosen input direction. Forward mode carries the pair $(v_i,\dot v_i)$ through each operation.

Split the example into $a=2x$, $b=e^a$, $c=x^3$ and $f=b-c$. Seed $\dot x=1$:

$$
\dot a=2,\qquad
\dot b=b\dot a=2b,\qquad
\dot c=3x^2,\qquad
\dot f=2b-3x^2.
$$

At $x=0$, the primal output is $f(0)=1$ and the tangent output is $\dot f=2$. These are different quantities.

For $f:\mathbb{R}^n\to\mathbb{R}^m$, an input seed $v$ gives $J_f(x)v$. One direction gives one Jacobian-vector product. Recovering every column of a dense Jacobian requires a basis of input seeds.

## reverse mode

First evaluate the primals. Then propagate output sensitivity backward, retaining or recomputing the intermediate values needed by the derivative rules. For this scalar output, define the **adjoint** $\bar v_i=\partial f/\partial v_i$. Starting with $\bar f=1$ gives

$$
\bar b=1,\qquad \bar c=-1,\qquad \bar a=b,
\qquad \bar x=2\bar a+3x^2\bar c=2b-3x^2.
$$

Both uses of $x$ contribute to its adjoint. For a vector output, an output seed $w$ gives $J_f(x)^\mathsf{T}w$. A scalar loss needs only one output seed to obtain the full gradient over its inputs, which explains reverse mode's use in neural-network training.

[[thoughts/Jax|JAX]] exposes these operations through `jax.jvp` and `jax.vjp`; its [JVP/VJP guide](https://docs.jax.dev/en/latest/jacobian-vector-products.html) develops the seed interpretation. See [[thoughts/Automatic Differentiation]] for a multivariable example.
