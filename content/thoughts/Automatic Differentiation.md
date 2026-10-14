---
date: '2024-02-07'
description: evaluating derivatives of a program by applying the chain rule to its operations.
id: Automatic Differentiation
modified: 2026-06-05 15:08:20 GMT-04:00
seealso:
  - '[[thoughts/Autograd|Autograd]]'
  - '[[thoughts/Jax|JAX]]'
tags:
  - math
title: Automatic Differentiation
---

Automatic differentiation (AD) evaluates derivatives of a function expressed as a program. Each operation has a local derivative rule. AD combines these rules along the computation, using the intermediate values from the program's execution.

For a chain of differentiable functions, write

$$
x_0 = x, \qquad x_i = f_i(x_{i-1}), \qquad
f(x) = (f_n \circ \cdots \circ f_1)(x).
$$

The chain rule gives

$$
J_f(x) = J_{f_n}(x_{n-1})\cdots J_{f_1}(x_0),
$$

where each $J$ is a Jacobian, the matrix of local partial derivatives. A program can also reuse intermediate values across several operations. Its computation then forms a graph, and contributions from those uses must be added when differentiating backward. [PyTorch's autograd documentation](https://docs.pytorch.org/docs/2.14/notes/autograd.html) describes this graph representation and the values saved for the backward pass.

## forward and reverse mode

Consider a small computation:

$$
a = xy, \qquad b = \sin x, \qquad f = a+b.
$$

**Forward mode** carries a derivative alongside each value. To differentiate with respect to $x$, seed $\dot{x}=1$ and $\dot{y}=0$, then follow the original evaluation order:

$$
\dot{a}=y\dot{x}+x\dot{y}=y, \qquad
\dot{b}=\cos x\,\dot{x}=\cos x, \qquad
\dot{f}=\dot{a}+\dot{b}=y+\cos x.
$$

For a vector input, a seed direction $v$ produces the Jacobian-vector product $J_f(x)v$.

**Reverse mode** first evaluates the function, then works backward from the output. Write $\bar{a}=\partial f/\partial a$, and likewise for the other variables. Seed $\bar{f}=1$. Addition sends this derivative to both inputs, giving $\bar{a}=\bar{b}=1$. The two uses of $x$ contribute separately:

$$
\bar{x}=\bar{a}y+\bar{b}\cos x=y+\cos x,
\qquad
\bar{y}=\bar{a}x=x.
$$

Both partial derivatives come from the same reverse pass. For a vector output, an output seed $w$ produces $J_f(x)^\mathsf{T}w$, a vector-Jacobian product written with column vectors.

This explains why reverse mode is useful for a scalar training loss with many parameters: one backward pass obtains the whole gradient. Forward mode obtains one input direction per pass. Reverse mode needs intermediate values from the forward evaluation, which must be stored or recomputed. [[thoughts/Jax|JAX's]] [JVP/VJP guide](https://docs.jax.dev/en/latest/jacobian-vector-products.html) develops this tradeoff.

Backpropagation applies this reverse-mode calculation to a neural network's loss.

AD applies derivative rules at the evaluated values. Floating-point rounding still applies, and an operation needs a derivative rule that the system supports. At a nondifferentiable point, such as the corner of a rectifier, a framework may use a specified convention; the returned number does not make the mathematical derivative exist.
