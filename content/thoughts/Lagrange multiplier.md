---
date: "2025-08-20"
description: finding local maxima/minima
id: Lagrange multiplier
modified: 2025-10-29 02:15:26 GMT-04:00
tags:
  - math
  - optimization
title: Lagrange multiplier
---

An [[thoughts/optimization]] method to finding local maxima and minima of a function subject to equation constraints.

You can think of it as one of the tool for [[thoughts/Convex function|convex]] optimization.

> [!note] intuition
>
> To essentially convert a constrained problem into a form such that the derivative test of an unconstrained problem can still be applied.

The relationship between [[thoughts/Vector calculus#gradient|gradient]] of a function and gradients of constraints would lead to a _Lagrangian function_. In general case, it is defined as:

$$
\mathcal{L}(x, \lambda) \equiv f(x) + \langle \lambda, g(x) \rangle
$$

for function $f, g$, and the notation $\langle \cdot, \cdot  \rangle$ denotes an [[thoughts/Inner product space|inner product]]. And the $\lambda$ is the **Lagrangian multiplier**

For simple case, where the inner product is defined as a _dot product_, we have the Lagrangian:

$$
\mathcal{L}(x, \lambda) \equiv f(x) + \lambda \cdot g(x)
$$
