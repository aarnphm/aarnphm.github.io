---
date: '2024-11-05'
description: useful for deriving upper bounds, e.g when analysing the error or convergence rate of an algorithm
id: Cauchy-Schwarz
modified: 2026-10-02 09:08:00 GMT-04:00
tags:
  - math
title: Cauchy-Schwarz
---

> [!abstract] format
>
> for all vectors $u$ and $v$ of an inner product space, we have
>
> $$
> \mid \langle u, v \rangle \mid ^2 \le \langle u, u \rangle \cdot \langle v, v \rangle
> $$
>
> Equality holds exactly when $u$ and $v$ are [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/tut/tut1#linear dependence of vectors|linearly dependent]].

For real vectors with the Euclidean norm:

$$
\mid x^T y \mid  \le \|x\|_2 \|y\|_2
$$

## proof

_using Pythagorean theorem_

Use the [[thoughts/Inner product space|inner product]] convention that is linear in the first argument and conjugate-linear in the second.

Special case: $v=0$. Both sides are zero, and the pair is linearly dependent.

Assume that $v \neq 0$. Subtract the component of $u$ along $v$:

$$
z \coloneqq u - \frac{\langle u, v \rangle}{\langle v, v \rangle} v.
$$

It follows from linearity of inner product that

$$
\langle z,v \rangle = \langle u - \frac{\langle u,v \rangle}{\langle v, v \rangle} v,v \rangle = \langle u,v \rangle - \frac{\langle u,v \rangle}{\langle v,v \rangle}\langle v,v \rangle = 0
$$

Thus $z$ is orthogonal to $v$. Apply Pythagoras to the decomposition

$$
u = \frac{\langle u,v \rangle}{\langle v,v \rangle} v + z
$$

which gives

$$
\begin{aligned}
\|u\|^{2} &= \left| \frac{\langle u,v \rangle}{\langle v,v \rangle} \right|^{2} \|v\|^{2} + \|z\|^2 \\
&= \frac{|\langle u, v \rangle|^2}{\|v\|^{2}} + \|z\|^2
\ge \frac{|\langle u,v \rangle|^2}{\|v\|^{2}}.
\end{aligned}
$$

Multiplying by $\|v\|^2$ gives the inequality. The gap is exactly $\|z\|^2\|v\|^2$, so equality holds if and only if $z=0$. By its definition, $z=0$ means that $u$ is a scalar multiple of $v$. Conversely, if $u=cv$, the residual $z$ vanishes.

q.e.d

See [Axler, §6A, Cauchy-Schwarz](https://linear.axler.net/LADR4e.pdf#page=203) for the same orthogonal-decomposition proof.
