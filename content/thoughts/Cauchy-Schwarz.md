---
date: '2024-11-05'
description: useful for derive upper bounds, e.g when analysing the error or convergence rate of an algorithm
id: Cauchy-Schwarz
modified: 2025-10-29 02:15:17 GMT-04:00
tags:
  - math
title: Cauchy-Schwarz
---

> [!abstract] format
>
> for all vectors $v$ and $v$ of an inner product space, we have
>
> $$
> \mid \langle u, v \rangle \mid ^2 \le \langle u, u \rangle \dot \langle v, v \rangle
> $$

In context of Euclidean norm:

$$
\mid x^T y \mid  \le \|x\|_2 \|y\|_2
$$

## proof

_using Pythagorean theorem_

special case of $v=0$. Then $\|u\|\|v\| =0$,

=> if $u$ and $v$ are [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/tut/tut1#linear dependence of vectors|linearly dependent]]., then q.e.d

Assume that $v \neq 0$. Let $z \coloneqq u - \frac{\langle u, v \rangle}{\langle v, v \rangle} v$

It follows from linearity of inner product that

$$
\langle z,v \rangle = \langle u - \frac{\langle u,v \rangle}{\langle v, v \rangle} v,v \rangle = \langle u,v \rangle - \frac{\langle u,v \rangle}{\langle v,v \rangle}\langle v,v \rangle = 0
$$

Therefore $z$ is orthogonal to $v$ (or $z$ is the projection onto the plane orthogonal to $v$). We can then apply Pythagorean theorem for the following:

$$
u = \frac{\langle u,v \rangle}{\langle v,v \rangle} v + z
$$

which gives

$$
\begin{aligned}
\|u\|^{2} &= \mid \frac{\langle u,v \rangle}{\langle v,v \rangle} \mid^{2} \|v\|^{2} + \|z\|^2 \\
&=\frac{\mid \langle u,v \rangle \mid^{2}}{(\|v\|^2)^{2}} \|v\|^{2} + \|z\|^2 \\
&= \frac{\mid \langle u, v \rangle\mid^2}{\|v\|^{2} } + \|z\|^2 \ge \frac{\mid \langle u,v \rangle \mid^2}{\|v\|^{2} }\\
\end{aligned}
$$

Follows $\|z\|^{2}=0 \implies z=0$, which establishes
linear dependences between $u$ and $v$.

q.e.d
