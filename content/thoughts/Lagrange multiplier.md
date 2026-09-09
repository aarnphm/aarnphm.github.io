---
date: '2025-08-20'
description: Finding stationary points under equality constraints by combining their normal vectors.
id: Lagrange multiplier
modified: 2026-09-08 09:12:38 GMT-04:00
tags:
  - math
  - optimization
title: Lagrange multiplier
---

At a constrained extremum, the objective can have a nonzero [[thoughts/Vector calculus#gradient|gradient]]. What must vanish is its derivative along every direction allowed by the constraints. Lagrange multipliers express that condition using the constraint gradients.

## stationary equations

Let $f:U\to\mathbb R$ and $g:U\to\mathbb R^m$ be continuously differentiable on an open set $U\subseteq\mathbb R^n$. To find extrema of $f$ subject to $g(x)=0$, define

$$
\mathcal L(x,\lambda)=f(x)+\lambda^{\mathsf T}g(x),
\qquad \lambda\in\mathbb R^m.
$$

If the constraint gradients are linearly independent at a local extremum $x^*$, there is a multiplier $\lambda^*$ satisfying

$$
\nabla_x\mathcal L(x^*,\lambda^*)
=\nabla f(x^*)+\sum_{j=1}^m\lambda_j^*\nabla g_j(x^*)=0,
\qquad g(x^*)=0.
$$

The independence assumption makes the feasible set a $C^1$ submanifold near $x^*$. Its tangent directions $v$ satisfy $Dg(x^*)v=0$. At an extremum, $\nabla f(x^*)^{\mathsf T}v=0$ for each such direction, so the objective gradient lies in the span of the constraint normals. This is the geometric argument in [MIT's constrained-optimization notes](https://math.mit.edu/~djk/18_022/chapter04/section03.html).

## example

Minimize $f(x,y)=x^2+y^2$ subject to $x+y=1$. The Lagrangian and stationary equations are

$$
\mathcal L=x^2+y^2+\lambda(x+y-1),
\qquad
2x+\lambda=0,\quad 2y+\lambda=0,\quad x+y=1.
$$

They give $x=y=1/2$ and $\lambda=-1$. Substituting the constraint into the objective verifies the minimum directly:

$$
f(x,1-x)=2\left(x-\frac12\right)^2+\frac12.
$$

## limits of the test

The multiplier equations give candidates. A stationary point still needs classification, and convexity is unnecessary for the first-order necessary condition.

Even in the example, the Lagrangian has a saddle point in the joint variables: along $(x,y,\lambda)=(1/2+t,1/2,-1-2t)$ its value decreases by $t^2$, while keeping $\lambda=-1$ makes it increase by $t^2$.

The constraint qualification also matters. Minimize $f(x)=x$ subject to $g(x)=x^2=0$. The only feasible point is $x=0$, yet $f'(0)+\lambda g'(0)=1$ for every $\lambda$. The extremum exists even though no multiplier solves the stationary equation. The full-rank hypothesis and this failure mode are covered by [MIT 14.102, Theorem 178](https://web.mit.edu/14.102/www/notes/lecturenotes1007.pdf#page=5).
