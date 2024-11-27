---
id: manifold
tags:
  - math
date: "2024-11-27"
modified: "2024-11-27"
title: manifold
---

a topological space that locally resembles Euclidean space near each point.

> an $n$-dimensional manifold is a topological space with the property that each point has a [[thoughts/manifold#neighborhood|neighbourhood]] that is [[thoughts/homeomorphism|homeomorphic]] to an open subset of $n$-dimensional Euclidean space.

Formally,a topological manifold is a ==second countable Hausdorff space== that is _locally homeomorphic_ to a Euclidean space.

> [!abstract] Locally homeomorphic to a Euclidean space
>
> every point has a neighborhood [[thoughts/homeomorphism|homeomorphic]] to an open subset of the Euclidean space $\mathbb{R}^n$ for some non-negative integer $n$

Implies that either the point is an isolated point $n=0$, or it has a neighborhood homeomorphic to the open ball:

$$
\mathbf{B}^n = \{(x_{1},x_{2},\ldots, x_n) \in \mathbb{R}^n : x_1^2 + x_2^2 + \ldots x_n^2 <1\}
$$

## differentiable manifold

_a topological manifold with a_ _==globally==_ defined differential structure.

### Pseudo-Riemannian manifold

abbrev: Lorentzian manifold

_with a metric tensor that is everywhere non-degenerate_

application used in general relativity is four-dimensional Lorentzian manifold for modeling space-time

![[thoughts/Tensor field#metric tensors]]

---

## neighborhood

think of open set or interior.

intuition: a set of point containing that point where one can move some amount in any direction away from that point without leaving the set.

> [!math] definition
>
> if $X$ is a topological space and $p$ is a point in $X$, then a **neighbourhood** of $p$ is a subset $V$ of $X$ that includes an ==open set== $U$ containing $p$:
>
> $$
> p \in U \subseteq V \subseteq X
> $$
>
> This is equivalent to the point $p \in X$ belonging to the topological interior of $V$ in $X$.

> [!important] properties
>
> the neighbourhood $V$ _need not be an open subset_ of $X$.
