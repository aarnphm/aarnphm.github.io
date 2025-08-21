---
id: norm
tags:
  - math
date: "2025-08-20"
modified: 2025-08-20 16:51:03 GMT-04:00
title: norm
---

A norm is function in real or complex vector space to non-negative numbers that behaves a certain way.

> [!abstract] definition
>
> Given a vector space $X$ over a subfield $F$ of a complex number $\mathcal{C}$, a _norm_ of $X$ is a real-valued function $f: X \to \mathbb{R}$ w/ the following properties:
>
> - sub-additivity/triangle inequality:
>   $p(x+y) \le  p(x) + p(y) \space \forall \space x,y \in X$ (1)
> - absolute homogeneity:
>   $p(sx) = |s|p(x) \space \forall \space x \in X$ and all scalars $s$ (2)
> - positive definiteness:
>   $\forall x \in X, \text{ if } p(x) = 0 \text{ then } x = 0$ [^alternative] (3)

[^alternative]: from (2) implies $p(0) = 0$, (3) can also be phrased as: "for every $x \in X, p(x) = 0 \iff x=0$"

A norm $p : X \to \mathbb{R}$ on a given vector space $X$, then the norm of a vector $z \in X$ is denoted as $\|z\|=p(z)$

> [!important]
>
> Every vector space admits a norm: $x_{\cdots} = (x_i)_{i \in I}$ is a Hamel [[thoughts/basis]]

## euclidean norm
