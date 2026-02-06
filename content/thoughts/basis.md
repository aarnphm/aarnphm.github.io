---
date: '2025-08-20'
description: a set of vectors as a linear combination of elements of B.
id: basis
modified: 2025-10-29 02:15:41 GMT-04:00
tags:
  - math
title: basis
---

> a set $B$ of elements of a vector space $V$ is called a _basis_ (pl.: bases) if every elements of $V$ can be written in a unique way as a finite linear combination of elements of $B$

Coefficient of these combination is referred as _components_ or _coordinates_ of the vector with respect to $B$.

> [!important]
>
> A set $B$ is a _basis_ if its elements are linearly independent and every element of $V$ is a linear combination of elements of $B$. Or, a basis is a _linearly independent spanning set_.

> [!abstract] definition
>
> A _basis_ $B$ of a vector space $V$ over a field $F$ is a linearly independent subset of $V$ that spans [^terminology] $V$
>
> This means it satisfies the following conditions:
>
> - linear independence:
>   for every finite subset $\{\mathbf{v}_{1},\ldots,\mathbf{v}_{m}\}$ of $B$, if $\sum_{i=1}^{m} c_i \mathbf{v}_i = 0$ for some $c_{1},\ldots,c_{m}$ in $F$, then $c_{1}=\ldots =c_m=0$
> - spanning property:
>   for every vector $\mathbf{v}$ in $V$, one can choose $a_{1},\ldots,a_n$ in $F$ and $\mathbf{v}_1,\ldots,\mathbf{v}_n$ in $B$ such that $\mathbf{v} = \sum_{i=1}^{n} a_i \mathbf{v}_n$

[^terminology]: We refer to linear span of a vector.
