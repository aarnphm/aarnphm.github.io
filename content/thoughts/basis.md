---
date: '2025-08-20'
description: A linearly independent spanning set gives each vector unique coordinates.
id: basis
modified: 2026-09-08 09:12:38 GMT-04:00
tags:
  - math
title: basis
---

> [!abstract] definition
>
> A basis $B$ of a vector space $V$ over a field $F$ is a linearly independent set that spans $V$.
>
> - **Spanning:** every vector in $V$ is a finite linear combination of elements of $B$.
> - **Linear independence:** for distinct $b_1,\ldots,b_m\in B$, the equality $\sum_{i=1}^m c_i b_i=0$ implies $c_1=\cdots=c_m=0$.
>
> These conditions make the coefficients unique: subtracting two representations gives a linear combination equal to zero, so every coefficient difference vanishes.

For a finite ordered basis $B=(b_1,\ldots,b_n)$, the coefficients form the coordinate vector $[v]_B$. Their values depend on the basis. [Axler's basis notes](https://linear.axler.net/Bases.pdf) develop this equivalence between a basis and unique representation.

For example, in $\mathbb R^2$ take $b_1=(1,1)$ and $b_2=(1,-1)$. Solving for the coefficients gives

$$
(x,y)=\frac{x+y}{2}b_1+\frac{x-y}{2}b_2,
\qquad
[(3,1)]_B=\begin{pmatrix}2\\1\end{pmatrix}.
$$

The vector has standard coordinates $(3,1)$ and coordinates $(2,1)$ in this basis. A basis need not be orthogonal or normalized; those are extra conditions supplied by an [[thoughts/Inner product space|inner product]].

In an infinite-dimensional space, the finite-sum definition is called a _Hamel basis_. Infinite-series expansions require a notion of convergence, as in an orthonormal basis of a Hilbert space.
