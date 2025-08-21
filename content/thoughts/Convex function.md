---
id: Convex function
tags:
  - math
description: a real-valued function is convex if its epigraph is a convex set
date: "2024-12-10"
modified: 2025-08-20 12:39:25 GMT-04:00
title: Convex function
---

> a convex set is essentially a set that intersects every line in a line segment [^lineseg]

[^lineseg]:
    fancy math names for a part of a straight line that is bounded by two _distinct_ end points.
    This is a special case of an _arc_ with zero curvature. The length is given by the Euclidean distance between its endpoints.

    If $V$ is a [[thoughts/Vector space|vector space]] over $\mathbb{R}$ or $\mathbb{C}$, and $L$ is a subset of $V$, then $L$ is a line segment if $L$ can be parameterised:

    $$
    L = \{\mathbf{u} + t\mathbf{v} \mid t \in [0,1]\}
    $$

    for some vectors $\mathbf{u}, \mathbf{v} \in V$ where $\mathbf{v}$ is nonzero.

ELI5: a convex function graph is shaped like a cup $\cup$, where as a concave function graph is shaped like a cap $\cap$

> [!math] formal definition
>
> Let $X$ be a convex subset of a real [[thoughts/Vector space|vector space]] and let $f: X \to \mathbb{R}$ be a function.
>
> Then $f$ is called **convex** iff any of the following equivalent holds:
>
> 1. $\forall 0 \le t \le 1 \cap x_{1}, x_{2} \in X \mid f(tx_{1}+(1-t)x_{2}) \le tf(x_{1}) + (1-t)f(x_{2})$
> 2. $\forall 0 \le t \le 1 \cap x_{1}, x_{2} \in X \text{ where } x_{1} \neq x_{2} \mid f(tx_{1}+(1-t)x_{2}) \le tf(x_{1}) + (1-t)f(x_{2})$

This is also known as the _Jensen's inequality_:

$$
f(tx_{1} + (1-t)x_{2}) \le tf(x_{1}) + (1-t)f(x_{2})
$$

> [!note] probability theory
>
> The form as follows:
>
> > Given $X$ is a random variable with $\varphi$ a convex function, then
> >
> > $$
> > \varphi(E[X]) \le E[\varphi{(X)}]
> > $$

## convex hull

Intersection of all convex set containing a given subset of an Euclidean space, or equivalently as the set of all convex combinations of points in a subset.

> [!abstract] definition
>
> A set of point in a Euclidean space is considered convex if it contains the line segments connecting each pair of its point. Therefore, the _convex hull_ of a given set $\mathcal{X}$ is given by:
>
> - The (unique) minimal convex set containing $\mathcal{X}$
> - The intersection of all convex set containing $\mathcal{X}$
> - The set of all convex combinations of points in $\mathcal{X}$
> - The union of all [[thoughts/Convex function#simplex|simplices]] with vertices in $\mathcal{X}$

## simplex

a generalization of the notion of a triangle or tetrahedron to arbitrary dimensions. The simplex is so-named because it represents the simplest possible polytope[^2] in any given dimension. For example:

- a 0-dimensional simplex is a point,
- a 1-dimensional simplex is a line segment,
- a 2-dimensional simplex is a triangle,
- a 3-dimensional simplex is a tetrahedron, and
- a 4-dimensional simplex is a 5-cell.

Specifically, a k-simplex is a k-dimensional polytope that is the convex hull of its $k+1$ vertices.
Formally, suppose the $k+1$ points $u_0, \ldots, u_k$ are _affinely independent_, which means the $k$ vectors $u_1-u_0, \ldots, u_k - u_0$ are _linearly independent_.
Then simplex determined by them is a set of points:

$$
C=\{\theta_0 u_0 + \ldots \theta_k u_k \mid \sum_{i=0}^{k} \theta_i = 1 \text{ and } \theta_i \ge 0 \text{ for } i=0,\ldots,k\}
$$

[^2]: a polytope is a geometric object with flat sides (faces). Polytopes are the generalization of three-dimensional polyhedra to any number of dimensions.
