---
id: Convex function
tags:
  - math
date: "2024-12-10"
description: a real-valued function is convex if its epigraph is a convex set
modified: 2024-12-10 23:37:38 GMT-05:00
title: Convex function
---

> a convex set is essentially a set that intersects every line in a line segment [^lineseg]

[^lineseg]:
    fancy math names for a part of a straight line that is bounded by two _distinct_ end points.description
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
> >   \varphi(E[X]) \le E[\varphi{(X)}]
> > $$
