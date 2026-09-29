---
date: '2024-12-10'
description: a real-valued function is convex if its epigraph is a convex set
id: Convex function
modified: 2026-06-05 15:08:27 GMT-04:00
tags:
  - math
title: Convex function
---

A set $C$ is convex when the line segment between any two of its points stays inside $C$:

$$
x,y\in C,\quad t\in[0,1]
\quad\Longrightarrow\quad
(1-t)x+ty\in C.
$$

The segment includes both endpoints and can reduce to a single point when $x=y$. Convex sets can also be unbounded: a line, a ray and the whole space are examples.[^sets]

For a function, look at the line segment joining two points **on its graph**. Convexity requires the graph to lie on or below that segment between the two inputs.

> [!definition] formal definition
>
> Let $X$ be a convex subset of a real [[thoughts/Vector space|vector space]]. A function $f:X\to\mathbb{R}$ is **convex** if, for every $x,y\in X$ and $t\in[0,1]$,
>
> $$
> f((1-t)x+ty)\le(1-t)f(x)+tf(y).
> $$
>
> It is **strictly convex** when the inequality is strict for distinct $x,y$ and $0<t<1$. Reversing the inequality defines a concave function.

For $f(x)=x^2$, the graph lies below each chord. The same definition includes $f(x)=|x|$, which has a corner, and affine functions $f(x)=ax+b$, which satisfy equality everywhere. The cup-shaped picture only covers some examples. Equivalently, $f$ is convex exactly when its [[thoughts/epigraph|epigraph]] is a convex set.[^functions]

## derivatives and minima

For differentiable $f$ on an open convex domain in $\mathbb{R}^d$, convexity is equivalent to

$$
f(y)\ge f(x)+\nabla f(x)^\top(y-x)
\qquad\text{for all }x,y\text{ in the domain}.
$$

The tangent approximation gives a lower bound everywhere. If $\nabla f(x)=0$, this inequality becomes $f(y)\ge f(x)$, so $x$ is a global minimizer. More generally, every local minimum of a convex function is global. Convexity still allows several minimizers or none at all: a constant function attains its minimum everywhere; $e^x$ on $\mathbb{R}$ never reaches its infimum zero.[^functions]

## Jensen's inequality

The defining inequality extends from two inputs to a weighted average:

$$
f\left(\sum_{i=1}^{n}p_i x_i\right)
\le\sum_{i=1}^{n}p_i f(x_i),
\qquad p_i\ge0,\quad\sum_{i=1}^{n}p_i=1.
$$

> [!note] probability theory
>
> For an integrable real random variable $Z$ taking values in an interval $I$, and a convex function $\varphi:I\to\mathbb{R}$, Jensen's inequality gives
>
> $$
> \varphi(\mathbb{E}[Z])\le\mathbb{E}[\varphi(Z)],
> $$
>
> provided $\mathbb{E}[Z]\in I$ and $\mathbb{E}[|\varphi(Z)|]<\infty$.

For example, applying it to $\varphi(z)=z^2$ gives $(\mathbb{E}[Z])^2\le\mathbb{E}[Z^2]$ when the second moment is finite. The gap is the variance.[^jensen]

## convex hull

The **convex hull** of a nonempty set $\mathcal{X}\subseteq\mathbb{R}^d$ consists of all finite convex combinations of its points:

$$
\operatorname{conv}(\mathcal{X})
=\left\{\sum_{i=1}^{m}\theta_i x_i
\;\middle|\;
 m\ge1,\ x_i\in\mathcal{X},\ \theta_i\ge0,\ \sum_{i=1}^{m}\theta_i=1\right\}.
$$

Every convex set containing $\mathcal{X}$ must contain these combinations. They form a convex set themselves, so the hull is the smallest convex set containing $\mathcal{X}$. This also makes it the intersection of all convex sets containing $\mathcal{X}$.[^sets]

Three noncollinear points in the plane have a filled triangle as their hull, including its edges. Three collinear points produce the segment between the two extreme points.

## simplex

A $k$-simplex is the convex hull of $k+1$ affinely independent vertices $u_0,\ldots,u_k$. Affine independence means the vectors $u_1-u_0,\ldots,u_k-u_0$ are linearly independent; it ensures the hull has dimension $k$.

$$
C=\left\{\sum_{i=0}^{k}\theta_i u_i
\;\middle|\;\theta_i\ge0,\ \sum_{i=0}^{k}\theta_i=1\right\}.
$$

The first examples are a point, a line segment, a filled triangle, a tetrahedron and, in four dimensions, a 5-cell. Affine independence matters: the three collinear points above cannot define a $2$-simplex.

The probability simplex is a useful instance:

$$
\Delta^{n-1}=\left\{p\in\mathbb{R}^{n}\;\middle|\;p_i\ge0,\ \sum_{i=1}^{n}p_i=1\right\}.
$$

Its vertices are the standard basis vectors. Each point represents a distribution over $n$ outcomes; the sum constraint leaves $n-1$ dimensions.[^sets]

[^sets]: Stephen Boyd and Lieven Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), sections 2.1.4 and 2.2.4: convex combinations, convex hulls and simplices.

[^functions]: Boyd and Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), sections 3.1.1, 3.1.3 and 3.1.7: convexity, first-order conditions and epigraphs; section 4.2.2 on local and global optima.

[^jensen]: Boyd and Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), section 3.1.8.
