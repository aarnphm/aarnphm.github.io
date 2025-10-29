---
date: "2025-08-28"
id: linear map
modified: 2025-10-29 02:15:49 GMT-04:00
tags:
  - seed
  - linalg
title: linear map
---

Linear maps preserve the algebra of vector spaces: addition and scalar multiplication. They are the core abstraction behind matrices, change of basis, projections, and differentiation (the derivative at a point is a linear map).

> [!abstract] definition
>
> Let $V,W$ be vector spaces over the same field $\mathbb F$.
> A map $T:V\to W$ is linear if for all $x,y\in V$ and $a,b\in\mathbb F$,
>
> $$
> T(ax+by)=a\,T(x)+b\,T(y).
> $$
>
> Equivalently, $T(0)=0$ and $T\big(\sum_i a_i x_i\big)=\sum_i a_i T(x_i)$.

See also [[thoughts/Vector space|vector spaces]] and [[thoughts/Inner product space|inner product spaces]].

## Basic examples

> [!example]
>
> - Identity: $I(x)=x$; Zero map: $0(x)=0$.
> - Coordinate projections $P_i:\mathbb F^n\to\mathbb F$, and block projections $P_S: \mathbb F^n\to\mathbb F^{|S|}$.
> - Differentiation on polynomials $D:\mathbb F[t]\to\mathbb F[t]$, $D(\sum a_k t^k)=\sum k a_k t^{k-1}$.
> - Matrix action: for $A\in\mathbb F^{m\times n}$, $T(x)=Ax$ is linear from $\mathbb F^n$ to $\mathbb F^m$.

## Kernel, image, rank

The kernel and image are subspaces:

$$
\ker T\coloneqq\{x\in V: T(x)=0\},\qquad
\operatorname{im} T\coloneqq\{T(x):x\in V\}.
$$

> [!math] Rank–Nullity Theorem (finite dimension)
>
> If $\dim V<\infty$, then
>
> $$
> \dim V
>
> = \underbrace{\dim \ker T}_{\text{nullity}(T)}
>
> + \underbrace{\dim\operatorname{im} T}_{\operatorname{rank}(T)}.
> $$
>
> In particular, $T$ is injective iff $\ker T=\{0\}$; $T$ is surjective iff $\operatorname{rank}(T)=\dim W$.

## Matrix representation and change of basis

Fix ordered bases $\mathcal B=(v_1,\dots,v_n)$ of $V$ and $\mathcal C=(w_1,\dots,w_m)$ of $W$.
There exists a unique matrix $[T]_{\mathcal B}^{\mathcal C}\in\mathbb F^{m\times n}$ such that for all $x\in V$,

$$
\big[ T(x) \big]_{\mathcal C}
 = [T]_{\mathcal B}^{\mathcal C} \,[x]_{\mathcal B}.
$$

If $\mathcal B'\!,\mathcal C'$ are new bases with change-of-basis matrices $P$ (domain) and $Q$ (codomain), then

$$
[T]_{\mathcal B'}^{\mathcal C'} = Q^{-1}\,[T]_{\mathcal B}^{\mathcal C}\,P.
$$

For endomorphisms ($T:V\to V$), this specializes to similarity: $[T]_{\mathcal B'}=S^{-1}[T]_{\mathcal B}S$.

> [!note] Composition
> For $S:U\to V$ and $T:V\to W$, $[T\circ S]=[T]\,[S]$ (with compatible bases), matching matrix multiplication.

## Invertibility

For $T:V\to W$ with $\dim V=\dim W=n<\infty$, the following are equivalent:

- $T$ is bijective (has a linear inverse $T^{-1}$).
- $[T]$ is an invertible $n\times n$ matrix; $\det[T]\ne0$.
- $\ker T=\{0\}$ and $\operatorname{im}T=W$.

## Operator norm and Lipschitzness

Assume norms $\|\cdot\|_V$ on $V$ and $\|\cdot\|_W$ on $W$ (see [[thoughts/norm]]). The operator norm of $T$ is

$$
\|T\|\;\coloneqq\;\sup_{x\ne0} \frac{\|T x\|_W}{\|x\|_V}

= \sup_{\|x\|_V=1} \|Tx\|_W.
$$

- Submultiplicativity: $\|T\circ S\|\le\|T\|\,\|S\|$.
- For matrices with vector $p$-norms, the induced matrix norms satisfy
  $\|Ax\|_q\le\|A\|_{q\leftarrow p}\,\|x\|_p$.
  Special cases: $\|A\|_1=\max_j\sum_i |a_{ij}|$ (max column sum),
  $\|A\|_\infty=\max_i\sum_j |a_{ij}|$ (max row sum), and
  $\|A\|_2$ equals the largest singular value (see [[thoughts/Singular Value Decomposition]]).

> [!math] Linear maps are Lipschitz (finite dimension)
>
> If $\dim V,\dim W<\infty$, then every linear $T:V\to W$ is globally Lipschitz:
>
> $$
> \|T x - T y\|_W \le L\,\|x-y\|_V\quad \text{for all }x,y,\;\text{with }L=\|T\|<\infty.
> $$
>
> Hence linear maps are continuous. This follows because all norms on a finite-dimensional space are equivalent, so $\|T\|$ is finite for any choice of norms.

> [!note] Boundedness vs continuity (general normed spaces)
> For linear maps between normed spaces, “bounded” $\iff$ “continuous” $\iff$ “Lipschitz with some constant”. In infinite dimensions not every linear map is bounded, but in linear algebra (finite dimension) this pathology does not occur.

### Computing a Lipschitz constant

Given a matrix $A$ representing $T$ and chosen vector norms, any induced matrix norm provides a Lipschitz constant: $L=\|A\|$. For the Euclidean norm, $L$ is the largest singular value of $A$.

## Projections and idempotents

A linear map $P:V\to V$ is a projection onto a subspace $S$ along a complement $S'$ if $P^2=P$, $\operatorname{im}P=S$, and $\ker P=S'$. Orthogonal projections require an inner product; see [[thoughts/Inner product space#Orthogonality, projections, Pythagoras/Parseval|orthogonal projection]].

## Affine vs linear

An affine map has the form $x\mapsto T x + b$ with $T$ linear and $b$ fixed. Linear maps are precisely the affine maps that fix the origin ($b=0$).

## See also

- [[thoughts/Vector space]] — bases, span, and dimension.
- [[thoughts/norm]] — norms and induced (operator/matrix) norms.
- [[thoughts/Inner product space]] — adjoint, orthogonality, and projections.
- [[thoughts/Singular Value Decomposition]] — spectral norm and low-rank structure of linear maps.
