---
date: '2025-08-20'
description: A dot-product structure for lengths and orthogonal projections, with one consistent complex convention.
id: Inner product space
modified: 2026-09-08 09:12:38 GMT-04:00
tags:
  - math
title: Inner product space
---

An inner product adds geometry to a vector space: length, orthogonality, and nearest-point projection.

## Definition (real/complex)

Let $V$ be a vector space over $\mathbb F\in\{\mathbb R,\mathbb C\}$. An inner product $\langle\cdot,\cdot\rangle:V\times V\to\mathbb F$ satisfies, for all vectors $x,y,z$ and scalars $a,b$,

$$
\begin{aligned}
\langle ax+by,z\rangle&=a\langle x,z\rangle+b\langle y,z\rangle,\\
\langle x,y\rangle&=\overline{\langle y,x\rangle},\\
\langle x,x\rangle&\ge 0,\qquad
\langle x,x\rangle=0\iff x=0.
\end{aligned}
$$

This note uses **linearity in the first argument**, so the second argument is conjugate-linear. For column vectors, the standard complex inner product is

$$
\langle x,y\rangle=y^\dagger x=\sum_i x_i\overline{y_i},
$$

where $\dagger$ denotes conjugate transpose. For example, $\langle i,1\rangle=i$ and $\langle1,i\rangle=-i$. A Hermitian positive-definite matrix $M$ gives the weighted inner product $\langle x,y\rangle_M=y^\dagger Mx$. Over the reals, this reduces to $x^{\mathsf T}My$. [Bindel's matrix-computations notes](https://www.cs.cornell.edu/courses/cs6210/2025fa/lec/2025-08-25.html) use the same convention.

## Induced norm and inequalities

The induced [[thoughts/norm|norm]] is $\lVert x\rVert=\sqrt{\langle x,x\rangle}$. Expanding a squared length gives

$$
\lVert x+y\rVert^2=\lVert x\rVert^2
+2\operatorname{Re}\langle x,y\rangle+\lVert y\rVert^2.
$$

[[thoughts/Cauchy-Schwarz]] gives $|\langle x,y\rangle|\le\lVert x\rVert\lVert y\rVert$, so the squared-length expansion implies the triangle inequality. If a positive-semidefinite form assigns length zero to a nonzero vector, it gives a seminorm; positive definiteness rules that out.

## Orthonormal bases & Gram–Schmidt

An orthonormal family satisfies $\langle q_i,q_j\rangle=\delta_{ij}$. Gram-Schmidt constructs one from a linearly independent list: subtract the projections onto the vectors already constructed, then normalize the residual. Applied to a finite [[thoughts/basis|basis]], it gives an orthonormal basis of the same space. The residual is nonzero because the next input vector lies outside the span of its predecessors. See [Axler, §6B](https://linear.axler.net/LADR4e.pdf#page=214).

## Orthogonality, projections, Pythagoras/Parseval

Vectors are orthogonal when $\langle x,y\rangle=0$. The squared-length expansion then gives Pythagoras, $\lVert x+y\rVert^2=\lVert x\rVert^2+\lVert y\rVert^2$.

For a finite-dimensional subspace $S$ with orthonormal basis $q_1,\ldots,q_k$, define

$$
P_Sx=\sum_{j=1}^k\langle x,q_j\rangle q_j.
$$

The residual $x-P_Sx$ is orthogonal to every vector in $S$. For any $s\in S$, Pythagoras therefore gives

$$
\lVert x-s\rVert^2
=\lVert x-P_Sx\rVert^2+\lVert P_Sx-s\rVert^2.
$$

The last term is minimized uniquely at $s=P_Sx$. This proves that the projection is the nearest point in $S$. In standard real or complex coordinates, $Q=[q_1\ \cdots\ q_k]$ gives $P_S=QQ^\dagger$. With the weighted inner product, $Q^\dagger MQ=I$ instead gives $P_S=QQ^\dagger M$. See [Axler, §6C, projection and minimization](https://linear.axler.net/LADR4e.pdf#page=229).

## Gram matrices and PSD

For vectors $x_1,\ldots,x_n$, define $G_{ij}=\langle x_j,x_i\rangle$. The index order follows our linear-first convention and makes

$$
c^\dagger Gc=\left\lVert\sum_j c_jx_j\right\rVert^2\ge0.
$$

Thus $G$ is Hermitian positive semidefinite, and it is positive definite exactly when the vectors are linearly independent. In standard coordinates, putting the vectors in the columns of $X$ gives $G=X^\dagger X$. [Bindel's Gram-matrix derivation](https://www.cs.cornell.edu/courses/cs6210/2025fa/lec/2025-08-25-slides.html#gram-matrices) keeps this index order explicit. These matrices appear in least squares and [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/principal component analysis|PCA]]; see also [[thoughts/Singular Value Decomposition]].

## Polarization identity

A norm comes from an inner product exactly when it satisfies the parallelogram law,

$$
\lVert x+y\rVert^2+\lVert x-y\rVert^2
=2\lVert x\rVert^2+2\lVert y\rVert^2.
$$

The inner product is then recovered from lengths. Over the reals,

$$
\langle x,y\rangle=\frac14\bigl(\lVert x+y\rVert^2-\lVert x-y\rVert^2\bigr).
$$

For our complex convention,

$$
\langle x,y\rangle=\frac14\Bigl(
\lVert x+y\rVert^2-\lVert x-y\rVert^2
+i\lVert x+iy\rVert^2-i\lVert x-iy\rVert^2\Bigr).
$$

## Relation to completeness (Hilbert spaces)

A Hilbert space is complete in its induced norm: every Cauchy sequence has a limit in the space.

For $a<b$, continuous functions on $[a,b]$ have inner product $\langle f,g\rangle=\int_a^b f(t)\overline{g(t)}\,dt$, yet they are incomplete in this norm. Their completion is $L^2([a,b])$, where functions equal almost everywhere are identified. Square-integrable random variables similarly use $\langle X,Y\rangle=\mathbb E[X\overline Y]$, modulo almost-sure equality.

In a Hilbert space, a pairwise orthogonal series converges precisely when $\sum_k\lVert u_k\rVert^2<\infty$. For its partial sums,

$$
\left\lVert\sum_{k=m+1}^n u_k\right\rVert^2
=\sum_{k=m+1}^n\lVert u_k\rVert^2.
$$

This turns the Cauchy condition for vectors into a condition on a scalar series. When it converges, the squared norm of the sum is the sum of squared norms.

For a countable orthonormal basis $(e_k)$ whose closed span is the whole space, this yields the expansion and Parseval identity

$$
x=\sum_k\langle x,e_k\rangle e_k,
\qquad
\lVert x\rVert^2=\sum_k|\langle x,e_k\rangle|^2.
$$

An arbitrary orthonormal family gives Bessel's inequality in place of equality. Projection onto a general subspace of a Hilbert space requires that subspace to be closed. [Axler, chapter 8](https://measure.axler.net/MIRA.pdf#page=234) develops completeness, projection, polarization, and orthogonal series.
