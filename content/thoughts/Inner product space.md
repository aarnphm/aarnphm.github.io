---
date: '2025-08-20'
description: Vector spaces with a generalized dot product that induces length, angles, orthogonality, and projections; the gateway from Euclidean geometry to Hilbert spaces.
id: Inner product space
modified: 2025-10-29 02:15:25 GMT-04:00
tags:
  - math
title: Inner product space
---

Inner products formalize dot products, letting us measure lengths, angles, and orthogonality in arbitrary vector spaces.

## Definition (real/complex)

A (real/complex) vector space $V$ with a map $\langle\cdot,\cdot\rangle:V\times V\to \mathbb F$ is an **inner product space** if for all $x,y,z\in V$ and $a,b\in\mathbb F$:

- **Conjugate symmetry:** $\langle x,y\rangle=\overline{\langle y,x\rangle}$.
- **Linearity in the first argument:** $\langle ax+by,z\rangle=a\langle x,z\rangle+b\langle y,z\rangle$ (hence, conjugate‑linearity in the second).
- **Positive‑definite:** $\langle x,x\rangle>0$ for all $x\neq 0$.

> [!note] Convention
>
> Many physics texts take linearity in the second argument. Results are equivalent after complex conjugation.

## Induced norm and core inequalities

Define $\|x\|:=\sqrt{\langle x,x\rangle}$. Then:

- **Absolute homogeneity:** $\|a x\|=|a|\,\|x\|$.
- **Triangle inequality:** $\|x+y\|\le \|x\|+\|y\|$ (via Cauchy–Schwarz).
- **Cauchy–Schwarz:** $|\langle x,y\rangle|\le \|x\|\,\|y\|$, with equality iff $x,y$ are linearly dependent. See [[thoughts/Cauchy-Schwarz]].
- **Parallelogram law:** $\|x+y\|^{2}+\|x-y\|^{2}=2\|x\|^{2}+2\|y\|^{2}$.

> [!important] Positive‑definiteness matters
>
> Without $\langle x,x\rangle>0$ (for $x\ne0$), $\sqrt{\langle x,x\rangle}$ is not a norm and geometry breaks.

## Polarization identity

A norm comes from an inner product **iff** it satisfies the parallelogram law (Jordan–von Neumann). Then the inner product is uniquely recovered by polarization:

- **Real spaces:** $\displaystyle \langle x,y\rangle=\tfrac14\big(\|x+y\|^{2}-\|x-y\|^{2}\big)$.
- **Complex spaces (math convention, linear in the first argument):**

  $$
  \langle x,y\rangle=\tfrac14\Big(\|x+y\|^{2}-\|x-y\|^{2}+i\|x+iy\|^{2}-i\|x-iy\|^{2}\Big).
  $$

> [!tip] Intuition
>
> Polarization is the “inverse” of building a norm from an inner product: it reconstructs $\langle\cdot,\cdot\rangle$ from $\|\cdot\|$.

## Orthogonality, projections, Pythagoras/Parseval

- **Orthogonality:** $x\perp y$ iff $\langle x,y\rangle=0$.
- **Pythagoras:** if $x\perp y$, then $\|x+y\|^{2}=\|x\|^{2}+\|y\|^{2}$.
- **Parseval (finite orthogonal families):** for pairwise orthogonal $\{x_k\}$, $\big\|\sum_k x_k\big\|^{2}=\sum_k\|x_k\|^{2}$.
  In Hilbert spaces (complete inner‑product spaces), the series version holds for infinite orthogonal families.

> [!math] Orthogonal projection onto a subspace
>
> If $Q=[q_1\,\cdots\,q_k]$ has orthonormal columns spanning $S\subseteq V$, then $P_S=QQ^{\!*}$ is the projector and $\operatorname{argmin}_{s\in S}\|x-s\|=P_S x$.

## Orthonormal bases & Gram–Schmidt

Every finite‑dimensional inner product space admits an **orthonormal basis**, obtained from any basis via Gram–Schmidt. In Hilbert spaces (complete), orthonormal bases support Parseval expansions.

> [!tip] Privileged basis
>
> Working in an orthonormal or eigenbasis simplifies computations (projections, diagonalisation, PCA/SVD).

## Canonical examples

- **$\mathbb R^{n}$:**
  $\displaystyle \langle x,y\rangle=x^{\mathsf T} M y$ for symmetric positive‑definite $M$ (the usual dot product uses $M=I$).

- **$\mathbb C^{n}$:**
  $\displaystyle \langle x,y\rangle=x^{\dagger} M y$ for Hermitian positive‑definite $M$ (standard choice $M=I$ gives $\sum_i \overline{x_i}y_i$).

- **Matrix space $\mathbb C^{m\times n}$: (Frobenius/Hilbert–Schmidt)**
  $\displaystyle \langle A,B\rangle=\operatorname{tr}(A B^{\dagger})$ with induced norm $\|A\|_{\!F}^{2}=\sum_{ij}|A_{ij}|^{2}$.

- **Continuous functions $C([a,b])$:**
  $\displaystyle \langle f,g\rangle=\int_a^b f(t)\,\overline{g(t)}\,dt$. This inner‑product norm is not complete on $C([a,b])$; the completion is $L^{2}([a,b])$.

- **Probability / $L^{2}$ random variables:**
  $\displaystyle \langle X,Y\rangle=\mathbb E[X\,\overline{Y}]$; modulo a.s. equality, this makes $L^{2}$ a Hilbert space.

## Gram matrices and PSD

Given $x_1,\ldots,x_n\in V$, the Gram matrix $G\in\mathbb{F}^{n\times n}$ with $G_{ij}=\langle x_i,x_j\rangle$ is Hermitian positive semidefinite: $c^{\!*} G c=\big\|\sum_i c_i x_i\big\|^2\ge 0$. In finite‑dimensional Euclidean spaces, $G=X^{\!*}X$ for data matrix $X$.

> [!note] Links
> Gram matrices appear in least squares and kernel methods; principal axes arise from eigen/SVD. See [[thoughts/Singular Value Decomposition]] and [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/principal component analysis|PCA]].

## Useful identities

- **Binomial expansion in IP spaces (real case):**
  $\displaystyle \|x+y\|^{2}=\|x\|^{2}+2\langle x,y\rangle+\|y\|^{2}$. (Complex: replace $2\langle x,y\rangle$ by $2\operatorname{Re}\langle x,y\rangle$.)

- **Angle via C–S (real spaces):**
  $\displaystyle \cos\theta=\frac{\langle x,y\rangle}{\|x\|\,\|y\|}\in[-1,1]$. (In complex spaces, use $\operatorname{Re}$ or $|\cdot|$ as needed.)

- **Polarization (again, because it’s that useful):**
  Real: $\displaystyle \frac14(\|x+y\|^{2}-\|x-y\|^{2})$;
  Complex: $\displaystyle \frac14(\|x+y\|^{2}-\|x-y\|^{2}+i\|x+iy\|^{2}-i\|x-iy\|^{2})$.

## Relation to completeness (Hilbert spaces)

An inner product space is a **Hilbert space** iff it’s complete in the induced norm. When complete, orthogonal series behave exactly like Euclidean Pythagoras in infinite sums:

$$
\Big\|\sum_{k=0}^{\infty} u_k\Big\|^{2}=\sum_{k=0}^{\infty}\|u_k\|^{2}
\quad\text{for orthogonal }\{u_k\}.
$$

If you are missing limits, complete it—every pre‑Hilbert space has a Hilbert completion.
