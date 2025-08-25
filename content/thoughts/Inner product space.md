---
id: Inner product space
tags:
  - seed
  - clippings
  - math
description: Vector space with generalized dot product operation that defines lengths, angles, and orthogonality, generalizing Euclidean spaces to infinite dimensions.
source: https://en.wikipedia.org/wiki/Inner_product_space
date: "2025-08-20"
created: "2025-08-20"
modified: 2025-08-24 08:18:05 GMT-04:00
published: "2001-09-29"
title: Inner product space
author:
  - "[[Contributors to Wikimedia projects]]"
---

> Geometric interpretation of the angle between two vectors defined using an inner product

## Definition (real or complex)

A (real/complex) vector space $V$ with a map $\langle\cdot,\cdot\rangle:V\times V\to \mathbb F$ is an **inner product space** if for all $x,y,z\in V$ and $a,b\in\mathbb F$:

- **Conjugate symmetry:** $\langle x,y\rangle=\overline{\langle y,x\rangle}$.
- **Linearity in the first argument:** $\langle ax+by,z\rangle=a\langle x,z\rangle+b\langle y,z\rangle$ (hence, conjugate‑linearity in the second).
- **Positive‑definite:** $\langle x,x\rangle>0$ for all $x\neq 0$.

> **Physicists’ convention:** often linear in the **second** argument; adapt the formulas accordingly. ([Wikipedia][1])

## Induced norm and core inequalities

Define $\|x\|:=\sqrt{\langle x,x\rangle}$. Then:

- **Absolute homogeneity:** $\|a x\|=|a|\,\|x\|$.
- **Triangle inequality:** $\|x+y\|\le \|x\|+\|y\|$ (via Cauchy–Schwarz).
- **Cauchy–Schwarz:** $|\langle x,y\rangle|\le \|x\|\,\|y\|$, with equality iff $x,y$ are linearly dependent.
- **Parallelogram law:** $\|x+y\|^{2}+\|x-y\|^{2}=2\|x\|^{2}+2\|y\|^{2}$.
  No positivity, no party: without positive‑definiteness, $\|x\|=\sqrt{\langle x,x\rangle}$ won’t be a norm. ([Wikipedia][2])

## Polarization identity (rebuild the inner product from $\|\cdot\|$)

A norm comes from an inner product **iff** it satisfies the parallelogram law (Jordan–von Neumann). Then the inner product is uniquely recovered by polarization:

- **Real spaces:** $\displaystyle \langle x,y\rangle=\tfrac14\big(\|x+y\|^{2}-\|x-y\|^{2}\big)$.
- **Complex spaces (math convention, linear in the first argument):**

  $$
  \langle x,y\rangle=\tfrac14\Big(\|x+y\|^{2}-\|x-y\|^{2}+i\|x+iy\|^{2}-i\|x-iy\|^{2}\Big).
  $$

Think of polarization as the “inverse FFT” from norms back to inner products. ([Wikipedia][3])

## Orthogonality and Pythagoras/Parseval

- **Orthogonality:** $x\perp y$ iff $\langle x,y\rangle=0$.
- **Pythagoras:** if $x\perp y$, then $\|x+y\|^{2}=\|x\|^{2}+\|y\|^{2}$.
- **Parseval (finite orthogonal families):** for pairwise orthogonal $\{x_k\}$, $\big\|\sum_k x_k\big\|^{2}=\sum_k\|x_k\|^{2}$.
  In Hilbert spaces (complete inner‑product spaces), the series version holds—completeness lets you sum infinitely many perpendicular “legs.” ([Wikipedia][4])

## Orthonormal bases & Gram–Schmidt

Every finite‑dimensional inner product space admits an **orthonormal basis**, obtained from any basis via **Gram–Schmidt**. In infinite dimension, Hilbert spaces support orthonormal bases with the usual Parseval/expansion theorems. ([Wikipedia][5])

## Canonical examples

- **$\mathbb R^{n}$:**
  $\displaystyle \langle x,y\rangle=x^{\mathsf T} M y$ for some symmetric positive‑definite matrix $M$ (the usual dot product is $M=I$). ([Wikipedia][1])

- **$\mathbb C^{n}$:**
  $\displaystyle \langle x,y\rangle=x^{\dagger} M y$ for Hermitian positive‑definite $M$ (standard choice $M=I$ gives $\sum_i \overline{x_i}y_i$). ([Wikipedia][1])

- **Matrix space $\mathbb C^{m\times n}$: (Frobenius/Hilbert–Schmidt)**
  $\displaystyle \langle A,B\rangle=\operatorname{tr}(A B^{\dagger})$ with induced norm $\|A\|_{\!F}^{2}=\sum_{ij}|A_{ij}|^{2}$. ([Wikipedia][6])

- **Continuous functions $C([a,b])$:**
  $\displaystyle \langle f,g\rangle=\int_a^b f(t)\,\overline{g(t)}\,dt$. This inner‑product norm is **not** complete on $C([a,b])$; the completion is $L^{2}([a,b])$. ([Wikipedia][1])

- **Probability / $L^{2}$ random variables:**
  $\displaystyle \langle X,Y\rangle=\mathbb E[X\,\overline{Y}]$; modulo a.s. equality, this makes $L^{2}$ a Hilbert space. ([Wikipedia][1])

## Quick identities you’ll actually use

- **Binomial expansion in IP spaces (real case):**
  $\displaystyle \|x+y\|^{2}=\|x\|^{2}+2\langle x,y\rangle+\|y\|^{2}$. (Complex: replace $2\langle x,y\rangle$ by $2\operatorname{Re}\langle x,y\rangle$.) ([Wikipedia][1])

- **Angle via C–S (real spaces):**
  $\displaystyle \cos\theta=\frac{\langle x,y\rangle}{\|x\|\,\|y\|}\in[-1,1]$. (In complex spaces, use $\operatorname{Re}$ or $|\cdot|$ as needed.) ([Wikipedia][2])

- **Polarization (again, because it’s that useful):**
  Real: $\displaystyle \frac14(\|x+y\|^{2}-\|x-y\|^{2})$;
  Complex: $\displaystyle \frac14(\|x+y\|^{2}-\|x-y\|^{2}+i\|x+iy\|^{2}-i\|x-iy\|^{2})$. ([Wikipedia][3])

## Relation to completeness (Hilbert spaces)

An inner product space is a **Hilbert space** iff it’s complete in the induced norm. When complete, orthogonal series behave exactly like Euclidean Pythagoras in infinite sums:

$$
\Big\|\sum_{k=0}^{\infty} u_k\Big\|^{2}=\sum_{k=0}^{\infty}\|u_k\|^{2}
\quad\text{for orthogonal }\{u_k\}.
$$

If you’re missing limits, complete it—every pre‑Hilbert space has a Hilbert completion. ([Wikipedia][7])

[1]: https://en.wikipedia.org/wiki/Inner_product_space "Inner product space - Wikipedia"
[2]: https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality "Cauchy–Schwarz inequality"
[3]: https://en.wikipedia.org/wiki/Polarization_identity "Polarization identity - Wikipedia"
[4]: https://en.wikipedia.org/wiki/Parseval%27s_identity "Parseval's identity"
[5]: https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process "Gram–Schmidt process"
[6]: https://en.wikipedia.org/wiki/Frobenius_inner_product "Frobenius inner product - Wikipedia"
[7]: https://en.wikipedia.org/wiki/Hilbert_space "Hilbert space"
