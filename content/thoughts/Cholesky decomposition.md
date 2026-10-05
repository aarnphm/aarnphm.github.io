---
date: '2024-10-28'
description: Cholesky factorization, a worked triangular factor, and how it generates correlated Gaussian samples.
id: Cholesky decomposition
modified: 2026-10-05 09:12:12 GMT-04:00
tags:
  - math
title: Cholesky decomposition
---

For a Hermitian positive-definite matrix $A$, the Cholesky decomposition gives

$$
A = LL^{*},
$$

where $L$ is lower triangular with real, positive diagonal entries, and $L^{*}$ is its conjugate transpose. These conditions make $L$ unique. For a real symmetric matrix, the conjugate transpose reduces to $L^T$.[^factor]

Positive definiteness means $x^{*}Ax > 0$ for every nonzero $x$. The factorization makes that condition visible:

$$
x^{*}Ax = x^{*}LL^{*}x = \|L^{*}x\|_2^2 > 0.
$$

The last inequality holds because the positive diagonal makes $L$ invertible.

## computing the factor

For a real matrix, fill the columns of $L$ from left to right. Matching entries of $A$ and $LL^T$ gives

$$
\begin{aligned}
L_{jj} &= \sqrt{A_{jj}-\sum_{k<j}L_{jk}^2}, \\
L_{ij} &= \frac{A_{ij}-\sum_{k<j}L_{ik}L_{jk}}{L_{jj}}, \qquad i>j.
\end{aligned}
$$

Each column subtracts the contributions already accounted for by earlier columns. Positive definiteness keeps every diagonal remainder positive in exact arithmetic.[^algorithm]

For example,

$$
A=\begin{pmatrix}4&2\\2&5\end{pmatrix},
\qquad
L=\begin{pmatrix}2&0\\1&2\end{pmatrix}.
$$

The first diagonal entry is $\sqrt{4}=2$. The entry below it is $2/2=1$, leaving $5-1^2=4$ for the second diagonal entry. Multiplication checks the result:

$$
LL^T=
\begin{pmatrix}2&0\\1&2\end{pmatrix}
\begin{pmatrix}2&1\\0&2\end{pmatrix}
=
\begin{pmatrix}4&2\\2&5\end{pmatrix}.
$$

## correlated samples

In [[thoughts/Monte-Carlo#simulations|Monte-Carlo simulations]], the same factor turns independent Gaussian draws into draws with covariance $A$. For the real case, take

$$
z\sim\mathcal{N}(0,I),
\qquad x=\mu+Lz.
$$

Then

$$
\mathbb{E}[x]=\mu,
\qquad
\operatorname{Cov}(x)=L\operatorname{Cov}(z)L^T=LL^T=A.
$$

The Gaussian distribution is preserved under this affine transformation, so $x\sim\mathcal{N}(\mu,A)$.[^sampling] In the example with $\mu=0$,

$$
x_1=2z_1,
\qquad x_2=z_1+2z_2.
$$

Both coordinates contain $z_1$, which produces covariance $2$. Their variances are $4$ and $5$, matching the diagonal of $A$.

The factor also solves $Ax=b$ through two triangular systems: first $Lu=b$, then $L^{*}x=u$. A singular positive-semidefinite matrix can have a zero pivot; the strictly positive-diagonal construction above assumes positive definiteness.[^algorithm]

[^factor]: LAPACK documents the real and complex factorizations in [DPOTRF](https://netlib.org/lapack/explore-html/d2/d09/group__potrf_ga84e90859b02139934b166e579dd211d4.html) and [ZPOTRF](https://netlib.org/lapack/explore-html/d2/d09/group__potrf_gae6842f47b241caed33a19ace77f482a4.html).

[^algorithm]: The scalar recurrence appears in LAPACK's [DPOTF2 implementation](https://netlib.org/lapack/lapack-3.1.1/html/dpotf2.f.html). A nonpositive diagonal remainder stops the factorization.

[^sampling]: Antti Honkela, [Computational Statistics I, chapter 3](https://www.cs.helsinki.fi/u/ahonkela/teaching/compstats1/book/multivariate-normal-distributions-and-numerical-linear-algebra.html), derives Gaussian sampling and triangular solves from the factorization.
