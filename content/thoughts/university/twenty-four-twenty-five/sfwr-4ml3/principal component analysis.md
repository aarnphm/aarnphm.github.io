---
date: '2024-10-07'
description: PCA as orthogonal projection, with covariance dimensions, reconstruction error, and the connection to SVD.
id: principal component analysis
modified: 2026-10-06 09:05:53 GMT-04:00
tags:
  - sfwr4ml3
title: principal component analysis
---

## problem statement

Given points $x^1,\ldots,x^n\in\mathbb{R}^d$, retain $q<d$ coordinates while minimising squared reconstruction error. Put observations in **columns**:

$$
\mu=\frac{1}{n}\sum_{i=1}^n x^i,
\qquad
X_c=\begin{bmatrix}x^1-\mu&\cdots&x^n-\mu\end{bmatrix}
\in\mathbb{R}^{d\times n}.
$$

The encoder $A\in\mathbb{R}^{q\times d}$ has orthonormal rows, so $AA^T=I_q$. Encoding and reconstruction are

$$
z=A(x-\mu)\in\mathbb{R}^q,
\qquad
\widehat x=\mu+A^Tz.
$$

The matrix $A^TA$ projects onto the retained subspace. Adding $\mu$ restores the location of the data.

## minimising reconstruction error

Within any chosen subspace, orthogonal projection gives the closest point. We can therefore find an optimal linear encoder and decoder by solving

$$
\min_{A:\,AA^T=I_q}
\left\|X_c-A^TAX_c\right\|_F^2
=\min_{A:\,AA^T=I_q}
\sum_{i=1}^n
\left\|(x^i-\mu)-A^TA(x^i-\mu)\right\|_2^2.
$$

One optimal encoder-decoder pair has $B=A^T$. This choice follows from finding the projection subspace; an arbitrary fixed encoder need not have its transpose as its best decoder.

Projection and residual are perpendicular. Pythagoras gives

$$
\left\|X_c-A^TAX_c\right\|_F^2
=\|X_c\|_F^2-\|AX_c\|_F^2.
$$

Thus minimising discarded squared length maximises retained variance. This is the reconstruction objective in [Guestrin's PCA lecture](https://cs229.stanford.edu/notes2022fall/pca.pdf).

## eigenvalue decomposition

Use the empirical covariance with denominator $n$:

$$
C=\frac{1}{n}X_cX_c^T
=\frac{1}{n}\sum_{i=1}^n(x^i-\mu)(x^i-\mu)^T
\in\mathbb{R}^{d\times d}.
$$

For $n>1$, using $n-1$ instead rescales every eigenvalue and leaves the principal directions unchanged. The unnormalised sum is the scatter matrix.

Since $C$ is symmetric and positive semidefinite, choose orthonormal eigenvectors as the **columns** of $U$:

$$
Cu_j=\lambda_j u_j,
\qquad
C=U\Lambda U^T,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_d),
\qquad
\lambda_1\geq\cdots\geq\lambda_d\geq0.
$$

The variance along a unit direction $u$ is $u^TCu$. Choosing the top $q$ eigenvectors maximises the sum of retained variances, as derived in [Ng's PCA notes](https://cs229.stanford.edu/notes2020spring/cs229-notes10.pdf).

Watch the dimensions: $X_c^TX_c\in\mathbb{R}^{n\times n}$ acts on sample coordinates. Its eigenvectors are not directly feature directions. With the full [[thoughts/Singular Value Decomposition|SVD]],

$$
X_c=U\Sigma V^T,
\qquad
C=U\frac{\Sigma\Sigma^T}{n}U^T,
\qquad
\lambda_j=\frac{\sigma_j^2}{n}.
$$

Here $\sigma_j$ are singular values, with zeros supplied where needed. The left singular vectors give the feature directions because observations are columns. Putting observations in rows makes them the right singular vectors instead.

## pca

For observations $x^i\in\mathbb{R}^d$, compute their mean $\mu$ and the covariance $C=\frac{1}{n}\sum_i(x^i-\mu)(x^i-\mu)^T$. Let $U_q=[u_1\ \cdots\ u_q]$ contain its top $q$ orthonormal eigenvectors. Then

$$
A=U_q^T,
\qquad
z^i=U_q^T(x^i-\mu),
\qquad
\widehat x^i=\mu+U_qz^i.
$$

The minimum total squared reconstruction error is

$$
\sum_{i=1}^n\|x^i-\widehat x^i\|_2^2
=n\sum_{j=q+1}^d\lambda_j.
$$

If the covariance eigenvalues are $9$ and $1$, retaining the first direction preserves $9/10$ of total variance and leaves per-observation squared reconstruction error $1$. Zero error needs $q\geq\operatorname{rank}(X_c)$; the data may already occupy fewer than $d$ dimensions. Tied eigenvalues at the cutoff allow several equally good subspaces.

Centering is part of this derivation. Scaling each feature to unit variance is a separate choice that changes the distance being minimised. Large variance alone also says nothing about whether a direction helps predict a label.
