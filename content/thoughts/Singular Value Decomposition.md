---
date: '2024-10-21'
description: singular vectors as paired input and output directions, with full and reduced matrix shapes, low-rank approximation, and eigenfaces.
id: Singular Value Decomposition
modified: 2026-10-06 09:12:36 GMT-04:00
tags:
  - ml
  - math/linalg
title: Singular Value Decomposition
---

![[https://www.youtube.com/watch?v=nbBvuuNVfco&ab_channel=SteveBrunton]]

A matrix maps input vectors to output vectors. The singular value decomposition (SVD) finds orthonormal directions in each space so that the map acts by scaling one input direction into one output direction.

## dimensions

Every real matrix $X\in\mathbb{R}^{m\times n}$ has a full SVD:

$$
X=U\Sigma V^T,\qquad
U\in\mathbb{R}^{m\times m},\quad
\Sigma\in\mathbb{R}^{m\times n},\quad
V\in\mathbb{R}^{n\times n}.
$$

The columns of $U$ and $V$ form orthonormal bases of the output and input spaces, respectively:

$$
U^TU=UU^T=I_m,\qquad V^TV=VV^T=I_n.
$$

Thus $U$ and $V$ are **orthogonal matrices**. All off-diagonal entries of $\Sigma$ are zero. Its diagonal holds the singular values, ordered as

$$
\sigma_1\geq\sigma_2\geq\cdots\geq\sigma_p\geq0,
\qquad p=\min(m,n).
$$

For complex matrices, use the conjugate transpose $V^*$ and unitary matrices instead. See the [LAPACK definition](https://netlib.org/lapack/lug/node53.html).

The **reduced SVD** keeps the first $p$ columns of each basis:

$$
X=U_p\Sigma_p V_p^T,\qquad
U_p\in\mathbb{R}^{m\times p},\quad
\Sigma_p\in\mathbb{R}^{p\times p},\quad
V_p\in\mathbb{R}^{n\times p}.
$$

Now $U_p^TU_p=V_p^TV_p=I_p$, while $U_pU_p^T$ and $V_pV_p^T$ are projections onto their column spaces. They equal the identity only when that basis spans the whole corresponding space. This is the shape convention used by [NumPy's `full_matrices=False`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html).

With $r=\operatorname{rank}(X)$, keeping only the positive singular values gives the **compact SVD** with $r$ columns. When $r<p$, this discards zero singular values. Both reductions still reconstruct $X$ exactly.

## what the vectors mean

For each singular-vector pair,

$$
Xv_i=\sigma_i u_i,\qquad X^Tu_i=\sigma_i v_i,
\qquad 1\leq i\leq p.
$$

$v_i$ is an input direction; $u_i$ is its output direction when $\sigma_i>0$. The singular value is the length of the output for that unit input. A zero singular value means the map erases the input direction. For an arbitrary input $z$,

$$
Xz=\sum_{i=1}^{r}\sigma_i u_i(v_i^Tz).
$$

Read off the input's component along each $v_i$, scale it by $\sigma_i$, then combine the output directions. This also explains the eigenvector connection:

$$
X^TXv_i=\sigma_i^2v_i,\qquad
XX^Tu_i=\sigma_i^2u_i.
$$

The right singular vectors diagonalize $X^TX$; the left singular vectors diagonalize $XX^T$. These symmetric positive-semidefinite matrices have orthonormal eigenbases even when $X$ is rectangular. [Strang's SVD lecture](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-29-singular-value-decomposition/) develops this construction.

### example

Consider a map from two coordinates to three:

$$
X=\begin{bmatrix}2&2\\1&-1\\0&0\end{bmatrix},\qquad
U=I_3,\qquad
\Sigma=\begin{bmatrix}2\sqrt{2}&0\\0&\sqrt{2}\\0&0\end{bmatrix},\qquad
V=\frac{1}{\sqrt{2}}\begin{bmatrix}1&1\\1&-1\end{bmatrix}.
$$

Multiplying $U\Sigma V^T$ recovers $X$. Writing $e_i$ for the $i$th coordinate unit vector, the input direction $v_1=(1,1)^T/\sqrt{2}$ maps to $2\sqrt{2}\,e_1$; $v_2=(1,-1)^T/\sqrt{2}$ maps to $\sqrt{2}\,e_2$. The first direction is stretched twice as much. Every output has a zero third coordinate, so the reduced SVD needs only the first two columns of $U$:

$$
U_p^TU_p=I_2,\qquad U_pU_p^T=\operatorname{diag}(1,1,0).
$$

## keeping fewer directions

Keeping the first $k<r$ terms gives

$$
X_k=\sum_{i=1}^{k}\sigma_i u_i v_i^T,\qquad
\|X-X_k\|_F^2=\sum_{i=k+1}^{r}\sigma_i^2.
$$

The squared Frobenius norm adds the squared errors in every matrix entry. The [Eckart–Young theorem](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-7-eckart-young-the-closest-rank-k-matrix-to-a/) says this truncation minimizes that error among matrices of rank at most $k$. In the example, keeping only the first direction gives

$$
X_1=\begin{bmatrix}2&2\\0&0\\0&0\end{bmatrix},\qquad
\frac{\|X-X_1\|_F^2}{\|X\|_F^2}=\frac{2}{10}=\frac15.
$$

Small singular values contribute little to this reconstruction criterion. Whether those directions contain noise or useful information depends on the data and the task.

## when these become eigenfaces

Flatten each aligned face image into a vector $x_j\in\mathbb{R}^m$. Put the images in columns and subtract the mean face:

$$
\bar{x}=\frac1n\sum_{j=1}^{n}x_j,\qquad
X=[x_1-\bar{x}\;\cdots\;x_n-\bar{x}].
$$

For $n>1$, the sample covariance is $C=XX^T/(n-1)$. Its eigenvectors are the left singular vectors of this centered matrix, with eigenvalues $\sigma_i^2/(n-1)$. Using the empirical covariance $XX^T/n$ rescales the eigenvalues and preserves the directions. Reshape a leading $u_i$ into the original image dimensions and it is an **eigenface**: a direction of pixel variation across the training faces. This is the PCA construction in [Turk and Pentland's _Eigenfaces for Recognition_](https://faculty.cc.gatech.edu/~hic/CS7616/Papers/Turk-Pentland.pdf).

A face's coefficients and reconstruction using the first $k$ directions are

$$
a=U_k^T(x-\bar{x}),\qquad \hat{x}=\bar{x}+U_ka.
$$

This interpretation requires face data and its pixel coordinates. An arbitrary matrix's singular vectors have whatever meaning its input and output spaces provide. Storing images as rows swaps the roles of $U$ and $V$; the eigenfaces then appear in $V$.
