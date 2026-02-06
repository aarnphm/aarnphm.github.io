---
date: '2024-10-07'
description: dimensionality reduction through eigenvalue decomposition minimizing reconstruction error with orthonormal transformations.
id: principal component analysis
modified: 2025-10-29 02:16:10 GMT-04:00
tags:
  - sfwr4ml3
title: principal component analysis
---

## problem statement

- map $x \in R^d$ to $z \in \mathbb{R}^q$ with $q < d$
- A $q \times d$ matrix can represent a linear mapping:
  $$
  z = Ax
  $$
  - Assume that $A A^T = I$ (orthonormal matrix)

## minimising reconstruction error

- Given $X \in \mathbb{R}^{d \times n}$, find $A$ that minimises the reconstruction error:
  $$
  \min\limits_{A,B} \sum_{i} \| x^i - B A x^i \|_2^2
  $$

> if $q=d$, then error is zero.

Solution:

- $B = A^T$
- $\min\limits_{A} \sum_i \| x^i - A^T A x^i \|^2$ is subjected to $A A^T = I_{q \times q}$
- assuming data is centered, or $\frac{1}{n} \sum\_{i} x^i = \begin{bmatrix} 0 & \cdots & 0 \end{bmatrix}^T $

## eigenvalue decomposition

$$
\begin{aligned}
X^T X \mathcal{u} &= \lambda \mathcal{u} \\
X^T X &= U^T \Lambda U \\
\\
\\
\because \Lambda &= \text{diag}(\lambda_1, \lambda_2, \cdots, \lambda_d) \\ &= \begin{bmatrix} \lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_q \end{bmatrix}
\end{aligned}
$$

## pca

Idea: given input $x^1, \cdots, x^n \in \mathbb{R}^d$, $\mu = \frac{1}{n} \sum_{i} x^i$

Thus

$$
C = \sum (x^i - \mu)(x^i - \mu)^T
$$

Find the eigenvectors/values of $C$:

$$
C = U^T \Lambda U
$$

Optimal $A$ is:

$$
A = \begin{bmatrix}
u_1^T \\
u_2^T \\
\vdots \\
u_q^T
\end{bmatrix}
$$
