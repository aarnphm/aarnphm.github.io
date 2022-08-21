---
created: "2025-09-15"
date: "2025-09-17"
description: Mathematical construct for Hermitian matrices relating eigenvalues to quadratic forms via ratios.
id: Rayleigh quotient
modified: 2025-10-29 02:15:33 GMT-04:00
published: "2003-08-27"
source: https://en.wikipedia.org/wiki/Rayleigh_quotient
tags:
  - math
  - seed
  - clippings
title: Rayleigh quotient
---

In mathematics, the **Rayleigh quotient** for a given complex Hermitian matrix $M$ and nonzero vector $x$ is defined as:

$$R(M,x) = \frac{x^*Mx}{x^*x}$$

For real matrices and vectors, the Hermitian condition reduces to symmetric, and the conjugate transpose $x^*$ becomes the usual transpose $x'$. Note that $R(M,cx) = R(M,x)$ for any non-zero scalar $c$.

Since Hermitian (or real symmetric) matrices are diagonalizable with only real eigenvalues, the Rayleigh quotient reaches its minimum value $\lambda_{\min}$ (the smallest eigenvalue of $M$) when $x$ is $v_{\min}$ (the corresponding eigenvector). Similarly, $R(M,x) \leq \lambda_{\max}$ and $R(M,v_{\max}) = \lambda_{\max}$.

## applications

The Rayleigh quotient is used in:

- The min-max theorem to get exact values of all eigenvalues
- Eigenvalue algorithms (such as Rayleigh quotient iteration) to obtain eigenvalue approximations from eigenvector approximations
- Quantum mechanics, where it gives the expectation value of the observable corresponding to operator $M$ for a system whose state is given by $x$

## bounds for Hermitian $m$

For any vector $x$, we have $R(M,x) \in [\lambda_{\min}, \lambda_{\max}]$, where $\lambda_{\min}, \lambda_{\max}$ are respectively the smallest and largest eigenvalues of $M$.

This follows from observing that the Rayleigh quotient is a weighted average of eigenvalues:

$$R(M,x) = \frac{x^*Mx}{x^*x} = \frac{\sum_{i=1}^n \lambda_i y_i^2}{\sum_{i=1}^n y_i^2}$$

where $(\lambda_i, v_i)$ is the $i$-th eigenpair after orthonormalization and $y_i = v_i^* x$ is the $i$-th coordinate of $x$ in the eigenbasis.

## special case of covariance matrices

An empirical covariance matrix $M$ can be represented as $A'A$ of the normalized data matrix $A$. Being positive semi-definite, $M$ has non-negative eigenvalues and orthogonal eigenvectors.

The Rayleigh quotient becomes:
$$R(M,x) = \sum_{i=1}^n \lambda_i \frac{(x'v_i)^2}{(x'x)}$$

This establishes that the Rayleigh quotient is the sum of squared cosines of angles between vector $x$ and each eigenvector $v_i$, weighted by corresponding eigenvalues.

### Formulation using Lagrange Multipliers

To find critical points, we maximize $x^T Mx$ subject to $\|x\|^2 = x^T x = 1$. Using Lagrange multipliers:

$$\mathcal{L}(x) = x^T Mx - \lambda(x^T x - 1)$$

Setting $\frac{d\mathcal{L}(x)}{dx} = 0$ yields:
$$Mx = \lambda x$$

Therefore: $R(M,x) = \lambda$

The eigenvectors of $M$ are the critical points of the Rayleigh quotient, and their corresponding eigenvalues are the stationary values.

## use in Sturm–Liouville Theory

For the linear operator:
$$L(y) = \frac{1}{w(x)}\left(-\frac{d}{dx}\left[p(x)\frac{dy}{dx}\right] + q(x)y\right)$$

The Rayleigh quotient is:
$$\frac{\langle y,Ly \rangle}{\langle y,y \rangle} = \frac{\int_a^b y(x)\left(-\frac{d}{dx}\left[p(x)\frac{dy}{dx}\right] + q(x)y(x)\right)dx}{\int_a^b w(x)y(x)^2 dx}$$

## generalizations

1. **Generalized Rayleigh quotient** for matrices $(A,B)$:
   $$R(A,B;x) := \frac{x^*Ax}{x^*Bx}$$

2. **Two-sided Rayleigh quotient** for vectors $(x,y)$ and Hermitian matrix $H$:
   $$R(H;x,y) := \frac{y^*Hx}{\sqrt{y^*y \cdot x^*x}}$$
