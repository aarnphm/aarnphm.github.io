---
aliases:
  - math/linalg
date: '2024-09-11'
description: linear algebra a la carte.
id: tut1
modified: 2026-10-07 09:14:19 GMT-04:00
tags:
  - sfwr4ml3
  - math/linalg
title: linalg review
transclude:
  title: false
---

See also [matrix cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).

## matrix representation of a system of linear equations

The rows of a matrix collect the coefficients of each equation:

$$
\begin{aligned}
x_1 + x_2 + x_3 &= 5 \\
x_1 - 2x_2 - 3x_3 &= -1 \\
2x_1 + x_2 - x_3 &= 3.
\end{aligned}
$$

Writing this system as $Ax=b$ gives

$$
A = \begin{bmatrix}
1 & 1 & 1 \\
1 & -2 & -3 \\
2 & 1 & -1
\end{bmatrix},
\qquad
x = \begin{bmatrix}x_1 \\ x_2 \\ x_3\end{bmatrix},
\qquad
b = \begin{bmatrix}5 \\ -1 \\ 3\end{bmatrix}.
$$

In general, $A \in \mathbb{R}^{m \times n}$ maps $x \in \mathbb{R}^n$ to $Ax \in \mathbb{R}^m$. There are $m$ equations and $n$ unknowns. Each row produces one entry of the output.

> [!important] Transpose of a matrix
>
> Transposing exchanges rows and columns. If $A \in \mathbb{R}^{m \times n}$, then $A^T \in \mathbb{R}^{n \times m}$, with $(A^T)_{ij}=A_{ji}$.

## dot product.

For $x,y \in \mathbb{R}^n$,

$$
\langle x,y\rangle = x^T y = \sum_{i=1}^{n} x_i y_i.
$$

The result is a scalar. Two vectors are orthogonal when their dot product is zero.

## linear combination of columns

Write $A=[a_1\ \cdots\ a_n]$, where each column $a_i \in \mathbb{R}^m$. Then

$$
Ax = \sum_{i=1}^{n} x_i a_i \in \mathbb{R}^m.
$$

Each entry of $x$ weights one column. Varying those weights gives every vector the matrix can produce.

## inverse of a matrix

An invertible square matrix $A \in \mathbb{R}^{n \times n}$ has a unique inverse $A^{-1}$ satisfying

$$
A^{-1}A = AA^{-1} = I_n.
$$

A square matrix is invertible exactly when its columns are linearly independent, or equivalently when $\operatorname{rank}(A)=n$. A singular matrix has no inverse: some nonzero input maps to zero, so the output cannot determine the input uniquely.

## euclidean norm

The Euclidean, or $L_2$, norm measures a vector's length:

$$
\|x\|_2 = \sqrt{\sum_{i=1}^{n}x_i^2} = \sqrt{x^T x}.
$$

The square matters: $x^T x=\|x\|_2^2$. For $x=(3,4)^T$, the norm is $5$ and its square is $25$.

$L_1$ norm: $\|x\|_1 = \sum_{i=1}^{n}|x_i|$. ^l1norm

$L_{\infty}$ norm: $\|x\|_{\infty} = \max_{1\leq i\leq n}|x_i|$.

For $1\leq p<\infty$, the $p$-norm is

$$
\|x\|_p = \left(\sum_{i=1}^{n}|x_i|^p\right)^{1/p}.
$$

> [!important] Comparison
>
> $$
> \|x\|_{\infty} \leq \|x\|_2 \leq \|x\|_1.
> $$

The largest squared entry is at most the sum of all squared entries. Squaring the sum of absolute entries adds nonnegative cross terms. Taking square roots gives the two inequalities.

## linear dependence of vectors

Vectors $x_1,\ldots,x_n \in \mathbb{R}^d$ are **linearly dependent** if some choice of scalar coefficients, with at least one nonzero coefficient, satisfies

$$
\sum_{i=1}^{n}\alpha_i x_i=0.
$$

They are **linearly independent** if the only such choice is

$$
\alpha_1=\cdots=\alpha_n=0.
$$

In a dependent family, choose an index $k$ with $\alpha_k\neq0$. Rearranging expresses that vector using the others:

$$
x_k = -\sum_{i\neq k}\frac{\alpha_i}{\alpha_k}x_i.
$$

For example, $(2,0)^T=2(1,0)^T$, so these two vectors are dependent.

## Span

The span is the set of all vectors obtainable by linear combinations:

$$
\operatorname{span}\{x_1,\ldots,x_n\}
=\left\{\sum_{i=1}^{n}\alpha_i x_i \;\middle|\; \alpha_1,\ldots,\alpha_n\in\mathbb{R}\right\}.
$$

Independent vectors form a basis for their span. To form a basis for all of $\mathbb{R}^d$, they must also span that space. This requires exactly $d$ independent vectors. For example, $(1,0,0)^T$ and $(0,1,0)^T$ are independent, and their combinations fill the plane with third coordinate zero. Adding $(0,0,1)^T$ gives a basis for $\mathbb{R}^3$. See [Strang's notes on independence, basis, and dimension](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/0bbc30e3f1d7933ea07a2d2e9ab050d9_MIT18_06SCF11_Ses1.9sum.pdf).

## Rank

For $A \in \mathbb{R}^{m\times n}$, the column rank counts the largest number of independent columns. The row rank counts the largest number of independent rows. These counts agree:

$$
\operatorname{rank}(A)=\operatorname{rank}(A^T)\leq\min(m,n).
$$

- The columns are independent exactly when $\operatorname{rank}(A)=n$.
- The rows are independent exactly when $\operatorname{rank}(A)=m$.
- The matrix has full rank when $\operatorname{rank}(A)=\min(m,n)$.

The inequality alone says nothing about independence. A zero matrix satisfies the bound and has rank zero. A tall full-rank matrix has independent columns; a wide full-rank matrix has independent rows.

## solving linear system of equations

If $A \in \mathbb{R}^{n\times n}$ is invertible, then every $b \in \mathbb{R}^n$ has exactly one solution:

$$
x=A^{-1}b.
$$

For a general matrix, an exact solution exists when $b$ lies in the span of its columns. A nontrivial null space makes any existing solution nonunique, since adding a vector that maps to zero leaves $Ax$ unchanged.

## Range and Projection

The range of $A \in \mathbb{R}^{m\times n}$ is its column span:

$$
\mathcal{R}(A)=\{Ax\mid x\in\mathbb{R}^n\}\subseteq\mathbb{R}^m.
$$

The orthogonal projection of $y \in \mathbb{R}^m$ onto this range is its unique closest vector in Euclidean distance:

$$
p=\operatorname*{argmin}_{v\in\mathcal{R}(A)}\|y-v\|_2^2.
$$

At the minimum, the residual $y-p$ is perpendicular to every column of $A$. Writing $p=A\hat{x}$ gives the normal equations:

$$
A^T(y-A\hat{x})=0,
\qquad
A^T A\hat{x}=A^Ty.
$$

If $A$ has independent columns, then

$$
p=A(A^TA)^{-1}A^Ty.
$$

With dependent columns, several coefficient vectors can produce the same projection. The closest vector $p$ remains unique. This is the geometry behind [least squares](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/least-squares-determinants-and-eigenvalues/projection-matrices-and-least-squares/).

## Null space of $A$

The null space contains the inputs that map to zero:

$$
\mathcal{N}(A)=\{x\in\mathbb{R}^n\mid Ax=0\}.
$$

It is a subspace of the input space. Its dimension and the rank account for all $n$ input dimensions:

$$
\dim\mathcal{N}(A)+\operatorname{rank}(A)=n.
$$

In particular, independent columns give $\mathcal{N}(A)=\{0\}$.
