---
id: tut1
aliases:
  - linalg
tags:
  - sfwr4ml3
date: "2024-09-11"
description: linear algebra a la carte.
modified: 2025-01-29 08:10:07 GMT-05:00
title: linalg review
transclude:
  title: false
---

See also [matrix cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

## matrix representation of a system of linear equations

$$
\begin{aligned}
x_1 + x_2 + x_3 &= 5 \\
x_1 - 2x_2 - 3x_3 &= -1 \\
2x_1 + x_2 - x_3 &= 3
\end{aligned}
$$

Equivalent matrix representation of $Ax = b$

$$
\begin{aligned}
A &= \begin{bmatrix}
1 & 1 & 1 \\
1 & -2 & -3 \\
2 & 1 & -1
\end{bmatrix} \\

x &= \begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix} \\

b &= \begin{bmatrix}
5 \\
-1 \\
3
\end{bmatrix}
\end{aligned}

\because A \in R^{m \times n}, x \in R^n, b \in R^m
$$

> [!important] Transpose of a matrix
>
> $A \in R^{m \times n}$ and $A^T \in R^{n \times m}$

## dot product.

$$
\begin{aligned}
\langle x, y \rangle &= \sum_{i=1}^{n} x_i y_i \\
&= \sum_{i=1}^{n} x_i \cdot y_i
\end{aligned}
$$

## linear combination of columns

Let $A \in R^{m \times n}$, $X \in R^n$, $Ax \in R^n$

Then $Ax = \sum_{i=1}^{n}{\langle a_i \rangle} x_i \in R^n$

## inverse of a matrix

The inverse of a square matrix $A \in R^{n \times n}$ is a **unique** matrix denoted by $A^{-1} \in \mathbb{R}^{n\times{n}}$

$$
A^{-1} A = I = A A^{-1}
$$

## euclidean norm

$L_{2}$ norm:

$$
\| x \|_{2} = \sqrt{\sum_{i=1}^{n}{x_i^2}} = X^TX
$$

L1 norm: $\| x \|_{1} = \sum_{i=1}^{n}{|x_i|}$ ^l1norm

$L_{\infty}$ norm: $\| x \|_{\infty} = \max_{i}{|x_i|}$

p-norm: $\| x \|_{p} = (\sum_{i=1}^{n}{|x_i|^p})^{1/p}$

> [!important] Comparison
> $ \|x\|_{\infty} \leq \|x\|_{2} \leq \|x\|\_{1}$

> One can prove this with Cauchy-Schwarz inequality

## linear dependence of vectors

Given $\{x_1, x_2, \ldots, x_n\} \subseteq \mathbb{R}^d$ and $\alpha_1, \alpha_2, \ldots, \alpha_n \in \mathbb{R}$

$$
\forall i \in [ n ], \forall \{a_1, a_2, \ldots, a_n\} \subseteq \mathbb{R}^d \space s.t. \space x_i \neq \sum_{j=1}^{n}{a_j x_j}
$$

## Span

> Given a set of vectors $\{x_1, x_2, \ldots, x_n\} \subseteq \mathbb{R}^d$, the span of the set is the set of all possible linear combinations of the vectors.
>
> $$
> \text{span}(\{x_1, x_2, \ldots, x_n\}) = \{ y: y =  \sum_{i=1}^{n}{\alpha_i x_i} \mid \alpha_i \in \mathbb{R} \}
> $$

If $x_{1}, x_{2}, \ldots, x_{n}$ are linearly independent, then the span of the set is the entire space $\mathbb{R}^d$

## Rank

For a matrix $A \in \mathbb{R}^{m \times n}$:

- column rank: max number of linearly independent columns of $A$
- row rank: max number of linearly independent rows of $A$

If $\text{rank}(A) \leq m$, then the rows are linearly independent. If $\text{rank}(A) \leq n$, then the columns are linearly independent.

> rank of a matrix $A$ is the number of linearly independent columns of $A$:
>
> - if $A$ is full rank, then $\text{rank}(A) = \min(m, n)$ ($\text{rank}(A) \leq \min(m, n)$)
> - $\text{rank}(A) = \text{rank}(A^T)$

## solving linear system of equations

If $A \in \mathbb{R}^{n}$ is invertible, there exists a solution:

$$
x = A^{-1}b
$$

## Range and Projection

Given a matrix $A \in \mathbb{R}^{m \times n}$, the range of $A$, denoted by $\mathcal{R}(A)$ is the span of columns of $A$:

$$
\mathcal{R}(A) = \{ y \in \mathbb{R}^m \mid y = Ax \mid x \in \mathbb{R}^m \}
$$

Projection of a vector $y \in \mathbb{R}^m$ onto $\text{span}(\{x_1, \cdots, x_n\})$, $x_i \in \mathbb{R}^m$ is a vector in the span that is as close as possible to $y$ wrt $l_2$ norm

$$
\text{Proj}(y; \{x_{1}, \cdots, x_n\}) = \argmin_{{v \in \text{span}(\{x_1, \cdots, x_n\})}} \| y - v \|_2
$$

## Null space of $A$

is the set of all vectors that satisfies the following:

$$
\mathcal{N}(A) = \{ x \in \mathbb{R}^n \mid Ax = 0 \}
$$
