---
id: tut1
tags:
  - sfwr4ml3
date: "2024-09-11"
title: linalg review
---

$$
\begin{aligned}
x_1 + x_2 + x_3 &= 5 \\
x_1 - 2x_2 - 3x_3 &= -1 \\
2x_1 + x_2 - x_3 &= 3
\end{aligned}
$$

Equivalent matrix representation of $A \times x = b$

$$
\begin{aligned}
A &= \begin{pmatrix}
1 & 1 & 1 \\
1 & -2 & -3 \\
2 & 1 & -1
\end{pmatrix} \\

X &= \begin{pmatrix}
x_1 \\
x_2 \\
x_3
\end{pmatrix} \\

b &= \begin{pmatrix}
5 \\
-1 \\
3
\end{pmatrix}
\end{aligned}
$$

with $A \in R^{m \times n}$

> [!important] Transpose of a matrix
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

## euclidean norm

$L_{2}$ norm:
$$
\| x \|_{2} = \sqrt{\sum_{i=1}^{n}{x_i^2}} = X^T \times X
$$

L1 norm: $\| x \|_{1} = \sum_{i=1}^{n}{|x_i|}$

$L_{\infty}$ norm: $\| x \|_{\infty} = \max_{i}{|x_i|}$

p-norm: $\| x \|_{p} = (\sum_{i=1}^{n}{|x_i|^p})^{1/p}$

> [!important] Comparison
> $ \|x\|_{\infty} \leq \|x\|_{2} \leq \|x\|_{1}$

> One can prove this with Cauchy-Schwarz inequality

## linear dependence of vectors

Given $\{x_1, x_2, \ldots, x_n\} \subseteq \mathbb{R}^d$ and $\alpha_1, \alpha_2, \ldots, \alpha_n \in \mathbb{R}$

$$
\forall i \in [ n ], \forall \{a_1, a_2, \ldots, a_n\} \subseteq \mathbb{R}^d \space s.t. \space x_i \neq \sum_{j=1}^{n}{a_j x_j}
$$
