---
date: '2024-02-06'
description: counts the right-half-plane roots of a real polynomial from sign changes in the first column of its Routh array, which decides stability without solving for roots.
id: Routh-Hurwitz criterion
modified: 2026-06-05 15:08:23 GMT-04:00
tags:
  - seed
title: Routh-Hurwitz criterion
---

A test for the stability of a linear time-invariant (LTI) [[tags/sfwr3dx4|control system]] that never computes a root.

Take $\hat{G}(s) = \frac{\mathcal{N}(s)}{\mathcal{D}(s)}$ with $\mathcal{N}$ and $\mathcal{D}$ sharing no factor. The system is input-output (BIBO) stable exactly when every root of $\mathcal{D}(s)$ lies in the open left half-plane. If a factor cancels, the cancelled root drops out of the transfer function but still lives in the state, so internal stability needs the uncancelled $\mathcal{D}$.

> [!important] Criterion
> If no entry of the first column of the Routh array is zero, the number of roots of $\mathcal{D}(s)$ in the right half-plane equals the number of sign changes down that column. All roots lie in the open left half-plane if and only if the first column has one sign.

The array starts with the coefficients of $\mathcal{D}(s) = a_n s^n + a_{n-1}s^{n-1} + \cdots + a_0$, alternating between two rows:

$$
\begin{array}{c|cccc}
s^n & a_n & a_{n-2} & a_{n-4} & \cdots \\
s^{n-1} & a_{n-1} & a_{n-3} & a_{n-5} & \cdots
\end{array}
$$

Each later row comes from the two rows above it. Write $c_{k,j}$ for entry $j$ of row $k$:

$$
c_{k,j} = \frac{c_{k-1,1}\,c_{k-2,j+1} - c_{k-2,1}\,c_{k-1,j+1}}{c_{k-1,1}}
$$

Positive coefficients are necessary and not sufficient. For $s^3 + s^2 + s + 2$ the rows are $(1, 1)$, $(1, 2)$, then $\frac{1\cdot 1 - 1\cdot 2}{1} = -1$, then $2$. The first column $1, 1, -1, 2$ changes sign twice, so two roots sit in the right half-plane.

Working the cubic $a_3s^3 + a_2s^2 + a_1s + a_0$ with $a_3 > 0$ once gives a rule worth remembering: the system is stable if and only if every $a_i > 0$ and $a_2 a_1 > a_3 a_0$. The example above fails the second condition, since $1 \cdot 1 < 1 \cdot 2$.

A zero in the first column, or a whole row of zeros, breaks the division. The fixes for both cases are in [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/stability#Routh-Hurwitz criterion|stability]].
