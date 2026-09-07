---
date: '2025-08-21'
description: constrained optimization and softmax identities for attention
id: notes
modified: 2026-09-07 13:56:24 GMT-04:00
tags:
  - workshop
title: supplement to 0[dot]200
---

supports:

- [[lectures/2/afp|attention from first principles]]
- [[lectures/2/convexity|convexity of attention]]
- [[thoughts/Attention]]
- [[thoughts/mechanistic interpretability]]

## equality constraints

Consider

$$
\min_x f(x)\qquad\text{subject to}\qquad h(x)=0,
$$

where $h:\mathbb{R}^n\to\mathbb{R}^m$ is differentiable. The Lagrangian is

$$
\mathcal{L}(x,\nu)=f(x)+\nu^\top h(x).
$$

If $x^*$ is a local optimum and the rows of $Dh(x^*)$ are linearly independent, then some multiplier $\nu^*$ satisfies

$$
\nabla f(x^*)+Dh(x^*)^\top\nu^*=0,
\qquad
h(x^*)=0.
$$

The first equation says that the objective gradient lies in the span of the active constraint normals. The rank condition matters: without a constraint qualification, an optimum need not admit such a multiplier.

## KKT conditions

For a differentiable convex program

$$
\min_x f(x)
\quad\text{subject to}\quad
g_i(x)\leq0,
\qquad
Ax=b,
$$

assume each $g_i$ is convex. The Lagrangian is

$$
\mathcal{L}(x,\lambda,\nu)
=f(x)+\sum_i\lambda_i g_i(x)+\nu^\top(Ax-b).
$$

The Karush-Kuhn-Tucker conditions are

$$
g_i(x^*)\leq0,
\qquad
Ax^*=b,
\qquad
\lambda_i^*\geq0,
$$

$$
\lambda_i^*g_i(x^*)=0,
$$

$$
\nabla f(x^*)+\sum_i\lambda_i^*\nabla g_i(x^*)+A^\top\nu^*=0.
$$

For this convex problem, KKT satisfaction is sufficient for optimality. Under Slater's condition, any attained finite optimum also has KKT multipliers. This is the assumption hidden by the loose phrase "KKT is necessary and sufficient."

## entropy and log-sum-exp

For $p$ in the probability simplex $\Delta_n$,

$$
H(p)=-\sum_i p_i\log p_i
$$

is concave. Its negative is strictly convex on $\Delta_n$. For $T>0$, define the scaled log-sum-exp

$$
\operatorname{LSE}_T(z)=T\log\sum_i e^{z_i/T}.
$$

Its variational form is

$$
\operatorname{LSE}_T(z)
=\max_{p\in\Delta_n}\left\{z^\top p+T H(p)\right\}.
$$

The unique maximizer and the gradient are the same probability vector:

$$
p^*=\operatorname{softmax}(z/T)
=\nabla\operatorname{LSE}_T(z).
$$

This identity keeps the temperature factors in one place. Writing $\log\sum_i e^{z_i/T}$ without the leading $T$ multiplies its gradient by $1/T$. [@gao2018propertiessoftmaxfunctionapplication; @blondel2019fenchelyoung; @blondel2020learningfenchelyounglosses]

## curvature and invariance

With $p=\operatorname{softmax}(z/T)$,

$$
\nabla^2\operatorname{LSE}_T(z)
=\frac1T\left(\operatorname{Diag}(p)-pp^\top\right).
$$

The matrix is positive semidefinite. It is the covariance matrix of a one-hot categorical vector divided by $T$. Its rows sum to zero, so

$$
\nabla^2\operatorname{LSE}_T(z)\mathbf1=0.
$$

This matches the shift invariance

$$
\operatorname{softmax}\!\left(\frac{z+c\mathbf1}{T}\right)
=\operatorname{softmax}(z/T).
$$

In Euclidean norm,

$$
\left\|\nabla^2\operatorname{LSE}_T(z)\right\|_2\leq\frac{1}{2T}.
$$

Thus the temperature-scaled softmax is $1/(2T)$-Lipschitz as a map from logits to probabilities. The bound controls only the weights; bounding value aggregation or later network blocks requires separate estimates. [@nair2025softmaxhalflipschitz]

For numerical evaluation, subtract $m=\max_i z_i$ before exponentiation:

$$
\operatorname{LSE}_T(z)
=m+T\log\sum_i e^{(z_i-m)/T}.
$$

The shift preserves the softmax and prevents an avoidable overflow.
