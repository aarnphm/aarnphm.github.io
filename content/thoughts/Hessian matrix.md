---
date: '2025-08-28'
description: the local quadratic part of a scalar function
id: Hessian matrix
modified: 2026-06-05 15:08:29 GMT-04:00
tags:
  - math
title: Hessian matrix
---

For a twice differentiable scalar function $f:\mathbb{R}^n\to\mathbb{R}$, the Hessian is the derivative of the gradient:

$$
H_f(\mathbf{x})
= D(\nabla f)(\mathbf{x})
= \left[
\frac{\partial^2 f}
{\partial x_i\,\partial x_j}(\mathbf{x})
\right]_{i,j=1}^{n}
$$

If the second partial derivatives are continuous near $\mathbf{x}$, then the mixed partials agree and $H_f(\mathbf{x})$ is symmetric.

## local quadratic model

When $f$ is $C^2$ near $\mathbf{x}$, the gradient gives the local linear change and the Hessian gives the next term:

$$
f(\mathbf{x}+\mathbf{h})
= f(\mathbf{x})
+ \nabla f(\mathbf{x})^{\mathsf T}\mathbf{h}
+ \frac{1}{2}\mathbf{h}^{\mathsf T}H_f(\mathbf{x})\mathbf{h}
+ o\left(\lVert\mathbf{h}\rVert^2\right)
$$

For a unit direction $\mathbf{u}$,

$$
\left.\frac{d^2}{dt^2}f(\mathbf{x}+t\mathbf{u})\right|_{t=0}
= \mathbf{u}^{\mathsf T}H_f(\mathbf{x})\mathbf{u}
$$

is the second directional derivative. It records the local curvature along that direction.

## stationary points

Suppose $\nabla f(\mathbf{x}_*)=0$.

- If $H_f(\mathbf{x}_*)$ is positive definite, then $\mathbf{x}_*$ is a strict local minimum.
- If $H_f(\mathbf{x}_*)$ is negative definite, then $\mathbf{x}_*$ is a strict local maximum.
- If $H_f(\mathbf{x}_*)$ is indefinite, then $\mathbf{x}_*$ is a saddle point.
- A semidefinite Hessian leaves the second derivative test inconclusive.

On an open convex domain, a twice differentiable function is convex exactly when its Hessian is positive semidefinite everywhere. This is the second-order condition used in [convex optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).

## example

For

$$
f(x,y)=3x^2+2xy+y^2,
$$

the Hessian is constant:

$$
H_f
= \begin{bmatrix}
6 & 2 \\
2 & 2
\end{bmatrix}
$$

Its eigenvalues are $4-2\sqrt{2}$ and $4+2\sqrt{2}$, both positive. The function therefore curves upward in every nonzero direction and is strictly convex.

At a point, the Hessian matrix represents a bilinear form in the chosen coordinates. Its determinant is one summary of that matrix. A functional determinant is a different object defined for operators on function spaces.

source: [MIT 18.S096 notes on Hessian matrices](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/resources/mit18_s096iap23_lec12_pdf/)
