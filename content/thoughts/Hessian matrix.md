---
id: Hessian matrix
tags:
  - math
description: second partial derivatives of an objective function.
date: "2025-08-28"
modified: 2025-08-30 22:51:34 GMT-04:00
title: Hessian matrix
---

> square matrix of second-order partial derivatives of a scalar-valued function: $D^{2}$ or $\nabla \nabla, \nabla \otimes \nabla, {\nabla}^{2}$

> [!abstract] definition
>
> Suppose a function $f : \mathbb{R}^{n} \longrightarrow \mathbb{R}$ is a function taking input vector $\textbf{x} \in \mathbb{R}^{n}$ and output a scalar $f(\textbf{x}) \in \mathbb{R}$.
> If the second-order partial derivatives of $f$ exists, then the Hessian matrix $\textbf{H}$ of $f$ follows a square matrix $n \times n$:
>
> $$
> H_f = \begin{bmatrix}
> \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
> \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
> \vdots & \vdots & \ddots & \vdots \\
> \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
> \end{bmatrix}
> $$
>
> That is, the entry of the $i^{\text{th}}$ row and $j^{\text{th}}$ column is
>
> $$
> (\textbf{H}_f)_{i,j} = \frac{{\partial}^{2} f}{\partial x_{i} \partial x_{j}}
> $$

intuition: functional determinants
