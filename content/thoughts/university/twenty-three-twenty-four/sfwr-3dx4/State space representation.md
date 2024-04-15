---
id: State space representation
tags:
  - sfwr3dx4
date: "2024-01-24"
title: State space representation
---

See also [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/state_space.pdf|sides]]

> time-domain technique

$$
\begin{align}
\dot{x} &= Ax + Bu \\\
y &= Cx + Du
\end{align}
$$

- _Linearly independent_
- _State vector_: $x = [x_{1},x_{2},\ldots, x_{n}]^{T}$

## transfer function to a state space representation

### controller form

Given
$$
G(s) = \frac{\sum_{i=1}^{n-1}b_is^i + b_{0}}{s^n + \sum_{i=1}^{n-1}a_is^{i} + a_{0}} = \frac{Y(s)}{U(s)}
$$

We get _controller canonical state space_ form:

$$
\begin{aligned}
\dot{x}(t) &= \begin{bmatrix}
0 & 1 & 0 & \cdots & 0 & 0 \\\
0 & 0 & 1 & \cdots & 0 & 0 \\\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\\
0 & 0 & 0 & \cdots & 1 & 0 \\\
0 & 0 & 0 & \cdots & 0 & 1 \\\
-a_0 & -a_1 & -a_2 & \cdots & -a_{n-2} & -a_{n-1}
\end{bmatrix} x(t) + \begin{bmatrix}
0 \\\
0 \\\
\vdots \\\
0 \\\
0 \\\
1
\end{bmatrix} u(t) \\\
y(t) &= \begin{bmatrix}
b_0 & b_1 & \cdots & b_{n-2} & b_{n-1}
\end{bmatrix} x(t).
\end{aligned}
$$

### observer form

We get _observer canonical state space_ form:

$$
\begin{aligned} \dot{x}(t) &= \begin{bmatrix} -a_{n-1} & 1 & 0 & \cdots & 0 & 0 \\ -a_{n-2} & 0 & 1 & \cdots & 0 & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ -a_2 & 0 & 0 & \cdots & 1 & 0 \\ -a_1 & 0 & 0 & \cdots & 0 & 1 \\ -a_0 & 0 & 0 & \cdots & 0 & 0 \end{bmatrix} x(t) + \begin{bmatrix} b_{n-1} \\ b_{n-2} \\ \vdots \\ b_2 \\ b_1 \\ b_0 \end{bmatrix} u(t) \\ y(t) &= \begin{bmatrix} 1 & 0 & \cdots & 0 & 0 \end{bmatrix} x(t). \end{aligned}
$$