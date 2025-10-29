---
date: "2024-10-21"
id: Singular Value Decomposition
modified: 2025-10-29 02:15:35 GMT-04:00
tags:
  - ml
title: Singular Value Decomposition
---

![[https://www.youtube.com/watch?v=nbBvuuNVfco&ab_channel=SteveBrunton]]

$$
\begin{aligned}
X &= \begin{bmatrix}
1 & 1 & \cdots & 1 \\
x_1 & x_2 & \cdots & x_m \\
\vdots & \vdots & \vdots & \vdots \\
1 & x_1 & \cdots & x_m
\end{bmatrix} = U \Sigma V^T \\
&= \begin{bmatrix}
1 & 1 & \cdots & 1 \\
u_{1} & u_{2} & \cdots & u_n \\
\vdots & \vdots & \vdots & \vdots \\
1 & 1 & \cdots & 1
\end{bmatrix} \begin{bmatrix}
\sigma_1 & \cdots & \cdots & \cdots \\
\vdots & \sigma_2 & \cdots & \cdots \\
\vdots & \cdots & \ddots & \cdots \\
\vdots & \cdots & \cdots & \sigma_m \\
0 & 0 & 0 & 0 \\
\end{bmatrix} {\begin{bmatrix}
\vdots & \vdots & \vdots & \vdots \\
v_{1} & v_{2} & \cdots & v_n \\
\vdots & \vdots & \vdots & \vdots
\end{bmatrix}}^T
\\
\\
x_k &\in \mathbb{R}^n \\
\\
\text{U, V } &: \text{unitary matrices} \\
\Sigma &: \text{diagonal matrix}
\end{aligned}
$$

where $\begin{bmatrix} 1 \\ u_{1} \\ \vdots \\ 1 \end{bmatrix}$ are "eigen-faces"

$U$ is orthonormal, meaning:

$$
\begin{aligned}
U U^T &= U^T U = \mathbb{I}_{n \times n} \\
V V^T &= V^T V = \mathbb{I}_{m \times m} \\
\\
\Sigma &: \text{diagonal} \quad \sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_m \geq 0
\end{aligned}
$$
