---
id: Support Vector Machine
tags:
  - sfwr4ml3
date: "2024-11-11"
modified: 2024-12-10 22:33:17 GMT-05:00
title: Support Vector Machine
---

idea: maximises margin and more robust to "perturbations"

Euclidean distance between two points $x$ and the hyperplane parametrised by $W$ is:

$$
\frac{\mid W^T x + b \mid }{\|W\|_2}
$$

> Assuming $\| W \|_2=1$ then the distance is $\mid W^T x + b \mid$

## maximum margin hyperplane

$W$ has $\gamma$ margin if

$$
\begin{aligned}
W^T x + b \ge \gamma \space &\forall \text{ blue x} \\
W^T x +b \le - \gamma \space &\forall \text{ red x}
\end{aligned}
$$

Margin:

$$
Z = \{(x^{i}, y^{i})\}_{i=1}^{n}, y \in \{-1, 1\}, \|W\|_2 = 1
$$

## hard-margin SVM

```pseudo
\begin{algorithm}
\caption{Hard-SVM}
\begin{algorithmic}
\REQUIRE Training set $(\mathbf{x}_1, y_1),\ldots,(\mathbf{x}_m, y_m)$
\STATE \textbf{solve:} $(w_{0},b_{0}) = \argmin\limits_{(w,b)} \|w\|^2 \text{ s.t } \forall i, y_{i}(\langle{w,x_i} \rangle + b) \ge 1$
\STATE \textbf{output:} $\hat{w} = \frac{w_0}{\|w_0\|}, \hat{b} = \frac{b_0}{\|w_0\|}$
\end{algorithmic}
\end{algorithm}
```

Note that this version is sensitive to outliers

## soft-margin SVM

```pseudo
\begin{algorithm}
\caption{Soft-SVM}
\begin{algorithmic}
\REQUIRE Input $(\mathbf{x}_1, y_1),\ldots,(\mathbf{x}_m, y_m)$
\STATE \textbf{parameter:} $\lambda > 0$
\STATE \textbf{solve:} $\min_{\mathbf{w}, b, \boldsymbol{\xi}}  \left( \lambda \|\mathbf{w}\|^2 + \frac{1}{m} \sum_{i=1}^m \xi_i \right)$
\STATE \textbf{s.t: } $\forall i, \quad y_i (\langle \mathbf{w}, \mathbf{x}_i \rangle + b) \geq 1 - \xi_i \quad \text{and} \quad \xi_i \geq 0$
\STATE \textbf{output:} $\mathbf{w}, b$
\end{algorithmic}
\end{algorithm}
```
