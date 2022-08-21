---
date: "2024-11-11"
id: Support Vector Machine
modified: 2025-10-29 02:16:08 GMT-04:00
tags:
  - sfwr4ml3
title: Support Vector Machine
---

idea: maximises margin and more robust to "perturbations"

Euclidean distance between two points $x$ and the hyperplane parametrised by $W$ is:

$$
\frac{\mid W^T x + b \mid }{\|W\|_2}
$$

> Assuming $\| W \|_2=1$ then the distance is $\mid W^T x + b \mid$

> [!abstract] regularization
>
> SVMs are good for high-dimensional data

We can probably use a solver, or [[thoughts/gradient descent]]

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

_this is the version with bias_

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

> it assumes that training set is linearly separable

## soft-margin SVM

_can be applied even if the training set is not linearly separable_

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

Equivalent form of soft-margin SVM:

$$
\begin{aligned}
\min_{w} &(\lambda \|w\|^2 + L_S^{\text{hinge}}(w)) \\[8pt]
L_{S}^{\text{hinge}}(w) &= \frac{1}{m} \sum_{i=1}^{m} \max{(\{0, 1 - y \langle w, x_i \rangle\})}
\end{aligned}
$$

## SVM with basis functions

$$
\min_{W} \frac{1}{n} \sum \max \{0, 1 - y^i \langle w, \phi(x^i) \rangle\} + \lambda \|w\|^2_2
$$

> $\phi(x^i)$ can be high-dimensional

## representor theorem

$$
W^{*} = \argmin_{W} \frac{1}{n} \sum \max \{0, 1- y^i \langle w, \phi (x^i) \rangle\} + \lambda \|w\|^2_2
$$

> [!abstract] theorem
>
> There are real values $a_{1},\ldots,a_{m}$ such that [^note1]
>
> $$
> W^{*} = \sum a_i \phi(x^i)
> $$

[^note1]: note that we can also write $a^T \phi$ where $\phi = [\phi(x^1),\ldots,\phi(x^n)]^T$

## kernelized SVM

from [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Support Vector Machine#representor theorem]], we have the kernel:

$$
K(x,z) = \langle \phi(x), \phi(z) \rangle
$$

## drawbacks

- prediction-time complexity
- need to store all training data
- Dealing with $\mathbf{K}_{n \times n}$
- choice of kernel, in which is tricky and pretty heuristic sometimes.
