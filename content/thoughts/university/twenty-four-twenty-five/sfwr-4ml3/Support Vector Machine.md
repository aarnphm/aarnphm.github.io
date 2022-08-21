---
id: SVM
tags:
  - sfwr4ml3
date: "2024-11-11"
modified: "2024-11-11"
title: Support Vector Machine
---

idea: maximizes margin and more robus to "perturbations"

Eucledian distance between two points $x$ and the hyperplan parametrized by $W$ is:

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
