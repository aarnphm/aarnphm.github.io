---
id: Stochastic gradient descent
tags:
  - sfwr4ml3
  - ml
date: "2024-11-11"
description: gradient descent but with smoothness properties (differentiable or sub-differentiable)
modified: 2025-01-29 07:48:58 GMT-05:00
permalinks:
  - /SGD
title: Stochastic gradient descent
---

See also [[thoughts/university/twenty-three-twenty-four/compsci-4x03/A4|that numerical assignment on ODEs and GD]], [[thoughts/PyTorch#SGD|SGD implementation in PyTorch]]

```pseudo
\begin{algorithm}
\caption{Stochastic Gradient Descent (SGD) update}
\begin{algorithmic}
\Require Learning rate schedule $\{\epsilon_1, \epsilon_2, \dots\}$
\Require Initial parameter $\theta$
\State $k \gets 1$
\While{stopping criterion not met}
    \State Sample a minibatch of $m$ examples from the training set $\{x^{(1)}, \dots, x^{(m)}\}$ with corresponding targets $y^{(i)}$.
    \State Compute gradient estimate: $\hat{g} \gets \frac{1}{m} \nabla_{\theta} \sum_{i} L\bigl(f(x^{(i)};\theta), y^{(i)}\bigr)$
    \State Apply update: $\theta \gets \theta - \epsilon_k \hat{g}$
    \State $k \gets k + 1$
\EndWhile
\end{algorithmic}
\end{algorithm}
```
