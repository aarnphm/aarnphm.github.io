---
id: Stochastic gradient descent
permalinks:
  - /SGD
tags:
  - sfwr4ml3
  - ml
description: gradient descent but with smoothness properties (differentiable or sub-differentiable)
date: "2024-11-11"
modified: 2025-08-28 07:32:04 GMT-04:00
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

Intuition: you can think of it as online [[thoughts/gradient descent]], where the true gradient $Q(w)$ is approximated by a gradient at a single sample[^step-size]:

$$
w \coloneqq w - \upeta \nabla Q_i(w)
$$

[^step-size]: $\upeta$ is often considered as step-size during a linear-regression training run.
