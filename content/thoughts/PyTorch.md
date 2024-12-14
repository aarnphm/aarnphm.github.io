---
id: PyTorch
tags:
  - ml
  - framework
date: "2024-11-11"
description: tidbits from PyTorch
modified: 2024-12-14 02:41:03 GMT-05:00
title: PyTorch
---

see also: [unstable docs](https://pytorch.org/docs/main/)

## MultiMarginLoss

Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss) between input $x$
(a 2D mini-batch `Tensor`) and output $y$ (which is a 1D tensor of target class indices, $0 \le y \le \text{x}.\text{size}(1) -1$):

For each mini-batch sample, loss in terms of 1D input $x$ and output $y$ is:

$$
\text{loss}(x,y) = \frac{\sum_{i} \max{0, \text{margin} - x[y] + x[i]}^p}{x.\text{size}(0)}
\\
\because i \in \{0, \ldots x.\text{size}(0)-1\} \text{ and } i \neq y
$$

## SGD

[[thoughts/Nesterov momentum]] is based on [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf)

```pseudo
\begin{algorithm}
\caption{SGD in PyTorch}
\begin{algorithmic}
\State \textbf{input:} $\gamma$ (lr), $\theta_0$ (params), $f(\theta)$ (objective), $\lambda$ (weight decay),
\State $\mu$ (momentum), $\tau$ (dampening), nesterov, maximize
\For{$t = 1$ to $...$}
    \State $g_t \gets \nabla_\theta f_t(\theta_{t-1})$
    \If{$\lambda \neq 0$}
        \State $g_t \gets g_t + \lambda\theta_{t-1}$
    \EndIf
    \If{$\mu \neq 0$}
        \If{$t > 1$}
            \State $b_t \gets \mu b_{t-1} + (1-\tau)g_t$
        \Else
            \State $b_t \gets g_t$
        \EndIf
        \If{$\text{nesterov}$}
            \State $g_t \gets g_t + \mu b_t$
        \Else
            \State $g_t \gets b_t$
        \EndIf
    \EndIf
    \If{$\text{maximize}$}
        \State $\theta_t \gets \theta_{t-1} + \gamma g_t$
    \Else
        \State $\theta_t \gets \theta_{t-1} - \gamma g_t$
    \EndIf
\EndFor
\State \textbf{return} $\theta_t$
\end{algorithmic}
\end{algorithm}
```
