---
id: Stochastic gradient descent
tags:
  - sfwr4ml3
  - ml
date: "2024-11-11"
modified: 2024-12-10 23:12:52 GMT-05:00
permalinks:
  - /SGD
title: Stochastic gradient descent
---

See also [[thoughts/university/twenty-three-twenty-four/compsci-4x03/A4|SGD and ODEs]]

[[thoughts/Nesterov momentum]] is based on [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf)

```pseudo
\begin{algorithm}
\caption{SGD}
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
