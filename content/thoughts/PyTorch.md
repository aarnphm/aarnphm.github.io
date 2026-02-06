---
date: '2024-11-11'
description: tidbits
id: PyTorch
modified: 2025-11-12 20:26:51 GMT-05:00
tags:
  - ml
  - framework
title: PyTorch
---

see also: [unstable docs](https://pytorch.org/docs/main/)

```python title="qk_score.py"
import torch

qk_scores_short = torch.randn(2048)
qk_scores_long = torch.randn(128000)

max_v = torch.max(qk_scores_short.max(), qk_scores_long.max())
qk_scores_short[0] = max_val
qk_scores_long[0] = max_val
qk_scores_short.softmax(0)[0], qk_scores_long.softmax(0)[0]
```

## `MultiMarginLoss`

Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss) between input $x$
(a 2D mini-batch `Tensor`) and output $y$ (which is a 1D tensor of target class indices, $0 \le y \le \text{x}.\text{size}(1) -1$):

For each mini-batch sample, loss in terms of 1D input $x$ and output $y$ is:

$$
\text{loss}(x,y) = \frac{\sum_{i} \max{0, \text{margin} - x[y] + x[i]}^p}{x.\text{size}(0)}
\\
\because i \in \{0, \ldots x.\text{size}(0)-1\} \text{ and } i \neq y
$$

## `SGD`

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

## [[thoughts/knowledge distillation]]

examples on CIFAR

![[thoughts/distill.py]]

### Cosine loss minimisation run

assumption: the teacher network will have a better internal [[thoughts/representations]] comparing to student's weights. Thus we need to artificially push the students' weight to "mimic" the teachers' weights.

We will apply `CosineEmbeddingLoss` such that students' internal representation would be a permutation of the teacher's:

$$
\text{loss}(x,y) = \begin{cases}
1 - \cos(x_1, x_2), & \text{if } y = 1 \\
\max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1
\end{cases}
$$

The updated loops as follow [^internal]:

[^internal]: Naturally, we have to update the hidden representation:

    ```python
    sample_input = torch.randn(128, 3, 32, 32).to(device) # Batch size: 128, Filters: 3, Image size: 32x32
    logits, hidden_representation = modified_nn_light(sample_input)

    print("Student logits shape:", logits.shape) # batch_size x total_classes
    print("Student hidden representation shape:", hidden_representation.shape) # batch_size x hidden_representation_size

    logits, hidden_representation = modified_nn_deep(sample_input)

    print("Teacher logits shape:", logits.shape) # batch_size x total_classes
    print("Teacher hidden representation shape:", hidden_representation.shape) # batch_size x hidden_representation_size
    ```

![[thoughts/modified_deep_cosine.py]]
