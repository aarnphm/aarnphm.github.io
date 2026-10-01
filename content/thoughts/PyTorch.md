---
date: '2024-11-11'
description: tidbits
id: PyTorch
modified: 2026-10-01 09:15:44 GMT-04:00
tags:
  - ml
  - framework
title: PyTorch
---

see also: [unstable docs](https://pytorch.org/docs/main/)

A fixed logit can receive different probabilities in sequences of different lengths. [Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html) normalizes over every score:

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}.
$$

This puts the same maximum at the start of two random score vectors. Their denominators differ.

```python title="qk_score.py"
import torch

torch.manual_seed(0)
qk_scores_short = torch.randn(2048)
qk_scores_long = torch.randn(128000)

max_v = torch.max(qk_scores_short.max(), qk_scores_long.max())
qk_scores_short[0] = max_v
qk_scores_long[0] = max_v
print(qk_scores_short.softmax(0)[0], qk_scores_long.softmax(0)[0])
```

## `MultiMarginLoss`

[`MultiMarginLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiMarginLoss.html) compares the target class score with each competing score. For one example, let $x \in \mathbb{R}^C$ and let $y \in \{0,\ldots,C-1\}$ be the target class index. With no class weights,

$$
\ell(x,y) = \frac{1}{C}\sum_{i \ne y}\max(0, m - x_y + x_i)^p,
\qquad p \in \{1,2\}.
$$

A competing class contributes zero once the target score exceeds it by at least the margin $m$. The default uses $m=1$, $p=1$, and averages the per-example losses over the batch. The division by $C$ still includes the target class in the class count, even though its term is excluded from the sum.

## `SGD`

PyTorch's [SGD implementation](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) applies the `maximize` sign to the objective gradient before adding weight decay. The parameter update then subtracts the resulting direction in both modes. This keeps decay pointing toward zero.

For [[thoughts/Nesterov momentum]], PyTorch cites [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf). Its momentum buffer is initialized to the first gradient. Dampening starts at the second update; Nesterov requires positive momentum and zero dampening.

```pseudo
\begin{algorithm}
\caption{SGD in PyTorch}
\begin{algorithmic}
\State \textbf{input:} $\gamma$ (lr), $\theta_0$ (params), $f(\theta)$ (objective), $\lambda$ (weight decay),
\State $\mu$ (momentum), $\tau$ (dampening), nesterov, maximize
\For{$t = 1$ to $...$}
    \State $g_t \gets \nabla_\theta f_t(\theta_{t-1})$
    \If{$\text{maximize}$}
        \State $g_t \gets -g_t$
    \EndIf
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
    \State $\theta_t \gets \theta_{t-1} - \gamma g_t$
\EndFor
\State \textbf{return} $\theta_t$
\end{algorithmic}
\end{algorithm}
```

## [[thoughts/knowledge distillation]]

CIFAR-10 examples from the [PyTorch distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html). These are excerpts: the loaders, trained teacher, and model instances must be set up before running the training loops.

![[thoughts/scripts/distill.py]]

### Cosine loss minimisation run

The experiment asks whether a trained teacher's hidden [[thoughts/representations]] help the student classify images. The teacher stays fixed. The student minimizes a weighted sum of classification loss and cosine loss between their hidden activations.

[`CosineEmbeddingLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html) compares equal-length vectors in their current coordinate order. It encourages aligned directions. It does not copy weights or find a permutation of the teacher's features:

$$
\text{loss}(x,y) = \begin{cases}
1 - \cos(x_1, x_2), & \text{if } y = 1 \\
\max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1
\end{cases}
$$

Here the loss target is $y=1$ for every teacher/student pair, independently of the image's class label. Positive rescaling leaves the cosine unchanged, so this objective does not require equal activation magnitudes.

The teacher produces $2048$ flattened features. Average pooling reduces these to $1024$, matching the student's $1024$ features.[^internal] That makes the loss computable; whether this alignment helps classification still needs a held-out comparison. The cosine term updates the student's feature extractor, while cross-entropy also trains its classifier.

[^internal]: Both modified models return logits and the hidden representation:

    ```python
    sample_input = torch.randn(128, 3, 32, 32).to(
      device
    )  # Batch size: 128, Channels: 3, Image size: 32x32
    logits, hidden_representation = modified_nn_light(sample_input)

    print('Student logits shape:', logits.shape)  # batch_size x total_classes
    print(
      'Student hidden representation shape:', hidden_representation.shape
    )  # batch_size x hidden_representation_size

    logits, hidden_representation = modified_nn_deep(sample_input)

    print('Teacher logits shape:', logits.shape)  # batch_size x total_classes
    print(
      'Teacher hidden representation shape:', hidden_representation.shape
    )  # batch_size x hidden_representation_size
    ```

![[thoughts/scripts/modified_deep_cosine.py]]
