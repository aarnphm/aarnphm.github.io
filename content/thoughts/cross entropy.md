---
id: cross entropy
tags:
  - ml
  - probability
description: criterion for finding difference between input logits and targets.
date: "2025-09-14"
modified: 2025-09-15 23:47:49 GMT-04:00
title: cross entropy
---

see also: [[thoughts/Entropy]], [[thoughts/Kullback-Leibler divergence]], [[thoughts/Maximum likelihood estimation]], [[thoughts/Logistic regression]], [[thoughts/Negative log-likelihood]] · [[thoughts/Perplexity]]

measures how well a predictive distribution $q(y\mid x)$ matches the true distribution $p(y\mid x)$.

- Lower is better; zero only when $q = p$.
- For supervised classification with one correct class per example, $p(y\mid x)$ is a one‑hot vector. The cross‑entropy per sample reduces to $-\log q(y^*\mid x)$ where $y^*$ is the true class.
- Information‑theoretic view: $H(p, q) = H(p) + \mathrm{KL}(p \parallel q)$.
  - $H(p)$ is constant w.r.t. the model
    > minimizing cross‑entropy is equivalent to minimizing $\mathrm{KL}(p \parallel q)$.

### derivation (KL and CE)

$$
\begin{aligned}
H(p, q) &:= - \, \mathbb{E}_{y\sim p}[\log q(y)] \\
H(p) &:= - \, \mathbb{E}_{y\sim p}[\log p(y)] \\
\mathrm{KL}(p\Vert q)
&= \mathbb{E}_{y\sim p}\Big[ \log \tfrac{p(y)}{q(y)} \Big] \\
&= \mathbb{E}_{y\sim p}[\log p(y)] - \mathbb{E}_{y\sim p}[\log q(y)] \\
&= -H(p) + H(p, q) \\
\Rightarrow\quad H(p,q) &= H(p) + \mathrm{KL}(p\Vert q)
\end{aligned}
$$

## maximum likelihood

- Empirical risk with cross‑entropy equals the negative log‑likelihood (NLL) of the dataset under the model.
- Minimizing cross‑entropy thus performs [[thoughts/Maximum likelihood estimation]] (MLE).
- Derivation sketch for a dataset $\{(x_i, y_i)\}$ with categorical model $q_\theta(y\mid x)$:
  - Objective: minimize
    $$
    L(\theta) = -\frac{1}{N} \sum_i \log q_\theta(y_i \mid x_i)
    $$
  - This is the sample average of $-\mathbb{E}_{p_{\mathrm{data}}}[\log q_\theta(y\mid x)]$
    - i.e., cross‑entropy between $p_{\mathrm{data}}(y\mid x)$ and $q_\theta(y\mid x)$.
  - For softmax classifiers with logits $z = f_\theta(x)$, $q_\theta(y{=}k\mid x) = \mathrm{softmax}(z)_k$.
    - [[thoughts/Vector calculus#gradient|gradient]] w.r.t. logits is $\partial L/\partial z = \mathrm{softmax}(z) - \mathrm{one\_hot}(y)$.

## multiclass cross‑entropy

i.e [[thoughts/optimization#softmax]]

- Model: logits $z \in \mathbb{R}^C$, probabilities $p = \mathrm{softmax}(z)$.
- Loss per example with hard labels $y \in \{0,\ldots,C-1\}$: $L = -\log p_y$.
- With soft/weighted targets $t \in \Delta^C$ (non‑negative, sum to 1):
  - $L = -\sum_k t_k \log p_k$.
  - includes label smoothing where $t = (1-\varepsilon)\cdot\mathrm{one\_hot}(y) + \varepsilon/C$.

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^C e^{z_j}},\quad
L_{\text{hard}} = -\log p_y,\quad
L_{\text{soft}} = - \sum_{k=1}^C t_k \log p_k.
$$

Gradient w.r.t. logits $z$:

$$
\frac{\partial L}{\partial z} = \mathrm{softmax}(z) - \mathrm{one\_hot}(y).
$$

### derivation (softmax gradient)

Let $L = -\log \mathrm{softmax}(z)_y = - z_y + \log \sum_{j} e^{z_j}$. For each $k$,

$$
\begin{aligned}
\frac{\partial L}{\partial z_k}
&= -\,\mathbf{1}[k{=}y] + \frac{1}{\sum_j e^{z_j}}\, e^{z_k} \\
&= -\,\mathbf{1}[k{=}y] + \mathrm{softmax}(z)_k \\
&= \mathrm{softmax}(z)_k - \mathbf{1}[k{=}y].
\end{aligned}
$$

In vector form: $\nabla_z L = \mathrm{softmax}(z) - \mathrm{one\_hot}(y)$.

### binary vs. multiclass vs. multilabel

- Binary single‑label: use binary cross‑entropy (logistic regression).
  - With logits $s$, probability $\sigma(s)$:

    | quantity | expression                                                                   |
    | -------- | ---------------------------------------------------------------------------- |
    | labels   | $y \in \{0,1\}$                                                              |
    | sigmoid  | $\sigma(s) = \dfrac{1}{1 + e^{-s}}$                                          |
    | loss     | $\ell(s, y) = -\big[y \log \sigma(s) + (1-y) \log\big(1-\sigma(s)\big)\big]$ |

- Multiclass single‑label (exactly one class):
  - use softmax cross‑entropy.
- Multilabel (independent classes, several can be 1):
  - use per‑class binary cross‑entropy (`BCEWithLogitsLoss`), not softmax CE.

[[thoughts/PyTorch]] usage:

- `nn.CrossEntropyLoss` expects raw logits and integer class indices. It internally applies `LogSoftmax` + `NLLLoss`.
- Shapes:
  - Input `logits`: `(N, C)` or `(N, C, d1, d2, ...)` for 2D/3D tasks.
  - Target `y`: `(N)` or `(N, d1, d2, ...)` with `dtype=torch.long` containing class indices `0..C-1`.
- Key args:
  - `weight`: tensor of shape `(C,)` for class weighting (helps class imbalance).
  - `ignore_index`: integer label to exclude (e.g., padding in NLP/segmentation).
  - `reduction`: `'mean' | 'sum' | 'none'`.
    - `'mean'` averages by batch (or sum of weights if provided).
  - `label_smoothing`: float in `[0, 1)`.
    - Smooths targets and can improve calibration/generalization.
- Soft targets: Recent PyTorch versions support probabilistic targets (same shape as logits).
  - For older versions, use `torch.nn.KLDivLoss` with `log_softmax`.

### reductions

- `reduction='none'`: returns per-element losses.
  - for 2D logits `(N, C)`, shape is `(N)`
  - for segmentation `(N, C, d1, d2, ...)`, shape is `(N, d1, d2, ...)`.
- `reduction='sum'`: sums over all non-ignored elements (and spatial dims if present).
- `reduction='mean'` (default): averages over non-ignored elements.
  - With `weight`, computes
    $$
    \frac{\sum_i w_{y_i} \, \ell_i}{\sum_i w_{y_i}}
    $$
    where $\ell_i$ is the un-reduced loss of sample/pixel $i$ and $w_{y_i}$ is the class weight.

> To average per-sample but not across batch, use `reduction='none'` and then reduce across spatial dims, e.g. `loss = loss_per_elem.flatten(1).mean(dim=1)`.

Numeric example (weighted mean)

```python
import torch

logits = torch.tensor([
  [2.0, 0.0],  # sample 0
  [0.2, 0.0],
])  # sample 1
y = torch.tensor([0, 1])  # true classes
w = torch.tensor([1.0, 2.0])  # class weights (for 0 and 1)

# Framework result (weighted mean):
crit = torch.nn.CrossEntropyLoss(weight=w, reduction='mean')
loss_fw = crit(logits, y)

# Manual check
p = torch.softmax(logits, dim=1)
ell = -torch.log(p[range(2), y])  # per-sample losses
weights = w[y]
loss_manual = (weights * ell).sum() / weights.sum()

print(loss_fw.item(), loss_manual.item())  # nearly equal
```

```python
import torch
import torch.nn as nn

# Multiclass CE with integer targets
logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.2, 0.0]])  # shape (N=2, C=3)
y = torch.tensor([0, 2])  # correct classes
loss = nn.CrossEntropyLoss()(logits, y)  # includes log_softmax + NLL

# With class weights and label smoothing
weights = torch.tensor([1.0, 2.0, 0.5])
crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
loss_ws = crit(logits, y)

# Soft targets (probability distributions)
targets = torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.2, 0.8]])
loss_soft = nn.functional.cross_entropy(logits, targets)

# Multilabel example uses BCEWithLogits (NOT CrossEntropy)
ml_logits = torch.tensor([[1.2, -0.7, 0.3]])  # per-class logits
ml_targets = torch.tensor([[1.0, 0.0, 1.0]])  # independent labels
bce = nn.BCEWithLogitsLoss()
ml_loss = bce(ml_logits, ml_targets)
```

### pitfalls

- Pass logits to `CrossEntropyLoss`.
- Do not apply `softmax` beforehand; doing so causes numerical issues and incorrect gradients.
- For segmentation or sequence tasks, ensure target tensor shape matches logits' spatial dims and uses class indices, not one‑hot.
- Use `ignore_index` to mask padding/void classes; verify the effective denominator when using `'mean'` reduction with weights.
- For heavy class imbalance, combine `weight` with sampling strategies or consider focal loss.
- Numerical stability is handled internally via `logsumexp`, but keep inputs in reasonable ranges and avoid manual `log(softmax(.))`.

### intuition

- Proper scoring rule: Cross‑entropy is a strictly proper scoring rule, incentivizing truthful probability estimates.
- Log‑loss emphasizes confident mistakes: assigning tiny probability to the true class incurs a large penalty, discouraging over‑confident wrong predictions.
- As data $\to \infty$, minimizing cross‑entropy recovers the data‑generating conditional distribution under correct model specification (consistency of MLE).
