---
date: '2025-01-29'
description: Training a student on teacher outputs, with the soft-target loss and its high-temperature limit.
id: knowledge distillation
modified: 2026-09-17 09:05:23 GMT-04:00
tags:
  - ml
title: knowledge distillation
---

_Knowledge distillation trains a student model on a teacher's outputs. A smaller student can reduce the computation needed at inference_ [@hinton2015distillingknowledgeneuralnetwork].

Distillation is a [[thoughts/Machine learning|training method]]; the student's architecture determines which parameters run. Matching the teacher's outputs gives no evidence that the teacher's parameters were [[thoughts/Attribution parameter decomposition|unused]].

## conceptually

For one input with $N$ classes, let $v_i$ be teacher logits and $z_i$ student logits. Apply [[thoughts/optimization#softmax|softmax]] at the same temperature $T > 0$:

$$
p_i = \frac{e^{v_i/T}}{\sum_{j=1}^{N} e^{v_j/T}},
\qquad
q_i = \frac{e^{z_i/T}}{\sum_{j=1}^{N} e^{z_j/T}}. \tag{1}
$$

For example, targets $(0.6, 0.3, 0.1)$ and $(0.6, 0.1, 0.3)$ share a top class. Keeping only that label discards which alternative the teacher considers more plausible.

With the teacher fixed, minimize cross-entropy on these soft targets:

$$
C_T = -\sum_{i=1}^{N} p_i \log q_i,
\qquad
\frac{\partial C_T}{\partial z_i} = \frac{q_i - p_i}{T}. \tag{2}
$$

Higher temperatures soften the distributions. The gradient shrinks approximately as $T^{-2}$ in the high-temperature limit.[^temperature-approx] With a known class label $y$, combine this loss with ordinary supervised training:

$$
\mathcal{L}
= \alpha T^2 C_T + (1-\alpha)(-\log q_y^{(1)}),
\qquad 0 \leq \alpha \leq 1,
$$

where $q_y^{(1)}$ uses temperature $1$. The $T^2$ factor keeps the soft-target gradient's scale roughly stable when changing $T$ [@hinton2015distillingknowledgeneuralnetwork].

## generated responses

[[thoughts/DeepSeek#Distill|R1 distillation]] used supervised fine-tuning of Qwen and Llama models on $800{,}000$ curated examples. Those students learned from response sequences. The report describes no reinforcement-learning stage for the released distill variants [@deepseekai2025deepseekr1incentivizingreasoningcapability, §2.4].

This distinction matters when reproducing the method: response training requires the selected text, while the soft-target loss above requires the teacher's probability distribution over the same classes as the student.

[^temperature-approx]:
    For logits small relative to $T$, use $e^{z_i/T} \approx 1 + z_i/T$ in both numerator and denominator:

    $$
    \frac{\partial C_T}{\partial z_i}
    \approx \frac{1}{T}\left(
    \frac{1 + z_i/T}{N + \sum_j z_j/T}
    - \frac{1 + v_i/T}{N + \sum_j v_j/T}
    \right). \tag{3}
    $$

    Subtracting each model's mean logit leaves its softmax unchanged. With $\sum_j z_j = \sum_j v_j = 0$:

    $$
    \frac{\partial C_T}{\partial z_i}
    \approx \frac{z_i-v_i}{NT^2}. \tag{4}
    $$

    Thus high-temperature distillation approaches squared-error matching of the centered logits, up to a scale factor.
