---
id: knowledge distillation
tags:
  - ml
date: "2025-01-29"
description: and improving smaller models performance through larger general systems.
modified: 2025-01-29 06:05:59 GMT-05:00
title: knowledge distillation
---

_a transfer [[thoughts/Machine learning|learning]] method to use quality data from larger system to improve specialist capabilities_ [@hinton2015distillingknowledgeneuralnetwork]. Even more popular now with the rise of [[thoughts/DeepSeek#Distill|R1]]

Unlike Mixture of Expert, these specialist will only includes activations that is related to that specific fields. Conceptually, most of parameters within a neural networks are [[thoughts/Attribution parameter decomposition|unused]].

## conceptually

usually NN produces a class probabilities via [[thoughts/optimization#softmax]] layer that converts logits $z_i$ computed for each class into probability $q_i$ by comparing $z_i$ with other logits:

$$
q_i = \frac{\text{exp}(z_i/T)}{\sum_{j} \text{exp}(z_j/T)} \tag{1}
$$

where temperature $T$ is often set to 1.

We "distill" the knowledge by training the base specialist systems with a soft target distribution of the transfer set.

Each case in a transfer sets contributes to ==cross-entropy gradient== $dC/d z_i$ with respects to each logit $z_i$ of the distilled model.

The gradient given by the training done at temperature $T$ that gives soft target probabilities $p_i$ [^temperature-approx]:

$$
\frac{\partial C}{\partial z_i} = \frac{1}{T} (q_i - p_i) = \frac{1}{T} (\frac{e^{z_i/T}}{\sum_{j}  e^{z_j/T}} - \frac{e^{v_i/T} }{\sum_{j} e^{v_j/T}}) \tag{2}
$$

[^temperature-approx]:
    If the temperature is high compared to the magnitude of the logits, then we can approximate the following:

    $$
    \frac{\partial C}{\partial z_i} \approx \frac{1}{T} (\frac{1 + z_i/T}{N + \sum_{j} e^{z_j/T}} - \frac{1 + v_i/T}{N + \sum_{j} e^{v_j/T}}) \tag{3}
    $$

    We further simplified Eq.3 assuming logits have been zero-mean separately per transfer case:

    $$
    \frac{\partial C}{\partial z_i} \approx \frac{1}{NT^2} (z_i -v_i) \tag{4}
    $$
