---
id: Kullback-Leibler divergence
tags:
  - math
  - probability
date: "2024-12-12"
description: also called relative entropy or I-divergence
modified: 2024-12-14 08:01:48 GMT-05:00
title: Kullback-Leibler divergence
---

_denoted as_ $D_{\text{KL}}(P \parallel Q)$

> [!math] definition
>
> The ==statistical distance== between a model probability distribution $Q$ difference from a true probability distribution $P$:
>
> $$
> D_{\text{KL}}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log (\frac{P(x)}{Q(x)})
> $$

alternative form:

$$
\begin{aligned}
\text{KL}(p \parallel q) &= E_{x \sim p}(\log \frac{p(x)}{q(x)}) \\
&= \int_x P(x) \log \frac{p(x)}{q(x)} dx
\end{aligned}
$$
