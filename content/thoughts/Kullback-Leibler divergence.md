---
id: Kullback-Leibler divergence
aliases:
  - kl divergence
tags:
  - math
  - probability
date: "2024-12-12"
description: also called relative entropy or I-divergence
modified: 2025-07-03 19:16:49 GMT-04:00
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

Alternative form [^discrete]:

[^discrete]: For _discrete probability distribution_ $P$ and $Q$ defined on the same sample space.

$$
\begin{aligned}
\text{KL}(p \parallel q) &= E_{x \sim p}(\log \frac{p(x)}{q(x)}) \\
&= \int_x P(x) \log \frac{p(x)}{q(x)} dx
\end{aligned}
$$

For relative entropy if $\forall x > 0, Q(x) = 0 \implies P(x) = 0$ _absolute continuity_

For distribution $P$ and $Q$ of a continuous random variable, then KL divergence is:

$$
D_{\text{KL}}(P \parallel Q) = \int_{-\infty}^{+ \infty} p(x) \log \frac{p(x)}{q(x)} dx
$$

where $p$ and $q$ denote probability densities of $P$ and $Q$
