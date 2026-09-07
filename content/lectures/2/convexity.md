---
date: '2025-08-21'
description: entropy-regularized attention on the probability simplex
id: convexity
modified: 2026-09-07 13:56:24 GMT-04:00
tags:
  - math/linalg
  - theory
title: convexity of attention
---

One row of attention weights solves a convex problem once its score vector is fixed. This gives softmax a precise variational meaning. It does not make the query, key, or model-training problem convex.

## entropy-regularized weights

For scores $s\in\mathbb{R}^n$, let

$$
\Delta_n=\left\{\alpha\in\mathbb{R}^n:\alpha_i\geq0,\ \sum_{i=1}^n\alpha_i=1\right\}.
$$

At temperature $\tau>0$, consider

$$
\min_{\alpha\in\Delta_n}
\left\{
\tau\sum_{i=1}^n\alpha_i\log\alpha_i-s^\top\alpha
\right\},
$$

with $0\log0=0$. Negative entropy is strictly convex on the simplex. The minimizer is unique, and its coordinates are positive because the one-sided derivative of $x\log x$ tends to $-\infty$ at zero.

The equality-constrained stationarity equation is

$$
\tau(1+\log\alpha_i)-s_i+\lambda=0.
$$

Exponentiating and enforcing $\sum_i\alpha_i=1$ gives

$$
\alpha_i^*(s,\tau)
=\frac{e^{s_i/\tau}}{\sum_j e^{s_j/\tau}}
=\operatorname{softmax}(s/\tau)_i.
$$

This is the gradient of the scaled log-partition function

$$
\operatorname{LSE}_\tau(s)=\tau\log\sum_j e^{s_j/\tau}.
$$

As $\tau\downarrow0$, the mass approaches the uniform distribution over the indices attaining $\max_i s_i$. If the maximum is unique, the limit is its basis vector. As $\tau\uparrow\infty$, the weights approach $\mathbf1/n$. [@gao2018propertiessoftmaxfunctionapplication; @blondel2019fenchelyoung]

## Fenchel-Young prediction maps

Let $\Omega$ be a proper closed convex regularizer on $\Delta_n$. Its regularized prediction correspondence is

$$
\widehat\alpha_\Omega(s)
\in\underset{\alpha\in\Delta_n}{\arg\max}
\left\{s^\top\alpha-\Omega(\alpha)\right\}.
$$

When the maximizer is unique, as it is for the regularizers below, this correspondence is a prediction map.

The regularizer controls how probability mass sits on the simplex:

- Shannon negative entropy, $\Omega(\alpha)=\tau\sum_i\alpha_i\log\alpha_i$, gives softmax and a dense solution.
- The squared Euclidean regularizer, $\Omega(\alpha)=\tfrac12\|\alpha\|_2^2$, gives sparsemax, the Euclidean projection of $s$ onto the simplex.
- Tsallis-entropy regularizers give the entmax family. Softmax and sparsemax appear at its two standard endpoints, while intermediate choices can produce sparse weights with a smoother support transition. [@blondel2019fenchelyoung; @peters2019sparsesequencetosequencemodels]

The convexity is in $\alpha$ with $s$ held fixed. In a transformer, $s_i=q^\top k_i/\sqrt{d_h}$ is bilinear in the projected query and key. Learning those projections remains a nonconvex parameter problem.

## output geometry

With fixed values $v_1,\ldots,v_n$, any simplex-valued attention rule returns

$$
y=\sum_i\alpha_i v_i\in\operatorname{conv}\{v_1,\ldots,v_n\}.
$$

This convex-hull restriction is the probability-cage observation. It concerns one head's weighted average before the output projection and residual connection; it is not a bound on the whole transformer. [@richter2020normalizedattentionprobabilitycage]

## matrix normalization

Sinkhorn attention starts with a positive score matrix and alternates row and column rescaling toward prescribed marginals. For a square matrix with uniform marginals, the limit is doubly stochastic under the usual positivity conditions. That constraint lives at matrix level, not inside each row's simplex. Sinkformers use it as an architectural normalization, not as another scalar entropy regularizer. [@sander2022sinkformers]
