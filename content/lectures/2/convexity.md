---
date: '2025-08-21'
description: Entropy-regularized attention as a convex program; Fenchel–Young view, geometry, and verification insights.
id: convexity
modified: 2026-01-15 08:29:33 GMT-05:00
tags:
  - math/linalg
  - theory
title: convexity of attention
---

We study attention weight computation as a convex program: entropy‑regularized linear maximization on the probability simplex. We derive uniqueness and closed‑form solutions, connect to Fenchel–Young regularizers (softmax/sparsemax/entmax), and outline geometric and verification consequences.

## setup

Let $s\in\mathbb{R}^n$ be scores (e.g., $s_i=q^\top k_i/\sqrt{d}$). Define the simplex

$$
\Delta=\{\alpha\in\mathbb{R}^n\mid \alpha\ge 0,\ \mathbf{1}^\top\alpha=1\}.
$$

Consider the entropy‑regularized problem

$$
\min_{\alpha\in\Delta}\quad f_\tau(\alpha)=\tau\sum_i \alpha_i\log\alpha_i - s^\top\alpha,\qquad \tau>0.
$$

> [!abstract] Proposition 1 (Strict convexity and uniqueness).
>
> $f_\tau$ is strictly convex on the relative interior of $\Delta$; the minimizer on $\Delta$ is unique.

Proof. $x\mapsto x\log x$ is convex on $x\ge 0$ with Hessian $\nabla^2 f_\tau(\alpha)=\tau\,\mathrm{diag}(1/\alpha_i)\succ0$ for $\alpha>0$; the feasible set is convex with nonempty interior, so the solution is unique.

> [!abstract] Proposition 2 (softmax solution).
>
> The unique minimizer is
>
> $$
> \alpha^*_i(s,\tau)=\frac{\exp(s_i/\tau)}{\sum_j \exp(s_j/\tau)}.
> $$

Proof. [[lectures/2/notes#KKT|KKT]] stationarity gives $\tau(1+\log\alpha_i)-s_i+\lambda-\mu_i=0$.

At optimality $\alpha_i>0$ so $\mu_i=0$; exponentiating yields $\alpha_i\propto e^{s_i/\tau}$ and normalization on $\Delta$ fixes the constant.

> [!abstract] Corollary 3 (temperature limits).
>
> As $\tau\downarrow 0$, $\alpha^*(s,\tau)\to e_k$ for any $k\in\arg\max_i s_i$ (extreme point/argmax). As $\tau\uparrow\infty$, $\alpha^*\to \tfrac{1}{n}\mathbf{1}$.

## Fenchel–Young view

Write attention as the regularized argmax

$$
\alpha^*(s)=\arg\max_{\alpha\in\Delta}\ \langle s,\alpha\rangle-\Omega(\alpha),
$$

with convex regularizer $\Omega$. Choices of $\Omega$ give different behaviors with convex losses and explicit solutions:

- Softmax: $\Omega(\alpha)=\tau\sum_i\alpha_i\log\alpha_i$ (dense, max‑entropy).
- Sparsemax/entmax: $\Omega$ inducing exact zeros in $\alpha$ (sparse attention; convex objectives; closed‑form/provably convergent solvers). [@martins2022sparsecontinuousdistributions; @peters2019sparsesequencetosequencemodels]
- Doubly‑stochastic attention: apply Sinkhorn to obtain row/column stochastic $W$ (OT connection; beneficial inductive biases). [@sander2022sinkformers]

## geometry and convexity

- Input non‑convexity. As a function of $(q,k)$, scores are bilinear and the softmax mapping is non‑convex; standard transformers remain non‑convex in parameters.
- Weight‑space convexity. For fixed scores, optimizing over $\alpha\in\Delta$ is convex; the output $y=\sum_i\alpha_i v_i$ lies in the convex hull of $\{v_i\}$ (the "probability cage"). [@richter2020normalizedattentionprobabilitycage]

## verification and robustness

Tight convex lower and concave upper bounds for softmax enable robustness verification with stronger certificates than linear relaxations; they integrate into $\alpha, \beta$‑CROWN/BaB verifiers.

- Guarantees: convex surrogates train to global optimality for the surrogate model. [@ergen2022convexifyingtransformersimprovingoptimization]
- Structure: choose $\Omega$ and constraints to promote sparsity, matching, or conservation (e.g., Sinkhorn) with principled convergence. [@martins2022sparsecontinuousdistributions; @sander2022sinkformers]
- Analysis: convex duality exposes implicit biases (e.g., low‑rank/clustering) and yields interpretable formulations. [@sahiner2022unravelingattentionconvexduality]
- Differentiation: optimization layers allow exact/implicit gradients with stable sensitivity. [@amos2017optnet]
