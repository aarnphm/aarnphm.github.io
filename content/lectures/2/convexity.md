---
id: convexity
tags:
  - seed
  - math
description: Entropy-regularized attention as a convex program; Fenchel–Young view, geometry, and verification insights.
date: "2025-08-21"
modified: 2025-09-14 23:13:14 GMT-04:00
title: convexity prose
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

## Fenchel–Young view and design knobs

Write attention as the regularized argmax

$$
\alpha^*(s)=\arg\max_{\alpha\in\Delta}\ \langle s,\alpha\rangle-\Omega(\alpha),
$$

with convex regularizer $\Omega$. Choices of $\Omega$ give different behaviors with convex losses and explicit solutions:

- Softmax: $\Omega(\alpha)=\tau\sum_i\alpha_i\log\alpha_i$ (dense, max‑entropy).
- Sparsemax/entmax: $\Omega$ inducing exact zeros in $\alpha$ (sparse attention; convex objectives; closed‑form/provably convergent solvers) ([1], [2]).
- Doubly‑stochastic attention: apply Sinkhorn to obtain row/column stochastic $W$ (OT connection; beneficial inductive biases) ([3]).

## geometry and convexity

- Input non‑convexity. As a function of $(q,k)$, scores are bilinear and the softmax mapping is non‑convex; standard transformers remain non‑convex in parameters.
- Weight‑space convexity. For fixed scores, optimizing over $\alpha\in\Delta$ is convex; the output $y=\sum_i\alpha_i v_i$ lies in the convex hull of $\{v_i\}$ (the “probability cage”) ([11]).

## verification and robustness

Tight convex lower and concave upper bounds for softmax enable robustness verification with stronger certificates than linear relaxations; they integrate into $\alpha, \beta$‑CROWN/BaB verifiers ([4], [5], [6]).

## Practical levers

- Guarantees: convex surrogates train to global optimality for the surrogate model ([9]).
- Structure: choose $\Omega$ and constraints to promote sparsity, matching, or conservation (e.g., Sinkhorn) with principled convergence ([1], [3]).
- Analysis: convex duality exposes implicit biases (e.g., low‑rank/clustering) and yields interpretable formulations ([10]).
- Differentiation: optimization layers allow exact/implicit gradients with stable sensitivity ([7], [8]).

---

[1]: https://jmlr.org/papers/v23/21-0879.html "Sparse Continuous Distributions and Fenchel-Young Losses"
[2]: https://arxiv.org/pdf/1905.05702 "From Softmax to Sparsemax/Entmax"
[3]: https://proceedings.mlr.press/v151/sander22a/sander22a.pdf "Sinkformers: Transformers with Doubly Stochastic Attention"
[4]: https://proceedings.mlr.press/v206/wei23c/wei23c.pdf "Convex Bounds on the Softmax Function with Applications to Robustness Verification"
[5]: https://research.ibm.com/publications/convex-bounds-on-the-softmax-function-with-applications-to-robustness-verification "IBM: Convex Bounds on Softmax"
[6]: https://github.com/Verified-Intelligence/alpha-beta-CROWN "alpha-beta-CROWN verifier"
[7]: https://stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf "Differentiable Convex Optimization Layers"
[8]: https://proceedings.mlr.press/v70/amos17a/amos17a.pdf "OptNet: Differentiable Optimization as a Layer"
[9]: https://arxiv.org/pdf/2211.11052 "Convexifying Transformers"
[10]: https://arxiv.org/abs/2205.08078 "Unraveling Attention via Convex Duality"
[11]: https://arxiv.org/abs/2005.09561 "Normalized Attention Without Probability Cage"
