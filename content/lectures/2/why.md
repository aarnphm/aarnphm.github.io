---
date: "2025-08-21"
description: motivations of attention
id: why
modified: 2025-10-29 02:14:22 GMT-04:00
tags:
  - seed
  - ml
  - math
title: why certain components in attention?
---

see also:

- Wainwright & Jordan, variational view of log‑partition. [^1]
- Boyd & Vandenberghe, log‑sum‑exp and convexity. [^2]
- [@vaswani2023attentionneed; @shazeer2019fasttransformerdecodingwritehead; @ainslie2023gqatraininggeneralizedmultiquery]

[^1]: https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf

[^2]: https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf

## Why dot products and softmax?

Let $q\in\mathbb{R}^{d}$ retrieve from a dictionary $\{(k_j,v_j)\}_{j=1}^n$. Logits: $z_j=\langle q,k_j\rangle/\sqrt d$.

Geometric identity. If $\|q\|=\|k_j\|=c$, then $\langle q,k_j\rangle=\tfrac12(\|q\|^2+\|k_j\|^2-\|q-k_j\|^2)=c^2-\tfrac12\|q-k_j\|^2$. Thus $\exp(z_j/T)\propto \exp(-\|q-k_j\|^2/(2T\sqrt d))$: row‑softmax yields Gaussian/RBF weights up to a constant, i.e., kernel smoothing on learned features. The $\sqrt d$ factor rescales the effective bandwidth.

MaxEnt derivation. For $z\in\mathbb{R}^n$, temperature $T>0$,

$$
p^*(z,T)=\arg\max_{p\in\Delta}\{\langle p,z\rangle + T H(p)\},\quad H(p)=-\sum_j p_j\log p_j,
$$

has unique solution $p^*(z,T)=\operatorname{softmax}(z/T)$. Equivalently, $\log\sum_j e^{z_j/T}=\max_{p\in\Delta}\{\tfrac1T\langle p,z\rangle+H(p)\}$ (Gibbs variational principle). Let $\mathrm{lse}_\lambda(z)=\lambda^{-1}\log\sum_j e^{\lambda z_j}$ with $\lambda=1/T$. Then $\nabla\mathrm{lse}_\lambda(z)=\operatorname{softmax}(\lambda z)$ and $\nabla^2\mathrm{lse}_\lambda(z)=\lambda(\mathrm{Diag}(p)-pp^\top)\succeq0$ with operator norm $\le \lambda$; hence $\operatorname{softmax}(\lambda\cdot)$ is Lipschitz with constant at most $\lambda$.

## Attention as an optimization

Row $i$ solves $p^{(i)}=\arg\max_{p\in\Delta}\{\langle p,z^{(i)}\rangle + T H(p)\}$ with $z^{(i)}_j=\langle q_i,k_j\rangle/\sqrt d$ and output $o_i=\sum_j p^{(i)}_j v_j$. Limits: $T\downarrow0$ gives hard nearest neighbor; $T\uparrow\infty$ gives uniform averaging.

## From single head to MHA and GQA

MHA: heads $h=1,\dots,H$ solve independent copies with head‑specific projections; $W_O$ combines outputs.

GQA/MQA: share $K,V$ across groups (or all heads). Only the dictionary changes; the per‑row entropic program remains the same.

Stability under grouping. With $z=K_h q/\sqrt d$ and $\tilde z=K_g q/\sqrt d$, $\delta z=\tilde z-z=(K_g-K_h)q/\sqrt d$. The softmax function $\sigma(\lambda \cdot)$ has Lipschitz constant $\lambda$ (with $\lambda=1/T$), so:

$$
\|p(\tilde z)-p(z)\|_2 = \|\sigma(z/T)-\sigma(\tilde z/T)\|_2 \le \tfrac{1}{T}\|\delta z\|_2 = \tfrac{1}{T\sqrt d}\,\|(K_g-K_h)q\|_2 \le \tfrac{1}{T\sqrt d}\,\|K_g-K_h\|_{op}\,\|q\|_2,
$$

so weight drift (and thus output drift) is controlled by dictionary mismatch, query norm, and temperature.

## Why the $1/\sqrt d$ scale?

If $q,k$ have i.i.d. zero‑mean, unit‑variance entries, $\mathrm{Var}(\langle q,k\rangle)=d$. Dividing by $\sqrt d$ keeps logit variance $O(1)$ as width grows, stabilizing the softmax and gradients.

> [!important]
>
> - Softmax arises as the unique MaxEnt solution; attention is a per‑row entropic program on the simplex.
> - With normalized features, attention weights equal RBF weights; bandwidth $\sigma^2=T\sqrt d$.
> - MHA decouples across heads; GQA/MQA share dictionaries with controlled stability loss.
> - The $1/\sqrt d$ scaling normalizes logits and preserves numerics.
