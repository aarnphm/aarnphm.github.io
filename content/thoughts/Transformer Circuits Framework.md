---
id: Transformer Circuits Framework
tags:
  - ml
  - interpretability
description: Anthropic, 2021
date: "2025-09-15"
modified: 2025-09-16 15:46:00 GMT-04:00
title: A Mathematical Framework for Transformer Circuits
---

The framework treats a transformer as linear updates to a shared residual stream, making algebraic structure explicit and suggesting “privileged” feature bases for interpretation.

see also:

- [[thoughts/induction heads]], superposition, logit lens
- Matrix perspectives and decompositions: [[thoughts/Singular Value Decomposition|SVD]], [[thoughts/Attention]], [[thoughts/Vector calculus#Jacobian matrix|Jacobian]].

## residual stream as a [[/tags/linalg]] space

For token position $t$, the residual vector $r^{(\ell)}_t\in\mathbb R^d$ evolves via skip connections as

$$
r^{(\ell+1)}_t \;=\; r^{(\ell)}_t \;+
\operatorname{Attn}^{(\ell)}(r^{(\ell)}_{\le t}) \;+
\operatorname{MLP}^{(\ell)}(r^{(\ell)}_t).
$$

The unembedding is linear: logits $\ell_t=U^\top r^{(L)}_t$ for unembedding matrix $U\in\mathbb R^{d\times |V|}$. The [[thoughts/Attention|attention]] and MLP blocks are (piecewise) linear maps that write directions into the residual stream.

> [!note] lens intuition
> “Logit lens” inspects $U^\top r^{(\ell)}$ mid‑stack; early layers partially align residual directions with token directions, revealing emergent features.

## heads and MLPs as linear writes

Single head (ignoring nonlinearity):

$$
\operatorname{Attn}(R) \approx \underbrace{\mathrm{softmax}\!\Big(\tfrac{QK^\top}{\sqrt d}\Big)}_{A}\, V\, W_O \;=\; A\,R\,W_V W_O,
$$

which is a data‑dependent low‑rank write into the residual stream. MLPs apply $W_2\,\sigma(W_1 r)$ — in linear regimes they also act as low‑rank writes.

## features, superposition, and privileged bases

- Features are directions in the residual stream. Neurons can be polysemantic (encode multiple features in superposition).
- A “privileged basis” aligns axes with sparse, interpretable features rather than raw neuron axes. [[thoughts/sparse autoencoder|Sparse autoencoders]] and dictionary learning aim to find such bases.
- Changing basis clarifies circuits: weight matrices are change‑of‑basis operators between feature spaces of successive blocks; see [[thoughts/Inner product space#Orthonormal bases & Gram–Schmidt|orthonormal bases]].

> [!tip] Circuits
>
> A circuit is a composed path of linear writes that implements a behavior (e.g., induction heads). Basis choices can make the path sparse and legible.

## takeaway

- Residual update: $r\leftarrow r + \Delta_{\text{attn}} + \Delta_{\text{mlp}}$.
- Logits: $\ell=U^\top r$; probabilities $\mathrm{softmax}(\ell)$.
- Projection onto a feature $a$: $\operatorname{proj}_a(r)=\dfrac{a^\top r}{a^\top a}a$; feature activation is a dot product.
