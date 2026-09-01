---
date: '2025-09-15'
description: Anthropic, 2021
id: mathematical framework transformers circuits
modified: 2026-09-01 09:15:00 GMT-04:00
tags:
  - ml
  - interp
title: mathematical framework for transformer circuits
---

The 2021 framework treats the residual stream as an additive, mostly basis-free communication channel. Once attention patterns and normalization scales are fixed, attention-head paths can be expanded into QK and OV matrix products.[@elhage2021mathematical]

see also:

- [[thoughts/induction heads]], [[thoughts/polysemantic|superposition]], logit lens
- Matrix perspectives and decompositions: [[thoughts/Singular Value Decomposition|SVD]], [[thoughts/Attention]], [[thoughts/Vector calculus#Jacobian matrix|Jacobian]].

## residual stream as a [[thoughts/Embedding|embedding]] space

For token position $t$, a pre-norm block has two serial residual updates:

$$
\begin{aligned}
a_t^{(\ell)}
&=r_t^{(\ell)}+
\operatorname{Attn}^{(\ell)}\!\left(\operatorname{LN}(r_{\leq t}^{(\ell)})\right),\\
r_t^{(\ell+1)}
&=a_t^{(\ell)}+
\operatorname{MLP}^{(\ell)}\!\left(\operatorname{LN}(a_t^{(\ell)})\right).
\end{aligned}
$$

The unembedding is linear: logits $z_t=U^\top r^{(L)}_t$ for $U\in\mathbb R^{d\times |V|}$. Additivity happens at the residual stream. Attention becomes linear in the value and output path only after its attention pattern is fixed. An MLP still contains its activation nonlinearity.

> [!note] logit lens
> Logit lens applies the unembedding to an intermediate residual state, $U^\top r_t^{(\ell)}$, to inspect the model's current next-token logits.

## heads and MLPs as linear writes

Let $R\in\mathbb R^{T\times d}$ store one residual vector per row. For one head,

$$
\begin{aligned}
A^{(h)}
&=\operatorname{softmax}\!\left(
\frac{R W_Q^{(h)}W_K^{(h)\mathsf T}R^{\mathsf T}}
{\sqrt{d_{\mathrm{head}}}}
\right),\\
h^{(h)}(R)
&=A^{(h)}R W_V^{(h)}W_O^{(h)}.
\end{aligned}
$$

The QK matrices determine the attention pattern. Once that pattern is held fixed, $R\mapsto A^{(h)}R W_V^{(h)}W_O^{(h)}$ is a linear OV write.

## attention heads as information movement

Attention heads move information from the residual stream at one token position to the residual stream at another.[@elhage2021mathematical]

- **Kronecker factorization.** Define the row-vector OV product as $\widetilde W_{OV}^{(h)}=W_V^{(h)}W_O^{(h)}$. With the attention pattern fixed,

  $$
  \operatorname{vec}\!\left(h^{(h)}(R)\right)
  =\left(\widetilde W_{OV}^{(h)\mathsf T}\otimes A^{(h)}\right)
  \operatorname{vec}(R).
  $$

  The first factor acts on residual features and the second acts on token positions. This is the typed version of the tensor-product claim.

- **QK routing.** The scores determine how much each destination token attends to each allowed source token.
- **OV transport.** $W_V^{(h)}W_O^{(h)}$ describes the feature map written from a source row into a destination row under this convention.
- **Paths.** Residual addition lets products of head matrices describe paths across layers. A path decomposition is exact only for the simplified or locally frozen model being analyzed.

## features, superposition, and privileged bases

A common working hypothesis is that many features correspond to directions or sparse dictionary elements in activation space. This claim is empirical, so examples and interventions have to carry it.

The 2021 algebra gives the residual stream no privileged basis because an invertible change of residual coordinates can be absorbed into adjacent weight matrices. Token identities, attention patterns, and MLP neuron activations do have privileged coordinates. Later [empirical work](https://transformer-circuits.pub/2023/privileged-basis/index.html) found some standard-basis artifacts in trained residual streams, likely tied to per-coordinate optimizer dynamics.

[[thoughts/sparse autoencoder|Sparse autoencoders]] and dictionary learning search for sparse feature dictionaries inside this activation space. Reconstruction quality alone is insufficient, so a feature still needs ablation or intervention tests.

> [!tip] circuits
>
> A circuit is a proposed mechanism made from model components and paths that causally explains a behavior. Linear path products describe one part of that mechanism after attention and other nonlinear terms have been fixed or replaced.

## connections to modern interpretability

Attribution graphs and [[thoughts/circuit tracing|circuit tracing]] start from this algebra. Current methods add learned feature dictionaries, transcoder replacement models, frozen attention patterns, explicit error nodes, graph pruning, and intervention tests. [[thoughts/sparse crosscoders|Crosscoders]] can represent features shared across layers. The learned graph remains an approximation of the original model.

## takeaway

- Residual sublayers write additively in sequence.
- QK determines the attention pattern. OV determines the residual feature map once that pattern is fixed.
- The residual stream is basis-free in the simplified algebra.
- The coefficient of $r$ along a feature direction $a$ is $a^\top r/(a^\top a)$. Learned dictionary features may also use biases and nonlinear encoders.
