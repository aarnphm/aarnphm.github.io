---
date: "2025-09-15"
description: Anthropic, 2021
id: mathematical framework transformers circuits
modified: 2025-11-10 06:36:51 GMT-05:00
tags:
  - ml
  - interpretability
title: mathematical framework for transformer circuits
---

The framework treats a transformer as linear updates to a shared residual stream, making algebraic structure explicit and suggesting “privileged” feature bases for interpretation.

see also:

- [[thoughts/induction heads]], superposition, logit lens
- Matrix perspectives and decompositions: [[thoughts/Singular Value Decomposition|SVD]], [[thoughts/Attention]], [[thoughts/Vector calculus#Jacobian matrix|Jacobian]].

## residual stream as a #linalg space

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

## attention heads as information movement

> “Attention heads move information between positions in the residual stream.”[@elhage2021mathematical]

- **Kronecker factorisation.** For head $h$, the linear part factors as

  $$
  H^{(h)}(R) = A^{(h)}(R) \otimes W_{OV}^{(h)} R,
  $$

  where $A^{(h)}$ routes tokens (query/key side) while $W_{OV}^{(h)}=W_O^{(h)}W_V^{(h)}$ transports the feature written into the residual stream. This tensor-product view makes the “information moves from source token via feature channel” story explicit.

- **Query/Key routing.** $QK^\top$ scores select _which_ source token sends information. The softmaxed attention matrix $A$ is a routing operator shaped by sequence content.
- **Value transport.** The value projection $W_V$ extracts a feature from the source token; the output projection $W_O$ determines how that feature is written into the destination residual vector.
- **Paths across layers.** Because each head writes additively into the residual stream, information can hop across tokens and layers, forming interpretable circuits (e.g., induction or copy heads).
- **Diagnostics.** Inspecting $W_V W_O$ (spectrum, singular vectors) reveals what kind of feature is transported, while attention heatmaps show where it moves — see [[lectures/412/notes#spectral diagnostics]] for tooling.

## features, superposition, and privileged bases

- Features are directions in the residual stream. Neurons can be polysemantic (encode multiple features in superposition).
- A “privileged basis” aligns axes with sparse, interpretable features rather than raw neuron axes. [[thoughts/sparse autoencoder|Sparse autoencoders]] and dictionary learning aim to find such bases.
- Changing basis clarifies circuits: weight matrices are change‑of‑basis operators between feature spaces of successive blocks; see [[thoughts/Inner product space#Orthonormal bases & Gram–Schmidt|orthonormal bases]].

> [!tip] Circuits
>
> A circuit is a composed path of linear writes that implements a behavior (e.g., induction heads). Basis choices can make the path sparse and legible.

## connections to modern interpretability

This mathematical framework underpins recent advances in interpretability:

- **[[thoughts/mechanistic interpretability#attribution graph|Attribution graphs]]**: Trace information flow through residual stream by analyzing how features compose across layers. The linear update view makes attribution well-defined when nonlinearities are frozen.

- **[[thoughts/sparse crosscoders|Crosscoders]]**: Exploit residual stream's additive property to learn shared features across layers, resolving cross-layer superposition by treating adjacent layers as "almost parallel branches."

- **[[thoughts/Attribution parameter decomposition|Parameter decomposition]]**: Decompose weight matrices as sums of mechanism-specific components, leveraging the framework's linear algebra structure to identify which parameters implement which computations.

- **[[thoughts/circuit tracing|Circuit tracing]]**: Use [[thoughts/mechanistic interpretability#transcoders|transcoders]] as replacement models that maintain the residual stream structure while enabling interpretable feature analysis.

The framework's emphasis on residual updates, privileged bases, and compositional structure provides the theoretical foundation for these empirical methods.

## takeaway

- Residual update: $r\leftarrow r + \Delta_{\text{attn}} + \Delta_{\text{mlp}}$.
- Logits: $\ell=U^\top r$; probabilities $\mathrm{softmax}(\ell)$.
- Projection onto a feature $a$: $\operatorname{proj}_a(r)=\dfrac{a^\top r}{a^\top a}a$; feature activation is a dot product.
- Framework enables: attribution graphs, crosscoders, parameter decomposition, circuit tracing.
