---
date: '2025-09-12'
description: linear algebra applications for transformers
id: notes
modified: 2026-06-07 01:29:37 GMT-04:00
seealso:
  - '[[thoughts/Transformers|Transformers]]'
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/LLMs|LLMs]]'
  - '[[lectures/412/tools|Tools]]'
slides: true
tags:
  - workshop
  - math/linalg
title: supplement to 0[dot]412
transclude:
  title: false
---

## introduction

- structure: embed tokens, inject positions, run masked multi-head attention, apply position-wise MLPs, and close with the unembedding that converts residual coordinates into logits.

> [!abstract] readings
> residual-stream linearization and MLP interpretation. [@elhage2021mathematical; @geva2021transformerfeedforwardlayerskeyvalue]
>
> - residual stream: Anthropic's decomposition of attention heads and unembedding matrices into analyzable paths to motivate each algebraic step. [@elhage2021mathematical]
> - MLP as KV memory: highlight how the ffns cache features, setting up the later mixture-of-experts discussion. [@geva2021transformerfeedforwardlayerskeyvalue]

## decoder-only stack as linear maps

Let $d_{model}$ denote the residual stream width and $T$ the sequence length. Represent the token sequence as a matrix $X_0\in\mathbb{R}^{T\times d_{model}}$ where each row is the residual vector at a position.

### embeddings and residual coordinates

- token embedding: $X_0 = S E$ where $S\in\mathbb{R}^{T\times |\mathcal{V}|}$ is a one-hot indicator matrix and $E\in\mathbb{R}^{|\mathcal{V}|\times d_{model}}$ is the learned embedding.
  - learned absolute and sinusoidal encodings add a position vector to the residual stream. RoPE instead rotates query and key coordinates inside attention, so it is not an additive matrix $P$ in $X_0$. [@vaswani2023attentionneed]
- layer norm: treat pre-normalization as an affine map $\mathcal{N}(x)=D(x)(x-\mu(x))$ with diagonal scaling $D(x)$. In practice we approximate $D$ as invertible on the support of interest to reason about nearby linearizations.
- residual stream: every block writes back to $X$ through additive updates, so we model the stack as

$$
X_{\ell+1} = X_{\ell} + F_{\ell}(X_{\ell}), \quad \ell = 0,\dots,L-1
$$

where $F_{\ell}$ sums attention and mlp contributions.

**Definition (token embedding map).** The map $\varepsilon: \{1,\dots,|\mathcal{V}|\}\to\mathbb{R}^{d_{model}}$ satisfies $\varepsilon(i)=e_i^\top E$ with $e_i$ the $i$th standard basis vector. For a one-hot token matrix $S$, the initial residual matrix is $X_0 = S E$.

**Lemma (injectivity for additive position vectors).** The map $\widetilde\varepsilon(a,i)=\varepsilon(a)+p_i$ is injective exactly when

$$
\varepsilon(a)-\varepsilon(b)\ne p_j-p_i
$$

for every distinct pair $(a,i)\ne(b,j)$.

_Proof._ Equality of two combined embeddings is equivalent to $\varepsilon(a)-\varepsilon(b)=p_j-p_i$. excluding that equality for every distinct pair is precisely injectivity. distinct token embeddings and distinct position vectors alone are insufficient because their differences can cancel. [^embed-proof]

**Lemma (layer norm linearization).** Let $z=x-\mu\mathbf1$ and $s=\sqrt{d^{-1}\lVert z\rVert_2^2+\epsilon}$. the Jacobian of $\mathcal N(x)=\gamma\odot z/s+\beta$ is

$$
J_{\mathcal N}(x)
=\frac{\operatorname{diag}(\gamma)}{s}
\left(I-\frac{\mathbf1\mathbf1^T}{d}-\frac{zz^T}{ds^2}\right).
$$

_Proof._ Differentiate $\mu$, $z$, and $s$, then collect the three terms. the bracketed matrix annihilates $\mathbf1$, so layer norm is not locally invertible. when $\epsilon>0$ it is not a projection: its eigenvalue along $z$ is $\epsilon/s^2$. [^layernorm-proof]

### single-head attention as a tensor product

Fix a head $h$ inside layer $\ell$. Following the transformer-circuits factorization we write the attention output as a Kronecker-style product[^Kronecker] acting on $X_{\ell}$:[@elhage2021mathematical]

$$
H^{(h)}(X_{\ell}) = A^{(h)}(X_{\ell})X_{\ell}W_{OV}^{(h)}
$$

[^Kronecker]:
    In machine learning, we often deal with matrices and tensors where we want to multiply “one side.”

    In transformers specifically, our activations are often 2D arrays representing vectors at different context indices, and we often want to multiply **per position** or **across positions**. Frequently, we want to do both! Tensor (or equivalently, Kronecker) products are a really clean way to denote this. We denote them with the $\otimes$ symbol.

    Very pragmatically:

    - A product like $\mathrm{Id} \otimes W$ (with identity on the left) represents multiplying each position in our context by a matrix $W$.
    - A product like $A \otimes \mathrm{Id}$ (with identity on the right) represents multiplying **across positions**.
    - A product like $A \otimes W$ multiplies the vector at each position by $W$ _and_ across positions with $A$. It doesn’t matter which order you do this in.
    - The products obey the **mixed-product property**:

      $$
        (A \otimes B) \cdot (C \otimes D) = (A C) \otimes (B D).
      $$

    There are several completely equivalent ways to interpret these products. If the symbol is unfamiliar, you can pick whichever you feel most comfortable with:

    - **Left-right multiplying**: Multiplying $x$ by a tensor product $A \otimes W$ is equivalent to simultaneously left and right multiplying:

      $$
        (A \otimes W)\, x = A\, x\, W^\top.
      $$

      When we sum them, it is equivalent to adding the results of this multiplication:

      $$
        (A_1 \otimes W_1 + A_2 \otimes W_2)\, x = A_1 x W_1^\top + A_2 x W_2^\top.
      $$

    - **Kronecker product**: The operations we want to perform are linear transformations on a flattened (“vectorized”) version of the activation matrix $x$. But flattening gives us a huge vector, and we need to map our matrices to a much larger block matrix that performs operations like “multiply the elements which previously corresponded to a vector by this matrix.” The correct operation to do this is the Kronecker product. So we can interpret $\otimes$ as a Kronecker product acting on the vectorization of $x$, and everything works out equivalently.

    - **Tensor product**: $A \otimes W$ can be interpreted as a tensor product turning the matrices $A$ and $W$ into a 4D tensor. In NumPy notation, it is equivalent to

      $$
      A[:, :, \mathrm{None}, \mathrm{None}] * W[\mathrm{None}, \mathrm{None}, :, :]
      $$

      (though one wouldn’t computationally represent them in that form). More formally, $A$ and $W$ are “type $(1,1)$” tensors (matrices mapping vectors to vectors), and $A \otimes W$ is a “type $(2,2)$” tensor (which can map matrices to matrices).

with $A^{(h)}(X_{\ell})\in\mathbb{R}^{T\times T}$ encoding token-to-token routing and $W_{OV}^{(h)} = W_V^{(h)}W_O^{(h)}\in\mathbb{R}^{d_{model}\times d_{model}}$ a low-rank map ($\mathrm{rank} \le d_{head}$).

- Because the only nonlinearity resides in the softmax defining $A^{(h)}$, freezing $A^{(h)}$ linearizes the head completely.
- the idea of [[thoughts/mathematical framework transformers circuits#attention heads as information movement|information movement]] comes from the fact attention compute value vectors for each tokens from residual stream, and linearly combine each of those vectors based on attention pattern.
  - i.e linear map $[n_\text{context}, d_{\text{model}}] \to [n_{\text{context}}, d_{\text{model}}]$
- This enables path analysis:
  - expand logits as sums of products of $W_{OV}$ and $W_{QK}$ matrices and map each product to a circuit acting on a basis vector in the residual stream. [@elhage2021mathematical]

- An attention head applies two linear operations, $A$ across positions and $W_VW_O$ across residual coordinates.
  - $A$ governs which token's information to move from and to
  - $W_VW_O$ governs which information is read from the source token and how it is written to the destination token.

> [!tip] intuition
> think of $W_{OV}$ as choosing a subspace to copy from the source token and another to write into at the destination. singular vectors of $W_{OV}$ highlight reusable features; tracking them across layers surfaces induction heads and copy circuits.

- causal mask: define $M\in\mathbb{R}^{T\times T}$ with $M_{ij}=0$ for $j\le i$ and $M_{ij}=-\infty$ otherwise. then $A^{(h)} = \operatorname{softmax}\!\left(\frac{Q^{(h)}K^{(h)\top}}{\sqrt{d_k}} + M\right)$ is lower triangular and can have a nonzero diagonal because a token may attend to itself.
- stochasticity: each row of $A^{(h)}$ sums to $1$. causal attention is generally not column-stochastic, so it does not lie in the Birkhoff polytope.[^birkhoff]

**Proposition (causal support is closed under composition).** If $A,B\in\mathbb R^{T\times T}$ are lower triangular attention matrices, then $AB$ is lower triangular.

_Proof._ For $j>i$,

$$
(AB)_{ij}=\sum_k A_{ik}B_{kj}=0,
$$

because a nonzero summand would require both $k\le i$ and $j\le k$, which is impossible. the product remains causal. it is not generally nilpotent because its diagonal can be nonzero.

#### observations on single-head attention

- $A$ is the **only non-linear part** of this equation (being computed from a softmax). This means that if we fix the attention pattern, attention heads perform a linear operation. This also means that, **without fixing** $A$, attention heads are “half-linear” in some sense, since the per-token linear operation is constant.
- $W_Q$ and $W_K$ always operate together. They’re never independent. Similarly, $W_O$ and $W_V$ always operate together as well.
  - Although they’re parameterized as separate matrices, $W_V W_O$ and $W_Q W_K^T$ can be treated as individual low-rank maps under the row-activation convention.
  - Keys, queries and value vectors are, in some sense, superficial. They're intermediary by-products of computing these low-rank matrices. One could easily reparameterize both factors of the low-rank matrices to create different vectors, but still function identically.
  - define $W_{OV}=W_VW_O$ and $W_{QK}=W_QW_K^T$ to keep the row-activation convention explicit.
- Products of attention heads behave much like attention heads themselves. By the distributive property,

  $$
    H^{(h_2)}(H^{(h_1)}(X))
    =(A^{h_2}A^{h_1})X(W_{OV}^{h_1}W_{OV}^{h_2}).
  $$

  the composite has sequence operator $A^{h_2}A^{h_1}$ and residual operator $W_{OV}^{h_1}W_{OV}^{h_2}$. this is the virtual-head bookkeeping device.

### block-level composition

multi-head aggregation: a layer with heads $h=1,\dots,H$ yields

$$
M_{\ell}(X_{\ell}) = \sum_{h=1}^{H} H^{(h)}(X_{\ell})
$$

and the residual update becomes $X_{\ell+1}=X_{\ell}+M_{\ell}(X_{\ell})+G_{\ell}(X_{\ell})$ where $G_{\ell}$ is the mlp contribution.

- mlp as rank-expansion: with row activations, $G_\ell(x)=\phi(xW_1)W_2$, where $W_1\in\mathbb R^{d_{model}\times d_{ff}}$ and $W_2\in\mathbb R^{d_{ff}\times d_{model}}$. freezing the diagonal activation Jacobian gives the local residual-coordinate map $W_1\operatorname{diag}(\phi'(xW_1))W_2$.
- key-value memory view: geva et al. show that feed-forward neurons implement key-value associative memories; $W_1$ encodes keys, the activation acts as a selector, and $W_2$ writes cached values back into the residual stream, matching the residual-circuit picture.[@geva2021transformerfeedforwardlayerskeyvalue]
- virtual heads: products like $H^{(h_2)}\circ H^{(h_1)}$ have attention map $A^{(h_2)}A^{(h_1)}$ and row-convention OV map $W_{OV}^{(h_1)}W_{OV}^{(h_2)}$.[@elhage2021mathematical]

**Proposition (linearized residual path expansion).** Freeze the attention patterns, layer-norm Jacobians, and MLP activation Jacobians at one input. each layer is then a linear operator $I+B_\ell$ on $\operatorname{vec}(X)$, where $B_\ell$ contains both sequence-routing and residual-space factors. the unembedded output is

$$
\operatorname{vec}(\operatorname{logits})
=(W_U^T\otimes I_T)
\prod_{\ell=0}^{L-1}(I+B_\ell)
\operatorname{vec}(X_0).
$$

_Proof._ Expanding the ordered product chooses either the identity or one update from each layer. every term is therefore an ordered residual path. attention terms retain both $A^{(\ell,h)}$ and $W_{OV}^{(\ell,h)}$; MLP terms retain their local Jacobians. the unembedding maps the final residual coordinates to logits.[@elhage2021mathematical][^path-proof]

### mlp superposition (geva et al.)

- Factor the feed-forward block as a key-value memory: $W_1$ supplies keys, $W_2$ provides values, and the non-linearity selects which keys fire.[@geva2021transformerfeedforwardlayerskeyvalue]
- Superposition appears because different features reuse the same neuron; activations overlap in the residual basis until you rotate into a privileged coordinate system (see the toy-model section below).

**Lemma (key-value decomposition).** Let $x\in\mathbb{R}^{d_{model}}$, $K=W_1$, $V=W_2^\top$, and $a=\phi(Kx)$ with $\phi$ applied elementwise. Then the MLP output can be written as

$$
G_{\ell}(x) = V^\top a = \sum_{j=1}^{d_{ff}} a_j\, v_j,
$$

where $v_j$ is the $j$th column of $V^\top$. Each neuron contributes a rank-one write $a_j v_j$; overlapping keys $K_j$ mean several semantic features can share the same neuron and therefore superpose inside $a$.[^geva-proof]

> [!note] spectral diagnostics
>
> singular values of $W_{OV}^{(h)}$ act like feature strength indicators.
>
> - The singular values $\sigma_i$ quantify how much a head stretches each principal direction; sharp decay (large $\sigma_1$, tiny tail) signals a nearly one-dimensional copier, while flat spectra imply superposed features with comparable gain.[^sigma-footnote]
> - Effective rank $r_{\text{eff}} = \exp\!\big(-\sum_i p_i \log p_i\big)$ with $p_i = \sigma_i^2 / \sum_j \sigma_j^2$ summarises the decay; it stays small for copy heads and grows when heads mix many directions.
> - The cumulative energy $E_k = \sum_{i\le k}\sigma_i^2 / \sum_i \sigma_i^2$ tells you how many singular directions you need to capture a target fraction of variance.
> - The diagnostic script [[lectures/412/tools#scripts]] computes these curves, logs spectra, and now measures effective rank so you can separate low-rank copiers from high-rank mixers in practice.
> - Example below: layer 12 head 8 on the prompt "The true meaning of life is absurdity, and suffering" shows both spectrum and attention routing for the same head.
>
> ![[thoughts/images/Qwen3-0.6B_layer12_head8.webp]]
> ![[thoughts/images/Qwen3-0.6B_layer12_head8_heatmap.webp]]
>
> ![[thoughts/images/Qwen3-0.6B_layer22_head4.webp]]
> ![[thoughts/images/Qwen3-0.6B_layer22_head4_heatmap.webp]]
>
> ![[thoughts/images/Qwen3-0.6B_layer22_head8.webp]]
> ![[thoughts/images/Qwen3-0.6B_layer22_head8_heatmap.webp]]
>
> ![[thoughts/images/Qwen3-0.6B_layer3_head3.webp]]
> ![[thoughts/images/Qwen3-0.6B_layer3_head3_heatmap.webp]]
>
> ![[thoughts/images/Qwen3-0.6B_layer3_head6.webp]]
> ![[thoughts/images/Qwen3-0.6B_layer3_head6_heatmap.webp]]
>
> ![[thoughts/images/Qwen3-0.6B_layer3_head8.webp]]
> ![[thoughts/images/Qwen3-0.6B_layer3_head8_heatmap.webp]]
>
> ![[lectures/412/images/qwen3-0.6b_layer12_head8_absurdity_spectrum.webp]]
> ![[lectures/412/images/qwen3-0.6b_layer12_head8_absurdity_heatmap.webp]]

[^embed-proof]: This condition is finite and exact for a fixed vocabulary and context window. a direct-sum assumption on the token-difference and position-difference spans would be sufficient, though stronger than necessary.

[^layernorm-proof]: Centering makes the constant direction a null direction. the variance term changes the centered radial direction, with the $\epsilon=0$ formula gaining a second null direction.

[^path-proof]: Expanding the ordered product yields one term for every choice of layer updates. a path term must keep its sequence and residual-coordinate operators; dropping the attention matrices loses token routing.

[^geva-proof]: Geva et al. show that each neuron behaves like a key–value pair: $K_j$ is a key direction, the activation gate decides whether to fire it, and $v_j$ is the value written back. Summing $a_j v_j$ across neurons reconstructs the residual update while exposing where features overlap.

[^privileged-proof]: Conjugating $W_{OV}^{(h)}$ by its eigenbasis just rotates into coordinates where the operator is diagonal. Any downstream linear probe composes in the same way, so analyzing logits in that basis preserves the output while making feature axes explicit.

[^moe-proof]: When the router enforces equal traffic per expert, the Gram matrix $R^\top R$ becomes diagonal. This means expert activations do not interfere (off-diagonal zero) and each diagonal entry reflects uniform utilisation, the condition required by the balancing loss.

[^birkhoff]: The Birkhoff polytope contains doubly stochastic matrices, whose rows and columns both sum to $1$. softmax supplies only the row condition.

[^sigma-footnote]: Singular values measure how much an operator stretches orthogonal directions; plotting them on a log scale reveals how quickly the energy decays. Interpreting $\sigma_i^2$ as variances explains why effective rank and cumulative energy summarise the head’s behaviour.

### privileged bases and feature disentanglement

- privileged basis claim: residual features admit bases where important circuits become sparse; Anthropic demonstrate that induction heads align with Fourier-like bases along sequence positions.[@elhage2023privilegedbasis]
- linear algebra: let $V$ be the residual subspace spanned by a circuit. Choosing a basis via eigenvectors of $W_{OV}^{(h)}$ diagonalizes its action, exposing feature directions with maximal gain. The change-of-basis matrix $S$ satisfies $S^{-1}W_{OV}^{(h)}S = \Lambda$, sharpening interpretability.

**Lemma.** If $W_{OV}^{(h)}$ is diagonalizable with eigenbasis $S$, then for any downstream linear probe $W$, analyzing in the privileged coordinates $S$ preserves logits: $W W_{OV}^{(h)} = (WS) \Lambda S^{-1}$.

_Proof._ Substitute $W_{OV}^{(h)} = S \Lambda S^{-1}$ and regroup. The transformed probe $WS$ operates in the privileged basis while $S^{-1}$ returns to the canonical coordinates, showing that feature attribution is invariant under the change of basis.[@elhage2023privilegedbasis][^privileged-proof]

### induction circuits and in-context learning

- induction heads: specific two-head motifs compute copy-and-advance algorithms identified in Anthropic's induction study.[@olsson2022context]
- algebraic signature: a query head $h_q$ focuses on the token preceding the current position, while a value head $h_v$ copies its value forward. The composite transfer is approximated by $W_{OV}^{(h_v)}W_{OV}^{(h_q)}$, yielding a shift operator along the sequence.
- causal proof sketch: restricting to length-2 subsequences, one shows that $A^{(h_q)}$ acts as a permutation matrix $P$ shifting attention left, and $W_{OV}^{(h_v)}$ approximates the identity on copied tokens. Hence $P$ composed with a copy map reproduces repeated substrings, explaining in-context generalization on algorithmic data.

### toy models and superposition

- superposition phenomenon: neuron activations store multiple features in the same subspace, resolved by sparse autoencoders. The toy models paper formalizes this by minimizing reconstruction loss under $L_1$ feature sparsity.[@elhage2022superposition]
- linear toy setup: consider feature matrix $F\in\mathbb{R}^{k\times d}$ and dictionary $D\in\mathbb{R}^{d\times m}$. The optimization $\min_D \|F - SD\|_F^2 + \lambda\|S\|_1$ shows how features combine linearly inside neurons; decoding recovers them by solving an overcomplete linear system.
- engineering tie-in: when examining $W_{OV}^{(h)}$ spectra with `latent_projection.py`, sparse directions often correspond to disentangled toy features, matching the Anthropic prediction that finding a privileged basis restores sparsity.

## sparse and latent extensions

### mixture-of-experts feed-forward layers

- router scoring: replace $G_{\ell}$ with $\sum_{e\in\mathcal{E}}\pi_e(X_{\ell}) G_e(X_{\ell})$, where each expert $G_e$ is a rank-expanded mlp and $\pi_e$ is computed via a learned gate. In DeepSeekMoE, $\pi$ performs top-$k$ routing with auxiliary load-balancing loss and combines shared (+ always-on) experts with routed ones.[@deepseekai2024deepseekv2strongeconomicalefficient]
- linear algebra view: the router multiplies token states by a gating matrix $R\in\mathbb{R}^{T\times |\mathcal{E}|}$ with sparse rows; the dispatched expert activations stack into a block-diagonal matrix, making the total Jacobian block-sparse. Inspecting $R^\top R$ diagnoses load imbalance and expert redundancy.
- conditioning: per-token activation of $k$ experts bounds the effective rank of the moe update by $k d_{ff}$, which informs cache planning and gradient checkpointing.

**Definition (top-$k$ routing matrix).** For gate logits $g\in\mathbb{R}^{T\times |\mathcal{E}|}$, define $R_{ie}=\frac{\exp(g_{ie})}{\sum_{e'\in\mathcal{N}_k(i)}\exp(g_{ie'})}$ if $e\in\mathcal{N}_k(i)$, the set of top-$k$ experts for token $i$, and $0$ otherwise. Then $R$ has exactly $k$ nonzero entries per row.

**Lemma (balanced expert utilization).** If the gating coefficients are binary ($R_{ie} \in \{0, 1\}$) and routing is top-$1$ ($k=1$), then a load-balancing constraint enforcing $\sum_i R_{ie} = \frac{T}{|\mathcal{E}|}$ for all $e$ implies $R^\top R$ is diagonal with constant entries $\frac{T}{|\mathcal{E}|}$. For soft gating or $k > 1$, $R^\top R$ contains positive off-diagonal entries representing co-activation overlap.

_Proof._ For binary top-1 routing, each token is assigned to exactly one expert, so $R_{ie_1} R_{ie_2} = 0$ for $e_1 \ne e_2$, establishing orthogonality. The diagonal entries are $\sum_i R_{ie}^2 = \sum_i R_{ie} = \frac{T}{|\mathcal{E}|}$. If $k > 1$, a single token activates multiple experts, yielding off-diagonal overlap. [^moe-proof]

### [[thoughts/MLA|MLA]]

- kv compression: MLA computes one latent $\mathbf c_t^{KV}\in\mathbb R^{d_c}$ per token and caches it with a shared RoPE key. content-key and value up-projections are absorbed into the query and output paths during inference.[@deepseekai2024deepseekv2strongeconomicalefficient]
- matrix factorization: stacking the dense content-key maps gives a shared factorization $W^K=UW^{DKV}$ with rank at most $d_c$. the factors are learned with the model; they are not produced by truncating the SVD of the realized key activations.
- latency gains: the cache width falls from $2n_hd_h$ to $d_c+d_R$. this reduces memory traffic and can admit larger batches. kernel shapes and the extra latent-width contractions determine the actual latency.

**Proposition (best fixed low-rank approximation).** For a fixed matrix $W$ with singular values $\sigma_1\ge\sigma_2\ge\cdots$, the best rank-$d_c$ approximation satisfies

$$
\min_{\operatorname{rank}(\widehat W)\le d_c}
\lVert W-\widehat W\rVert_F^2
=\sum_{i>d_c}\sigma_i^2.
$$

_Proof._ This is the Eckart-Young-Mirsky theorem. it bounds a chosen linear map, not the output error of a trained attention layer.

---

## residual stream/highway networks

### motivation & intuition

- Highway Networks were developed to allow _very deep feedforward_ models to train by giving the network control over how much information to transform vs. carry forward. [@srivastava2015highwaynetworks]
- They generalize residual connections by introducing **learned gates** that modulate how much of the _transformed signal_ vs. the _identity (carry) signal_ passes through.

### architecture & equations

A highway block for a vector $x \in \mathbb{R}^d$ computes:

$$
\begin{aligned}
H(x) &= \mathrm{Transform}(x) = \phi(W_H x + b_H) \\
T(x) &= \mathrm{Gate}(x) = \sigma(W_T x + b_T) \\
C(x) &= 1 - T(x) \quad (\text{carry gate}) \\
y &= H(x) \odot T(x) \;+\; x \odot C(x) \\
  &= H(x) \odot T(x) \;+\; x \odot (1 - T(x))
\end{aligned}
$$

Here:

- $W_H$ and $b_H$ define the “transform” path (like an MLP).
- $W_T$ and $b_T$ define the gating function (sigmoid output in $[0,1]$).
- The gate $T(x)$ regulates _how much of the transformed version vs. the original version_ is used. If $T(x)\approx 1$, the block transforms heavily; if $T(x)\approx 0$, it mostly carries $x$ unchanged.

Put differently, it’s a gated residual: not always add everything, but _weight_ how much residual vs. new transformation enters.

A common variant (the “coupled” version) enforces $C(x) = 1 - T(x)$ so we don’t need a separate carry gate. [@greff2017highwayresidualnetworkslearn]

when $T(x)\equiv1$, the block returns $H(x)$. when $T(x)\equiv0$, it returns $x$. a residual block has the different form $x+F(x)$, so the coupled highway equation does not become a residual block merely by fixing its gate. [@greff2017highwayresidualnetworkslearn]

### derivation & path decomposition

From the path-analysis lens, each block contributes:

- A **carry path**: $x$ flows unmodified (scaled by $C(x)$).
- A **transform path**: $H(x)$ flows scaled by $T(x)$.

Because $T(x)$ depends on $x$ (nonlinear gating), the block is not strictly linear. However, conditional on the gate’s activation regime (for a local neighborhood of $x$), you can treat $T(x)$ as (approximately) constant, making the block behave as an _affine combination_ of identity + transform. That is, for inputs in a region where $T(x)$ is fairly stable, the block is approximately:

$$
y \approx (1 - t)\, I\, x \;+\; t\, H(x), \quad \text{with } t \approx T(x)
$$

Thus the block is a **mixture of two linear/affine pathways** whose weights shift with input. From a circuits perspective, highway layers add _gated split paths_ to the residual stream.

One useful insight from @greff2017highwayresidualnetworkslearn is the view of highway / residual networks as **unrolled iterative estimation**: later layers refine rather than completely overwrite previous representations.

You can embed a mini-proof or derivation slide showing that, under layer gating stability, you can linearize the block and attribute contributions to carry vs. transform paths.

### comparisons in the transformer context

- Transformers use **ungated residual addition** (always add the sublayer’s output). A highway-style gating mechanism inserted into transformer sublayers could let the network dynamically decide whether a sublayer should apply or skip.
- In MoE/conditional layers, gating is already a core idea (choice of expert). Highway gating is conceptually similar: gate whether to apply transformation.
- From interpretability: highway gating gives more granular control of _where_ the network “chooses to intervene” vs. _let the residual pass cleanly_. You could imagine analyzing when gated self-attention or gated MLP is active, akin to diagnosing which tokens “asked” for transformation.

## napkin math for finding $d_{\text{model}}$ to meet pareto frontier

1. **Total parameters**
   - For language models, cross-entropy loss follows smooth **scaling laws** with model size, data, and compute.
   - Within broad ranges, _fine architectural choices like exact width vs depth have second-order effect compared to total params and tokens_.
   - So pick `d_model` primarily to hit your **parameter budget** that’s compute-/data-optimal (then tune depth). [@kaplan2020scalinglawsneurallanguage]

2. **Compute-optimal training ties model size to data size.**
   - Chinchilla showed: for a fixed training FLOPs budget, you should **scale model size and tokens together**
     - roughly “tokens $\approx [20,25] \times \text{params}$”
   - That means your `d_model` (which drives params) should be chosen with the dataset size in mind. [@hoffmann2022trainingcomputeoptimallargelanguage]

3. **Depth >> width (until it doesn’t).**
   - Holding total params fixed, multiple studies find **deeper** transformers often generalize better than purely **wider** ones, up to stability/optimization limits. Practically, don’t try to get all capacity from `d_model`; spread it across more layers and moderate width.

4. **Attention head geometry and KV cache cost constrain width.**
   - Usually $d_\text{model} = n_\text{heads} \times \text{head\_dim}$.
   - The original Transformer used **head_dim $\approx 64$**; many LLMs still target ~64–80.
     - dot-products well scaled (the $\sqrt{d_k}$ factor) and constrains **KV-cache memory/bandwidth** at inference, which scales with $n_\text{heads} \times \text{head\_dim} \times \text{sequence\_length}$.
   - So when you raise `d_model`, know you’re also lifting that cache and bandwidth bill unless you adjust heads.

5. **Hardware likes certain widths.**
   - TensorCores thrive on matrix sizes that align with their MMA tile shapes; practical throughput is better when `d_model` (and intermediate MLP sizes) are multiples of these tile granularities (commonly multiples of 8/16/64, architecture-dependent).
   - Picking `d_model` like 3072, 4096, 5120, 6144, 8192, … isn’t arbitrary (mapping neatly to GEMM tiles and keeping you near peak compute flops).

6. **Tie input and output embeddings unless you have a reason not to.**
   - Use [[thoughts/weight tying]].
   - It reduces parameters (big win when vocab is large) and _improves perplexity/regularization_ in language models.
   - This makes the **embedding dim = unembedding dim = d_model** the default.

### actual napkin math

> [!note] Params from $d_{\text{model}}$, depth, vocab

Roughly, for a decoder-only transformer:

- Embedding + unembedding: $V \times d$ (if not tied: $2Vd$; if tied: $Vd$)
- Attention layers: for each layer, you have Q, K, V, O projections: $4 d^2$ (or more precisely $d\times d_k + d\times d_k + d \times d_v + (h d_v)\times d$)
- MLP layers: often two linear maps of sizes $d \rightarrow d_{\text{ff}}$ and $d_{\text{ff}} \rightarrow d$. If $d_{\text{ff}} = k d$ (e.g. $k=4$), then params $\approx$ $2 k d^2$.
- Residual and LayerNorm biases are smaller and can be ignored for roughness.

So for $L$ layers:

$$
\text{Params} \approx V d + L \big(4 d^2 + 2k d^2\big) = V d + L \big( (4 + 2k) d^2 \big)
$$

If you tie input and output embeddings, drop one $Vd$.

**Example**: Suppose $d = 2048$, $k=4$, and $L = 48$, vocab $V=50{,}000$. Then:

- $(4 + 2k) = 4 + 8 = 12$
- Layer params $\approx$ $48 \times 12 \times 2048^2 = 48 \times 12 \times 4.194 \times 10^6 \approx 2.42 \times 10^{9}$ ($\approx$ 2.4B)
- Embedding params $\approx$ $50{,}000 \times 2048 = 1.024 \times 10^8$ ($\approx$ 100M)
- Total $\approx$ 2.5B parameters (embedding is ~4% of total)

So you see that for large $d$ and $L$, the embedding term is relatively small in well-scaled models.

> [!note] Scaling laws $\to$ balancing data and params

From the Chinchilla insight: for **compute-optimal** training, one should scale **number of tokens** and **number of parameters** roughly proportionally (every doubling of params, double the tokens). (Training-optimal result)

Let $N$ = number of parameters, $D$ = number of training tokens, and $C$ = total compute (FLOPs). If you assume $C \propto N \cdot D$ (i.e., training cost scales linearly in param × tokens), then the optimal solution under some loss model gives

$$
N \sim D
$$

Thus:

$$
C \sim N^2
$$

Therefore:

- $N \sim \sqrt{C}$
- $D \sim \sqrt{C}$

Hence increases in compute should be split between growing the model and seeing more data, not putting all into one side.

| Compute budget scale factor | Param scale | Token scale |
| --------------------------- | ----------- | ----------- |
| 4×                          | 2×          | 2×          |
| 16×                         | 4×          | 4×          |
| 100×                        | 10×         | 10×         |

This is a rough guide: you don't want to overexpand parameters while starving data or vice versa.

> [!note] Attention compute & memory cost for context length $n$

At inference, cost and memory scale with context length $n$. Suppose:

- $d = d_{\text{model}}$
- Number of heads $h$, head dimension $d_k$ (so $d \approx h\,d_k$)

Per layer:

- **Compute**: building $QK^T$ is $O(n^2 d_k)$ (for each head, $n \times d_k$ times $d_k \times n$), then multiply by $V$: $O(n^2 d_v)$. So total $\sim O(h \, n^2 d_k)$.
- **Memory / KV cache**: must store $K$ and $V$ for each token in context: memory ~ $n \times (d_k + d_v) \times h \approx n \times 2d$ roughly.

So if you double context length $n$, compute cost quadruples ($n^2$), memory doubles.

---

## taxonomy of products

| Name                                                      | Domain / Type                                       | Definition / Description                                                                                                                                                                                                                                                                                                           | Key Properties                                                                                                                                                                   |
| --------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Standard / ordinary / (matrix) product**                | Matrices / linear maps                              | If $A$ is $m \times p$ and $B$ is $p \times n$, then $(AB)_{ij} = \sum_{k=1}^p A_{ik} B_{kj}$.                                                                                                                                                                                                                                     | Associative, non-commutative in general; has identity; distributes over addition.                                                                                                |
| **Dot (scalar) product**                                  | Vectors in $\mathbb{R}^n$ (or inner-product spaces) | $\mathbf{x} \cdot \mathbf{y} = \sum_i x_i y_i$ (real) or involves conjugation if complex.                                                                                                                                                                                                                                          | Bilinear (or sesquilinear), symmetric (or Hermitian symmetric), positive definite, yields norm via $\|\cdot\| = \sqrt{\langle x, x\rangle}$.                                     |
| **Inner product**                                         | Real or complex vector spaces                       | A map $\langle\cdot,\cdot\rangle:V\times V\to\mathbb F$ that is linear in one argument, conjugate-linear in the other over $\mathbb C$, conjugate symmetric, and positive definite: $\langle x,x\rangle>0$ for $x\ne0$.                                                                                                            | Induces norm, angle, and orthogonality; the convention determines which argument is linear.                                                                                      |
| **Frobenius inner product**                               | Matrices of same size                               | $\langle A, B \rangle_F = \sum_{i,j} \overline{A}_{ij} B_{ij} = \mathrm{Tr}(A^\dagger B)$.                                                                                                                                                                                                                                         | It’s an inner product on the vector space of matrices; induces Frobenius norm; behaves nicely w.r.t. vectorization.                                                              |
| **Hadamard product (entry-wise product / Schur product)** | Matrices (same size)                                | $(A \odot B)_{ij} = A_{ij} \cdot B_{ij}$.                                                                                                                                                                                                                                                                                          | Commutative, associative; distributes over addition; different from matrix multiplication. Preserves positive semi-definiteness under some conditions (“Schur product theorem”). |
| **Kronecker product**                                     | Two matrices (any sizes) → block matrix             | If $A$ is $m \times n$, $B$ is $p \times q$: $ A \otimes B$ is an $mp \times nq$ block matrix where each entry $a_{ij}$ is replaced by $a_{ij} B$.                                                                                                                                                                                 | Large dimension blow-up; satisfies mixed product properties; related to tensor product; useful for vectorization tricks.                                                         |
| **Block-wise product / block matrix product**             | Block-partitioned matrices                          | You treat big matrices as partitioned into submatrices (“blocks”) and multiply as if treating each block as an entry, provided the partitioning is compatible. The formal rules match standard matrix multiplication, just on blocks.                                                                                              | Allows organize computation; modularity; if subblocks large, this helps computational performance.                                                                               |
| **Tensor product**                                        | Vectors / modules / vector spaces / Hilbert spaces  | Generalization of bilinear product: $u \otimes v$ generates a larger space; extends bilinearly. For matrices, corresponds often to Kronecker. In Hilbert spaces, one defines an inner product on the tensor product by $\langle u_1 \otimes u_2, v_1 \otimes v_2 \rangle = \langle u_1, v_1\rangle \cdot \langle u_2, v_2\rangle$. | Not commutative; but associative up to canonical isomorphism; basis expansion; useful in multilinear algebra, quantum mechanics, etc.                                            |
| **Petersson inner product**                               | Spaces of modular (or cusp) forms                   | $\langle f, g \rangle = \int_{F} f(\tau)\overline{g(\tau)} (\operatorname{Im}\tau)^k d\nu(\tau)$ (with fundamental domain etc.)                                                                                                                                                                                                    | Hermitian form; positive definite; invariance under modular transformations; central in number theory, modular forms.                                                            |

### Additional “products” worth knowing

- **Convolution product** (for functions, sequences): $(f * g)(t) = \int f(s) g(t-s)\, ds$ or discrete sum.
- **Outer product** of vectors: For $u \in \mathbb{R}^m, v \in \mathbb{R}^n$, the outer product is the $m \times n$ matrix $u v^\top$.
- **Tensor contraction**: combining indices of tensors, generalizing matrix multiplication.
- **Wedge product / exterior product**: in differential geometry / algebra, antisymmetric product of vectors/forms.
- **Cross product** (in 3D): vector product giving a vector orthogonal to inputs.
- **Hadamard / elementwise product** (we already listed).
- **Composition of functions** (product in a category sense).
- **Direct sum / direct product** (though “product” here is categorical / structural, not multiplicative).
- **Boolean product**: e.g. in Boolean algebra/matrices, product where multiplication = AND, addition = OR.
- **Convolution / group-algebra product**: for functions on a group.
- **Dotting / pairing of dual vectors**: pairing a vector in $V$ with one in the dual space $V^*$ to give a scalar.

### relationship & hierarchy

- the dot product is the standard inner product on $\mathbb{R}^n$; the general inner product extends it, and the matrix case is the Frobenius inner product.
- Hadamard is pointwise product as opposed to standard matrix multiplication.
- Kronecker / tensor product enlarge dimension; outer product is a simple tensor product between vectors.
- Blockwise product is really just standard matrix multiplication viewed on blocks, or sometimes Kronecker-type constructions.
