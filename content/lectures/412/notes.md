---
date: "2025-09-12"
description: linear algebra applications for transformers
id: notes
modified: 2025-11-11 06:58:34 GMT-05:00
slides: true
tags:
  - workshop
  - linalg
title: supplement to 0.412
transclude:
  title: false
---

see also: [[thoughts/Transformers]], [[thoughts/Attention]], [[thoughts/LLMs]], [[lectures/412/deep research|walkthrough]], [[lectures/412/tools]]

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
  - Rotary or sinusoidal position encodings add $P\in\mathbb{R}^{T\times d_{model}}$ before the first block, keeping the update linear in $E$ and $P$. [@vaswani2023attentionneed]
- layer norm: treat pre-normalization as an affine map $\mathcal{N}(x)=D(x)(x-\mu(x))$ with diagonal scaling $D(x)$. In practice we approximate $D$ as invertible on the support of interest to reason about nearby linearizations.
- residual stream: every block writes back to $X$ through additive updates, so we model the stack as

$$
X_{\ell+1} = X_{\ell} + F_{\ell}(X_{\ell}), \quad \ell = 0,\dots,L-1
$$

where $F_{\ell}$ sums attention and mlp contributions.

**Definition (token embedding map).** The map $\varepsilon: \{1,\dots,|\mathcal{V}|\}\to\mathbb{R}^{d_{model}}$ satisfies $\varepsilon(i)=e_i^\top E$ with $e_i$ the $i$th standard basis vector. For a one-hot token matrix $S$, the initial residual matrix is $X_0 = S E$.

**Lemma (distinguishability with positional encodings).** Suppose positions $0\le i<j<T_{max}$ have distinct positional vectors $p_i\ne p_j$ and that $\operatorname{span}\{E\}-\operatorname{span}\{P\}$ intersects only at $0$. Then the combined embedding map $\tilde\varepsilon(i,\text{pos}) = \varepsilon(i)+p_{\text{pos}}$ is injective over token-position pairs.

_Proof._ If $\tilde\varepsilon(i,\text{pos})=\tilde\varepsilon(j,\text{pos}')$, subtracting yields $\varepsilon(i)-\varepsilon(j)=p_{\text{pos}'}-p_{\text{pos}}$. By the direct-sum assumption the two sides lie in complementary subspaces, forcing both to be zero. Distinct positional vectors give $\text{pos}=\text{pos}'$, and injectivity of $\varepsilon$ then implies $i=j$. [^embed-proof]

**Lemma (layer norm linearization).** For inputs constrained to a neighborhood where $\sigma(x)>0$, the layer norm $\mathcal{N}(x) = \gamma\odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$ admits a first-order linearization $J_{\mathcal{N}}(x) = \operatorname{diag}(\gamma)B_x$ where $B_x$ is a doubly stochastic matrix projecting onto the mean-zero subspace.

_Proof._ Differentiate the mean and variance terms to obtain $\nabla_x \mu = \tfrac{1}{d}\mathbf{1}$ and $\nabla_x \sigma = \tfrac{1}{d\sigma}(x-\mu\mathbf{1})$. Substituting into the Jacobian yields $B_x = I - \tfrac{1}{d}\mathbf{1}\mathbf{1}^\top - \frac{(x-\mu\mathbf{1})(x-\mu\mathbf{1})^\top}{d\sigma^2}$, which is linear and projects onto the hyperplane orthogonal to $\mathbf{1}$. [^layernorm-proof]

### single-head attention as a tensor product

Fix a head $h$ inside layer $\ell$. Following the transformer-circuits factorization we write the attention output as a Kronecker-style product[^Kronecker] acting on $X_{\ell}$:[@elhage2021mathematical]

$$
H^{(h)}(X_{\ell}) = A^{(h)}(X_{\ell}) \otimes W_{OV}^{(h)} X_{\ell}
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

with $A^{(h)}(X_{\ell})\in\mathbb{R}^{T\times T}$ encoding token-to-token routing and $W_{OV}^{(h)} = W_O^{(h)}W_V^{(h)}\in\mathbb{R}^{d_{model}\times d_{model}}$ a low-rank map ($\mathrm{rank} \le d_{head}$).

- Because the only nonlinearity resides in the softmax defining $A^{(h)}$, freezing $A^{(h)}$ linearizes the head completely.
- the idea of [[thoughts/mathematical framework transformers circuits#attention heads as information movement|information movement]] comes from the fact attention compute value vectors for each tokens from residual stream, and linearly combine each of those vectors based on attention pattern.
  - i.e linear map $[n_\text{context}, d_{\text{model}}] \to [n_{\text{context}}, d_{\text{model}}]$
- This enables path analysis:
  - expand logits as sums of products of $W_{OV}$ and $W_{QK}$ matrices and map each product to a circuit acting on a basis vector in the residual stream. [@elhage2021mathematical]

- An attention head is really applying two linear operations,$A$ and $W_O W_V$ which operate on different dimensions and act independently.
  - $A$ governs which token's information to move from and to
  - $W_O W_V$ governs which information is read from the source token and how it is written to the destination token.

> [!tip] intuition
> think of $W_{OV}$ as choosing a subspace to copy from the source token and another to write into at the destination. singular vectors of $W_{OV}$ highlight reusable features; tracking them across layers surfaces induction heads and copy circuits.

- causal mask: define $M\in\mathbb{R}^{T\times T}$ with $M_{ij}=0$ for $j\le i$ and $M_{ij}=-\infty$ otherwise. Then $A^{(h)} = \operatorname{softmax}\!\left(\frac{Q^{(h)}K^{(h)\top}}{\sqrt{d_k}} + M\right)$ is strictly lower triangular, ensuring autoregressive causality.
- stochasticity: each row of $A^{(h)}$ sums to 1 because $\operatorname{softmax}$ outputs a probability simplex vector, so attention matrices lie in the Birkhoff polytope[^birkhoff] intersected with the causal mask.

**Proposition (masked attention operators).** Let $L\in\mathbb{R}^{T\times T}$ be strictly lower triangular. Then $A = \operatorname{softmax}(L)$ is nilpotent of index $T$ when interpreted as an operator on the quotient space mod residual self-contributions.

_Proof._ The masked structure forces $A$ to have zeros on and above the diagonal. Because multiplication of strictly lower triangular matrices increases the subdiagonal width, $A^T=0$.

#### observations on single-head attention

- $A$ is the **only non-linear part** of this equation (being computed from a softmax). This means that if we fix the attention pattern, attention heads perform a linear operation. This also means that, **without fixing** $A$, attention heads are “half-linear” in some sense, since the per-token linear operation is constant.
- $W_Q$ and $W_K$ always operate together. They’re never independent. Similarly, $W_O$ and $W_V$ always operate together as well.
  - Although they’re parameterized as separate matrices, $W_O W_V$ and $W_Q^T W_K$ can always be thought of as individual, low-rank matrices.
  - Keys, queries and value vectors are, in some sense, superficial. They're intermediary by-products of computing these low-rank matrices. One could easily reparameterize both factors of the low-rank matrices to create different vectors, but still function identically.
  - Because $W_O W_V$ and $W_Q W_K$ always operate together, we like to define variables representing these combined matrices, $W_{OV} = W_O W_V$ and $W_{QK} = W_Q^T W_K$.
- Products of attention heads behave much like attention heads themselves. By the distributive property,

  $$
    (A^{h_2} \otimes W_{OV}^{h_2}) \cdot (A^{h_1} \otimes W_{OV}^{h_1})
    = (A^{h_2} A^{h_1}) \otimes (W_{OV}^{h_2} W_{OV}^{h_1}).
  $$

  The result of this product can be seen as functionally equivalent to an attention head, with an attention pattern which is the composition of the two heads $A^{h_2} A^{h_1}$ and an output-value matrix $W_{OV}^{h_2} W_{OV}^{h_1}$. We call these “virtual attention heads”, discussed in more depth later.

### block-level composition

multi-head aggregation: a layer with heads $h=1,\dots,H$ yields

$$
M_{\ell}(X_{\ell}) = \sum_{h=1}^{H} H^{(h)}(X_{\ell})
$$

and the residual update becomes $X_{\ell+1}=X_{\ell}+M_{\ell}(X_{\ell})+G_{\ell}(X_{\ell})$ where $G_{\ell}$ is the mlp contribution.

- mlp as rank-expansion: in decoder-only GPT blocks, $G_{\ell}(x)=W_2\,\phi(W_1 x)$ with $W_1\in\mathbb{R}^{d_{ff}\times d_{model}}$, $W_2\in\mathbb{R}^{d_{model}\times d_{ff}}$. linearize by freezing the diagonal Jacobian of $\phi$ to obtain a low-rank update governed by $W_2\operatorname{diag}(\phi'(W_1 x))W_1$.
- key-value memory view: geva et al. show that feed-forward neurons implement key-value associative memories; $W_1$ encodes keys, the activation acts as a selector, and $W_2$ writes cached values back into the residual stream, matching the residual-circuit picture.[@geva2021transformerfeedforwardlayerskeyvalue]
- virtual heads: products like $H^{(h_2)}\circ H^{(h_1)}$ manifest as new low-rank operators with attention $A^{(h_2)}A^{(h_1)}$ and OV matrix $W_{OV}^{(h_2)}W_{OV}^{(h_1)}$. these virtual heads explain induction circuits spanning multiple layers.[@elhage2021mathematical]

**Proposition (Residual path expansion).** Freezing softmax weights in every attention head induces a linear map $T:\mathbb{R}^{T\times d_{model}}\to\mathbb{R}^{T\times d_{model}}$ whose matrix factors into a sum over directed paths $p$ through attention and mlp nodes: $T=\sum_{p}W_U^{(p)} W_{OV}^{(p_k)}\cdots W_{OV}^{(p_1)}$.

_Proof._ If we rewrite each attention head as $A^{(h)}W_{OV}^{(h)}$ and each mlp as $D_{\phi}(x)W_2W_1$. When the routing matrices $A^{(h)}$ are held fixed, every block becomes affine-linear. Unrolling the $L$ residual blocks yields

$$
X_L = X_0 + \sum_{\ell=0}^{L-1} F_{\ell}(X_{\ell})
$$

and inductively substituting $X_{\ell}$ produces a Neumann-like expansion where each term is a product of head/ff matrices applied to the seed $X_0$. Grouping by unique sequences of heads gives the stated sum; tails that end at the unembedding $W_U$ map residual features into logits.[@elhage2021mathematical][^path-proof]

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

[^embed-proof]: The argument simply states that the column space of $E$ (token features) and the span of positional vectors intersect only at the zero vector. Thus the only way a token difference can equal a positional difference is if both sides vanish, forcing the tokens and positions to match.

[^layernorm-proof]: Taking derivatives shows layer norm subtracts the mean direction and rescales by variance. The Jacobian therefore projects onto the mean-zero hyperplane while applying the learned gain $\gamma$, which is the linear approximation used in analyses.

[^path-proof]: Expanding the residual recursion is equivalent to unrolling a power series: every term corresponds to a path that alternates attention and MLP updates before finally mapping to logits via $W_U$.

[^geva-proof]: Geva et al. show that each neuron behaves like a key–value pair: $K_j$ is a key direction, the activation gate decides whether to fire it, and $v_j$ is the value written back. Summing $a_j v_j$ across neurons reconstructs the residual update while exposing where features overlap.

[^privileged-proof]: Conjugating $W_{OV}^{(h)}$ by its eigenbasis just rotates into coordinates where the operator is diagonal. Any downstream linear probe composes in the same way, so analyzing logits in that basis preserves the output while making feature axes explicit.

[^moe-proof]: When the router enforces equal traffic per expert, the Gram matrix $R^\top R$ becomes diagonal. This means expert activations do not interfere (off-diagonal zero) and each diagonal entry reflects uniform utilisation, the condition required by the balancing loss.

[^birkhoff]: The Birkhoff polytope is the convex hull of permutation matrices. Any row-stochastic attention matrix (non-negative rows summing to one) lies inside it, meaning attention can be interpreted as a probabilistic mixture of permutations that route information between tokens (with causality enforced via masking).

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

**Lemma (balanced expert utilization).** If the auxiliary load-balancing loss enforces $\sum_i R_{ie} = \frac{kT}{|\mathcal{E}|}$ for all $e$, then $R^\top R$ is diagonal with constant entries $\frac{kT}{|\mathcal{E}|}$, minimizing interference between experts.

_Proof._ Orthogonality of different experts follows from disjoint token assignments; equal column sums give identical diagonal entries in $R^\top R$. [^moe-proof]

### multi-head latent attention (mla)

- kv compression: mla computes token-level latent codes $Z\in\mathbb{R}^{T\times d_{latent}}$ via a learned projection $W_{latent}$, storing $Z$ instead of full key/value tensors. During inference, queries multiply the latent cache through shared mixing matrices, approximating the original attention scores while shrinking memory by >90% on DeepSeek-V2.[@deepseekai2024deepseekv2strongeconomicalefficient]
- matrix factorization: interpret mla as factoring the key matrix $K=W_K X$ into $K=U Z$ with $U\in\mathbb{R}^{d_{model}\times d_{latent}}$. Selecting $d_{latent}\ll d_{model}$ enforces a low-rank structure; the reconstruction error aligns with the neglected singular values of $K$.
- latency gains: the compressed cache allows wider batch sizes because attention now costs $O(T d_{latent})$ per head instead of $O(T d_{model})$. The trade-off surfaces as condition number growth in $U$; monitor $\kappa_2(U)$ to guard against numerical drift at long context lengths.

**Proposition (spectral error bound).** Let $K = U Z$ be the rank-$d_{latent}$ approximation from the mla factorization and $\sigma_i$ the singular values of the original key matrix. Then $\|K - K_{\text{full}}\|_F^2 = \sum_{i>d_{latent}} \sigma_i^2$, so truncating at $d_{latent}$ discards precisely the tail energy.

_Proof._ Apply the Eckart–Young–Mirsky theorem to the SVD of $K_{\text{full}}$.

> [!example] experiment stub
> use `python content/lectures/412/latent_projection.py --head 3` to sample a random head from a checkpoint shard, compute its $W_{OV}$ spectrum, and visualize the latent cache reconstruction error.

## practice agenda

- revisit parsing tasks from [[lectures/2/attention first principle]] using the residual stream basis; show how induction heads form through $Q$-composition.
- feed real activation statistics from a GPT-style checkpoint through the new tools to inspect $W_{OV}$ singular values and moe router sparsity.
- derive the logit lens: multiply the post-block residual by the unembedding $U\in\mathbb{R}^{d_{model}\times |\mathcal{V}|}$ to map subspace contributions directly to token probability shifts; tie this back to the path expansion pathways.[@elhage2021mathematical]
- challenge: extend mla analysis by swapping rotary embeddings with learned affine positional encodings and measure changes in latent rank.
