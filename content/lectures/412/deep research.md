---
id: deep research
tags:
  - seed
  - workshop
  - augmented
description: made with gpt-5 and claude
date: "2025-09-26"
modified: 2025-09-26 16:35:00 GMT-04:00
noindex: true
title: supplement for supplement for 0.412
---

see also: [[lectures/412/notes]], [[lectures/412/tools]], [[thoughts/transformer circuits framework]], [[thoughts/attention]], [[thoughts/induction heads]]

## introduction

> [!summary] framing
> transformers carry "an enormous amount of linear structure"; we read a GPT-style decoder by spelling out the matrices that move information through the residual stream.

We stay inside a decoder-only transformer with causal masking, the GPT archetype that Anthropic unpack in their transformer-circuits programme. Each subsection pairs the intuitive story with the algebraic map it instantiates so practising engineers can trace the matrices actually acting on the residual stream.

## token embeddings and positional encodings

> [!tip] mental model
> embeddings map discrete ids into vectors; positional encodings shift those vectors so every (token, position) pair lands in a unique spot.

**linear map.** With $W_e \in \mathbb{R}^{V \times d_{\text{model}}}$ the token embedding is $e_i$. Stacking sequence one-hots $S$ yields $S W_e \in \mathbb{R}^{n\times d_{\text{model}}}$.

**positional offsets.** Learned (or sinusoidal) vectors $p_i$ add elementwise:

$$
\mathrm{inputVector}_i = \mathrm{tokenEmbedding}(t_i) + p_i.
$$

This breaks permutation symmetry and feeds order information to attention.

**intuition.** After this stage each row already says “what” and “where”; attention only has to remix rows via matrix multiplications.

## self-attention mechanism (scaled dot-product attention)

> [!note] two step view
> attention scores relevance, then transports value vectors along those scores.

1. **projections.** For token representation $x_i$ compute queries $q_i = x_i W^Q$, keys $k_i = x_i W^K$, values $v_i = x_i W^V$.
2. **masked scores.** Form $(q_i k_j^T) / \sqrt{d_k}$ and add the causal mask $M$ so positions cannot look ahead.
3. **softmax weights.** Row-wise softmax produces the attention matrix $A$ with non-negative rows summing to $1$.
4. **weighted aggregation.** Stack values into $V$ and return

$$
O = \mathrm{softmax}\!\Big(\frac{Q K^T}{\sqrt{d_k}}\Big)\; V. \tag{1}
$$

The only nonlinearity is the softmax; everything else is linear algebra. Freeze $A$ and a single head becomes a pure linear operator on the residual stream.

> [!example] diagnostic
> plotting $A$ as a heatmap reveals structures such as diagonal bands for induction heads or column spikes for copy heads.

## multi-head attention

> [!idea] parallel focus
> multiple heads share the same input but learn distinct routing patterns and value subspaces.

- **per-head projections.** Each head has its own $W^Q_i, W^K_i, W^V_i$ and produces $O_i = \operatorname{softmax}(Q_i K_i^	op / \sqrt{d_k}) V_i$.
- **mixing.** Concatenate head outputs and project back with $W^O$:

  $$
  M = [O_1 \parallel O_2 \parallel \cdots \parallel O_h] W^O.
  $$

- **equivalent view.** Because $W^O$ is linear, this is the same as summing head contributions in the residual stream—exactly the perspective used in transformer circuits analyses.

## residual connections and layer normalization

> [!important] plumbing
> residual adds keep earlier information accessible; layernorm re-centres and rescales to stabilise depth.

- **residual add.** Each sublayer outputs $F(x)$ and writes $x + F(x)$ back into the residual stream, ensuring gradients have a short path and features persist when updates are small.
- **layer normalization.** Pre-norm GPT variants normalise each token vector to zero mean and unit variance before attention/MLP, using an affine map $\gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \eta$. The Jacobian projects onto the mean-zero subspace and scales by $\gamma$, preserving linear structure locally.

## position-wise feed-forward network (mlp layer)

> [!note] local experts
> attention mixes across tokens; the MLP refines each position independently.

- **structure.** Apply $h = W_1 x + b_1$, nonlinearity $\phi$ (ReLU or GELU), then $y = W_2 \phi(h) + b_2$ with $W_1 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$ and $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$.
- **linear view.** Freeze $\phi'$, and the MLP acts as $W_2 \operatorname{diag}(\phi'(h)) W_1$: a low-rank update writing features back into the residual stream.
- **interpretation.** Geva et al. model these neurons as key-value memories—$W_1$ encodes keys, the activation selects them, $W_2$ writes the stored values.

## mixture-of-experts feed-forward layers (advanced)

> [!special] sparse capacity
> moe swaps the single MLP for a pool of experts and a routing gate, letting different tokens trigger different parameter subsets.

- **routing.** A gate produces logits $g_{ie}$; keep top-$k$ experts per token, normalise with softmax, and form a sparse routing matrix $R$.
- **per-expert processing.** Each chosen expert $G_e$ runs its own MLP on the token; outputs recombine by the routing weights.
- **linear algebra view.** Dispatch can be written as block-diagonal multiplication followed by $R$ and $R^\top$; $R^\top R$ diagnoses expert balance and interference.

## conclusion

> [!summary] takeaways
> the decoder stack is a composition of linear maps plus softmax/nonlinear gates, so analysing it is largely an exercise in linear algebra.

- embeddings + positional offsets: place tokens in a residual basis.
- attention: linear mixing guided by a softmax-controlled routing matrix.
- multi-head + residual + layernorm: parallel low-rank updates that accumulate additively.
- mlp and moe layers: position-wise (or routed) low-rank maps with nonlinear selectors.

Thinking in these terms keeps transformers interpretable: you can trace circuits, find privileged bases, and reason about how architectural tweaks shift the underlying matrices without changing the core intuition.

---

## residual stream/highway networks

### motivation & intuition

- Highway Networks were developed to allow _very deep feedforward_ models to train by giving the network control over how much information to transform vs. carry forward. [@srivastava2015highwaynetworks]
- They generalize residual connections by introducing **learned gates** that modulate how much of the _transformed signal_ vs. the _identity (carry) signal_ passes through.

### architecture & equations

A highway block for a vector $x \in \mathbb{R}^d$ computes:

$$
\begin{aligned}
H(x) &= \mathrm{Transform}(x) = x W_H + b_H \quad (\text{linear + nonlinearity}) \\
T(x) &= \mathrm{Gate}(x) = \sigma(x W_T + b_T) \\
C(x) &= 1 - T(x) \quad (\text{carry gate}) \\
y &= H(x)\, T(x) \;+\; x\, C(x) \\
  &= H(x)\, T(x) \;+\; x\, (1 - T(x))
\end{aligned}
$$

Here:

- $W_H$ and $b_H$ define the “transform” path (like an MLP).
- $W_T$ and $b_T$ define the gating function (sigmoid output in $[0,1]$).
- The gate $T(x)$ regulates _how much of the transformed version vs. the original version_ is used. If $T(x)\approx 1$, the block transforms heavily; if $T(x)\approx 0$, it mostly carries $x$ unchanged.

Put differently, it’s a gated residual: not always add everything, but _weight_ how much residual vs. new transformation enters.

A common variant (the “coupled” version) enforces $C(x) = 1 - T(x)$ so we don’t need a separate carry gate. [@greff2017highwayresidualnetworkslearn]

When $T(x)\equiv 1$ always, the highway block reduces to a plain transformed layer; when $T(x)\equiv 0$, it passes identity (just skip). If gates are always “open” (always carry), it reduces to residual. Thus highway nets generalize residuals by _learning_ gating. [@greff2017highwayresidualnetworkslearn]

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
   - Usually `d_model = n_heads × head_dim`.
   - The original Transformer used **head_dim $\approx$ 64**; many LLMs still target ~64–80.
     - dot-products well scaled (the $\sqrt{d_k}$ factor) and constrains **KV-cache memory/bandwidth** at inference, which scales with `n_heads × head_dim × sequence_length`.
   - So when you raise `d_model`, know you’re also lifting that cache and bandwidth bill unless you adjust heads.

5. **Hardware likes certain widths.**
   - TensorCores thrive on matrix sizes that align with their MMA tile shapes; practical throughput is better when `d_model` (and intermediate MLP sizes) are multiples of these tile granularities (commonly multiples of 8/16/64, architecture-dependent).
   - Picking `d_model` like 3072, 4096, 5120, 6144, 8192, … isn’t arbitrary—it maps neatly to GEMM tiles and keeps you near peak compute flops

6. **Tie input and output embeddings unless you have a reason not to.**
   - Use [[thoughts/weight typing]].
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

| Name                                                      | Domain / Type                                                            | Definition / Description                                                                                                                                                                                                                                                                                                           | Key Properties                                                                                                                                                                   |
| --------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Standard / ordinary / (matrix) product**                | Matrices / linear maps                                                   | If $A$ is $m \times p$ and $B$ is $p \times n$, then $(AB)_{ij} = \sum_{k=1}^p A_{ik} B_{kj}$.                                                                                                                                                                                                                                     | Associative, non‐commutative in general; has identity; distributes over addition.                                                                                                |
| **Dot (scalar) product**                                  | Vectors in $\mathbb{R}^n$ (or inner‐product spaces)                      | $\mathbf{x} \cdot \mathbf{y} = \sum_i x_i y_i$ (real) or involves conjugation if complex.                                                                                                                                                                                                                                          | Bilinear (or sesquilinear), symmetric (or Hermitian symmetric), positive definite, yields norm via $\|\cdot\| = \sqrt{\langle x, x\rangle}$.                                     |
| **Inner product**                                         | In any real/complex vector space (or other modules) with extra structure | Generalizes dot product: a map $\langle \cdot , \cdot \rangle: V \times V \to \mathbb{F}$ satisfying linearity, conjugate symmetry (if over $\mathbb{C}$), positive definiteness.                                                                                                                                                  | Induces norm, angle, orthogonality; symmetric/Hermitian; positive definite; linear in first (or second) argument depending on convention.                                        |
| **Frobenius inner product**                               | Matrices of same size                                                    | $\langle A, B \rangle_F = \sum_{i,j} \overline{A}_{ij} B_{ij} = \mathrm{Tr}(A^\dagger B)$.                                                                                                                                                                                                                                         | It’s an inner product on the vector space of matrices; induces Frobenius norm; behaves nicely w.r.t. vectorization.                                                              |
| **Hadamard product (entry-wise product / Schur product)** | Matrices (same size)                                                     | $(A \odot B)_{ij} = A_{ij} \cdot B_{ij}$.                                                                                                                                                                                                                                                                                          | Commutative, associative; distributes over addition; different from matrix multiplication. Preserves positive semi-definiteness under some conditions (“Schur product theorem”). |
| **Kronecker product**                                     | Two matrices (any sizes) → block matrix                                  | If $A$ is $m \times n$, $B$ is $p \times q$: $ A \otimes B$ is an $mp \times nq$ block matrix where each entry $a_{ij}$ is replaced by $a_{ij} B$.                                                                                                                                                                                 | Large dimension blow-up; satisfies mixed product properties; related to tensor product; useful for vectorization tricks.                                                         |
| **Block-wise product / block matrix product**             | Block‐partitioned matrices                                               | You treat big matrices as partitioned into submatrices (“blocks”) and multiply as if treating each block as an entry, provided the partitioning is compatible. The formal rules match standard matrix multiplication, just on blocks.                                                                                              | Allows organize computation; modularity; if subblocks large, this helps computational performance.                                                                               |
| **Tensor product**                                        | Vectors / modules / vector spaces / Hilbert spaces                       | Generalization of bilinear product: $u \otimes v$ generates a larger space; extends bilinearly. For matrices, corresponds often to Kronecker. In Hilbert spaces, one defines an inner product on the tensor product by $\langle u_1 \otimes u_2, v_1 \otimes v_2 \rangle = \langle u_1, v_1\rangle \cdot \langle u_2, v_2\rangle$. | Not commutative; but associative up to canonical isomorphism; basis expansion; useful in multilinear algebra, quantum mechanics, etc.                                            |
| **Petersson inner product**                               | Spaces of modular (or cusp) forms                                        | $\langle f, g \rangle = \int_{F} f(\tau)\overline{g(\tau)} (\operatorname{Im}\tau)^k d\nu(\tau)$ (with fundamental domain etc.)                                                                                                                                                                                                    | Hermitian form; positive definite; invariance under modular transformations; central in number theory, modular forms.                                                            |

### Additional “products” worth knowing

To be thorough, here are more “product”‐like operations that occur often in math / ML / algebra.

- **Convolution product** (for functions, sequences): $(f * g)(t) = \int f(s) g(t−s) ds$ or discrete sum.
- **Outer product** of vectors: For $u \in \mathbb{R}^m, v \in \mathbb{R}^n$, the outer product is the $m \times n$ matrix $u v^\top$.
- **Tensor contraction**: combining indices of tensors, generalizing matrix multiplication.
- **Wedge product / exterior product**: in differential geometry / algebra, antisymmetric product of vectors/forms.
- **Cross product** (in 3D): vector product giving a vector orthogonal to inputs.
- **Hadamard / elementwise product** (we already listed).
- **Composition of functions** (product in a category sense).
- **Direct sum / direct product** (though “product” here is categorical / structural, not multiplicative).
- **Boolean product**: e.g. in Boolean algebra/matrices, product where multiplication = AND, addition = OR.
- **Convolution ‒ group algebra product**: for functions on a group.
- **Dotting / pairing of dual vectors**: pairing a vector in V with one in the dual space V\* to give scalar.

### relationship & hierarchy

- Inner product ⊂ scalar / dot product for vectors; generalizes to matrices (Frobenius).
- Hadamard is pointwise product as opposed to standard matrix multiplication.
- Kronecker / tensor product enlarge dimension; outer product is a simple tensor product between vectors.
- Blockwise product is really just standard matrix multiplication viewed on blocks, or sometimes Kronecker‐type constructions.
