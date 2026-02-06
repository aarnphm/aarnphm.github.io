---
created: '2025-09-17'
date: '2025-09-17'
description: Shorthand notation for tensor operations using repeated indices to imply summation
id: einstein notation
modified: 2025-12-18 01:35:52 GMT-05:00
published: '2003-03-12'
source: https://en.wikipedia.org/wiki/Einstein_notation
tags:
  - math
---

> a notational convention that implies summation over a set of indexed terms in a formula, achieving brevity.

It's a notational subset of Ricci calculus, often used in physics applications. Albert Einstein introduced it to physics in 1916.

## statement of convention

When an index variable appears twice in a single term and is not otherwise defined, it implies summation over all values of that index. For indices ranging over {1, 2, 3}:

$$y = \sum_{i=1}^{3} x^i e_i = x^1 e_1 + x^2 e_2 + x^3 e_3$$

is simplified to:

$$y = x^i e_i$$

The upper indices are not exponents but indices of coordinates, coefficients, or basis vectors. In this context, $x^2$ represents the second component of $x$, not $x$ squared.

### index conventions

- **Greek alphabet** ($\mu, \nu, \ldots$): space-time components, indices take values 0, 1, 2, 3
- **Latin alphabet** ($i, j, \ldots$): spatial components only, indices take values 1, 2, 3

### index types

- **Summation index** (dummy index): summed over, can be replaced by any symbol
- **Free index**: appears only once per term, usually appears in every term of an equation

Example: In $v_i = a_i b_j x^j$, "i" is free and "j" is summed.

## vector representations

### superscripts vs subscripts

In covariance/contravariance contexts:

- **Upper indices**: contravariant vector components (vectors)
- **Lower indices**: covariant vector components (covectors)

$$
\begin{aligned}
v &= v^i e_i = \begin{bmatrix} e_1 & e_2 & \cdots & e_n \end{bmatrix} \begin{bmatrix} v^1 \\ v^2 \\ \vdots \\ v^n \end{bmatrix} \\
w &= w_i e^i = \begin{bmatrix} w_1 & w_2 & \cdots & w_n \end{bmatrix} \begin{bmatrix} e^1 \\ e^2 \\ \vdots \\ e^n \end{bmatrix}
\end{aligned}
$$

### mnemonics

- "**Up**per indices go **up** to down; **l**ower indices go **l**eft to right"
- "**Co**variant tensors are **row** vectors with indices **below**"
- Covectors are row vectors: $[w_1 \cdots w_k]$
- Contravariant vectors are column vectors: $\begin{bmatrix} v^1 \\ \vdots \\ v^k \end{bmatrix}$

## common operations

### inner product

$$\langle \mathbf{u}, \mathbf{v} \rangle = u_j v^j$$

For orthonormal basis: $\langle \mathbf{u}, \mathbf{v} \rangle = u_j v^j$

### vector cross product

In 3D with positively oriented orthonormal basis:

$$\mathbf{u} \times \mathbf{v} = \varepsilon_{jk}^i u^j v^k \mathbf{e}_i$$

where $\varepsilon_{jk}^i = \varepsilon_{ijk}$ is the Levi-Civita symbol.

### matrix-vector multiplication

$$u^i = {A^i}_j v^j$$

### matrix multiplication

$${C^i}_k = {A^i}_j {B^j}_k$$

### trace

For square matrix ${A^i}_j$: $\text{tr}(A) = {A^i}_i$

### Outer Product

$${A^i}_j = u^i v_j$$

### raising and lowering indices

Using metric tensor $g_{\mu u}$:

- Lower an index: $g_{\mu\sigma} {T^\sigma}_\beta = T_{\mu\beta}$
- Raise an index: $g^{\mu\sigma} {T_\sigma}^\alpha = T^{\mu\alpha}$

## computational `einsum` interfaces

The core of an `einsum` API is a compiler from symbolic Einstein notation into explicit contraction loops. Each backend shares index grammar but differs in broadcasting rules, optimization strategies, and device execution.

### `numpy.einsum`

- Signature: `einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe', optimize=False)`.
- Subscripts grammar allows comma-separated input terms followed by `->` and output indices, e.g. `"ij,jk->ik"` mirrors $C^i{}_k = A^i{}_j B^j{}_k$.
- Repeated labels within a term contract that axis; labels absent from output disappear (summed out). Ellipsis `...` maps to unmatched leading dimensions, matching Einstein's implicit summation over all coordinates of a dummy index family.
- `optimize=True` invokes the [Brock et al. optimized path search](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html#numpy.einsum) to reorder contractions, analogous to choosing an efficient parenthesization of a multi-index expression. The resulting operation fuses elementwise multiplications and reductions into a single kernel, minimizing temporaries.

#### mapping examples

- Inner product $u_j v^j$: `numpy.einsum("i,i->", u, v)` produces a scalar; the empty output index list expresses total contraction.
- Trace ${A^i}_i$: `numpy.einsum("ii->", A)` matches the Einstein contraction $\text{tr}(A)$.
- Batch matrix multiply with broadcasting: `numpy.einsum("bij,bjk->bik", A, B)` extends the dummy index set with batch label `b`, mirroring a family of independent contractions.

### `torch.einsum`

- PyTorch copies NumPy's syntax but lowers to ATen tensor iterator kernels. Signature: `torch.einsum(equation, *operands)`.
- Gradients propagate through the symbolic contraction graph because the backward pass re-applies the same Einstein pattern with complementary free indices. This aligns with viewing $z = x^i y_i$ as a bilinear form whose differentials follow the same index wiring.
- Device semantics: computations execute on the operands' device (CPU, CUDA, MPS). Mixed-device inputs are rejected, unlike continuous Einstein notation which is device-agnostic.
- PyTorch supports uppercase/lowercase labels uniformly; it reserves no special symbols aside from ellipsis. Broadcasting follows PyTorch semantics prior to contraction, so you can emulate $T^{ij}{}_{k\ell} u^k v^\ell$ with `torch.einsum("ijab,b->ija", T, v)` by leaving the contracted labels absent from the output term.

#### performance notes

- For static shapes, `opt_einsum` or `torch.compile` can fuse repeated contractions. Conceptually this is choosing a better summation tree for the Einstein expression.
- On CUDA, reduce dimensions should be contiguous to avoid extra transposes; re-labeling indices to align with the backend's memory layout matches the tensor-density intuition that Einstein summations prefer compatible basis orderings.

### `einops`

- `einops.rearrange`, `reduce`, and `einsum` maintain the same symbolic idea but treat axis names as semantic tags rather than single characters. `einops.einsum("b t h, h d -> b t d", queries, weights)` reads closer to natural-language Einstein notation where labels are words.
- The library separates _pattern_ from _equation_: `rearrange` handles pure re-indexing (no summation), while `reduce` introduces explicit aggregations (`sum`, `mean`, `max`). `einops.einsum` unifies both: any label appearing multiple times triggers the reduction chosen in `einsum` (default `sum`).
- Broadcasting is explicit: absent labels correspond to newly created axes, avoiding implicit dummy indices. This mirrors the pedagogy that every contraction should state its surviving indices; you cannot accidentally drop an axis without naming it.

#### cross-library translation

- Map `einops` multi-character axes to single-character labels when moving to NumPy/PyTorch: `time -> t`, `heads -> h`, preserving the order of appearance to keep basis orientation untouched.
- When translating from `numpy.einsum` to `einops`, rewrite the equation as `pattern, reduction`. Example: `numpy.einsum("bij,jk->bik", A, B)` becomes `einops.einsum(A, B, "batch i j, j k -> batch i k")`—the Einstein logic is identical; only the label alphabet changes.

## abstract description

Einstein notation represents invariant quantities with simple notation. Scalars remain invariant under basis transformations, while vector components transform linearly. The convention extends to tensor products: any tensor $\mathbf{T}$ in $V \otimes V$ can be written as:

$$\mathbf{T} = T^{ij} \mathbf{e}_{ij}$$

The dual space $V^*$ has basis $\mathbf{e}^1, \mathbf{e}^2, \ldots, \mathbf{e}^n$ satisfying:

$$\mathbf{e}^i(\mathbf{e}_j) = \delta_j^i$$

where $\delta$ is the Kronecker delta.
