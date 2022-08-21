---
date: "2025-10-17"
description: a basic theorem proof
id: mla-rope-proofs
modified: 2025-11-11 06:56:02 GMT-05:00
tags:
  - ml
title: proof for MLA RoPE
---

### motivation

standard multi-head attention (MHA) stores separate key-value pairs for each attention head during inference, creating a memory bottleneck. for a model with $n_h = 128$ heads, $d_h = 128$ dimensions per head, the KV cache size per token is $2 \times n_h \times d_h = 32{,}768$ values. MLA compresses this dramatically.

### core formulation

MLA decomposes the key-value computation through low-rank joint compression into a shared latent space.

**compression phase:**

for input hidden state $\mathbf{h}_t \in \mathbb{R}^d$ at position $t$, we compress into a low-dimensional latent:

$$
\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t
$$

where:

- $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ is the compressed KV latent
- $W^{DKV} \in \mathbb{R}^{d_c \times d}$ is the down-projection matrix
- $d_c \ll d$ (typically $d_c \approx 512$ while $d \approx 7168$ in DeepSeek-V3)

**decompression phase:**

keys and values are reconstructed from the latent:

$$
\begin{aligned}
\mathbf{k}_t^{(i)} &= W_i^{UK} \mathbf{c}_t^{KV} \\
\mathbf{v}_t^{(i)} &= W_i^{UV} \mathbf{c}_t^{KV}
\end{aligned}
$$

where:

- $W_i^{UK}, W_i^{UV} \in \mathbb{R}^{d_h \times d_c}$ are head-specific up-projection matrices
- $\mathbf{k}_t^{(i)}, \mathbf{v}_t^{(i)} \in \mathbb{R}^{d_h}$ are the reconstructed key and value for head $i$

**query computation:**

queries follow a similar compression-decompression pattern (during training) or can be computed directly:

$$
\mathbf{q}_t^{(i)} = W_i^{UQ} (W^{DQ} \mathbf{h}_t)
$$

where $W^{DQ} \in \mathbb{R}^{d_c' \times d}$ and $W_i^{UQ} \in \mathbb{R}^{d_h \times d_c'}$.

### memory reduction proof

> [!theorem] 1.1
>
> MLA reduces KV cache size by a factor of $\frac{n_h \cdot d_h}{d_c}$.

_proof:_

standard MHA stores per token:

$$
\text{MHA cache} = 2 \times n_h \times d_h \text{ values}
$$

MLA stores per token:

$$
\text{MLA cache} = d_c \text{ values}
$$

reduction factor:

$$
r = \frac{2 n_h d_h}{d_c}
$$

**concrete example (DeepSeek-V3):**

parameters:

- $d = 7168$ (model dimension)
- $n_h = 128$ (number of heads)
- $d_h = 128$ (dimension per head)
- $d_c = 512$ (latent dimension)

standard MHA: $2 \times 128 \times 128 = 32{,}768$ values/token

MLA: $512$ values/token

reduction: $\frac{32{,}768}{512} = 64\times$ or equivalently, MLA uses only $\frac{512}{32{,}768} \approx 1.56\%$ of the original cache size.

this exceeds the stated 5-13% range because DeepSeek actually uses additional components (RoPE heads, discussed below). $\square$

### computational complexity

> [!theorem] 1.2
>
> MLA requires more FLOPs but achieves higher throughput due to memory bandwidth savings.

let $n$ be sequence length. standard attention performs:

$$
\text{QKV projection: } O(n \cdot d \cdot n_h \cdot d_h) = O(n \cdot d^2)
$$

MLA performs:

$$
\begin{aligned}
\text{down-projection: } &O(n \cdot d \cdot d_c) \\
\text{up-projection: } &O(n \cdot d_c \cdot n_h \cdot d_h) \\
\text{total: } &O(n(d \cdot d_c + d_c \cdot n_h \cdot d_h))
\end{aligned}
$$

when $d = n_h \cdot d_h$, the ratio is:

$$
\frac{\text{MLA FLOPs}}{\text{MHA FLOPs}} = \frac{d_c + n_h d_h}{n_h d_h} = 1 + \frac{d_c}{n_h d_h}
$$

for DeepSeek-V3: $1 + \frac{512}{128 \times 128} \approx 1.03$ (3% more FLOPs).

however, memory-bound operations dominate inference. MLA reads $512$ values vs $32{,}768$ values per token, achieving $\approx 64\times$ bandwidth reduction. $\square$

### weight absorption optimization

during inference, matrix multiplications can be fused to eliminate intermediate latent computation.

**query-key absorption:**

define the absorbed attention pattern matrix:

$$
W^{KQ}_i = (W_i^{UK})^T W_i^{UQ} \in \mathbb{R}^{d_c \times d_c'}
$$

then the attention score becomes:

$$
\text{score}_{t,s}^{(i)} = (\mathbf{c}_s^{KV})^T W^{KQ}_i \mathbf{c}_t^Q
$$

**value absorption:**

similarly, absorb value projection:

$$
W^{VQ}_i = W_i^{UV} \in \mathbb{R}^{d_h \times d_c}
$$

this eliminates the need to explicitly compute and store $\mathbf{k}_t^{(i)}, \mathbf{v}_t^{(i)}$ during inference, operating directly on compressed representations.

### algebraic equivalence proof

> [!theorem] 1.3
>
> MLA produces identical outputs to an equivalent MHA with constrained weight structure.

_proof:_

standard MHA computes:

$$
\mathbf{k}_t^{(i)} = W_i^K \mathbf{h}_t, \quad \mathbf{v}_t^{(i)} = W_i^V \mathbf{h}_t
$$

MLA computes:

$$
\mathbf{k}_t^{(i)} = W_i^{UK}(W^{DKV} \mathbf{h}_t) = (W_i^{UK} W^{DKV}) \mathbf{h}_t
$$

setting $W_i^K = W_i^{UK} W^{DKV}$ shows MLA is equivalent to MHA where all head projection matrices share a common low-rank structure factorized as:

$$
W_i^K = W_i^{UK} W^{DKV}, \quad \text{rank}(W_i^K) \leq d_c
$$

the key insight: MLA enforces this low-rank constraint explicitly, enabling compression during inference while maintaining expressiveness during training. $\square$

## rotary position embeddings (RoPE)

### motivation and requirements

transformers are position-agnostic—attention mechanism treats sequences as sets. we need position encoding $f(\mathbf{x}, m)$ that:

1. encodes absolute position $m$
2. induces relative position information in attention scores
3. generalizes to arbitrary sequence lengths

### mathematical derivation

**setup:**

we seek a function $f: \mathbb{R}^d \times \mathbb{N} \to \mathbb{R}^d$ such that the inner product captures relative position:

$$
\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m - n)
$$

for some function $g$ depending only on relative position $m - n$.

**2D case:**

treat 2D embeddings as complex numbers: $\mathbf{x} = (x_0, x_1) \leftrightarrow x_0 + i x_1$.

> [!theorem] 2.1 (modulus invariance)
>
> $|f(\mathbf{x}, m)| = |\mathbf{x}|$.

_proof:_ setting $m = n$ in the requirement:

$$
\langle f(\mathbf{x}, m), f(\mathbf{x}, m) \rangle = g(\mathbf{x}, \mathbf{x}, 0)
$$

the right side depends only on $\mathbf{x}$, so $|f(\mathbf{x}, m)|^2 = |\mathbf{x}|^2$. $\square$

> [!theorem] 2.2 (rotation structure)
>
> $f(\mathbf{x}, m) = \mathbf{x} e^{i m \theta}$ for some constant $\theta \in \mathbb{R}$.

_proof:_ write $f(\mathbf{x}, m) = |\mathbf{x}| e^{i(\phi(\mathbf{x}) + \psi(m))}$ where $\phi$ depends on $\mathbf{x}$ and $\psi$ depends on $m$.

computing the inner product:

$$
\begin{aligned}
\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle &= \text{Re}(|\mathbf{q}| |\mathbf{k}| e^{i(\phi(\mathbf{q}) - \phi(\mathbf{k}) + \psi(m) - \psi(n))}) \\
&= |\mathbf{q}| |\mathbf{k}| \cos(\phi(\mathbf{q}) - \phi(\mathbf{k}) + \psi(m) - \psi(n))
\end{aligned}
$$

for this to depend only on $m - n$, we need $\psi(m) - \psi(n) = \psi(m - n)$, implying $\psi(m) = m\theta$ for some constant $\theta$.

without loss of generality, absorb $\phi(\mathbf{x})$ into $\mathbf{x}$'s phase, yielding:

$$
f(\mathbf{x}, m) = \mathbf{x} e^{i m \theta}
$$

$\square$

### rotation matrix formulation

**2D rotation matrix:**

expanding $e^{i m \theta} = \cos(m\theta) + i \sin(m\theta)$:

$$
f\begin{pmatrix} x_0 \\ x_1 \end{pmatrix}_m = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \end{pmatrix}
$$

this is a rotation by angle $m\theta$ in the 2D plane.

**higher dimensions:**

for $d$-dimensional vectors, apply independent 2D rotations to pairs of dimensions:

$$
R_\Theta^{(d)}(m) = \begin{pmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) & 0 & 0 & \cdots & 0 \\
\sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 & \cdots & 0 \\
0 & 0 & \cos(m\theta_1) & -\sin(m\theta_1) & \cdots & 0 \\
0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos(m\theta_{d/2-1}) \\
0 & 0 & 0 & 0 & \cdots & \sin(m\theta_{d/2-1})
\end{pmatrix}
$$

with frequencies:

$$
\theta_j = \frac{1}{\text{base}^{2j/d}}, \quad j = 0, 1, \ldots, \frac{d}{2} - 1
$$

where $\text{base} = 10{,}000$ (standard) or $\text{base} = 5 \times 10^6$ (DeepSeek-V3 for long context).

### relative position property proof

> [!theorem] 2.3
>
> RoPE attention scores depend only on relative position.

_proof:_

queries and keys at positions $m, n$ after RoPE:

$$
\mathbf{q}_m = R_\Theta(m) \mathbf{q}, \quad \mathbf{k}_n = R_\Theta(n) \mathbf{k}
$$

attention score:

$$
\begin{aligned}
\text{score}_{m,n} &= \mathbf{q}_m^T \mathbf{k}_n \\
&= (R_\Theta(m) \mathbf{q})^T (R_\Theta(n) \mathbf{k}) \\
&= \mathbf{q}^T R_\Theta(m)^T R_\Theta(n) \mathbf{k} \\
&= \mathbf{q}^T R_\Theta(n - m) \mathbf{k}
\end{aligned}
$$

the last equality holds because:

$$
R_\Theta(m)^T R_\Theta(n) = R_\Theta(-m) R_\Theta(n) = R_\Theta(n - m)
$$

thus the score depends only on $\Delta = n - m$, not absolute positions $m$ or $n$. $\square$

### practical implementation

**element-wise formulation:**

rather than matrix multiplication, RoPE is computed element-wise:

$$
\begin{pmatrix} q_0 \\ q_1 \end{pmatrix}_m = \begin{pmatrix} q_0 \cos(m\theta) - q_1 \sin(m\theta) \\ q_0 \sin(m\theta) + q_1 \cos(m\theta) \end{pmatrix}
$$

**vectorized implementation:**

for efficiency, precompute $\cos(m\theta_j)$ and $\sin(m\theta_j)$ for all positions and frequencies:

```python
def rotate_half(x):
  x1, x2 = x.chunk(2, dim=-1)
  return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
  q_rotated = (q * cos) + (rotate_half(q) * sin)
  k_rotated = (k * cos) + (rotate_half(k) * sin)
  return q_rotated, k_rotated
```

## integration: MLA with RoPE

### decoupled RoPE

MLA cannot apply standard RoPE to latent representations because:

1. rotation matrices require fixed dimensions
2. latent space $d_c$ is shared across heads with different semantic roles

**solution:** decoupled RoPE separates position-carrying components from content components.

### architecture

**position-carrying components:**

allocate $d_R$ dimensions per head for RoPE:

$$
\mathbf{q}_{t,i}^R = \text{RoPE}(W_i^{QR} \mathbf{c}_t^Q) \in \mathbb{R}^{d_R}
$$

**shared position key:**

use a single key head for all queries (inspired by multi-query attention):

$$
\mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{c}_t^{KV}) \in \mathbb{R}^{d_R}
$$

**content components:**

non-rotated components capture semantic information:

$$
\mathbf{q}_{t,i}^C = W_i^{QC} \mathbf{c}_t^Q \in \mathbb{R}^{d_h - d_R}
$$

$$
\mathbf{k}_{t,i}^C = W_i^{KC} \mathbf{c}_t^{KV} \in \mathbb{R}^{d_h - d_R}
$$

### attention computation

full attention score combines position and content:

$$
\text{score}_{t,s}^{(i)} = \frac{1}{\sqrt{d_h}} \left( (\mathbf{q}_{t,i}^R)^T \mathbf{k}_s^R + (\mathbf{q}_{t,i}^C)^T \mathbf{k}_{s,i}^C \right)
$$

**key insight:** position information comes from a shared RoPE component, while per-head content components capture semantic relationships. this enables:

- efficient KV cache (only $d_R$ additional values for position)
- relative position encoding via RoPE
- head-specific content modeling via $\mathbf{k}_t^C$

### concrete dimensions (DeepSeek-V3)

parameters:

- $d = 7168$ (model dimension)
- $n_h = 128$ (attention heads)
- $d_h = 128$ (head dimension)
- $d_c = 512$ (KV latent dimension)
- $d_c' = 1536$ (Q latent dimension)
- $d_R = 64$ (RoPE dimension per head)

**per-token KV cache:**

$$
\text{cache size} = d_c + d_R = 512 + 64 = 576 \text{ values}
$$

**compression ratio:**

$$
\frac{576}{2 \times 128 \times 128} = \frac{576}{32{,}768} \approx 1.76\%
$$

**per-token query (not cached):**

$$
\text{query size} = d_c' + n_h \times d_R = 1536 + 128 \times 64 = 9{,}728 \text{ values}
$$

queries are recomputed each step so they don't accumulate in cache.

## theoretical properties

### expressiveness

> [!theorem] 3.1
>
> MLA with rank $d_c$ can approximate any attention mechanism with bounded approximation error.

_sketch:_ by singular value decomposition, any matrix $W \in \mathbb{R}^{m \times n}$ can be written:

$$
W = \sum_{i=1}^{\min(m,n)} \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

taking the top $d_c$ singular values:

$$
W \approx \sum_{i=1}^{d_c} \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

with error $\epsilon = \sqrt{\sum_{i=d_c+1}^{\min(m,n)} \sigma_i^2}$.

MLA learns this factorization end-to-end, potentially finding better low-rank approximations than SVD for the specific task. $\square$

### length generalization

> [!theorem] 3.2
>
> RoPE generalizes to sequences longer than training.

_intuition:_ rotation angles $m\theta_j$ scale linearly with position. frequencies $\theta_j$ decay exponentially across dimensions, providing both fine-grained (small $j$, large $\theta_j$) and coarse-grained (large $j$, small $\theta_j$) position signals.

for positions $m > m_{\text{train}}$, rotations continue smoothly, unlike learned absolute embeddings that have no defined behavior beyond training range.

## performance analysis

### training efficiency

**parameter comparison:**

standard MHA:

$$
\text{params} = 3 \times n_h \times d \times d_h = 3d^2 \text{ (for } d = n_h d_h\text{)}
$$

MLA:

$$
\text{params} = d \times (d_c + d_c') + n_h \times d_h \times (d_c + d_c')
$$

for DeepSeek-V3:

$$
\text{MLA params} = 7168 \times (512 + 1536) + 128 \times 128 \times (512 + 1536) \approx 48M
$$

comparable to standard attention, slightly higher due to latent projections.

### inference throughput

**memory bandwidth:**

modern GPUs are memory-bound for transformer inference. per token:

standard MHA reads: $32{,}768$ KV values

MLA reads: $576$ KV values

**throughput gain:**

empirically, DeepSeek-V2 reports 5.76× higher generation throughput with MLA, confirming memory bandwidth as the primary bottleneck.

## implementation notes

### initialization

low-rank projections should be initialized to approximate identity:

$$
W^{UK} W^{DKV} \approx I_d
$$

one strategy: initialize $W^{DKV}$ with SVD of identity (top $d_c$ components), initialize $W^{UK}$ as pseudoinverse.

### numerical stability

rotation matrices have eigenvalues on unit circle, preserving gradient norms. for very long sequences, use mixed precision (FP32 for position encoding, FP16/BF16 for content).

### fused kernels

optimal performance requires custom CUDA kernels fusing:

1. latent projection
2. RoPE rotation
3. attention score computation

avoiding intermediate materialization of full $\mathbf{k}_t^{(i)}, \mathbf{v}_t^{(i)}$ tensors.

## references

- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (2024)
- DeepSeek-V3 Technical Report (2024)
- RoFormer: Enhanced Transformer with Rotary Position Embedding, Su et al. (2021)

## further reading

for implementation details:

- [[thoughts/attention|attention mechanisms]]
- [[thoughts/structured outputs|structured outputs and constrained generation]]

for theoretical foundations:

- [[thoughts/autoencoder-diagrams-intuition/index|autoencoder diagrams and intuition]]
- low-rank matrix approximation and SVD decomposition
