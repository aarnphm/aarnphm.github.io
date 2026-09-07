---
date: '2025-08-21'
description: self-attention from scores to cache layout
id: attention
modified: 2026-09-07 13:56:24 GMT-04:00
tags:
  - ml
title: attention primer
---

Self-attention turns one residual sequence into a weighted average of learned value vectors. Queries choose which key rows receive mass; those weights then mix the values. The same equations explain the stable-softmax trick and the KV-cache accounting used at inference time.

## scores and normalization

Let $X\in\mathbb{R}^{n\times d}$ be a sequence of $n$ residual vectors. For one head,

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

with $Q,K,V\in\mathbb{R}^{n\times d_h}$. Scaled dot-product attention is

$$
\operatorname{Attn}(Q,K,V)
=\operatorname{softmax}_{\mathrm{row}}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V,
$$

where $M=0$ for unrestricted attention. A causal mask sets entries above the current position to $-\infty$ before the row-wise softmax. [@vaswani2023attentionneed]

For temperature $T>0$, define

$$
\operatorname{LSE}_T(z)=T\log\sum_j e^{z_j/T}.
$$

Then

$$
\nabla\operatorname{LSE}_T(z)=\operatorname{softmax}(z/T).
$$

If $m=\max_jz_j$, the stable evaluation is

$$
\operatorname{LSE}_T(z)=m+T\log\sum_j e^{(z_j-m)/T}.
$$

Subtracting $m$ changes neither the softmax nor the normalized weights. The factor $1/\sqrt{d_h}$ serves a different purpose from temperature: under an isotropic initialization it keeps the variance of a query-key dot product near one.

[[thoughts/RoPE|RoPE]] applies paired rotations to queries and keys before their dot product. The resulting inner product depends on relative position while preserving each rotated pair's norm. [@su2023roformerenhancedtransformerrotary]

## two exact properties

> [!proposition] Permutation equivariance
>
> Remove positional information and any position-dependent mask. For a permutation matrix $P$,
>
> $$
> \operatorname{Attn}(PQ,PK,PV)=P\operatorname{Attn}(Q,K,V).
> $$

The score matrix becomes $PQK^\top P^\top$. Row-wise softmax commutes with the matching row and column permutations, so

$$
\operatorname{softmax}_{\mathrm{row}}(PZP^\top)
=P\operatorname{softmax}_{\mathrm{row}}(Z)P^\top.
$$

Multiplying by $PV$ gives the result. A causal mask is tied to sequence order and therefore removes this full permutation symmetry.

> [!proposition] Initialization variance
>
> Let $q,k\in\mathbb{R}^{d_h}$ be independent, centered random vectors with covariances $\Sigma_q$ and $\Sigma_k$. Then
>
> $$
> \operatorname{Var}(q^\top k)=\operatorname{tr}(\Sigma_q\Sigma_k).
> $$
>
> Under $\Sigma_q=\Sigma_k=I$, dividing by $\sqrt{d_h}$ changes the variance from $d_h$ to $1$.

The independence assumption describes an initialization calculation. Queries and keys later come from the same residual stream, so it is not a distributional law for a trained model.

## Gaussian-kernel identity

Fix a query $q$ and keys $k_1,\ldots,k_n$ of equal norm. Gaussian kernel weights satisfy

$$
\exp\!\left(-\frac{\|q-k_j\|_2^2}{2\sigma^2}\right)
=C(q)\exp\!\left(\frac{q^\top k_j}{\sigma^2}\right),
$$

where $C(q)$ is independent of $j$. After normalization, the weights match dot-product attention when

$$
\sigma^2=T\sqrt{d_h}.
$$

The equal-key-norm condition does the work. Without it, $\|k_j\|_2^2$ contributes a key-dependent term, so ordinary dot-product attention is not exactly Gaussian Nadaraya-Watson regression.

## heads and KV groups

Multi-head attention uses $h$ query heads and concatenates their outputs:

$$
\operatorname{MHA}(X)=W_O[O_1;\ldots;O_h].
$$

MHA gives each query head its own key and value head. [[thoughts/GQA|Grouped-query attention]] uses $G<h$ key-value heads, with several query heads assigned to each group. Multi-query attention is the case $G=1$. These layouts reduce per-token KV-cache reads during decoding. [@ainslie2023gqatraininggeneralizedmultiquery; @shazeer2019fasttransformerdecodingwritehead]

For one token in one layer, ignoring batch, precision, and metadata, the cached element counts are

| Layout | Cached elements |
| ------ | --------------: |
| MHA    |         $2hd_h$ |
| GQA    |         $2Gd_h$ |
| MQA    |          $2d_h$ |

### a weight-perturbation bound

Let $S_T(z)=\operatorname{softmax}(z/T)$. Its Jacobian is

$$
J(z)=\frac1T\left(\operatorname{Diag}(p)-pp^\top\right),
\qquad p=S_T(z),
$$

and $\|J(z)\|_2\leq 1/(2T)$. If replacing a head's key matrix $K$ by a shared matrix $\widetilde K$ changes its logits by

$$
\delta z=\frac{q(\widetilde K-K)^\top}{\sqrt{d_h}},
$$

then

$$
\|S_T(z+\delta z)-S_T(z)\|_2
\leq\frac{1}{2T}\|\delta z\|_2.
$$

This only bounds the change in one attention-weight vector. It does not by itself bound the output of the layer or the loss of the full model. [@nair2025softmaxhalflipschitz]

## Multi-head Latent Attention

DeepSeek-V2's Multi-head Latent Attention (MLA) stores a joint low-rank key-value latent instead of one key and one value vector per head:

$$
c_t^{KV}=W^{DKV}h_t\in\mathbb{R}^{d_c}.
$$

Head-specific up-projections recover the content keys and values from $c_t^{KV}$. A separate RoPE key $k_t^R\in\mathbb{R}^{d_h^R}$ carries the position-dependent part. During decoding, the content-key up-projection can be folded into the query projection, and the value up-projection can be fused with the output path. Decode kernels can attend over the latent cache without materializing every cached content key and value. Prefill implementations may still reconstruct or chunk full keys and values when that path is more compute-efficient. The RoPE branch stays separate because its rotation depends on position. [@deepseekai2024deepseekv2strongeconomicalefficient]

The cache per token per layer is

$$
d_c+d_h^R
$$

elements. With the DeepSeek-V2 dimensions $h=128$, $d_h=128$, $d_c=512$, and $d_h^R=64$, MHA stores $32{,}768$ elements and MLA stores $576$; MHA's cache is about $56.9\times$ larger. This compares element counts only, not end-to-end memory or throughput.

## reference kernels

The local kernels separate the three operations in ordinary attention:

- score products: ![[lectures/2/qk_scores.cu]]
- row-wise softmax: ![[lectures/2/row_softmax.cu]]
- value aggregation: ![[lectures/2/apply_values.cu]]

[[thoughts/flash attention|FlashAttention]] computes the same attention result by tiling these operations and maintaining an online row maximum and normalizer. It avoids writing the full $n\times n$ score and probability matrices to high-bandwidth memory. [@dao2022flashattentionfastmemoryefficientexact]

Triton reference: ![[lectures/2/attention_triton.py]]
