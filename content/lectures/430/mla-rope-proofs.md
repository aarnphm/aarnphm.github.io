---
date: '2025-10-17'
description: a derivation of multi-head latent attention, decoupled RoPE, and the inference-time absorption trick
id: mla-rope-proofs
modified: 2026-06-06 01:10:23 GMT-04:00
tags:
  - ml
title: proof for MLA RoPE
---

## what MLA stores

standard multi-head attention stores one key and one value for every head and cached token. with $n_h$ heads and head width $d_h$, that is

$$
C_{\mathrm{MHA}}=2n_hd_h
$$

values per token.

[[thoughts/MLA|Multi-head latent attention]] first maps the hidden state $\mathbf h_t\in\mathbb R^d$ into one shared latent:

$$
\mathbf c_t^{KV}=W^{DKV}\mathbf h_t,
\qquad
W^{DKV}\in\mathbb R^{d_c\times d}.
$$

content keys and values can be reconstructed for head $i$:

$$
\mathbf k_{t,i}^{C}=W_i^{UK}\mathbf c_t^{KV},
\qquad
\mathbf v_{t,i}^{C}=W_i^{UV}\mathbf c_t^{KV},
$$

where $W_i^{UK},W_i^{UV}\in\mathbb R^{d_h\times d_c}$. inference does not need to store those reconstructed vectors. it stores $\mathbf c_t^{KV}$ and uses weight absorption during attention.

DeepSeek-V3 also compresses queries:

$$
\mathbf c_t^Q=W^{DQ}\mathbf h_t,
\qquad
\mathbf q_{t,i}^{C}=W_i^{UQ}\mathbf c_t^Q.
$$

the query latent is temporary. it is recomputed for the current token and does not grow with the sequence.

## cache reduction

decoupled RoPE adds one shared positional key $\mathbf k_t^R\in\mathbb R^{d_R}$ to the cache. the actual MLA cache width is therefore

$$
C_{\mathrm{MLA}}=d_c+d_R,
$$

and the reduction factor relative to dense MHA is

$$
r=\frac{2n_hd_h}{d_c+d_R}.
$$

for DeepSeek-V3, $n_h=128$, $d_h=128$, $d_c=512$, and $d_R=64$:

$$
C_{\mathrm{MHA}}=2\cdot128\cdot128=32{,}768,
$$

$$
C_{\mathrm{MLA}}=512+64=576,
$$

$$
r=\frac{32{,}768}{576}\approx56.9.
$$

the cache is about $1.76\%$ of the dense MHA cache at the same number and width of heads. the earlier $64$-fold result omitted the shared RoPE key.

## projection work

count matrix multiplications before weight absorption. dense MHA spends

$$
F_{KV}^{\mathrm{MHA}}=2dn_hd_h
$$

multiply-accumulates per token on its key and value projections. MLA spends

$$
F_{KV}^{\mathrm{MLA}}
=dd_c+2d_cn_hd_h+dd_R.
$$

the terms are the latent down-projection, the content key/value up-projections, and the shared RoPE-key projection. with the DeepSeek-V3 dimensions,

$$
\frac{F_{KV}^{\mathrm{MLA}}}{F_{KV}^{\mathrm{MHA}}}
=\frac{7168\cdot512+2\cdot512\cdot128\cdot128+7168\cdot64}
{2\cdot7168\cdot128\cdot128}
\approx0.089.
$$

this ratio describes projection work. decode has another tradeoff: absorption lets attention read the compressed cache, while some contractions operate at latent width $d_c$. the throughput result depends on sequence length, batch shape, kernel choice, and memory bandwidth.

## weight absorption

### content scores

the content part of one attention score is

$$
\begin{aligned}
s_{t,u,i}^{C}
&=(\mathbf q_{t,i}^{C})^T\mathbf k_{u,i}^{C}\\
&=(\mathbf c_t^Q)^T(W_i^{UQ})^TW_i^{UK}\mathbf c_u^{KV}.
\end{aligned}
$$

define the absorbed matrix

$$
A_i=(W_i^{UQ})^TW_i^{UK}
\in\mathbb R^{d_c'\times d_c}.
$$

then

$$
s_{t,u,i}^{C}=(\mathbf c_t^Q)^TA_i\mathbf c_u^{KV}.
$$

an equivalent implementation transforms the current query with $(W_i^{UK})^T$ and takes its dot product with every cached latent. both forms avoid materializing a full key vector for every past token.

### values and output projection

for head $i$, let $p_{t,u,i}$ be the attention probability assigned to cached position $u$. the head output is

$$
\begin{aligned}
\mathbf o_{t,i}
&=\sum_u p_{t,u,i}\mathbf v_{u,i}^{C}\\
&=W_i^{UV}\left(\sum_u p_{t,u,i}\mathbf c_u^{KV}\right).
\end{aligned}
$$

partition the output projection by head as $W^O=[W_1^O\;\cdots\;W_{n_h}^O]$. then

$$
\mathbf y_t
=\sum_i W_i^OW_i^{UV}
\left(\sum_u p_{t,u,i}\mathbf c_u^{KV}\right).
$$

the products $W_i^OW_i^{UV}$ can be formed once for inference. this moves value reconstruction after the weighted sum and removes the per-token value expansion.

## RoPE

for one pair of coordinates, [[thoughts/RoPE|RoPE]] applies

$$
R_j(t)=
\begin{bmatrix}
\cos(t\theta_j)&-\sin(t\theta_j)\\
\sin(t\theta_j)&\cos(t\theta_j)
\end{bmatrix}.
$$

for an even width $d_R$, the full operator is block diagonal:

$$
R_{\Theta}(t)=\operatorname{diag}
\left(R_0(t),R_1(t),\ldots,R_{d_R/2-1}(t)\right),
$$

with a frequency schedule such as

$$
\theta_j=b^{-2j/d_R}.
$$

the base $b$ and any long-context scaling rule are model configuration. they should not be inferred from the head dimensions.

each block is orthogonal, so

$$
R_j(t)^TR_j(t)=I
$$

and

$$
R_j(m)^TR_j(n)=R_j(n-m).
$$

therefore a rotated query and key satisfy

$$
\begin{aligned}
(R_{\Theta}(m)\mathbf q)^T(R_{\Theta}(n)\mathbf k)
&=\mathbf q^TR_{\Theta}(m)^TR_{\Theta}(n)\mathbf k\\
&=\mathbf q^TR_{\Theta}(n-m)\mathbf k.
\end{aligned}
$$

the positional part of the score depends on relative displacement. this algebra defines the operator at positions outside the training window; it does not guarantee that a trained model will extrapolate there.

## why RoPE is decoupled from the latent

an arbitrary rotation between the down-projection and up-projection cannot usually be absorbed into fixed weights:

$$
W^{UK}R_{\Theta}(t)W^{DKV}.
$$

the middle factor changes with $t$, so there is no single position-independent matrix product to precompute. DeepSeek keeps the absorbable content path unrotated and adds a separate positional path:

$$
\mathbf q_{t,i}^{R}
=R_{\Theta}(t)W_i^{QR}\mathbf c_t^Q,
$$

$$
\mathbf k_t^{R}
=R_{\Theta}(t)W^{KR}\mathbf h_t.
$$

the positional key is shared across heads and projects directly from the hidden state. it is not reconstructed from $\mathbf c_t^{KV}$.

the full score is

$$
s_{t,u,i}
=\frac{
(\mathbf q_{t,i}^{C})^T\mathbf k_{u,i}^{C}
+(\mathbf q_{t,i}^{R})^T\mathbf k_u^{R}
}{\sqrt{d_h+d_R}}.
$$

the implementation can fold a model-specific scaling factor into this normalization. the cache still contains only

$$
[\mathbf c_u^{KV};\mathbf k_u^R]\in\mathbb R^{d_c+d_R}.
$$

## what the low-rank constraint proves

stack the dense key maps for all heads into $W^K$. a shared latent factorization has the form

$$
W^K=UW^{DKV},
$$

so

$$
\operatorname{rank}(W^K)\leq d_c.
$$

if a target matrix has rank at most $d_c$, this factorization can represent it exactly. for a fixed higher-rank target matrix $W$ with singular values $\sigma_1\geq\sigma_2\geq\cdots$, the best rank-$d_c$ approximation in Frobenius norm has error

$$
\min_{\operatorname{rank}(\widehat W)\leq d_c}
\lVert W-\widehat W\rVert_F
=\left(\sum_{j>d_c}\sigma_j^2\right)^{1/2}.
$$

this is a statement about approximating one linear map. it is not a bound on the output of an arbitrary attention mechanism, because softmax, queries, values, and the data distribution also affect the result. MLA learns the factors jointly with the rest of the model instead of approximating a trained dense attention matrix after the fact.

## parameter count

include query, key, value, and output projections. for the DeepSeek-V3 dimensions, MLA has

$$
\begin{aligned}
P_{\mathrm{MLA}}
={}&dd_c+dd_c'+n_hd_c'(d_h+d_R)\\
&+n_hd_cd_h+n_hd_cd_h+dd_R+dn_hd_h\\
\approx{}&187.1\text{ million parameters}.
\end{aligned}
$$

a dense attention layer with query, key, and value widths $n_hd_h$ has

$$
P_{\mathrm{MHA}}=3dn_hd_h+dn_hd_h
\approx469.8\text{ million parameters}.
$$

these counts omit biases and normalization parameters. they compare the projection structures at the stated widths.

## initialization and kernels

the product $W^{UK}W^{DKV}$ has rank at most $d_c$. when $d_c<d$, it cannot equal or approximate the identity on every direction in $\mathbb R^d$. the MLA papers do not require an SVD-of-identity initialization. use the checkpoint implementation's ordinary projection initialization when training from scratch.

efficient inference needs kernels that keep the compressed cache through the score and value contractions. a fallback that reconstructs every historical key and value is algebraically correct and gives back most of the memory-traffic cost that MLA was designed to remove.

## references

- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437)
- [RoFormer](https://arxiv.org/abs/2104.09864)
