---
date: '2025-08-21'
description: Self-attention from first principles, formal properties, and efficiency bounds.
id: attention
modified: 2026-06-06 00:04:40 GMT-04:00
tags:
  - ml
title: attention primer
---

Self-attention is a routing mechanism. We map queries to keys to determine value weighting. The architecture requires tracking precise memory ratios per token to keep inference within bounds.

## preliminaries

Let $X\in\mathbb{R}^{n\times d}$ be the input sequence. Project using $W_Q,W_K,W_V$. For $h$ heads, head dimension is $d_h$.

For $z\in\mathbb{R}^m$, the softmax and logsumexp (LSE) functions are:

$$
\operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}},\quad \operatorname{LSE}(z)=\log\sum_j e^{z_j},\quad \nabla\!\operatorname{LSE}(z)=\operatorname{softmax}(z).
$$

Temperature $T>0$ scales the logits as $z/T$. Stability requires shifting by the maximum before exponentiation: $z\leftarrow z-\max(z)$. [@dabah2025temperaturescalingconformalprediction]

[[thoughts/RoPE|RoPE]] (rotary position embeddings) adds relative position by rotating $(q,k)$ pairs in the complex plane before the dot product. [@su2023roformerenhancedtransformerrotary]

[[thoughts/Kullback-Leibler divergence]] controls total variation distance (Pinsker's inequality) and is defined as:

$$
D_{\mathrm{KL}}(P\|Q)=\sum_i P_i\log\frac{P_i}{Q_i}
$$

## scaled dot-product self-attention

Let $Q=XW_Q,\;K=XW_K,\;V=XW_V$.

$$
\mathrm{Attn}(Q,K,V)=\sigma\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,\quad \sigma=\text{row-softmax}.
$$

> [!proposition] Permutation Equivariance
> Without positional encodings, for any permutation matrix $P\in\mathbb{R}^{n\times n}$:
>
> $$
> \mathrm{Attn}(PQ,PK,PV)=P\,\mathrm{Attn}(Q,K,V).
> $$

Proof. $PQ(PK)^\top=PQK^\top P^\top$. Row-wise softmax satisfies $\sigma_{\text{row}}(PZP^\top)=P\,\sigma_{\text{row}}(Z)\,P^\top$ because $P$ is orthogonal ($P^\top P = I$) and permuting both rows and columns preserves row-wise sums. Right-multiply by $PV$ to yield $P \sigma_{\text{row}}(Z) P^\top P V = P \mathrm{Attn}(Q,K,V)$.

> [!proposition] Variance Scaling
> Assume $q,k\in\mathbb{R}^d$ are statistically independent at initialization. Let $\mathbb{E}[q]=\mathbb{E}[k]=0$, $\mathrm{Cov}(q)=\Sigma_q$, $\mathrm{Cov}(k)=\Sigma_k$. For $S=q^\top k$ and $Z=S/\sqrt{d}$:
>
> $$
> \mathrm{Var}(S)=\operatorname{tr}(\Sigma_q\Sigma_k),\quad \mathrm{Var}(Z)=\frac{1}{d}\operatorname{tr}(\Sigma_q\Sigma_k).
> $$
>
> Under isotropy $\Sigma_q=\Sigma_k=I_d$, $\mathrm{Var}(S)=d$ and $\mathrm{Var}(Z)=1$. [@vaswani2023attentionneed]

Proof. Use $\mathbb{E}[S]=0$. By iterated expectation (relying strictly on the independence of $q$ and $k$), $\mathbb{E}[S^2] = \mathbb{E}_q[\mathbb{E}_k[q^\top k k^\top q \mid q]] = \mathbb{E}_q[q^\top \Sigma_k q]$. Since this is a scalar, we apply the trace trick: $\mathbb{E}_q[\operatorname{tr}(q^\top \Sigma_k q)] = \operatorname{tr}(\Sigma_k \mathbb{E}[q q^\top]) = \operatorname{tr}(\Sigma_k \Sigma_q)$. Scale the result by $1/d$.

_Note_: In real models, $q$ and $k$ are derived from the same residual stream, so this independence assumption only loosely holds post-initialization.

## kernel regression

> [!proposition] RBF Weights under Strict Normalization
> Fix $q\in\mathbb{R}^d$ and keys $\{k_j\}_{j=1}^n$. If key norms are strictly uniform such that $\|k_j\|=\beta$ for all $j$, then scaled dot-product attention at temperature $T$ exactly mirrors Nadaraya-Watson Gaussian kernel regression:
>
> $$
> w_j(q)=\frac{\exp(-\|q-k_j\|^2/(2\sigma^2))}{\sum_{\ell}\exp(-\|q-k_\ell\|^2/(2\sigma^2))}\quad\text{with}\quad \sigma^2=T\sqrt{d}.
> $$
>
> [^tweet]

[^tweet]: https://x.com/chastronomic/status/1995604876823593374

Proof. Expand squared distance: $\|q-k_j\|^2 = \|q\|^2+\|k_j\|^2-2\langle q,k_j\rangle$. The $\|q\|^2$ term is constant across all $j$ and cancels in the softmax fraction. The $\|k_j\|^2$ term only cancels if it is a uniform constant $\beta$. If uniform, the kernel weight $\exp(-\|q-k_j\|^2/(2\sigma^2))$ isolates to $\exp(\langle q,k_j\rangle/\sigma^2)$. Equating this to the attention exponent $\langle q,k_j\rangle/(T\sqrt{d})$ forces $\sigma^2=T\sqrt{d}$. Without uniform key norms, standard attention deviates from true Gaussian regression.

## architectural variants

Multi-Head Attention (MHA) concatenates outputs from $h$ independent routing spaces:

$$
\text{MHA}(X)=W_O [O_1;\ldots;O_h],\quad O_i=\sigma\!\left(\frac{Q_i K_i^\top}{\sqrt{d_h}}\right)V_i.
$$

Memory bounds during decoding run into KV cache limits. [[thoughts/GQA|Grouped-Query Attention]] (GQA) shares keys and values across $G$ groups, while Multi-Query Attention (MQA) shares one set across all heads ($G=1$). [@ainslie2023gqatraininggeneralizedmultiquery; @shazeer2019fasttransformerdecodingwritehead]

> [!proposition] GQA Robustness Bound
> Replacing exact $K_i$ with a group-shared $K_g$ shifts logits by $\delta z= q(K_g-K_i)^\top/\sqrt{d_h}$. The softmax function is $(\lambda/2)$-Lipschitz, where $\lambda$ is inverse temperature.
>
> $$
> \|\operatorname{softmax}(z+\delta z)-\operatorname{softmax}(z)\|_2\le \frac{\lambda}{2}\,\|\delta z\|_2.
> $$

Proof. The Jacobian of softmax is $J = \lambda(\operatorname{diag}(\sigma) - \sigma\sigma^\top)$. The maximum eigenvalue (spectral norm) of this matrix is at most $\lambda/2$.

### Multi-Head Latent Attention (MLA)

DeepSeek's [[thoughts/MLA|MLA]] caches a compressed latent $c^{KV}_t\in\mathbb{R}^{d_c}$ instead of raw heads. To maintain spatial information, [[thoughts/RoPE|RoPE]] requires a decoupled key vector $k_t^R \in\mathbb{R}^{d_h^R}$.

During inference, up-projections $W_U^K$ and $W_U^V$ map the latent back to full dimension. Because matrix multiplication is associative, these projections are absorbed into $W_Q$ and $W_O$:
$W_Q' = W_{UQ} (W_U^K)^\top$ and $W_O' = W_U^V W_O$.

This strips the up-projection from the critical path entirely.

### memory limits

Elements cached per token per layer:

- **MHA**: $2 h d_h$
- **GQA**: $2 G d_h$
- **MQA**: $2 d_h$
- **MLA**: $d_c + d_h^R$

_Example:_ At $h=128$, $d_h=128$, MHA caches 32,768 elements. MLA ($d_c=512$, $d_h^R=64$) caches 576 elements, a 56:1 reduction ratio.

## reference cuda kernels

> [!note] implementation
> See reference CUDA kernels for exact score matrices and stable block logic.
>
> - Scores: ![[lectures/2/qk_scores.cu]]
> - Softmax: ![[lectures/2/row_softmax.cu]]
> - Apply: ![[lectures/2/apply_values.cu]]
>
> [[thoughts/flash attention|FlashAttention]] prevents intermediate materialization by streaming tiles through shared memory, calculating softmax scaling blocks directly. [@dao2022flashattentionfastmemoryefficientexact]
>
> Triton equivalent: ![[lectures/2/attention_triton.py]]
