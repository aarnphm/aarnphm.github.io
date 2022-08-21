---
date: "2025-08-21"
description: Self-attention from first principles with formal properties and efficient variants.
id: attention
modified: 2025-11-11 07:02:17 GMT-05:00
tags:
  - ml
title: attention primer
---

I present a concise derivation of scaled dot-product self-attention, prove basic invariance and scaling properties, establish an exact equivalence to Gaussian kernel regression under mild normalization, and summarize multi-head attention as a multi-kernel ensemble. We then formalize efficiency variants (MQA/GQA/MLA) and decode-time complexity, and note IO-aware optimizations.

## preliminaries

Notation. Let $X\in\mathbb{R}^{n\times d}$, with projections $W_Q,W_K,W_V$. For $h$ heads, head dimension is $d_h$.

Softmax/LSE. For $z\in\mathbb{R}^m$,

$$
\operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}},\quad \operatorname{LSE}(z)=\log\sum_j e^{z_j},\quad \nabla\!\operatorname{LSE}(z)=\operatorname{softmax}(z).
$$

Stability uses $z\leftarrow z-\max(z)$. Temperature $T>0$ uses $\operatorname{softmax}(z/T)$ (entropy increases with $T$). [@dabah2025temperaturescalingconformalprediction]

RoPE. Rotary position embeddings add relative position via complex-plane rotations of $(q,k)$ before the dot product. [@su2023roformerenhancedtransformerrotary]

[[thoughts/Kullback-Leibler divergence]], i.e: KL divergence.

$$
D_{\mathrm{KL}}(P\|Q)=\sum_i P_i\log\frac{P_i}{Q_i}
$$

is jointly convex and controls total variation (Pinsker).

## scaled dot-product self-attention

Let $Q=XW_Q,\;K=XW_K,\;V=XW_V$. Define

$$
\mathrm{Attn}(Q,K,V)=\sigma\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,\quad \sigma=\text{row-softmax}.
$$

> [!propos] 1 (Permutation Equivariance).
>
> Without positional encodings, for any permutation matrix $P\in\mathbb{R}^{n\times n}$,
>
> $$
> \mathrm{Attn}(PQ,PK,PV)=P\,\mathrm{Attn}(Q,K,V).
> $$

Proof. $PQ(PK)^\top=PQK^\top P^\top$ and row-wise softmax satisfies $\sigma_{\text{row}}(PZP^\top)=P\,\sigma_{\text{row}}(Z)\,P^\top$ (since $P$ is orthogonal and permutes both rows and columns consistently, preserving row-wise normalization). Multiply by $PV$ on the right to complete the proof.

> [!propos] 2 (Variance Scaling).
>
> Let $q,k\in\mathbb{R}^d$ with $\mathbb{E}[q]=\mathbb{E}[k]=0$, $\mathrm{Cov}(q)=\Sigma_q$, $\mathrm{Cov}(k)=\Sigma_k$, and $q\perp k$. For $S=q^\top k$ and $Z=S/\sqrt{d}$,
>
> $$
> \mathrm{Var}(S)=\operatorname{tr}(\Sigma_q\Sigma_k),\quad \mathrm{Var}(Z)=\frac{1}{d}\operatorname{tr}(\Sigma_q\Sigma_k).
> $$
>
> In particular, under isotropy $\Sigma_q=\Sigma_k=I_d$, $\mathrm{Var}(S)=d$ and $\mathrm{Var}(Z)=1$. [@vaswani2023attentionneed]

Proof. Use $\mathbb{E}[S]=0$ and $\mathbb{E}[S^2]=\operatorname{tr}(\Sigma_q\Sigma_k)$ via $S^2=k^\top (qq^\top)k$ and iterated expectation; scale by $1/d$.

## kernel regression

> [!propos] 3 (Exact RBF Weights under Normalization).
>
> Fix $q\in\mathbb{R}^d$ and keys $\{k_j\}_{j=1}^n$. If $\|q\|=\alpha$ and $\|k_j\|=\beta$ for all $j$, then scaled dot-product attention at temperature $T$ yields
>
> $$
> w_j(q)=\frac{\exp(-\|q-k_j\|^2/(2\sigma^2))}{\sum_{\ell}\exp(-\|q-k_\ell\|^2/(2\sigma^2))}\quad\text{with}\quad \sigma^2=T\sqrt{d}.
> $$
>
> Hence $o(q)=\sum_j w_j(q)v_j$ is the Nadaraya–Watson estimator with Gaussian kernel.

Proof. Use polarization $\langle q,k_j\rangle = \tfrac12(\|q\|^2+\|k_j\|^2-\|q-k_j\|^2)$, cancel row-constants in softmax, and match with $K_\sigma(x,y)=\exp(-\|x-y\|^2/(2\sigma^2))$; identify $\sigma^2=T\sqrt{d}$. For small norm variation, the multiplicative deviation is controlled by $\exp(\|k_j\|^2/(2T\sqrt{d}))$ before renormalization.

## multi‑head attention (MHA) as multi‑kernel learning

With heads $i=1,\dots,h$,

$$
\text{MHA}(X)=W_O [O_1;\ldots;O_h],\quad O_i=\sigma\!\left(\frac{Q_i K_i^\top}{\sqrt{d_h}}\right)V_i.
$$

Each head induces a kernel smoother in its own feature space $(W_Q^i,W_K^i,W_V^i)$; concatenation and $W_O$ implement a learned multi-kernel combination. [@vaswani2023attentionneed]

## grouped‑query & multi‑query attention (GQA/MQA)

During decode, KV-cache bandwidth dominates. MQA shares one $K,V$ across all heads; GQA shares KV across $G$ groups ($1<G<h$). In GQA, queries have $h$ heads while keys/values have $G$ heads; each group of $h/G$ query heads attends to shared $(K_g,V_g)$. [@ainslie2023gqatraininggeneralizedmultiquery; @shazeer2019fasttransformerdecodingwritehead]

Memory per token per layer (FP16 elements counted, constants omitted):

- MHA: $2h d_h$ (all $K,V$ cached)
- GQA: $2G d_h$
- MQA: $2 d_h$ (the $G\!=\!1$ special case)

Thus GQA reduces KV cache by factor $\approx h/G$ at comparable $Q K^{T}$ compute.

Robustness bound. For a query row $q$ with logits $z= q(K_i)^\top/\sqrt{d_h}$, replacing $K_i$ by group-shared $K_g$ yields $\delta z= q(K_g-K_i)^\top/\sqrt{d_h}$. Since softmax $=\nabla\!\operatorname{LSE}$ is $\lambda$-Lipschitz with $\lambda$ the inverse temperature,

$$
\|\operatorname{softmax}(z+\delta z)-\operatorname{softmax}(z)\|_2\le \lambda\,\|\delta z\|_2.
$$

## DeepSeek's multi-latent attention (MLA)

MLA compresses KV while preserving quality. A compressed latent $c^{KV}_t\in\mathbb{R}^{d_c}$ is cached and up-projected when needed:

$$
c^{KV}_t = W^{KV}_D h_t,\quad d_c \ll h\,d_h,\qquad k^{C}_t = W^{K}_U c^{KV}_t,\; v^{C}_t = W^{V}_U c^{KV}_t.
$$

During inference, only $c^{KV}_t$ is cached, giving $d_c$ elements per token per layer (vs $2hd_h$). Up-projections can be absorbed into $W_Q,W_O$ on the critical path. RoPE requires a decoupled scheme to avoid entanglement with $W_U^K$.

## formal derivations

D.1 Self‑attention as gradient (when $V=K$). Define $F(q)=\log\sum_j \exp(\langle q,k_j\rangle/T)$. Then $\nabla_q F(q)=\sum_j p_j\,k_j$ with $p=\operatorname{softmax}(\langle q,K\rangle/T)$. Thus, with $V=K$, attention equals the gradient of a convex potential (smooth max).

D.2 Nadaraya–Watson equivalence (LN). If $\|q\|=\|k_j\|=c$ (approx true after LN), attention weights equal the RBF kernel in $\|q-k_j\|$ up to a row-constant; see Proposition 3. [@choromanski2022rethinkingattentionperformers; @katharopoulos2020transformers]

D.3 Lipschitz bound for group replacement (GQA). With inverse-temperature $\lambda$, $\|\sigma(z+\delta z)-\sigma(z)\|_2 \le \lambda\,\|\delta z\|_2$. Insert $\delta z= q(K_g-K_i)^\top/\sqrt{d_h}$.

> [!note] summary
>
> - MHA: read/write $K,V\in\mathbb{R}^{n\times(hd_h)}$ ⇒ HBM‑limited at long $n$.
> - GQA (G groups): memory $\sim 2G d_h$; bandwidth improves, approaching MQA when $G$ is small. [@ainslie2023gqatraininggeneralizedmultiquery]
> - MLA (latent $d_c$): memory $\sim d_c$; up‑projections absorbed into $W_Q,W_O$ at inference.
> - IO‑aware kernels: FlashAttention tiles Q/K/V and fuses softmax; exact attention with reduced HBM traffic. [@dao2022flashattentionfastmemoryefficientexact; @shah2024flashattention3fastaccurateattention]

## reference cuda kernels (single head, single batch)

Three kernels: scores $S=QK^\top/\sqrt{d}$, row‑softmax, and apply to values $O=SV$.

### Scores $S=QK^\top/\sqrt{d}$

![[lectures/2/qk_scores.cu]]

### Row‑wise softmax (stable; in‑place on S)

![[lectures/2/row_softmax.cu]]

### Apply to values $O = S V$

![[lectures/2/apply_values.cu]]

Masked causal attention sets $S_{ij}=-\infty$ for $j>i$.

## triton implementation

Two‑pass Triton kernel: (1) running row‑max/sum for softmax; (2) recompute tiles to form $O$ without materializing $S$.

![[lectures/2/attention_triton.py]]

## notes

- IO‑aware tiling (FlashAttention): stream K/V tiles, maintain running $(m_i,\ell_i)$, avoid $n\times n$ intermediates. [@dao2022flashattentionfastmemoryefficientexact; @shah2024flashattention3fastaccurateattention]
- KV paging + sharing: paged layouts with GQA/MQA/MLA reduce HBM traffic. [@ainslie2023gqatraininggeneralizedmultiquery; @shazeer2019fasttransformerdecodingwritehead]
- RoPE in fused kernels: apply rotations within tiles; with MLA use decoupled scheme.
- Mixed precision: FP16/FP8 with LSE stabilization.

---

## appendix

E.1 MHA as mixture of kernel smoothers. For head $i$, $\kappa_i(q,k)=\exp(\langle W_Q^i q, W_K^i k\rangle/\sqrt{d_h})$ and

$$
\mathrm{MHA}(x_t)=W_O\,[\underbrace{\textstyle\sum_{j\le t}\frac{\kappa_i(h_t,h_j)}{\sum_{\ell\le t}\kappa_i(h_t,h_\ell)}\, (W_V^i h_j)}_{\text{kernel smoother of }V_i}]_{i=1}^h.
$$

E.2 MLA cache accounting. Per‑token, per‑layer: MHA $2hd_h$ vs. MLA $d_c$. Example: $h{=}64,d_h{=}128,d_c{=}1024$ → $\approx16\times$ reduction.
