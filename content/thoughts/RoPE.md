---
date: '2025-08-07'
description: rotary position embeddings as pairwise Q/K rotations, the relative-offset identity, and frequency rescaling for longer contexts.
id: RoPE
modified: 2026-09-10 09:10:35 GMT-04:00
seealso:
  - '[[thoughts/positional embeddings|positional embeddings]]'
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/MLA|MLA]]'
socials:
  eleuther: https://blog.eleuther.ai/rotary-embeddings/
tags:
  - ml
  - seed
title: RoPE
---

RoPE [@su2023roformerenhancedtransformerrotary] rotates query and key coordinates in 2D pairs before the attention dot product. position $m$ gives a pair angle $m\theta$. the dot product cancels the absolute angles and leaves a signed offset, $n-m$.

qualification: the rotation depends on relative position. the score still depends on $q$ and $k$, which carry token content and earlier-layer state.

## the 2D case

for a frequency $\theta$, write

$$
R_m(\theta)=\begin{pmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)&\cos(m\theta)
\end{pmatrix}.
$$

this is an orthogonal matrix: $R_m^\top R_m=I$, so rotation preserves norm. for fixed $q,k\in\mathbb{R}^2$,

$$
\langle R_m q,R_n k\rangle
=q^\top R_m^\top R_n k
=q^\top R_{n-m}k.
$$

a shared position shift leaves this expression unchanged. moving the rotation to the query reverses its sign:

$$
\langle q,R_{n-m}k\rangle=\langle R_{m-n}q,k\rangle.
$$

sign check: take $q=(1,0)^\top$, $k=(0,1)^\top$, and $(n-m)\theta=\pi/2$. the rotated key is $(-1,0)^\top$, giving a score of $-1$. rotating the query forward by the same angle would give $+1$.

## $d$-dimensional generalisation

let rotary dimension $d$ be even. split the coordinates into $d/2$ pairs and give pair $i$ the frequency

$$
\theta_i=b^{-2i/d},\qquad i=0,\ldots,d/2-1.
$$

the original schedule uses $b=10000$. stack the pairwise rotations into a block-diagonal matrix $R_m^{(d)}$. applying the two-dimensional identity to each block gives

$$
\left\langle R_m^{(d)}q,R_n^{(d)}k\right\rangle
=\sum_{i=0}^{d/2-1}q_i^\top R_{n-m}(\theta_i)k_i.
$$

here $q_i,k_i\in\mathbb{R}^2$ are coordinate pairs. their wavelengths, measured in units of token positions, are

$$
\lambda_i=\frac{2\pi}{\theta_i},\qquad
\lambda_0=2\pi,\qquad
\lambda_{d/2-1}=2\pi b^{1-2/d}.
$$

the schedule supplies several distance scales. it adds no learned position vectors. the model learns how to use those scales through its query and key projections.

## scaling to longer contexts

RoPE is defined at every integer position. that algebra alone gives no guarantee about accuracy at longer distances. a slow pair may have covered only part of a turn during training; extending the sequence changes which phases and combinations the model encounters.

let $L$ be the original context length, $L'$ the target, and $s=L'/L>1$. position interpolation replaces position $m$ by $m/s$, equivalently replacing every $\theta_i$ by $\theta_i/s$. adjacent positions then have $1/s$ of their original angular separation. Chen et al. used this transformation with fine-tuning. [@chen2023extendingcontextwindowlarge]

### NTK-aware scaling

bloc97's NTK-aware base scaling keeps the fastest pair fixed and stretches the slowest wavelength by $s$. for $d>2$,

$$
(b')^{1-2/d}=s b^{1-2/d}
\quad\Longrightarrow\quad
b'=b s^{d/(d-2)}.
$$

consequently $\theta'_i=\theta_i s^{-2i/(d-2)}$: the first frequency stays fixed and the last is divided by $s$. YaRN's appendix gives this derivation for bloc97's NTK-aware proposal. this is endpoint matching; accuracy at the target context length still has to be measured. [@peng2023yarnefficientcontextwindow]

### YaRN

YaRN combines frequency-dependent interpolation with attention scaling. it leaves high frequencies unchanged, divides low frequencies by $s$, and blends between them. the bands use the number of rotations within $L$.

for the LLaMA-family experiments, its fitted amplitude factor is

$$
a=1+0.1\ln s,\qquad
\operatorname{softmax}\!\left(\frac{a^2\tilde Q\tilde K^\top}{\sqrt{d_h}}\right),
$$

here $\tilde Q,\tilde K$ use the rescaled frequencies before amplitude scaling, and $d_h$ is the attention head dimension. multiplying both by $a$ multiplies their dot product by $a^2$. the equivalent temperature is $t=a^{-2}$. the amplitude fit is empirical. [@peng2023yarnefficientcontextwindow]

### LongRoPE

LongRoPE searches rescaling factors across rotary dimensions and a prefix of token positions to leave unscaled. pairs crossing that prefix boundary lose the pure relative-offset property because the two positions follow different scaling rules.

its LLaMA2 procedure fine-tunes through a $256\mathrm{k}$ context, then searches another interpolation to reach $2048\mathrm{k}$; here $\mathrm{k}$ denotes $1024$ tokens. a separate short-context adjustment recovers performance at $8\mathrm{k}$. the paper reports $1000$ fine-tuning steps for this procedure; steps and training-token counts are different budgets. [@ding2024longropeextendingllmcontext]

## implementation note

in standard decoder implementations, RoPE rotates $Q$ and $K$; $V$ stays unrotated. queries and keys must use the checkpoint's coordinate-pairing convention. adjacent-pair and split-half layouts require the corresponding weight convention.

[[thoughts/MLA|DeepSeek's decoupled MLA]] caches a separate RoPE-bearing key alongside its compressed KV latent. this keeps the position-dependent key term separate from the content term whose projection can be absorbed into the query computation.

> [!todo]+ follow-ups
>
> - compare position interpolation and base scaling at $2L$, $4L$, and $8L$, recording both perplexity and retrieval accuracy.
> - test cached decoding when the scaling factor changes mid-sequence. previously rotated keys must remain consistent with the new queries.
