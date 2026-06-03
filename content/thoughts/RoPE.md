---
date: '2025-08-07'
description: rotary position embedding — pairwise rotation of Q/K so inner products encode relative offset, plus context-extension via frequency rescaling.
id: RoPE
modified: 2026-06-02 09:30:00 GMT-04:00
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

rotary position embedding [@su2023roformerenhancedtransformerrotary]. instead of adding a position vector to the input embedding, RoPE rotates query and key in $\mathbb{R}^2$ planes by an angle that depends on absolute position. the inner product $\langle R_m q, R_n k\rangle$ then depends only on the offset $m - n$, so attention sees relative position without any extra parameter.

## the 2D case

for a single 2-vector $q \in \mathbb{R}^2$ at position $m$, define the rotation

$$
R_m = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}, \qquad \tilde q_m = R_m q.
$$

the rotation is a unitary, so $\lVert \tilde q_m \rVert = \lVert q \rVert$. for two positions $m,n$,

$$
\langle R_m q, R_n k\rangle = q^\top R_m^\top R_n k = q^\top R_{n-m} k,
$$

because $R_m^\top R_n = R_{n-m}$ (rotations are an abelian one-parameter group). the dot product carries the offset $n-m$, not the absolute positions.

## $d$-dimensional generalisation

split the head dimension $d$ into $d/2$ disjoint pairs $\{(2i, 2i+1)\}_{i=0}^{d/2-1}$ and rotate each pair by its own frequency $\theta_i$. with the geometric schedule

$$
\theta_i = b^{-2i/d}, \qquad b = 10000,
$$

the rotation matrix is block-diagonal,

$$
R_m^{(d)} = \mathrm{diag}\bigl(R_m(\theta_0), R_m(\theta_1), \dots, R_m(\theta_{d/2-1})\bigr),
$$

and applied identically to $q$ and $k$ before the attention dot. the lowest-index pairs spin fast (period $\approx 2\pi$); the highest-index pairs spin slowly (period $\approx 2\pi b$). a model trained on context length $L$ has seen each frequency exercised over the angular interval $[0, L\theta_i)$.

> [!important] relative-offset property
>
> for any $m, n$ and any pair index $i$,
>
> $$
> \langle R_m^{(d)} q, R_n^{(d)} k\rangle = \sum_{i=0}^{d/2-1} \langle R_{n-m}(\theta_i)\, q_i, k_i\rangle.
> $$
>
> attention reads ==only the offset==. there is no learned position parameter; the inductive bias lives in the schedule $\{\theta_i\}$.

## scaling to longer contexts

the failure mode at test-time is that positions $m > L$ rotate the slow pairs into angles the model never saw at training. the three families below all reshape the schedule rather than retrain from scratch.

### NTK-aware scaling

stretch the base $b$ instead of compressing positions. for a context-extension factor $s = L'/L$, set

$$
b' = b \cdot s^{d/(d-2)}.
$$

high frequencies are nearly untouched (so short-range structure survives) while low frequencies pick up the slack. derived from a neural-tangent-kernel argument that the highest-frequency coordinates dominate the local interpolation error.

### YaRN

[@peng2023yarnefficientcontextwindow] partitions the pair indices into three bands by wavelength $\lambda_i = 2\pi / \theta_i$ relative to the training length $L$:

- **high-frequency** ($\lambda_i \ll L$, fully exercised) — leave $\theta_i$ untouched.
- **low-frequency** ($\lambda_i \gtrsim L$, never completed a period) — divide $\theta_i$ by $s$, the position-interpolation move.
- **mid-band** — ramp linearly between the two extremes.

YaRN also rescales the attention temperature by $\sqrt{1 + 0.1\,\log s}$ to compensate for the wider rotational support shrinking softmax mass. small fine-tune (typically $\sim 100$ steps) recovers the original perplexity at the new length.

### LongRoPE

[@ding2024longropeextendingllmcontext] keeps YaRN's per-band treatment but searches the per-dimension rescaling factors with an evolutionary loop on a small calibration set, then trains progressively from $L$ to $4L$ to $\dots$ to the target. reported context windows reach $2\text{M}$ tokens with sub-1B parameter-updates.

| scheme                 | what is rescaled                          | extra training           |
| ---------------------- | ----------------------------------------- | ------------------------ |
| position interpolation | all $\theta_i$ uniformly by $s$           | yes, small               |
| NTK-aware              | base $b \to b s^{d/(d-2)}$                | none                     |
| YaRN                   | per-band $\theta_i$ + softmax temperature | $\sim 100$ steps         |
| LongRoPE               | per-dim $\theta_i$ found by search        | progressive, multi-stage |

## implementation note

RoPE applies to $Q$ and $K$ only; values $V$ are left alone, which is what lets MLA cache one RoPE-bearing duplicate $k_t^R$ alongside the latent $c_t^{KV}$ without rotating the reconstructed $v_{t,i}^C$. see [[thoughts/MLA|MLA]] eq. (3)–(4) for the decoupling that makes the cache compatible with rotary embeddings.

> [!todo]+ follow-ups
>
> - derive the NTK rescaling $b' = b s^{d/(d-2)}$ from the kernel-interpolation argument.
> - compare position interpolation vs. NTK-aware on a fixed eval (perplexity at $2L, 4L, 8L$); which dims dominate the loss?
> - read Chen et al. (2023) for the original PI proposal and bloc97's reddit thread for NTK-aware.
