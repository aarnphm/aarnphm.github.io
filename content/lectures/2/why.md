---
id: why
tags:
  - seed
  - ml
  - math
description: motivations of attention
date: "2025-08-21"
modified: 2025-08-26 15:54:20 GMT-04:00
title: why
---

---

## why dot products? why _softmax_?

Let a query $q\in\mathbb{R}^{d}$ retrieve from a dictionary $\{(k_j,v_j)\}_{j=1}^n$, with keys $k_j\in\mathbb{R}^{d}$ and values $v_j\in\mathbb{R}^{d_v}$. Define logits

$$
z_j \;=\; \frac{\langle q,k_j\rangle}{\sqrt d}\in\mathbb{R}.
$$

**Geometric fact.** If $\|q\|=\|k_j\|=c$ (LayerNorm makes this approximately true in practice), then

$$
\langle q,k_j\rangle \;=\; \tfrac12\big(\|q\|^2+\|k_j\|^2-\|q-k_j\|^2\big)
= c^2-\tfrac12\|q-k_j\|^2.
$$

Hence $\exp(z_j/T)\propto \exp\big(-\|q-k_j\|^2/(2T\sqrt d)\big)$, i.e. **softmax‑weights are Gaussian/RBF weights in feature space** up to a per‑row constant that cancels out. Attention is thus kernel smoothing on a learned feature map. (This identity is purely algebraic; the $\sqrt d$ in the logits rescales the “bandwidth”.)

## softmax from entropy‑regularized maximization (Gibbs variational principle)

For a fixed logit vector $z\in\mathbb{R}^n$ and temperature $T>0$, consider the **entropy‑regularized linear program**

$$
\boxed{\qquad
p^\star(z,T)\;=\;\arg\max_{p\in\Delta^{n-1}}\ \Big\{\,\langle p,z\rangle \;+\; T\,H(p)\,\Big\}\!,
\qquad H(p)=-\sum_{j=1}^n p_j\log p_j,
\qquad}
\tag{1}
$$

where $\Delta^{n-1}$ is the probability simplex.

**Solution.** Lagrange/KKT gives

$$
0 = \frac{\partial}{\partial p_j}\Big(\langle p,z\rangle+T H(p)+\alpha(\textstyle\sum_i p_i-1)\Big)
= z_j - T(1+\log p_j) + \alpha,
$$

so $p_j\propto e^{z_j/T}$, i.e.

$$
\boxed{\,p^\star(z,T)=\operatorname{softmax}(z/T)\, .\,}
$$

This is the **Gibbs distribution**; identity (1) is the discrete form of the **Gibbs variational principle** / variational representation of the **log‑partition** (log‑sum‑exp). A classic statement is

$$
\log\!\sum_{j=1}^n e^{z_j/T}
= \max_{p\in\Delta^{n-1}}\ \Big\{ \tfrac1T\langle p,z\rangle + H(p) \Big\},
$$

so softmax is the **unique** maximizer, and log‑sum‑exp is the **Fenchel conjugate** of negative entropy restricted to the simplex. ([People at EECS Berkeley][1], [Stanford University][2], [ICDST][3])

**Equivalent KL form.** Since

$$
\langle p,z\rangle + T H(p)
= T\!\left( - D_{\mathrm{KL}}\!\big(p\,\|\,\operatorname{softmax}(z/T)\big)\right) + T\log\!\sum_j e^{z_j/T},
$$

the same solution is

$$
p^\star = \arg\min_{p\in\Delta^{n-1}} D_{\mathrm{KL}}\!\big(p\,\|\,\operatorname{softmax}(z/T)\big).
$$

Thus **softmax is the KL‑projection** of any candidate distribution onto the Gibbs density induced by $z$. (Variational/entropy references.) ([People at EECS Berkeley][1], [willsky.lids.mit.edu][4])

**Convex‑analysis differential facts.** Let $\mathrm{lse}_\lambda(z)=\frac1\lambda\log\sum_j e^{\lambda z_j}$ (inverse temperature $\lambda=1/T$). Then

$$
\nabla \mathrm{lse}_\lambda(z)=\operatorname{softmax}(\lambda z),\qquad
\nabla^2 \mathrm{lse}_\lambda(z)=\lambda\big(\operatorname{Diag}(p)-pp^\top\big).
$$

Hence softmax is the gradient of a convex potential; it is **$\lambda$‑Lipschitz** and **$1/\lambda$‑cocoercive** in $\ell_2$, with spectrum governed by $\operatorname{Diag}(p)-pp^\top$. For $\lambda=1$ (the usual case), softmax is non‑expansive. These properties are standard and useful for stability bounds below.&#x20;

## attention as a solved optimization

For each query row $q_i$, define logits $z^{(i)}\in\mathbb{R}^n$ by

$$
z^{(i)}_j = \frac{\langle q_i, k_j\rangle}{\sqrt d}.
$$

**Row‑wise problem.**

$$
p^{(i)} \;=\; \arg\max_{p\in\Delta^{n-1}}\ \big\{\langle p, z^{(i)}\rangle + T\,H(p)\big\}
\;\;=\;\; \operatorname{softmax}\!\big(z^{(i)}/T\big),
$$

and the attention output is the _expected value_

$$
o_i \;=\; \sum_{j=1}^n p^{(i)}_j\, v_j \;=\; \mathbb{E}_{j\sim p^{(i)}}[\,v_j\,].
$$

Thus **self‑attention computes the entropy‑regularized best convex combination of the values**, where “utility” is logit score $\langle q_i,k_j\rangle/\sqrt d$ and regularizer is Shannon entropy. Two useful limits:

- $T\to 0$: $p^{(i)}$ collapses on $\arg\max_j z^{(i)}_j$ (hard nearest neighbor).
- $T\to\infty$: $p^{(i)}$ tends to uniform over keys (global averaging).

This variational picture is exactly the exponential‑family view where log‑sum‑exp is the cumulant generating function and entropy is the dual potential. ([People at EECS Berkeley][1])

## From single head => MHA => GQA

### Multi‑Head Attention (MHA) = parallel entropic programs

For head $h=1,\dots,H$, with its own projections $W_Q^h,W_K^h,W_V^h$, we solve

$$
p_h^{(i)}=\arg\max_{p\in\Delta^{n-1}}\Big\{\langle p, z_h^{(i)}\rangle + T H(p)\Big\},\qquad
z_{h,j}^{(i)}=\tfrac{\langle q_i^h,k_j^h\rangle}{\sqrt{d_h}},
$$

and return $o_i = W_O[\,\sum_j p_{1,j}^{(i)} v_{1,j}\;;\dots; \sum_j p_{H,j}^{(i)} v_{H,j}\,]$.
**Key point.** The head‑wise problems **decouple** (same entropy but different logits/values per head); coupling enters linearly through $W_O$. (Transformer original.) ([NeurIPS Papers][5])

### Grouped‑Query / Multi‑Query Attention (GQA/MQA)

In **MQA**, all heads share a single $K,V$; in **GQA**, heads are partitioned into $G$ groups, each sharing $K_g,V_g$. For a head $h$ in group $g(h)$,

$$
p_h^{(i)}=\arg\max_{p\in\Delta^{n-1}}\Big\{\underbrace{\langle p, z_{g(h)}^{(i)}\rangle}_{\text{shared }K_g} + T H(p)\Big\},
\quad z_{g(h),j}^{(i)}=\tfrac{\langle q_i^h,k_{g(h),j}\rangle}{\sqrt{d_h}}.
$$

So **GQA changes only the dictionary** (shared $K,V$), not the entropy‑regularized decision rule. (Primary sources for MQA/GQA.) ([arXiv][6])

---

## 5) Stability of weights when sharing keys (GQA vs MHA)

Let a head $h$ originally use $K_h$ and, under GQA, use $K_g$. For a fixed query $q$, logits change by

$$
\delta z \;=\;\frac{1}{\sqrt d}\, \big( K_g - K_h\big)q.
$$

Since softmax is $\lambda$‑Lipschitz with $\lambda=1/T$ (in $\ell_2$), we have

$$
\|p_g - p_h\|_2 \;\le\; \lambda\,\|\delta z\|_2
\;\le\; \tfrac{1}{T\sqrt d}\,\|K_g-K_h\|_{op}\,\|q\|_2.
$$

And the output difference obeys

$$
\|o_g-o_h\|_2 \;\le\; \|p_g-p_h\|_1\cdot \max_j \|v_j\|_2
\;\le\; \sqrt n\,\|p_g-p_h\|_2 \cdot \max_j \|v_j\|_2.
$$

Thus the **quality loss** from grouping heads is controlled by (i) how different the original head‑specific keys were (operator norm $\|K_g-K_h\|_{op}$), (ii) query norm, and (iii) inverse temperature $1/T$. (Lipschitz/cocoercivity facts from convex analysis of softmax.)&#x20;

## Why the $\tfrac{1}{\sqrt d}$ scale?

If $q,k$ have i.i.d. zero‑mean, unit‑variance entries, then $\mathrm{Var}(\langle q,k\rangle)=d$; dividing by $\sqrt d$ keeps the **logit variance $O(1)$** as width grows, stabilizing the entropy‑regularized problem and its gradients. (Transformer §3.2.) ([NeurIPS Papers][5])

## takeaway

- **Self‑attention** on row $i$ =
  $\displaystyle \arg\max_{p\in\Delta}\big\{ \langle p,\tfrac{1}{\sqrt d}K q_i\rangle + T H(p)\big\}$
  $\implies p=\operatorname{softmax}((K q_i)/(\sqrt d\,T))$, $o_i=\mathbb{E}_{p}[V]$.
- **MHA** = $H$ independent copies of the same convex program on different feature maps, linearly combined by $W_O$.
- **GQA/MQA** = replace per‑head dictionaries by shared group dictionaries in the _same_ convex program.
- **Kernel view** (with normalized $q,k$): attention weights are **RBF weights** with bandwidth $\sigma^2 = T\sqrt d$.

## minimal code: run the **kernel equivalence**

The cell below **runs** a sanity check. With $\|q_i\|=\|k_j\|=1$, it shows that

$$
\operatorname{softmax}\!\Big(\tfrac{QK^\top}{\sqrt d\,T}\Big)
\quad\equiv\quad
\text{row}-\text{normalize}\Big(\exp\!\big(-\tfrac{\|q_i-k_j\|^2}{2\sigma^2}\big)\Big)
$$

when $\sigma^2=T\sqrt d$. The reported differences are at machine precision.

**What it does**

1. Samples $Q,K,V$.
2. L2‑normalizes rows of $Q$ and $K$.
3. Computes standard attention vs. RBF kernel smoother ($\sigma^2=T\sqrt d$).
4. Prints norms of the differences in weights and outputs.

**Result from running it just now (n=32, d=64, dv=16, T=1):**

```text
sigma^2 used = 8.0
max|ΔW| ≈ 1.39e-17,  ||ΔW||_F ≈ 1.22e-16
max|ΔO| ≈ 1.66e-16,  ||ΔO||_F ≈ 8.29e-16
```

- Use (1) to **motivate softmax** as the unique solution of a **maximum‑entropy** selection over candidates, not an ad‑hoc normalization.
- Show $\nabla\!\operatorname{lse}=$ softmax and the Hessian $\lambda(\mathrm{Diag}(p)-pp^\top)$ to connect smoothness, conditioning, and robustness.&#x20;
- For **MHA**, emphasize **parallel entropic programs** (independence across heads) and that **GQA** modifies only the dictionary. Cite the primary GQA & MQA sources. ([arXiv][7])
- If you later discuss **MLA/DeepSeek‑V2**, keep the same optimization view but note that MLA _compresses the dictionary_ (KV) into a latent and reconstructs on the fly; the entropic program itself stays row‑wise. ([arXiv][8])

---

## Appendix A — two short, slide‑ready proofs

**A.1 Softmax solves a MaxEnt problem.**
Maximize $\langle p,z\rangle+T H(p)$ on $\Delta^{n-1}$.
The Lagrangian $L(p,\alpha)=\sum_j p_j z_j + T\sum_j (-p_j\log p_j)+\alpha(\sum_j p_j-1)$ yields
$\partial L/\partial p_j= z_j - T(1+\log p_j)+\alpha = 0$.
Therefore $p_j=\exp((z_j+\alpha-T)/T)\propto e^{z_j/T}\Rightarrow p=\operatorname{softmax}(z/T).$
Uniqueness follows because the objective is strictly concave on the simplex. (Gibbs variational principle / conjugacy.) ([People at EECS Berkeley][1])

**A.2 Lipschitz of softmax.**
Let $\mathrm{lse}_\lambda(z)=\lambda^{-1}\log\sum_j e^{\lambda z_j}$. Then $\nabla \mathrm{lse}_\lambda=\sigma(\lambda z)$ and
$\nabla^2\mathrm{lse}_\lambda(z)=\lambda(\mathrm{Diag}(p)-pp^\top)\succeq 0$ with operator norm $\le\lambda$.
Hence $\sigma(\lambda\cdot)$ is $\lambda$-Lipschitz and $1/\lambda$-cocoercive in $\ell_2$. Set $\lambda=1/T$.&#x20;

## References

- **Convex/variational foundation for softmax/log‑sum‑exp:**
  Gao & Pavel, _On the Properties of the Softmax Function_ (gradient, Hessian, Lipschitz, conjugacy).&#x20;
  Wainwright & Jordan, _Graphical Models, Exponential Families, and Variational Inference_ (log‑partition as sup of linear + entropy). ([People at EECS Berkeley][1])
  Boyd & Vandenberghe, _Convex Optimization_ (log‑sum‑exp properties). ([Stanford University][2])
  Cover & Thomas, _Elements of Information Theory_ (Gibbs/maximum‑entropy principles). ([ICDST][3])

- **Architectures:**
  Vaswani et al., _Attention is All You Need_ (scaled dot‑product; multi‑head). ([NeurIPS Papers][5])
  Shazeer, _Fast Transformer Decoding: One Write‑Head is All You Need_ (MQA). ([arXiv][6])
  Ainslie et al., _GQA_ (grouped‑query attention). ([arXiv][7], [ACL Anthology][9])
  DeepSeek‑V2 (MLA / decoupled RoPE, KV compression). ([arXiv][8])

[1]: https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf "Graphical Models, Exponential Families, and Variational ..."
[2]: https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf "Convex Optimization"
[3]: https://dl.icdst.org/pdfs/files/aea72e61329cd4684709fa24f15ac098.pdf "elements of information theory"
[4]: https://willsky.lids.mit.edu/publ_pdfs/180_pub_IEEE.pdf "A New Class of Upper Bounds on the Log Partition Function"
[5]: https://papers.neurips.cc/paper_files/paper/2020/file/1bd413de70f32142f4a33a94134c5690-Paper.pdf "Smoothness Tradeoffs for Soft-Max Functions"
[6]: https://arxiv.org/abs/1911.02150 "Fast Transformer Decoding: One Write-Head is All You Need"
[7]: https://arxiv.org/abs/2305.13245 "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
[8]: https://arxiv.org/pdf/2405.04434 "DeepSeek-V2"
[9]: https://aclanthology.org/2023.emnlp-main.298/ "GQA: Training Generalized Multi-Query Transformer ..."
