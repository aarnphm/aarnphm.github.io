---
id: attention
tags:
  - seed
date: "2025-08-21"
modified: 2025-08-25 10:40:32 GMT-04:00
title: attention primer
---

## primer

- **Notation.** Sequence $X\in\mathbb{R}^{n\times d}$. Projections $W_Q,W_K,W_V$. Heads $h$, head dim $d_h$.
- **Softmax/LSE.** $\operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$; $\operatorname{LSE}(z)=\log\sum_j e^{z_j}$.
- **Stable softmax.** Use $z\leftarrow z-\max(z)$ (log‑sum‑exp trick).
- **RoPE** injects relative position by rotating $(q,k)$ in 2‑D frequency planes before the dot product. [@su2023roformerenhancedtransformerrotary]
- **Why the $\frac{1}{\sqrt{d_k}}$** scale? It variance‑normalizes dot products so logits don’t blow up with $d_k$ (see Prop. A1 below; and [@vaswani2023attentionneed; §3.2.1])
- **KL** $D_{\mathrm{KL}}(P\|Q)=\sum_i P_i\log\frac{P_i}{Q_i}$. Jointly convex in $(P,Q)$; bounds total variation via Pinsker. Use it to (i) align attention distributions, (ii) measure head agreement, (iii) distill approximations (e.g., MHA→GQA). ([home.ttic.edu][3], [people.lids.mit.edu][4])
- **Softmax temperature $T$.** $\operatorname{softmax}(z/T)$ controls entropy smoothly; $T\!\uparrow\Rightarrow$ higher entropy. (Theory & practice refs.) ([Cross Validated][5], [arXiv][6])
- **Softmax = $\nabla$LSE.** The Jacobian is $J=\lambda(\operatorname{Diag}(p)-pp^\top)$; Lipschitz constant scales with inverse temperature parameter $\lambda$. Useful for error bounds later.&#x20;

## self‑attention from first principles

Let $Q=XW_Q,\;K=XW_K,\;V=XW_V$. Scaled dot‑product attention:

$$
\mathrm{Attn}(Q,K,V)\;=\;\sigma\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,\quad
\sigma(\cdot)=\text{row‑softmax}.
$$

**A.1 Proposition (Permutation equivariance).**
Without positional encodings, self‑attention is permutation‑equivariant:
for any permutation matrix $P\in\mathbb{R}^{n\times n}$,

$$
\mathrm{Attn}(PQ,PK,PV)=P\,\mathrm{Attn}(Q,K,V).
$$

_Proof._

$PQ(PK)^\top=PQK^\top P^\top$. Row‑softmax respects row permutations, i.e., $\sigma(PZP^\top)=P\sigma(Z)P^\top$. Then $P\sigma(QK^\top)P^\top \cdot PV = P(\sigma(QK^\top)V) \; \boxed{}$.

**A.2 Proposition (Why the $\tfrac{1}{\sqrt{d_k}}$ scale?).**

Assume $q,k\in\mathbb{R}^{d_k}$ have i.i.d. entries with $\mathbb{E}[q_i]=\mathbb{E}[k_i]=0$, $\operatorname{Var}(q_i)=\operatorname{Var}(k_i)=1$. Then
$\operatorname{Var}(q^\top k)=d_k$. Scaling by $1/\sqrt{d_k}$ keeps logit variance $\approx 1$, stabilizing softmax and gradients as width grows. (Matches the original motivation. ) ([NeurIPS Papers][2])

> [!abstract]- Detailed
>
> Let $q,k\in\mathbb{R}^d$. Define the (unscaled) logit $S:=q^\top k = \sum_{i=1}^d q_i k_i$ and the scaled logit
>
> $$
> Z:=\frac{q^\top k}{\sqrt{d}}\;.
> $$
>
> Assume $\mathbb{E}[q]=\mathbb{E}[k]=0$ and $\mathrm{Cov}(q)=\Sigma_q$, $\mathrm{Cov}(k)=\Sigma_k$, with $q$ and $k$ **independent as random vectors** (no independence across coordinates is required). Then
>
> $$
> \mathrm{Var}(S)=\operatorname{tr}(\Sigma_q\Sigma_k),\qquad
> \mathrm{Var}(Z)=\frac{1}{d}\operatorname{tr}(\Sigma_q\Sigma_k).
> $$
>
> In particular, under **isotropy** $\Sigma_q=\Sigma_k=I_d$ one gets
>
> $$
> \mathrm{Var}(S)=d,\qquad \mathrm{Var}(Z)=1.
> $$

_Proof._

$\operatorname{Var}\big(\sum_i q_i k_i\big)=\sum_i \operatorname{Var}(q_i k_i)=\sum_i \mathbb{E}[q_i^2]\mathbb{E}[k_i^2]=d_k \; \boxed{}$.

Write $S=q^\top k = k^\top q$. Since $q$ and $k$ are independent and mean‑zero,

$$
\mathbb{E}[S]=\mathbb{E}[q]^\top \mathbb{E}[k]=0.
$$

For the second moment,

$$
S^2=(q^\top k)^2=(k^\top q)(q^\top k)=k^\top (qq^\top)k.
$$

Take conditional expectation given $k$ and then average:

$$
\mathbb{E}[S^2]
= \mathbb{E}_k\!\left[\,k^\top \,\mathbb{E}_q[qq^\top]\, k\,\right]
= \mathbb{E}_k\!\left[\,k^\top \Sigma_q\, k\,\right]
= \mathbb{E}\!\left[\operatorname{tr}\!\big(\Sigma_q\, k k^\top\big)\right]
= \operatorname{tr}\!\big(\Sigma_q\, \mathbb{E}[k k^\top]\big)
= \operatorname{tr}(\Sigma_q \Sigma_k).
$$

Since $\mathbb{E}[S]=0$, we have $\mathrm{Var}(S)=\mathbb{E}[S^2]=\operatorname{tr}(\Sigma_q\Sigma_k)$. Dividing by $d$ yields the claimed variance for $Z=S/\sqrt{d}$.

_Remarks._ (i) If $\mu_q:=\mathbb{E}[q]$ and $\mu_k:=\mathbb{E}[k]$ are nonzero, the same computation gives

$$
\mathrm{Var}(S)=\operatorname{tr}(\Sigma_q\Sigma_k)+\mu_k^\top \Sigma_q\,\mu_k+\mu_q^\top \Sigma_k\,\mu_q,
$$

so the isotropic, zero‑mean case reduces to $\operatorname{tr}(I\cdot I)=d$. (ii) A special case of this expectation identity appears throughout high‑dimensional probability; see, e.g., Vershynin’s notes on quadratic/bilinear forms.&#x20;

**A.3 The kernel/regression view (why attention “works”).**
If vectors are **length‑normalized** (LayerNorm makes this plausible in practice), then

$$
q^\top k=\tfrac12(\|q\|^2+\|k\|^2-\|q-k\|^2)=\text{const}-\tfrac12\|q-k\|^2.
$$

Consequently,

$$
\operatorname{softmax}_j\!\left(\frac{q^\top k_j}{T}\right)
\;\propto\; \exp\!\left(-\frac{\|q-k_j\|^2}{2T}\right),
$$

i.e., **Gaussian/RBF kernel weights** up to a per‑row constant that cancels in softmax. Thus, self‑attention is a **Nadaraya–Watson kernel smoother** of $V$ evaluated at query $q$. This gives an intuitive statistical footing: attention does _local averaging in feature space_. (Kernel regression and “linear/softmax‑as‑kernel” references.) ([Wikipedia][7], [arXiv][8], [Proceedings of Machine Learning Research][9])

> [!math] claim
>
> Fix a query $q\in\mathbb{R}^d$ and keys $\{k_j\}_{j=1}^n\subset\mathbb{R}^d$. Consider _scaled dot‑product attention_ at temperature $T>0$:
>
> $$
> w_j(q)\;=\;\frac{\exp\!\big(\langle q,k_j\rangle/(\sqrt{d}\,T)\big)}{\sum_{\ell=1}^n \exp\!\big(\langle q,k_\ell\rangle/(\sqrt{d}\,T)\big)},\qquad
> o(q)=\sum_{j=1}^n w_j(q)\,v_j.
> $$
>
> If $\|q\|=\alpha$ and $\|k_j\|=\beta$ for all $j$ (i.e., **row‑wise $\ell_2$ normalization to a constant norm**), then
>
> $$
> w_j(q)\;=\;\frac{\exp\!\big(-\|q-k_j\|^2/(2\sigma^2)\big)}{\sum_{\ell=1}^n\exp\!\big(-\|q-k_\ell\|^2/(2\sigma^2)\big)}\quad\text{with}\quad
> \boxed{\ \sigma^2 = T\,\sqrt{d}\ }.
> $$

Hence $o(q)$ is exactly the **Nadaraya–Watson kernel smoother** with Gaussian (RBF) kernel and bandwidth $\sigma^2$, evaluated at the “input” $q$ using the “samples” $\{(k_j,v_j)\}_{j=1}^n$. ([NeurIPS Papers][1], [Wikipedia][2])

_Proof._

1. **Start from scaled dot‑product attention.**
   Per Vaswani et al. (2017), row‑wise attention weights are softmax of scaled dot products:

$$
w_j(q)\;\propto\;\exp\!\Big(\frac{\langle q,k_j\rangle}{\sqrt{d}\,T}\Big).
$$

(We write $\propto$ because the common normalization factor is the row sum.) ([NeurIPS Papers][1])

2. **Rewrite the inner product as a squared‑distance expression.**
   For real inner‑product spaces, the **polarization identity** gives

$$
\langle q,k_j\rangle\;=\;\tfrac12\big(\|q\|^2+\|k_j\|^2-\|q-k_j\|^2\big).
$$

This is a standard identity equivalent to the dot‑product axioms. ([Wikipedia][3])

3. **Insert into the logits and separate constants.**
   Using Step 2,

   $$
   \frac{\langle q,k_j\rangle}{\sqrt{d}\,T}
   = \underbrace{\frac{\|q\|^2}{2\sqrt{d}\,T}}_{\text{row‑constant}}
   + \frac{\|k_j\|^2}{2\sqrt{d}\,T}
   -\frac{\|q-k_j\|^2}{2\,T\,\sqrt{d}}.
   $$

The first term is constant in $j$ and therefore **cancels inside softmax** (adding a constant to all logits leaves softmax unchanged).

4. **Impose equal‑norm keys (and query) for exact equivalence.**
   If $\|k_j\|\equiv\beta$ for all $j$ (and $\|q\|=\alpha$), then the second term is also constant in $j$ and **cancels as well** inside softmax. Thus

$$
w_j(q)\;\propto\;\exp\!\Big(-\frac{\|q-k_j\|^2}{2\,T\,\sqrt{d}}\Big).
$$

Recognizing the **Gaussian/RBF kernel**

$$
K_\sigma(x,y)=\exp\!\Big(-\frac{\|x-y\|^2}{2\sigma^2}\Big)
$$

and matching exponents gives the bandwidth $\sigma^2=T\sqrt{d}$. Normalizing across $j$ yields exactly

$$
w_j(q)=\frac{K_\sigma(q,k_j)}{\sum_{\ell=1}^n K_\sigma(q,k_\ell)},\qquad
\sigma^2=T\sqrt{d}.
$$

(This is the standard RBF form.) ([Wikipedia][2])

5. **Identify the estimator.**
   The output is a **kernel‑weighted average**

$$
o(q)=\sum_j w_j(q)\,v_j
=\frac{\sum_j K_\sigma(q,k_j)\,v_j}{\sum_\ell K_\sigma(q,k_\ell)},
$$

which is precisely the **Nadaraya–Watson (NW) kernel regression estimator** evaluated at $q$ with “design points” $k_j$ and “responses” $v_j$. ([Wikipedia][114])
∎

### remarks and refinements

- **Why the $1/\sqrt{d}$ in attention?** It couples with temperature $T$ to set the _effective_ bandwidth $\sigma^2=T\sqrt{d}$. Keeping logit variance $O(1)$ as $d$ grows stabilizes the softmax (and thus the kernel bandwidth), which is the original scaling motivation. ([NeurIPS Papers][2])

- **Without exact length normalization.**
  If keys have small norm variation, say $|\|k_j\|^2-\beta^2|\le \varepsilon$ for all $j$, then the attention weights differ from the RBF‑normalized weights by at most a multiplicative factor $\exp(\varepsilon/(2T\sqrt{d}))$ _before_ renormalization; after renormalization, the total deviation is correspondingly controlled. (This follows directly from Step 3 by factoring the extra $\exp(\|k_j\|^2/(2\sqrt{d}T))$ term.)

- **MHA and GQA interpretations.**
  Each head $h$ has its own feature map $(W_Q^h,W_K^h)$; applying the same derivation in that head’s space shows head‑wise attention is an NW smoother with bandwidth $\sigma_h^2=T\sqrt{d_h}$. **MHA** linearly combines $H$ such smoothers via $W_O$ (an ensemble of kernels). **GQA/MQA** _share the dictionary_ $\{k_j,v_j\}$ across heads (or groups) but the optimization/softmax and thus the kernel view are unchanged. ([NeurIPS Papers][2])

[114]: https://en.wikipedia.org/wiki/Kernel_regression "Kernel regression"

> **Takeaway.** With LN and dot‑product softmax, attention approximates RBF‑kernel regression on learned features, explaining why it smoothly interpolates relevant tokens and scales with width.

## Multi‑Head Attention (MHA) as multi‑kernel learning

With heads $i=1,\dots,h$,

$$
\text{MHA}(X)=W_O [O_1;\ldots;O_h],\quad
O_i=\sigma\!\left(\frac{Q_i K_i^\top}{\sqrt{d_h}}\right)V_i.
$$

**Interpretation 1 — Multi‑kernel ensemble.** Each head supplies a distinct feature map (different $W_Q^i,W_K^i,W_V^i$), hence a distinct kernel smoother; concatenation + $W_O$ learns how to combine them (akin to multiple kernel learning over token neighborhoods).

**Interpretation 2 — Block‑diagonal linearization.** Stack per‑head spaces as a direct sum; MHA equals single‑head attention on a block‑diagonal parameterization followed by $W_O$. This clarifies which capacity increases are due to _diverse subspaces_ vs depth.

(The base formulation and scaling are from Vaswani et al., 2017.) ([NeurIPS Papers][2])

## Grouped‑Query & Multi‑Query Attention (GQA/MQA)

During **decode**, KV cache bandwidth dominates. **MQA** shares **one** $K,V$ across all heads; **GQA** shares KV across **$G$** groups ($1<G<h$): a speed/quality compromise. Formal definition from Ainslie et al. (2023): queries have $h$ heads, but keys/values have $G$ heads; each group of $\tfrac{h}{G}$ query heads attends to the same $K_g,V_g$. ([arXiv][10])

**Memory per token per layer (FP16 elements counted, constants omitted):**

- **MHA:** $2h d_h$ (all $K,V$ cached)
- **GQA:** $2G d_h$
- **MQA:** $2 d_h$ (the $G\!=\!1$ special case)
  So GQA reduces KV cache by factor $\approx \frac{h}{G}$ at similar compute for QKᵀ.

**A small robustness bound.**
Let a query row $q$ produce logits $z= \frac{1}{\sqrt{d_h}} q K_i^\top$ under per‑head keys $K_i$. If we replace $K_i$ by group‑shared $K_g$, logits change by $\delta z=\frac{1}{\sqrt{d_h}}q(K_g-K_i)^\top$. Since softmax is the gradient of LSE with Jacobian $J=\lambda(\operatorname{Diag}(p)-pp^\top)$ whose spectral norm is bounded (scales with $\lambda$, i.e., inverse temperature), we get

$$
\|\operatorname{softmax}(z+\delta z)-\operatorname{softmax}(z)\|_2
\;\le\; L \,\|\delta z\|_2
$$

for $L=\lambda$ (non‑expansiveness at $\lambda\!=\!1$). This shows head‑sharing error is controlled by how different per‑head $K_i$ are in the subspace seen by $q$. (Rigorous Lipschitz facts from “On the Properties of Softmax”.)&#x20;

## DeepSeek’s **Multi‑Head Latent Attention (MLA)**

**Problem.** Even GQA leaves a non‑trivial KV cache. **MLA** aggressively compresses KV while _improving_ quality vs MHA in DeepSeek‑V2. Key idea: **learn a low‑rank joint latent for $K$ and $V$** that is cached; reconstruct full $K,V$ via cheap up‑projections when needed.

From DeepSeek‑V2 (§2.1): define a **compressed latent** $c^{KV}_t\in\mathbb{R}^{d_c}$ for token $t$ via a down‑projection,

$$
c^{KV}_t = W^{KV}_D h_t,\qquad d_c \ll h\cdot d_h.
$$

Then **up‑project** to per‑head keys and values:

$$
k^{C}_t = W^{K}_U\, c^{KV}_t,\quad
v^{C}_t = W^{V}_U\, c^{KV}_t.
$$

During **inference**, only $c^{KV}_t$ is cached, so KV cache per token is $d_c$ (vs $2hd_h$ for MHA). Moreover, $W^{K}_U$ can be **absorbed into** $W_Q$ and $W^{V}_U$ into $W_O$, avoiding materializing $K,V$ explicitly in the hot path. (Equations (9)–(13) and discussion.)&#x20;

> **Effect:** KV cache reduces by up to **$O(hd_h/d_c)$** while preserving head‑wise expressivity through the learned up‑projections. DeepSeek‑V2 reports 93.3% KV‑cache reduction and substantial throughput gains at equal/better quality.&#x20;

**RoPE gotcha & fix.** RoPE is position‑sensitive; naïvely applying it to $k^C_t$ would entangle RoPE with $W_U^K$. DeepSeek introduces **decoupled RoPE** to keep low‑rank compression compatible with rotary embeddings.&#x20;

**Context with GQA/MQA.** MQA/GQA share $K,V$; MLA _compresses_ them into a learned latent cache with **joint low‑rank structure**, a different axis of efficiency. (Background: MQA by Shazeer; GQA by Ainslie et al.) ([arXiv][11])

## formal derivations

**D.1 Self‑attention as gradient of a potential (when $V=K$).**
Define $F(q)=\log\sum_j \exp(\langle q,k_j\rangle/T)$. Then $\nabla_q F(q)=\sum_j p_j\,k_j$ with $p=\operatorname{softmax}(\langle q,K\rangle/T)$. Thus, with $V=K$, the attention output is **∇ of a convex potential** (smooth max). Good slide for mathematical motivation. (Softmax = ∇LSE.)&#x20;

**D.2 Nadaraya–Watson equivalence (LN assumption).**
If $\|q\|=\|k_j\|=c$ (approx true after LN), then the attention weights equal an RBF kernel in $\|q-k_j\|$ up to a row‑constant, giving a formal link to kernel regression estimators. ([Wikipedia][7])

**D.3 Lipschitz bound for head/group replacement (GQA).**
Using Prop. 4 of “Softmax properties”, with inverse‑temperature $\lambda$ (often 1 in practice):
$\|\sigma(z+\delta z)-\sigma(z)\|_2 \le \lambda\,\|\delta z\|_2$. Insert $\delta z=\tfrac{1}{\sqrt{d_h}}q(K_g-K_i)^\top$ to state a clean robustness bound to KV sharing.&#x20;

## Complexity summary (decode‑phase, per token)

- **MHA:** read/write $K,V\in\mathbb{R}^{n\times(hd_h)}$ ⇒ **HBM‑limited** at long $n$.
- **GQA (G groups):** memory $\sim 2G d_h$ per token per layer; improves bandwidth, close to MQA when $G$ small. ([arXiv][14])
- **MLA (latent $d_c$):** memory $\sim d_c$ per token per layer; can be orders smaller; up‑proj absorbed into $W_Q,W_O$ at inference.&#x20;
- **IO‑aware optimization:** FlashAttention reduces **HBM traffic** by tiling Q/K/V and fusing softmax; exact attention, big wall‑clock gains. ([arXiv][15], [Tri Dao][16])

## CUDA: correctness‑first kernels (single head, single batch)

> **Design goal:** maximum clarity; three kernels (scores → softmax → output). You can later fuse and tile.

### compute scores $S=QK^\top/\sqrt{d}$

```cpp
// qk_scores.cu
#include <cuda_runtime.h>
#include <cmath>

__global__ void qk_scores_kernel(
    const float* __restrict__ Q,  // [n, d]
    const float* __restrict__ K,  // [n, d]
    float* __restrict__ S,        // [n, n]
    int n, int d, float inv_sqrt_d) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;  // query index i
  int col = blockIdx.y * blockDim.y + threadIdx.y;  // key   index j
  if (row >= n || col >= n) return;

  const float* q = Q + row * d;
  const float* k = K + col * d;

  float acc = 0.f;
  // naive dot-product; correctness first
  for (int t = 0; t < d; ++t) acc += q[t] * k[t];

  S[row * n + col] = acc * inv_sqrt_d;
}

extern "C" void qk_scores(const float* Q, const float* K, float* S, int n, int d) {
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
  float inv_sqrt_d = 1.f / std::sqrtf((float)d);
  qk_scores_kernel<<<grid, block>>>(Q, K, S, n, d, inv_sqrt_d);
}
```

### row‑wise softmax (numerically stable; in‑place on S)

```cpp
// row_softmax.cu
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void row_softmax_kernel(float* __restrict__ S, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // 1) max
  float m = -FLT_MAX;
  for (int j = 0; j < n; ++j) m = fmaxf(m, S[i*n + j]);

  // 2) exp & sum
  float sum = 0.f;
  for (int j = 0; j < n; ++j) {
    float e = expf(S[i*n + j] - m);
    S[i*n + j] = e;
    sum += e;
  }

  // 3) normalize
  float inv = 1.f / fmaxf(sum, 1e-12f);
  for (int j = 0; j < n; ++j) S[i*n + j] *= inv;
}

extern "C" void row_softmax(float* S, int n) {
  int block = 256;
  int grid  = (n + block - 1) / block;
  row_softmax_kernel<<<grid, block>>>(S, n);
}
```

### apply to values $O = S V$

```cpp
// apply_values.cu
#include <cuda_runtime.h>

__global__ void apply_values_kernel(
    const float* __restrict__ S,  // [n, n]
    const float* __restrict__ V,  // [n, d_v]
    float* __restrict__ O,        // [n, d_v]
    int n, int d_v) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;  // row in S / O
  int t = blockIdx.y * blockDim.y + threadIdx.y;  // value dim
  if (i >= n || t >= d_v) return;

  float acc = 0.f;
  for (int j = 0; j < n; ++j) acc += S[i*n + j] * V[j*d_v + t];
  O[i*d_v + t] = acc;
}

extern "C" void apply_values(const float* S, const float* V, float* O, int n, int d_v) {
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (d_v + block.y - 1) / block.y);
  apply_values_kernel<<<grid, block>>>(S, V, O, n, d_v);
}
```

> **Correctness notes:**
> – Handles any $n,d,d_v$, stable softmax.
> – Separate kernels make memory traffic explicit for teaching.
> – To extend to **masked causal** attention, set $S_{ij}=-\infty$ for $j>i$ before softmax.

**Where to speed up later (talking points):**

- Tile Q/K/V to **shared memory**, use warp‑level reductions for max/sum.
- Use **WMMA** (tensor cores) with FP16/FP8 accumulators; align $d$ to MMA tile sizes.
- **Kernel fusion**: compute partial QKᵀ → streaming softmax → accumulate $OV$ without storing $S$ (FlashAttention idea). ([arXiv][15])
- **Paged KV** & **GQA/MQA/MLA** reduce HBM pressure; quantify with a roofline plot.

## triton: a compact, fused attention (clarity‑first)

Below is a didactic **two‑pass** Triton kernel: pass 1 computes running row‑max/sum for softmax; pass 2 recomputes tiles to form the output $O$ (avoids storing $S$). Single head, no mask, one batch for clarity.

![[lectures/2/attention_triton.py]]

> **Why two passes?** It matches the pedagogy of stable softmax (compute row‑max & denom, then normalize) yet avoids materializing $S$. You can evolve it to a **single‑pass streaming** kernel (à la FlashAttention) once students grasp this version. (For background on IO‑aware attention, see FlashAttention.) ([arXiv][15])

## Where to push performance (after correctness)

- **IO‑aware tiling (FlashAttention):** stream K/V tiles; maintain running $(m_i,\ell_i)$; write O blocks; no $n\times n$ intermediates. Latest FA‑3 saturates H100 tensor cores, includes FP8 paths. ([arXiv][17], [Tri Dao][16])
- **KV layout & paging:** lay out KV in **paged** blocks (PagedAttention) + **GQA/MQA** or **MLA** to tame cache. ([arXiv][14])
- **RoPE in fused kernels:** apply rotations to $Q,K$ **inside** tiles to avoid extra global reads; with MLA use decoupled scheme.&#x20;
- **Mixed precision:** FP16/FP8 accumulations, per‑row max/sub; ensure numerics via LSE trick. (See stable LSE references.) ([Oxford Academic][12])

### Appendix: one more formal nugget you might like on a blackboard

**E.1 MHA $\approx$ mixture of kernel smoothers.** For head $i$, define kernel
$\kappa_i(q,k)=\exp(\langle W_Q^i q, W_K^i k\rangle/\sqrt{d_h})$. Then

$$
\mathrm{MHA}(x_t)=W_O\,[\underbrace{\textstyle\sum_{j\le t}\frac{\kappa_i(h_t,h_j)}{\sum_{\ell\le t}\kappa_i(h_t,h_\ell)}\, (W_V^i h_j)}_{\text{kernel smoother of }V_i}]_{i=1}^h.
$$

This makes clear that **diversity across $W_{\{\cdot\}}^i$** is exactly diversity of kernels/contexts.

**E.2 MLA cache accounting.**
Per‑token, per‑layer KV cache:

- MHA: $2hd_h$
- MLA: $d_c$
  With $h{=}64,d_h{=}128,d_c{=}1024$, reduction $\approx \frac{2\cdot 64\cdot 128}{1024}=16\times$ (and higher in practice because MLA further absorbs up‑projections at inference). (Architecture details & inference trick in paper.)&#x20;

[1]: https://arxiv.org/pdf/2104.09864 "Enhanced Transformer with Rotary Position Embedding"
[2]: https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf "Attention is All you Need"
[3]: https://home.ttic.edu/~madhurt/courses/infotheory2021/l5.pdf "Lecture 5: January 26, 2021 1 Convexity of KL-divergence"
[4]: https://people.lids.mit.edu/yp/homepage/data/LN_fdiv.pdf "7.1 Definition and basic properties of f-divergences - People"
[5]: https://stats.stackexchange.com/questions/527080/what-is-the-role-of-temperature-in-softmax "What is the role of temperature in Softmax? - Cross Validated"
[6]: https://arxiv.org/html/2402.05806v2 "On Temperature Scaling and Conformal Prediction of Deep Classifiers"
[7]: https://en.wikipedia.org/wiki/Kernel_regression "Kernel regression"
[8]: https://arxiv.org/abs/2009.14794 "[2009.14794] Rethinking Attention with Performers"
[9]: https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf "Fast Autoregressive Transformers with Linear Attention"
[10]: https://arxiv.org/abs/2305.13245 "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
[11]: https://arxiv.org/pdf/1911.02150 "Fast Transformer Decoding: One Write-Head is All You Need"
[12]: https://academic.oup.com/imajna/article/41/4/2311/5893596 "Accurately computing the log-sum-exp and softmax functions"
[13]: https://transformer-circuits.pub/2021/framework/index.html "A Mathematical Framework for Transformer Circuits"
[14]: https://arxiv.org/pdf/2305.13245 "arXiv:2305.13245v3 [cs.CL] 23 Dec 2023"
[15]: https://arxiv.org/abs/2205.14135 "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[16]: https://tridao.me/publications/flash3/flash3.pdf "FlashAttention-3: Fast and Accurate Attention with ..."
[17]: https://arxiv.org/abs/2407.08608 "[2407.08608] FlashAttention-3: Fast and Accurate Attention ..."
[18]: https://arxiv.org/abs/1911.02150 "Fast Transformer Decoding: One Write-Head is All You Need"
