---
date: '2026-05-27'
description: shared-prefix KV split into a batched multi-query pass over the common prefix and per-request suffix passes, merged by the online-softmax attention state operator.
id: attention-cascade
modified: 2026-06-17 12:55:19 GMT-04:00
seealso:
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/flash attention|FlashAttention]]'
  - '[[thoughts/radix attention|radix attention]]'
  - '[[thoughts/tree attention|tree attention]]'
socials:
  blog: https://flashinfer.ai/2024/02/02/cascade-inference.html
tags:
  - ml
  - llm
  - technical
title: Cascade Attention
---

idea: when a whole batch shares a prefix (a system prompt, a few-shot preamble, a retrieved document), the KV cache splits into one common prefix $P$ and per-request suffixes $S_r$. Cascade inference attends each query against $P$ and against its own $S_r$ separately, then merges the two partial results with the online-softmax state operator. Exact, because $P$ and $S_r$ partition the key set and the merge is associative.

The merge is the same $(m,\ell)$ summary [[thoughts/flash attention|FlashAttention]] uses to tile softmax, read here as an attention state $(\mathbf{v}, s)$: the normalised partial output $\mathbf{v}(I)=\sum_{j\in I}\operatorname{softmax}(s_j)\,V_j$ over key set $I$, and its log-sum-exp $s(I)=\log\sum_{j\in I}e^{s_j}$. Two disjoint states combine as

```jsx imports={Zoomable,CascadeFilter}
<Zoomable label="cascade filter pipeline">
  <CascadeFilter
    caption="drag the threshold $\tau$: only blocks scoring $s_j \ge \tau$ join the survivor set $\mathcal{S}$ and reach exact softmax, so a spiky score distribution buys a large $n/k$ speedup at near-full recall."
    tiles={16}
  />
</Zoomable>
```

$$
\begin{aligned}
s(I\cup J) &= \log\big(e^{s(I)} + e^{s(J)}\big),\\
\mathbf{v}(I\cup J) &= \sigma\big(s(I)-s(J)\big)\,\mathbf{v}(I) + \sigma\big(s(J)-s(I)\big)\,\mathbf{v}(J),
\end{aligned}
$$

with $\sigma$ the logistic sigmoid, so each branch is weighted by its share of the total mass. The operator is commutative and associative with identity $(\mathbf{0}, -\infty)$, which is why the prefix and suffix passes can run independently and recombine in any order.

For a query $q_i$ in request $r$,

$$
O_i = \underbrace{(\mathbf{v}(P_i), s(P_i))}_{\text{shared, batched once}} \;\oplus\; \underbrace{(\mathbf{v}(S_{r,i}), s(S_{r,i}))}_{\text{per request}}.
$$

The win is in where the prefix KV lives. The prefix pass is a multi-query attention: every query in the batch reads the _same_ $K_P, V_P$, so the kernel stages that block once into SMEM and serves the whole thread block from there, at roughly $10\times$ the bandwidth of streaming it from global memory per request. The suffix pass stays an ordinary batched decode against each request's own slice. FlashInfer reports up to $31\times$ over a uniform decode kernel when the shared prompt is long and the batch is wide.

> [!math] when cascade pays
> Let the prefix be a fraction $\rho = |P| / (|P| + |S_r|)$ of each request's context, batch size $b$. A monolithic decode re-reads the prefix once per request: $b\,|P|$ KV loads from global memory. The cascade reads it once into SMEM and amortises across the batch, so prefix IO drops by $b{:}1$ while the suffix IO is unchanged. The speedup tracks $\rho$ and $b$ together: shared system prompts ($\rho \to 1$) on large batches are where the $31\times$ shows up, and a batch of distinct prompts ($\rho \to 0$) collapses cascade back to plain decode with no penalty.

The prefix need not be a single level. Shared structure is usually a tree: a system prompt common to everyone, a few-shot block shared by a subgroup, a per-conversation history below that. Each shared node is one multi-query pass; the cascade merges states up the tree from the deepest suffix to the root, which is the same prefix reuse [[thoughts/radix attention|radix attention]] schedules dynamically, here resolved into a static reduction over known cache levels.

> [!note] not block pruning
> Cascade inference is exact, where it reorders where KV is read. It is unrelated to sparse or top-$k$ schemes (Quest, SnapKV, [[thoughts/razor attention|razor attention]]) that score key blocks and drop the cold ones to approximate the softmax.
