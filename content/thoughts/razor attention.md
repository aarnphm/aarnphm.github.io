---
date: '2026-05-27'
description: training-free, head-wise KV cache compression — keep the full cache on retrieval heads, drop remote tokens on the rest and fold them into a single compensation token.
id: attention-razor
modified: 2026-06-06 01:39:16 GMT-04:00
seealso:
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/RoPE|RoPE]]'
  - '[[thoughts/KV compression|KV compression]]'
  - '[[thoughts/radix attention|radix attention]]'
tags:
  - ml
  - llm
  - technical
title: Razor Attention
---

RazorAttention compresses the KV cache by sorting attention heads into two castes and caching each caste differently [@tang2024razorattentionefficientkvcache]. most heads only ever read a short local window plus the attention sink; a thin minority, the _retrieval heads_, reach across the entire context to fetch whatever the query points at. so keep the full cache where the reaching happens and shrink it everywhere else. nothing is decided per-token — the head is the unit of eviction, and the call is made once, before a single query arrives.

the organizing claim is a _retrieve-and-process_ split, where the model first uses retrieval heads to gather the span the query needs, then leans on the remaining heads to process that span into an answer. if that picture is right, then the non-retrieval heads were never using the far context anyway, so dropping it there costs almost nothing.

this is the opposite of importance-based token-dropping. H2O, [[thoughts/KV compression#Snap-KV|SnapKV]] and [[thoughts/KV compression#Streaming LLM|StreamingLLM]] keep a fixed token budget and evict whatever scores low on the attention map so far [@zhang2023h2oheavyhitteroracleefficient], which bakes in a guess about what future queries will want. ask for something off the main theme and the evidence is already gone. the paper's sharpest demonstration: bury "Bob's favorite number is 7690" in an 8K document about military appropriations, then ask for it. H2O answers `!!!!!!!`; RazorAttention answers `7690`, because it never threw Bob away — it only shrank the heads that weren't reading him.

## ALiBi

for [ALiBi](https://arxiv.org/abs/2108.12409) models the caste split falls out of the arithmetic. head $h$ scores a query at position $m$ against a key at position $n \le m$ with a linear distance penalty,

$$
S_{m \to n}(q; k) = q_m k_n^{\top} - l_h\,(m - n),
$$

where $l_h$ is the head's fixed slope. the content term $q_m k_n^{\top}$ is bounded (the inputs pass through LayerNorm, so they live on a sphere), while the penalty $l_h (m-n)$ grows without limit. past some distance the penalty swamps the content and the softmax weight collapses.

$$
\mathrm{Attn}_{m \to n}(q; k) = \frac{\exp\!\big(S_{m \to n}(q; k)\big)}{\sum_{j=0}^{m} \exp\!\big(S_{m \to j}(q; k)\big)} \le \epsilon \quad\text{whenever}\quad m - n \ge L_h,
$$

$$
L_h := \frac{2\,\lVert W_{Q_h} W_{K_h}\rVert_2\,\big(\lVert \gamma\rVert^2 + \lVert b\rVert^2\big) - \log \epsilon}{l_h}.
$$

here $W_{Q_h}, W_{K_h}$ are the head's query and key projections, $\gamma$ and $b$ are the LayerNorm weight and bias ($b = 0$ for RMSNorm), and $\lVert \cdot \rVert_2$ is the spectral norm.[^bound]

It reads $L_h$ as the head's _vision scope_, where every token older than $L_h$ contributes attention weight below $\epsilon$, so caching it is wasted bytes. because $L_h \propto 1/l_h$, a small slope buys a wide scope. the retrieval heads are exactly the shallow-slope heads; the local heads have steep slopes and tiny scopes. for ALiBi you read the caste straight off the slope and clip each head's cache to its own $L_h$.

## RoPE

[[thoughts/RoPE|RoPE]] carries no such guarantee. rotating $q$ and $k$ by their positions,

$$
S_{m \to n}(q; k) = q_m k_n^{\top}, \qquad q_m = R_m q, \quad k_n = R_n k,
$$

leaves the score bounded but not monotone in distance, so no theorem hands you a vision scope. yet the empirical split is stark anyway: only about 15% of heads make real use of long-range information; the rest stay local. the asymmetry is easy to measure. clip the cache on the long-range heads and accuracy drops 16%; clip the same amount on the other 85% and it drops 1.5% (their Table 1). the castes are real in RoPE models too — you just have to find them rather than derive them.

### finding the retrieval heads

the two head types that matter for recall are well-known interpretability objects.

```jsx imports={Zoomable,RazorHeadTaxonomy}
<Zoomable label="echo and induction heads">
  <RazorHeadTaxonomy caption="the probe repeats a random block, so each head's habit is legible. landing on the second $B$, an echo head attends back to the earlier $B$ (the duplicate); an induction head attends to the $C$ that followed that earlier $B$ and copies $C$ out as the next token. RazorAttention protects both kinds." />
</Zoomable>
```

an _echo head_ attends from the current token back to its previous identical occurrence. an _induction head_ attends to whatever token followed that previous occurrence and copies it forward—the standard induction move from in-context learning.[^induction]

To score every head for these habits, feed the model a clean stress test: $K \approx 2500$ random tokens, repeated four times, which strips away semantics so the copy behaviour stands out.

For each head measure its echo score (attention mass on the duplicate) and its induction score (mass on the token after it).

the retrieval set is the top 1% by echo score plus the top 14% by induction score—about 15% of heads kept {{sidenotes[in full]: induction heads seem to depend on echo heads upstream, and removing the 1% noticeably dents recall.}}.

## compression scheme

with the castes labelled, the algorithm is short. retrieval heads keep an untouched cache. every non-retrieval head keeps a sliding window of the most recent $L_h = \max(S_0, N/C)$ tokens plus the first few sink tokens, drops the remote span in between, and replaces that whole span with one _compensation token_.

```jsx imports={Zoomable,RazorCompression}
<Zoomable label="razor attention cache scheme">
  <RazorCompression caption="a non-retrieval head keeps its sink tokens and a recent window $L_h = N/C$, drops the remote middle, and folds it into a single compensation token $\{\hat{k}, \hat{v}\}$. retrieval heads (15% of heads) keep the full cache. slide $C$ to trade window length against compression; the paper's $C = 5$ keeps $0.15 + 0.85 \cdot \tfrac{1}{5} \approx 0.32$ of the cache, a $3.125\times$ shrink." />
</Zoomable>
```

the compensation token is the mean of the dropped keys and values,

$$
\hat{k} = \frac{1}{N_d} \sum_{m \in D} k_m, \qquad \hat{v} = \frac{1}{N_d} \sum_{m \in D} v_m,
$$

where $D$ is the set of dropped indices and $N_d = |D|$. it gets appended to the surviving cache $\{K, \hat{k}\}, \{V, \hat{v}\}$ and weighted in the softmax as though it were $N_d$ separate tokens:

$$
\mathrm{Attn}\big(q_m, \{K, \hat{k}\}, \{V, \hat{v}\}\big) = \frac{N_d\, e^{q_m \hat{k}^{\top}}\, \hat{v} + \sum_{n \notin D} e^{q_m k_n^{\top}}\, v_n}{N_d\, e^{q_m \hat{k}^{\top}} + \sum_{n \notin D} e^{q_m k_n^{\top}}}.
$$

the $N_d$ factor restores the dropped span's share of the normalizer, so the surviving tokens don't get artificially loud.[^jensen]

the headline ratio comes out of two numbers. keep 15% of heads whole and clip the other 85% to a fifth, and the retained fraction is $0.15 + 0.85 \cdot \tfrac{1}{5} \approx 0.32$, a $3.125\times$ compression — "70% off" with no retraining and roughly {{sidenotes[no accuracy cost]: for [[thoughts/GQA|GQA]] models a whole group is promoted to retrieval if any head in it qualifies}} from 8K to 100K tokens.

[^bound]: the proof is a one-liner once you bound the content term. since $q = W_{Q_h} x$ and $k = W_{K_h} x$ with $x$ post-LayerNorm, $q k^{\top} \le \lVert W_{Q_h} W_{K_h}\rVert_2 \lVert x\rVert^2 \le \lVert W_{Q_h} W_{K_h}\rVert_2 (2\lVert \gamma\rVert^2 + 2\lVert b\rVert^2)$, a constant. set the bounded content minus the growing penalty below $\log \epsilon$ and solve for $m - n$.

[^induction]: this is the induction circuit from [@olsson2022context]: a previous-token (or duplicate-token) head feeds an induction head that completes `[A][B] … [A] → [B]`. RazorAttention reuses a mechanistic-interpretability finding as a systems heuristic — the same retrieval heads were shown to causally drive long-context factuality in [@wu2024retrievalheadmechanisticallyexplains].

[^jensen]: it is a zeroth-order approximation. collapsing $\sum_{m \in D} e^{q k_m^{\top}} v_m$ into $N_d\, e^{q \hat{k}^{\top}} \hat{v}$ is exact only if every dropped key is identical; by convexity of $\exp$ it understates the true mass otherwise, with error scaling in the spread of the dropped keys. it works because the dropped span is the _far_ context of a _local_ head, where the keys are both low-weight and fairly uniform.

[^flash]: this is why the paper drops SnapKV from its main comparison — SnapKV further assumes the query is known before compression, which fails in multi-round chat where each turn queries a different slice of the context.
