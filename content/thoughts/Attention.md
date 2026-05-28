---
abstract: The reason for Attention comparing to LSTM is that its ability to encode additional positional data into the inputs, in which it helps with longer context length and better memory retrieval. Note that most LLMs are decoder-only, given its superior benchmark in zero-shot tasks.
date: '2024-02-07'
description: and posteriori information retrieval.
id: Attention
modified: 2026-05-27 22:56:33 GMT-04:00
seealso:
  - '[[lectures/2/convexity|emperical finding]]'
socials:
  efficient: https://www.youtube.com/watch?v=Y-o545eYjXM
tags:
  - technical
  - llm
  - ml
title: Attention
transclude:
  title: false
---

Attention operates on a sequence of query $Q$, key $K$ and value $V$ vector. Attention matrix of a sequence then computed as [@vaswani2023attentionneed]:

$$
A(Q, K, V) = \operatorname{softmax}(\frac{Q \cdot K^{T}}{\sqrt{d}})V \space \space \text{ for } Q_{L \times d}, K_{L \times d}, V_{L \times d}
$$

First introduced in @vaswani2023attentionneed. One can think of attention for QKV as:

- Q: what I'm looking for
- K: what information do I have
- V: what information do I need to share to each other.

> [!note]- equivalent
>
> We can probably arrange the attention function (composed of multiple [[thoughts/induction heads|attention-heads]]) according to @elhage2021mathematical:
>
> $$
> \text{Attn}^{\vec{l,h}}(X_{\leq i}^{l-1}) = \sum_{j \leq i}a^{l,h}_{i,j} x^{l-1}_j W^{l,h}_{V} W_{O}^{l,h}
> $$
>
> where the ==learnable== weight matrices $W_{V}^{l,h} \in \mathbb{R}^{d \times d_h}$ and $W_{O}^{l,h} \in \mathbb{R}^{d_h \times d}$, $d_h$ is the dimension per head, are combined OV matrix

```jsx imports={Zoomable,AttentionCircuits}
<Zoomable label="QK/OV circuit decomposition">
  <AttentionCircuits caption="The same attention layer two ways: textbook softmax on the left, the Anthropic circuit decomposition on the right. Hover a circuit name to see which weights it covers: $QK$ decides where to look, $OV$ decides what to move." />
</Zoomable>
```

## Multi-head Attention

Allows the model to jointly attend to information from different representation subspaces at different positions:

$$
\begin{aligned}
\text{MHA}(Q,K,V) &= \operatorname{concat}(\text{head}_1, \ldots, \text{head}_h)\, W_O,\\
\text{head}_i &= \operatorname{softmax}\!\left(\frac{Q W_{Q,i}\,(K W_{K,i})^{\top}}{\sqrt{d_h}}\right)\, V W_{V,i},\\
& W_O \in \mathbb{R}^{(h d_h) \times d_{\text{model}}},\; W_{Q,i},W_{K,i} \in \mathbb{R}^{d_{\text{model}} \times d_h},\; W_{V,i} \in \mathbb{R}^{d_{\text{model}} \times d_h}.
\end{aligned}
$$

Each head can specialise on a distinct relational pattern in the same context window.

```jsx imports={Zoomable,MultiHeadAttention}
<Zoomable label="multi-head attention diagram">
  <MultiHeadAttention
    caption="Slide $h$ to rebalance the $d_m$ budget across heads. Each head's softmax is an independent normaliser; $h$ heads $\neq$ one wider head."
    heads={4}
  />
</Zoomable>
```

One may focus on positional offsets (e.g., "next token" dependencies) while another emphasises semantic alignment (e.g., subject $\leftrightarrow$ predicate links). The concatenation and final projection $W^O$ then recombine the perspectives into the [[thoughts/Transformers|transformer]] [[thoughts/mechanistic interpretability#residual stream|residual stream]].

> [!motivation]- why split the model into heads?
>
> Each head learns a slightly different relational probe over the same sequence. One head might focus on syntactic structure,
> another on long-distance coreference.
>
> By projecting $Q$, $K$, and $V$ into lower dimensional spaces, we allow those probes to specialise without paying
> the quadratic cost of a single massive head. Empirically this improves data efficiency because the model can
> reuse a single context to answer multiple "questions" about it in parallel, rather than re-reading the sequence each time.

> [!question]- internalise multi-head behaviour
>
> - [ ] Visualise attention heatmaps from several heads on the same prompt; annotate the linguistic or algorithmic pattern each head locks onto [@voita2019analyzingmha].
> - [ ] Compare perplexity of a single-head transformer to a multi-head variant while keeping parameter count fixed; relate to redundancy/prunability of heads [@michel2019sixteenheads].
> - [ ] Derive how residual mixing $W_O$ recombines the per-head outputs and why separate softmax normalisers per head change expressivity [@elhage2021mathematical].

> [!example]- Worked toy: two offset heads vs one big head
> Consider a length-$L$ sequence with two heads: head 1 attends to the next token $(+1)$ and head 2 attends to the previous token $(-1)$. Let $\beta \gg 0$ so each head's softmax is nearly an argmax on its offset.
>
> $$
> S^{(+1)}_{ij} = \beta\,[j=i+1], \quad S^{(-1)}_{ij} = \beta\,[j=i-1], \quad P^{(\pm1)} = \operatorname{softmax}_j S^{(\pm1)}.
> $$
>
> With values $V \in \mathbb{R}^{L\times d_h}$ and output projection blocks $W_O^{(1)}, W_O^{(2)}$, the MHA output is
>
> $$
> Y_{\text{MHA}} = \big(P^{(+1)} V\big) W_O^{(1)} + \big(P^{(-1)} V\big) W_O^{(2)}.
> $$
>
> A single head with scores $S = S^{(+1)} + S^{(-1)}$ yields $P=\operatorname{softmax}(S)$ and output $Y_{\text{SH}} = (PV)\,\tilde W_O$. Because $\operatorname{softmax}(A+B) \neq \operatorname{softmax}(A)+\operatorname{softmax}(B)$ in general, $Y_{\text{SH}}$ cannot match $Y_{\text{MHA}}$ for all inputs even if $\tilde W_O$ is chosen adversarially; one normaliser is coupled where two normalisers are separable. This structural independence of normalisers increases expressivity [@cordonnier2019relationshipselfattentionconvolution; @yun2019universaltransformers].
>
> ```python shell
> import torch, math
> torch.set_printoptions(precision=3, sci_mode=False)
> L, d_h = 6, 4
> beta = 10.0  # high temperature -> near-argmax
>
> # Build (+1) and (-1) score matrices
> S_p1 = torch.full((L,L), -float('inf'))
> S_m1 = torch.full((L,L), -float('inf'))
> for i in range(L-1): S_p1[i, i+1] = beta
> for i in range(1, L):  S_m1[i, i-1] = beta
>
> softmax = lambda S: (S - S.max(dim=-1, keepdim=True).values).softmax(dim=-1)
> P1, P2 = softmax(S_p1), softmax(S_m1)
>
> V = torch.randn(L, d_h)
> WO1, WO2 = torch.randn(d_h, d_h), torch.randn(d_h, d_h)
>
> Y_mha = P1@V@WO1 + P2@V@WO2
>
> # Single-head surrogate: add scores and use one normaliser
> P = softmax(S_p1 + S_m1)
> WOt = torch.randn(d_h, d_h)
> Y_sh = P@V@WOt
>
> print('||Y_mha - Y_sh||_F =', torch.linalg.norm(Y_mha - Y_sh).item())
> ```
>
> On random seeds this norm is typically O(1). No choice of a single post-projection can remove the coupling induced by the single softmax normaliser; two heads give two independent distributions you can recombine downstream. Empirically, heads do specialise and can be pruned selectively [@voita2019analyzingmha; @michel2019sixteenheads].

```jsx imports={Zoomable,OffsetHeadsToy}
<Zoomable label="two-head softmax demo">
  <OffsetHeadsToy
    caption="The same demo, live: two heads ($P^{(+1)}$, $P^{(-1)}$) and the single-head surrogate ($P = \operatorname{softmax}(S^{(+1)} + S^{(-1)})$). Reseed $V$ and the projection matrices to watch $\|Y_{\text{MHA}} - Y_{\text{SH}}\|_F$ stay stubbornly $O(1)$; no choice of single-head projection closes the gap."
    length={6}
  />
</Zoomable>
```

> [!math]- softmax factorisation barrier
>
> Let $S_i = QW_{Q,i}(KW_{K,i})^\top/\sqrt{d_h}$ and $P_i = \operatorname{softmax}(S_i)$.
>
> If a single-head self-attention with some score matrix $S$ and post-projection $\tilde W_O$ reproduced an $h$-head layer for all $Q,K,V$, then we would need $\operatorname{softmax}(S) V\tilde W_O = \sum_{i=1}^h P_i V W_{V,i} W_O^{(i)}$ for all $V$.
>
> This forces $\operatorname{softmax}(S) = \sum_i P_i M_i$ for some fixed matrices $M_i$ independent of inputs.
>
> But since softmax is not additive and $P_i$ depend on disjoint parameter sets, there exist inputs making $\sum_i P_i M_i$ violate row-stochasticity or attention symmetry constraints unless $h=1$ or all $S_i$ are affinely dependent.
>
> Hence in one layer, multi-head is strictly more expressive due to independent normalisers. See also [@cordonnier2019relationshipselfattentionconvolution; @yun2019universaltransformers].

> [!note]- Parameter and compute at fixed $d_{\text{model}}$
>
> - Params (packed projections): $W_Q,W_K,W_V,W_O \in \mathbb{R}^{d_m\times d_m}$ $\Rightarrow$ about $4d_m^2$ weights, essentially independent of head count $h$ (implementation splits the columns into $h$ groups).
> - FLOPs (naive): $\Theta(L^2 d_m)$ per layer per sequence; choosing $h$ changes per-head tile sizes and kernel efficiency, not asymptotics.
> - KV cache: per token per layer stores $K,V$ of size $2d_m$ (in bytes: $2d_m$ times dtype size). Changing $h$ does not change the sum dimension, but affects the layout and can impact IO-bound kernels in practice.

```jsx imports={Zoomable,AttentionCostCalculator}
<Zoomable label="attention cost calculator">
  <AttentionCostCalculator caption="Dial $d_{\text{model}}$, layers, $h$, sequence length, batch and dtype to see params, FLOPs, and KV cache update live. The $h$ knob is free in params; the cache is $L \times B \times N \times 2d_{\text{model}} \times \operatorname{bytes}$." />
</Zoomable>
```

> [!todo]- tasks to deepen multi-head understanding
>
> - Work through a two-token toy example where one head tracks positional offsets while another tracks part-of-speech, then visualise the resulting attention heatmaps.
> - Summarise how head specialisation emerges in practice (e.g., induction heads, name mover heads) by referencing case studies in [[thoughts/mathematical framework transformers circuits|Transformer Circuits]].
> - Compare the compute/memory footprint of doubling the number of heads versus increasing the hidden dimension, highlighting when each trade-off is preferable.

## variants

organised by their main concern:

```jsx imports={Zoomable,AttentionFamilyMap}
<Zoomable label="attention family map">
  <AttentionFamilyMap caption="four bottlenecks, four families. hover a chip for the math signature; click to follow." />
</Zoomable>
```

## optimization

> [!note]
> Exact kernels cut memory traffic; sparse/local reduce edges; linear/approximate trade exactness for O(L) time.

### Exact, IO-aware kernels

- FlashAttention v1/v2/v3: tile/fuse softmax, read each tile once from HBM; FA-3 exploits FP8 and hardware copy engines for higher throughput [@dao2022flashattentionfastmemoryefficientexact; @dao2023flashattention2fasterattentionbetter; @shah2024flashattention3fastaccurateattention].
- Flash-Decoding (+ +): specialised kernels for decode that parallelise across KV blocks and fuse reductions.

### Sparse/local attention (exact on a pattern)

- Longformer: sliding window with optional global tokens for linear scaling [@beltagy2020longformerlongdocumenttransformer].
- BigBird: block-sparse (window + random + global) with theoretical guarantees [@zaheer2021bigbirdtransformerslonger].

### Linear/approximate attention

- Reformer: LSH buckets for sub-quadratic attention + reversible layers [@kitaev2020reformerefficienttransformer].
- Linformer: low-rank projection along sequence dimension [@wang2020linformerselfattentionlinearcomplexity].
- Performer: FAVOR+ kernel features approximate softmax for linear time [@choromanski2022rethinkingattentionperformers].
- Nystromformer: landmark-based Nystrom approximation [@xiong2021nystromformernystrombasedalgorithmapproximating].

### Multi-device and prefix-aware inference

- Ring/Striped Attention: partition long sequences across devices, overlapping compute and communication [@liu2023ringattentionblockwisetransformers; @brandon2023stripedattentionfasterring].
- Cascade/Tree-aware kernels: exploit shared prefixes and tree layouts to reuse KV IO [@zheng2024sglangefficientexecutionstructured; @shyam2025treeattentiontopologyawaredecoding].

## cheatsheet

| Method              | Type         |      Complexity (seq) | Key idea                                    | Typical win                    |
| ------------------- | ------------ | --------------------: | ------------------------------------------- | ------------------------------ |
| FlashAttention-3    | exact kernel |              $O(L^2)$ | tiled IO-minimal attention; FP8/TMA overlap | large train speedups on Hopper |
| Flash-Decoding / ++ | exact decode |      $O(L)$ per token | block-parallel KV, fused reductions         | multi-x decode on long ctx     |
| Longformer          | sparse       | $\approx O(L\cdot w)$ | local window + global tokens                | linear scaling for long docs   |
| BigBird             | sparse       | $\approx O(L\cdot w)$ | window + random + global blocks             | theory + strong practice       |
| Reformer            | approx       |        $O(L \log{L})$ | LSH attention; reversible layers            | memory/time reductions         |
| Linformer           | approx       |         $O(L\cdot k)$ | low-rank K/V along L                        | linear time/space              |
| Performer           | approx       |         $O(L\cdot d)$ | FAVOR+ random features                      | linear attention               |
| Nystromformer       | approx       |         $O(L\cdot m)$ | landmark Nystrom approximation              | fewer tokens, good quality     |
| MQA/GQA             | arch/infer   |     $O(H_k\; d_h)$ KV | share K/V across heads/groups               | KV/bandwidth savings           |
| Ring/Striped        | parallel     |              $O(L^2)$ | pipeline across devices                     | million-token context          |
| Cascade/Tree-aware  | kernel       |              $O(L^2)$ | KV reuse on shared prefixes                 | big wins on shared prompts     |
