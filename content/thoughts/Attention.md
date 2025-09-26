---
id: Attention
tags:
  - technical
  - llm
  - ml
description: and posteriori information retrieval.
date: "2024-02-07"
abstract: The reason for Attention comparing to LSTM is that its ability to encode additional positional data into the inputs, in which it helps with longer context length and better memory retrieval. Note that most LLMs are decoder-only, given its superior benchmark in zero-shot tasks.
modified: 2025-09-21 02:21:43 GMT-04:00
title: Attention
---

Attention operates on a sequence of query $Q$, key $K$ and value $V$ vector. Attention matrix of a sequence then computed as [@vaswani2023attentionneed]:

$$
A(Q, K, V) = \operatorname{softmax}(\frac{Q \cdot K^{T}}{\sqrt{d}})V \space \space \text{ for } Q_{L \times d}, K_{L \times d}, V_{L \times d}
$$

First introduced in [@vaswani2023attentionneed]. One can think of attention for QKV as:

- Q: what I'm looking for
- K: what information do I have
- V: what information do I need to share to each other.

> [!note]+ equivalent
>
> We can probably arrange the attention function (composed of multiple [[thoughts/induction heads|attention-heads]]) according to @elhage2021mathematical:
>
> $$
> \text{Attn}^{\vec{l,h}}(X_{\leq i}^{l-1}) = \sum_{j \leq i}a^{l,h}_{i,j} x^{l-1}_j W^{l,h}_{V} W_{O}^{l,h}
> $$
>
> where the ==learnable== weight matrices $W_{V}^{l,h} \in \mathbb{R}^{d \times d_h}$ and $W_{O}^{l,h} \in \mathbb{R}^{d_h \times d}$, $d_h$ is the dimension per head, are combined OV matrix

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

One may focus on positional offsets (e.g., "next token" dependencies) while another emphasises semantic alignment (e.g., subject ↔ predicate links).
This diversity reduces the risk that a single head saturates and misses important cues, analogous to how [[thoughts/ensemble learning|ensembles]] provide robustness by blending multiple predictors.

The concatenation and final projection $W^O$ then recombine the perspectives into the [[thoughts/Transformers|transformer]] residual stream.

> [!tip]+ building intuition
> Imagine each attention head as a separate spotlight scanning the same stage from a slightly different angle. None of the spotlights alone illuminates the full choreography, yet together they produce a richer, multi-view understanding of the performance.

> [!motivation]+ why split the model into heads?
> Each head learns a slightly different relational probe over the same sequence. One head might focus on syntactic structure, another on long-distance coreference. By projecting $Q$, $K$, and $V$ into lower dimensional spaces, we allow those probes to specialise without paying the quadratic cost of a single massive head. Empirically this improves data efficiency because the model can reuse a single context to answer multiple "questions" about it in parallel, rather than re-reading the sequence each time.

> [!question]+ tasks to internalise multi-head behaviour (updated)
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
> A single head with scores $S = S^{(+1)} + S^{(-1)}$ yields $P=\operatorname{softmax}(S)$ and output $Y_{\text{SH}} = (PV)\,\tilde W_O$. Because $\operatorname{softmax}(A+B) \neq \operatorname{softmax}(A)+\operatorname{softmax}(B)$ in general, $Y_{\text{SH}}$ cannot match $Y_{\text{MHA}}$ for all inputs even if $\tilde W_O$ is chosen adversarially—one normaliser vs two. This structural independence of normalisers increases expressivity [@cordonnier2019relationshipselfattentionconvolution; @yun2019universaltransformers].
>
> ```python
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

> [!math]+ Proof sketch: softmax factorisation barrier
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

> [!note]+ Parameter and compute at fixed $d_{\text{model}}$
>
> - Params (packed projections): $W_Q,W_K,W_V,W_O \in \mathbb{R}^{d_m\times d_m}$ $\Rightarrow$ about $4d_m^2$ weights, essentially independent of head count $h$ (implementation splits the columns into $h$ groups).
> - FLOPs (naive): $\Theta(L^2 d_m)$ per layer per sequence; choosing $h$ changes per-head tile sizes and kernel efficiency, not asymptotics.
> - KV cache: per token per layer stores $K,V$ of size $2d_m$ (in bytes: $2d_m$ times dtype size). Changing $h$ does not change the sum dimension, but affects the layout and can impact IO-bound kernels in practice.

> [!tip]- Minimal PyTorch MHA (for reproducing plots quickly)
>
> ```python
> import torch, torch.nn as nn
> L, d_model, nhead = 64, 256, 8
> mha = nn.MultiheadAttention(d_model, nhead, batch_first=True, bias=False)
> x = torch.randn(1, L, d_model)
> y, attn = mha(x, x, x, need_weights=True, average_attn_weights=False)
> # attn shape: [1, nhead, L, L]; visualise per-head heatmaps
> ```

> [!todo]+ tasks to deepen multi-head understanding
>
> - Work through a two-token toy example where one head tracks positional offsets while another tracks part-of-speech, then visualise the resulting attention heatmaps.
> - Summarise how head specialisation emerges in practice (e.g., induction heads, name mover heads) by referencing case studies in [[thoughts/mathematical framework transformers circuits|Transformer Circuits]].
> - Compare the compute/memory footprint of doubling the number of heads versus increasing the hidden dimension, highlighting when each trade-off is preferable.

## Group-Query Attention

Group-Query Attention [@ainslie2023gqatraininggeneralizedmultiquery]

idea: reduce number of KV heads $n_k$ to a fraction $n_k^{'} = \frac{n_q}{k}$ of number of query heads $n_q$ (evenly dividing the query heads into $n_k$ groups with $r$ heads)

This technique targets the decode-time bottleneck in autoregressive generation. During prefill we still benefit from many $Q$ heads to capture diverse patterns, but at decode time the key/value caches dominate GPU memory bandwidth. Sharing a smaller set of $K,V$ representations across multiple $Q$ heads keeps latency low and unlocks larger batch sizes without sacrificing the nuanced querying capacity. In other words, GQA trades a tiny amount of representational flexibility for a substantial improvement in cache locality.

GQA keeps a rich set of query projections so that each decoding token can still ask a nuanced question, but it shares keys and values among groups of queries. The insight is that auto-regressive decoding is bottlenecked by memory bandwidth rather than compute—duplicating $K$ and $V$ per head is expensive, yet their content is often redundant because neighbouring heads tend to look at similar context tokens. By amortising $K$ and $V$ across a group, we trade a small loss in expressivity for a large win in cache locality and throughput on modern GPUs.

> [!motivation] decoding-first design pressure
> During inference each new token only appends a single query but must read the entire cached key/value tensors. Grouping $Q$ heads while keeping fewer $K/V$ copies reduces the size of that cache and the amount of memory traffic per step, which is exactly the term that dominates latency for long-context models.

> [!question]- explore the design space
>
> - [ ] Starting from a vanilla transformer decoder, implement grouped keys/values and benchmark the decode tokens-per-second improvement as context length grows.
> - [ ] Analyse how grouping interacts with rotary or ALiBi positional encodings—does sharing $K/V$ across heads degrade positional resolution?
> - [ ] Reproduce the ablation table from @ainslie2023gqatraininggeneralizedmultiquery to see how aggressively $n_k^{'}$ can be reduced before accuracy drops on your domain.

> [!todo]+ follow-up questions
>
> - Derive how the attention matrix factorises when queries are grouped and quantify the approximation error introduced by shared $K,V$ pairs.
> - Collect empirical results comparing [[thoughts/KV compression|KV cache]] sizes for MHA, Multi-Query, and GQA across popular decoder-only models.
> - Investigate hardware implications: how does GQA interact with tensor parallelism or speculative decoding pipelines?

## Tree Attention

Tree Attention [@shyam2025treeattentiontopologyawaredecoding] derives an energy formulation of attention and evaluates the softmax reduction through a communication tree. Keys and values are sharded along the sequence dimension; each query reduces over shards in $\log p$ stages for $p$ devices, cutting communication steps relative to the linear pipeline used in RingAttention. The method stays exact and can reuse single-GPU kernels such as FlashAttention-2, yielding up to $4\times$ decoder speedups on Llama-scale models while lowering peak memory traffic.

> [!motivation] topology matches hardware
> NVLink and InfiniBand fabrics already provide efficient tree collectives, so aggregating per-shard softmax statistics along that topology overlaps communication with compute instead of circulating full KV blocks around the ring.

> [!question]- tasks
>
> - [ ] Implement a tree-reduction decode for a toy sharded KV setup and compare against RingAttention on 2–8 GPUs; measure communication steps and wall-clock latency.
> - [ ] Profile sensitivity to interconnect bandwidth and block sizes; verify the $N/p + \log p$ scaling predicted in [@shyam2025treeattentiontopologyawaredecoding].
> - [ ] Explore compatibility with prefix-reuse systems (Paged/RadixAttention) when K/V are paged or cached.

> [!note] related but different
> Hierarchical token routing or coarse-to-fine attention over document structure is orthogonal to this topology-aware multi-GPU scheduling trick.

## Sliding Window Attention

Sliding window (or local) attention constrains each token to attend only to neighbours within a fixed radius $w$. The computational cost drops from $\mathcal{O}(L^2)$ to $\mathcal{O}(L \cdot w)$, which is crucial for extremely long sequences where full-context attention is prohibitive. Intuitively, this mimics how we read a long novel: we usually only need to relate a sentence to nearby sentences, occasionally jumping back to earlier chapters.

- Motivation: maximise throughput on long-context tasks where relevant information is clustered locally (e.g., speech, DNA sequences).
- Intuition: local convolutions but with content-aware weighting rather than fixed kernels.
- Challenge: ensuring important long-range dependencies are not lost—often solved by adding a handful of global tokens or dilation patterns.

> [!todo]+ experiments to run
>
> - Implement a toy [[thoughts/Autoregressive models|autoregressive]] model with sliding window attention and track perplexity as $w$ varies.
> - Document hybrid strategies (e.g., dilated windows, stride patterns) and how they impact the receptive field.
> - Collect references on how models like Longformer or [[thoughts/Transformers|BigBird-style transformers]] mix local and global tokens.

See also Longformer [@beltagy2020longformerlongdocumenttransformer] and BigBird [@zaheer2021bigbirdtransformerslonger].

## FlashAttention & IO-Aware Kernels

FlashAttention [@dao2022flashattentionfastmemoryefficientexact] reframes attention as a tiled matrix multiplication that keeps intermediate results in high-speed SRAM rather than slower GPU DRAM. The key insight is that recomputing softmax denominators on-the-fly avoids materialising the full attention matrix, drastically reducing memory traffic. As sequence lengths grow, attention becomes more IO-bound than FLOP-bound, so this optimisation yields both speedups and numerical stability (via online normalisation). See also FlashAttention‑2 [@dao2023flashattention2fasterattentionbetter] and FlashAttention‑3 [@shah2024flashattention3fastaccurateattention].

- Motivation: eliminate memory bandwidth bottlenecks so that longer contexts fit on commodity GPUs.
- Intuition: compute attention in blocks, never storing more than necessary—like reading a massive spreadsheet through a moving window rather than printing the entire sheet.
- Extension: variants such as FlashAttention-2/3, xFormers, and Triton kernels specialise for [[thoughts/GPU programming|GPU]] architectures and sparse layouts.

> [!todo]+ future notes
>
> - Re-derive the online softmax algorithm that maintains running maxima and partition functions per tile.
> - Benchmark FlashAttention against naive attention under identical hardware to quantify the IO savings.
> - Explore how FlashAttention integrates with techniques above (e.g., can GQA heads share tiles efficiently?).

## RadixAttention

RadixAttention [@zheng2024sglangefficientexecutionstructured] maintains an LRU eviction policy to keep relevant [[thoughts/KV compression|KV cache]] entries for all requests within a [[thoughts/Radix tree|radix tree]], implemented in https://github.com/sgl-project/sglang and detailed in the SGLang paper and LMSYS blog (Jan 17, 2024).

radix tree setup:

- key: sequence of tokens
- value: KV cache tensor (stored in GPU in a paged layout)

![[thoughts/images/vllm/radix-attention.webp]]

_dynamic evolution of the radix tree in response to various requests._

> [!abstract]- explanation of RadixAttention with LRU eviction policy
>
> These requests include two chat ses,sions, a batch of few-shot learning inquiries, and a self-consistency sampling. Each tree edge carries a label denoting a substring or a sequence of tokens. The nodes are color-coded to reflect different states: green for newly added nodes, blue for cached nodes accessed during the time point, and red for nodes that have been evicted.
>
> [full explanation](https://lmsys.org/blog/2024-01-17-sglang/#backend-automatic-kv-cache-reuse-with-radixattention)

> [!motivation] amortising repeated prefixes
> Shared prefixes across requests mean the expensive prefill phase has already computed the relevant keys and values. A radix tree indexes those prefixes so the runtime can instantly reuse them, and the LRU policy discards only the paths that have fallen out of active use. This keeps latency predictable even when the workload mixes long chats, few-shot prompts, and batched sampling.

> [!question]- exercises for deployment engineers
>
> - [ ] Simulate a workload with replayed prefixes and measure cache-hit rate as you vary the tree eviction threshold; plot how it affects end-to-end throughput.
> - [ ] Implement instrumentation that surfaces when two requests could share a prefix but fail to because of tokenisation mismatch.
> - [ ] Extend the scheduling algorithm above with priority weights so latency-sensitive requests pre-empt background sampling without trashing the cache.

### cache-aware scheduling

We define the hit rate as

$$
\begin{aligned}
\text{hit rate} &= \frac{\sum_{r \in R} \text{number of cached prefill tokens in } r}{\sum_{r \in R} \text{number of prefill tokens in } r} \\[8pt]
&=1 - \frac{C}{\sum_{r \in R} \text{number of prefill tokens}}
\end{aligned}
$$

_in batch settings: sort requests by matching prefix length and prioritise one with longer matched prefixes instead of FIFO schedule._

```pseudo lineNumber=false
\begin{algorithm}
\caption{Cache-Aware Scheduling}
\begin{algorithmic}
\State \textbf{Input:} Radix tree $T$, Memory pool $P$.
\State \textbf{Input:} current running batch $B$, waiting queue $Q$.
\State \textbf{Output:} Finished requests and updated system state.
\State // Get all requests from the waiting queue
\State requests $\gets Q.\text{get\_all\_requests}()$
\State // Search for prefix matching for all waiting request
\For{req $\in$ requests}
    \State req.prefix\_node, req.prefix\_len $\gets$ T.match\_prefix(req.input\_tokens)
\EndFor
\State // Sort the request according to matched prefix lengths
\State requests.sort()
\State // Select requests for the next batch
\State available\_size $\gets$ T.evictable\_size() + P.available\_size()
\State current\_size $\gets$ 0
\State new\_batch $\gets$ []
\For{req $\in$ requests}
    \If{req.size() + current\_size $\le$ available\_size}
        \State new\_batch.append(req)
        \State $\delta \gets T.\text{increase\_ref\_counter}(req.\text{prefix\_node})$
        \State available\_size $\gets$ available\_size + $\delta$
    \EndIf
\EndFor
\State Q.remove\_requests(new\_batch)
\State // Insert requests into the current running batch
\State B.merge(new\_batch)
\State // Allocate new memory and do eviction if necessary
\State needed\_size $\gets$ B.needed\_size()
\State success, buffer $\gets$ P.alloc(needed\_size)
\If{$\neg \text{success}$}
    \State T.evict(needed\_size)
    \State success, buffer $\gets$ P.alloc(needed\_size)
\EndIf
\State B.run(buffer)
\State // Process finished requests
\State finished\_requests $\gets$ B.drop\_finished\_requests()
\For{req $\in$ finished\_requests}
    \State T.decrease\_ref\_counter(req.prefix\_node)
    \State T.insert(req)
\EndFor
\State \Return finished\_requests
\end{algorithmic}
\end{algorithm}
```

We got lower bound:

$$
C \ge \sum_{e \in \text{edges}(T)} \mid e \mid
$$

Consider we visit radix tree $T$ in DFS order. For each edge $e$ of $T$, the first time we compute KV cache associated with $e$, then we will compute the whole subtree of $e$.

During computation of $e$ subtree, then edge $e$ will be continuously hit, thus no additional computation will happen.

> [!important] cache hit
>
> with cache size $\ge$ maximum request length (which will equals to longest path in radix tree), edge $e$ **WILL NOT** be evicted during computation of its subtree
> since the common prefix including $e$ of the subtree will be continuously hit.

We can show that longest-shared-prefix-first order is equivalent to DFS order by induction [^proof]

[^proof]: _base_: a random request correspond to node $x \in T$ will be processed.

    - All requests correspond to nodes $\{v_{1}, \ldots, v_{n}\}$ on path $x \gets \text{root}$ doesn't need recomputation.
    - Thus, computation complexity for requests of nodes $\{v_{1}, \ldots, v_{n}, x\}$ is aligned with DFS

    _induction_: assume we visit node $y \in T$, and the visited node align with DFS order. Let $P$ denote _path of_ $y \gets \text{root}$.

    - Each node that has not been visited has the lowest common ancestor with visited nodes on $P$.
    - Since nodes on $P$ are cached, a node $z$ that has yet to be visited with lowest common accestor on $P$ will have the _longest shared prefix_
    - longest-shared-prefix-first order will select $z$, which is a valid DFS
      q.e.d

![[thoughts/structured outputs#compressed FSM for jump-ahead tokens.]]

## Multi-head Latent Attention (MLA)

![[thoughts/images/mla-comparison.webp]]

low-rank joint compression for attention ==keys and values== to reduce KV cache during inference [@deepseekai2025deepseekv3technicalreport, see Section 2.1.1; @deepseekai2024deepseekv2strongeconomicalefficient]

- $d$ denote the embedding dimension
- $n_h$ denotes number of attention heads
- $d_h$ denotes dimension per heads
- $h_t \in \mathbb{R}^d$ denotes the attention input for the $t$-th token at a given attention layer

$$
\begin{align}
    \boxed{\textcolor{blue}{\mathbf{c}_t^{KV}}} &= W^{DKV} \mathbf{h}_t, \tag{1} \\
    [\mathbf{k}_{t,1}^{C}; \mathbf{k}_{t,2}^{C}; \dots; \mathbf{k}_{t, n_h}^{C}] &= \mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}, \tag{2} \\
    \boxed{\textcolor{blue}{\mathbf{k}_t^{R}}} &= \mathrm{RoPE}(W^{KR} \mathbf{h}_t), \tag{3} \\
    \mathbf{k}_{i,t} &= [\mathbf{k}_{t,i}^{C}; \mathbf{k}_t^{R}], \tag{4} \\
    [\mathbf{v}_{t,1}^{C}; \mathbf{v}_{t,2}^{C}; \dots; \mathbf{v}_{t,n_h}^{C}] &= \mathbf{v}_t^{C} = W^{UV} \mathbf{c}_t^{KV}. \tag{5}
\end{align}
$$

- _where_ $c_{t}^{KV} \in \mathbb{R}^{d_{c}}$ is the compression latent for keys and values
- $d_c \ll d_h n_h$ indicates KV [[thoughts/Compression|compression]] dimension
- $W^{DKV} \in  \mathbb{R}^{d_c \times d}$ denotes down-projection matrix
- $W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$ are the up-projection matrices to keys and values, respectively
- $W^{KR} \in \mathbb{R}^{d^R_h \times d}$ is the matrix used to produced the duplicate key that carries [[thoughts/RoPE|RoPE]]
- $\mathrm{RoPE}(.)$ denotes operations for RoPE matrices, and $[;]$ denotes ==concatenation==
- Note that only $\boxed{\textcolor{blue}{\mathbf{c}_t^{KV}}}, \boxed{\textcolor{blue}{\mathbf{k}_t^{R}}}$ needs to be cached

> [!important] cached generations
>
> Both $\textcolor{blue}{\mathbf{c}_t^{KV}}$ and $\textcolor{blue}{\mathbf{k}_t^{R}}$ should be cached to reduce KV cache while maintaining performance with [[thoughts/Attention#Multi-head Attention|MHA]]

> [!motivation] why latent compression matters
> MLA recognises that most of the information inside $K$ and $V$ lies in a low-dimensional manifold. By learning shared latent codes ($\mathbf{c}_t^{KV}$ and $\mathbf{c}_t^{Q}$) and lightweight up-projections, the model keeps the expressivity of many heads without storing every head explicitly. This decouples compute (which stays similar) from memory footprint (which shrinks drastically), enabling deployment on GPUs with limited KV cache.

> [!question]- MLA study plan
>
> - [ ] Re-derive equations (1)–(9) starting from a standard attention layer and show how the compression matrices factorise the original weight tensors.
> - [ ] Measure perplexity and cache usage on a long-context benchmark when toggling MLA on/off for the same base model.
> - [ ] Investigate whether sharing $\mathbf{c}_t^{KV}$ across layers compounds errors or if independent latents per layer yield better stability.

For attention ==queries==, we can perform the same operation:

$$
\begin{align}
    \mathbf{c}_t^{Q} &= W^{DQ} \mathbf{h}_t, \tag{6} \\
    [\mathbf{q}_{t,1}^{C}; \mathbf{q}_{t,2}^{C}; \dots; \mathbf{q}_{t, n_h}^{C}] &= \mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^{Q}, \tag{7} \\
    [\mathbf{q}_{t,1}^{R}; \mathbf{q}_{t,2}^{R}; \dots; \mathbf{q}_{t, n_h}^{R}] &= \mathrm{RoPE}(W^{QR} \mathbf{c}_t^Q), \tag{8} \\
    \mathbf{q}_{i,t} &= [\mathbf{q}_{t,i}^{C}; \mathbf{q}_t^{R}], \tag{9}
\end{align}
$$

- $c_t^Q$ is the compressed latent of queries
- $d_c \ll d_h n_h$ indicates queries compression dimension
- $W^{DQ} \in \mathbb{R}^{d^{'}_c \times d}, W^{UQ} \in \mathbb{R}^{d_h n_h \times d^{'}_c}$ are the up and down [[thoughts/geometric projections|projections]] matrices
- $W^{QR} \in \mathbb{R}^{d_{h}^R n_{h} \times d_{c}^{'}}$ is the matrix that produce _decompiled queries that carry RoPE_

> [!abstract] Attention output
>
> The attention output $\mathbf{u}_{t}$ can be calculated with the following:
>
> $$
> \begin{align}
>     \mathbf{o}_{t,i} &= \sum_{j=1}^{t} \mathrm{Softmax}_j (\frac{q_{t,i}^T k_{j,i}}{\sqrt{d_h + d_h^R}}) v_{j_i}^C, \tag{10} \\
>     \mathbf{u}_t &= \mathbf{W}^O [o_{t,1}; o_{t,2}; \dots; o_{t, n_h}] \tag{11}
> \end{align}
> $$

![[lectures/3/quantisation basics#multi-latent attention]]

## CascadeAttention

https://flashinfer.ai/2024/02/02/cascade-inference.html

CascadeAttention builds a two-stage filter for attention scores. A cheap scorer (for example, a low-rank approximation or sparse lookup) first estimates which key blocks are likely to matter. Only those candidates are passed to the expensive exact attention, meaning most tokens never touch the quadratic computation. This mirrors cascade classifiers in computer vision: fast heuristics prune the search space so heavy models only run on promising regions.

> [!motivation]+ matching cost to usefulness
> Long contexts often contain filler tokens. Spending equal compute on every position wastes FLOPs, so a cascade keeps throughput high by adapting compute to token importance.

> [!question]- tasks
>
> - [ ] Implement a toy cascade with random feature hashing as the coarse scorer; report recall of true top-$k$ attention weights.
> - [ ] Explore how to set the threshold dynamically based on query entropy so we do not over-prune when the model is uncertain.
> - [ ] Compare latency of the cascade versus FlashAttention when processing prompts containing repetitive boilerplate plus a short critical instruction.

## RingAttention

RingAttention [@liu2023ringattentionblockwisetransformers] shards long contexts across devices in a ring pipeline. Striped Attention [@brandon2023stripedattentionfasterring] improves load balance with alternating shard ownership.

RingAttention shards a long sequence across multiple devices and circulates key/value blocks in a logical ring. Each GPU holds only a slice of the cache, streams neighbouring slices just in time, and discards them after use. The ring topology overlaps communication with computation, so no single device ever needs the full context resident in memory.

> [!motivation] hardware-aligned parallelism
> Transformer memory demands grow with sequence length, but GPU memory per device is fixed. RingAttention trades redundant KV storage for bandwidth-efficient peer-to-peer communication, letting us scale to million-token contexts without out-of-memory errors.

> [!question]- experiments
>
> - [ ] Simulate RingAttention with three devices and measure throughput as you vary block size; look for the sweet spot where communication overlaps compute.
> - [ ] Analyse failure cases when network latency spikes—how resilient is the ring schedule compared to fully sharded data parallelism?
> - [ ] Derive how gradient checkpointing interacts with the ring; can we reuse streamed activations during backprop?

## RazorAttention

RazorAttention [@tang2024razorattentionefficientkvcache] maintains a fixed-size KV cache by scoring tokens with a learned eviction policy. Instead of evicting whole prefixes (like radix trees) or oldest tokens (pure LRU), it "shaves" the least impactful tokens anywhere in the sequence. Importance scores come from lightweight predictors trained to approximate how much each token will contribute to future attention.

> [!motivation] selective forgetting
> Decoder-only models often attend mostly to recent or semantically salient tokens. By identifying and dropping low-utility KV entries, RazorAttention keeps cache usage bounded while preserving the signal that matters for prediction.

> [!question]- sharpening intuition
>
> - [ ] Replicate the token-importance predictor on a small dataset and visualise which positions it consistently removes.
> - [ ] Evaluate quality versus cache size trade-offs when combining RazorAttention with grouped-query attention—do their savings compound?
> - [ ] Formally compare RazorAttention's policy with standard LRU by computing expected retained attention mass under a synthetic long-conversation workload.

## Paged Attention

Paged Attention [@kwon2023efficient]

In conjunction with [[thoughts/Continuous batching|continuous batching]], implemented in [[thoughts/vllm|vLLM]]

> The goal is to reduce internal and external fragmentation in LLM inference. see [[lectures/3/infer-0.3.pdf|this workshop]] for more information.

Reduce memory usage of attention mechanism by swapping kv-cache in and out of memory. A block manager is similar to those of _virtual memory_ in OS.

Essentially, it's a form of **paging**, such that attention can be stored in contiguous memory.
Partitions the KV cache of each sequence into KV blocks.

Another optimization is to use [[thoughts/KV compression|KV compression]] to reduce the size of the KV cache for longer context.

Given:

- each block contains KV vectors for fixed number of tokens, denoted as block size $B$.
- Key block $K_j= (k_{(j-1)B+1}, \ldots, k_{jB})$
- Value block $V_j= (v_{(j-1)B+1}, \ldots, v_{jB})$

$$
A_{ij} = \frac{\exp(q_i^T K_j / \sqrt{d})}{\sum_{t=1}^{i//B} \exp(q_i^T K_t / \sqrt{d})}, \quad o_i = \sum_{j=1}^{i//B} V_j A_{ij}^T
$$

where $A_{ij}=(a_{i,(j-1)B+1}, \ldots a_{i,jB})$ is row vector of attention score on j-th KV block.

> [!motivation] keeping inference steady under load
> Serving many users at once means the GPU must juggle dozens of sequences with wildly different lengths. Paging prevents a single long request from monopolising memory by moving cold KV blocks to host memory while keeping hot blocks on device.

> [!question]- hands-on prompts
>
> - [ ] Trace how vLLM's block manager migrates pages during a mixed workload (long chat + short tool call); sketch the timeline.
> - [ ] Experiment with different block sizes $B$ and record the impact on fragmentation and swap frequency.
> - [ ] Prototype a heuristic that prefetches likely-needed pages based on the query length distribution.

## positional encodings & length extrapolation

> [!summary]
> Positional encodings determine how models reason about order. Good schemes train short but generalise to longer contexts.

- Absolute/learned: add a learned vector per position; simple, weak extrapolation.
- Relative position bias: learn bias as a function of pairwise distance; stable encoders.
- [[thoughts/RoPE|RoPE]]: rotate Q/K in complex plane; inner products encode relative offsets; robust long‑range behavior [@su2023roformerenhancedtransformerrotary].
- ALiBi: add linear distance penalties to attention logits, inducing recency bias that extends context without re‑training [@press2022trainshorttestlong].

### RoPE scaling to extend context

- NTK‑aware scaling: adjust RoPE frequency base to stretch periods for longer contexts.
- YaRN: segment and rescale RoPE dims; efficient extension with small fine‑tune [@peng2023yarnefficientcontextwindow].
- LongRoPE: search-guided rescaling with progressive training to reach 1M–2M+ tokens [@ding2024longropeextendingllmcontext].

See also: [[lectures/3/quantisation basics#multi-latent attention|MLA]] for architectural KV reduction complementary to PE scaling.

---

## Efficient Attention Families

> [!note]
> Exact kernels cut memory traffic; sparse/local reduce edges; linear/approximate trade exactness for O(L) time.

### Exact, IO‑aware kernels

- FlashAttention v1/v2/v3: tile/fuse softmax, read each tile once from HBM; FA‑3 exploits FP8 and hardware copy engines for higher throughput [@dao2022flashattentionfastmemoryefficientexact; @dao2023flashattention2fasterattentionbetter; @shah2024flashattention3fastaccurateattention].
- Flash‑Decoding (+ +): specialised kernels for decode that parallelise across KV blocks and fuse reductions.

### Sparse/local attention (exact on a pattern)

- Longformer: sliding window with optional global tokens for linear scaling [@beltagy2020longformerlongdocumenttransformer].
- BigBird: block-sparse (window + random + global) with theoretical guarantees [@zaheer2021bigbirdtransformerslonger].

### Linear/approximate attention

- Reformer: LSH buckets for sub-quadratic attention + reversible layers [@kitaev2020reformerefficienttransformer].
- Linformer: low-rank projection along sequence dimension [@wang2020linformerselfattentionlinearcomplexity].
- Performer: FAVOR+ kernel features approximate softmax for linear time [@choromanski2022rethinkingattentionperformers].
- Nyströmformer: landmark-based Nyström approximation [@xiong2021nystromformernystrombasedalgorithmapproximating].

### Multi‑device and prefix‑aware inference

- Ring/Striped Attention: partition long sequences across devices, overlapping compute and communication [@liu2023ringattentionblockwisetransformers; @brandon2023stripedattentionfasterring].
- Cascade/Tree-aware kernels: exploit shared prefixes and tree layouts to reuse KV IO [@zheng2024sglangefficientexecutionstructured; @shyam2025treeattentiontopologyawaredecoding].

## Multi-Matrix Factorization Attention

First proposed in [[thoughts/MoE#Step3]]

The idea is to approximate the dense attention matrix by factorising it into multiple low-rank products, each specialised for a subset of heads or positions. Instead of computing $QK^T$ directly, we learn bases $U_i V_i^T$ whose weighted sum reconstructs the attention pattern. This reduces quadratic cost to a series of matrix multiplications with much smaller inner dimensions.

> [!motivation] sharing structure across heads
> Attention maps often lie near a union of low-dimensional subspaces (e.g., monotonic alignments, locality patterns). Factorisation captures those templates explicitly so the model reuses them instead of re-deriving them per head.

> [!question]- further work
>
> - [ ] Derive the computational complexity of using $m$ factors with rank $r$ and compare it to dense attention for typical $m, r$.
> - [ ] Implement a small transformer with multi-matrix factors and inspect whether each factor aligns with an interpretable pattern (locality, copying, etc.).
> - [ ] Investigate how the factorisation interacts with sparsity—can the same bases support both global and local attention if we gate them per token?

## convexity

see also [[lectures/2/convexity|emperical finding]]

---

## cheatsheet

| Method              | Type         | Complexity (seq) | Key idea                                    | Typical win                    |
| ------------------- | ------------ | ---------------: | ------------------------------------------- | ------------------------------ |
| FlashAttention‑3    | exact kernel |           O(L^2) | tiled IO‑minimal attention; FP8/TMA overlap | large train speedups on Hopper |
| Flash‑Decoding / ++ | exact decode |   O(L) per token | block‑parallel KV, fused reductions         | multi‑× decode on long ctx     |
| Longformer          | sparse       |          ~O(L·w) | local window + global tokens                | linear scaling for long docs   |
| BigBird             | sparse       |          ~O(L·w) | window + random + global blocks             | theory + strong practice       |
| Reformer            | approx       |       O(L log L) | LSH attention; reversible layers            | memory/time reductions         |
| Linformer           | approx       |           O(L·k) | low‑rank K/V along L                        | linear time/space              |
| Performer           | approx       |           O(L·d) | FAVOR+ random features                      | linear attention               |
| Nyströmformer       | approx       |           O(L·m) | landmark Nyström approximation              | fewer tokens, good quality     |
| MQA/GQA             | arch/infer   |    O(H_k d_h) KV | share K/V across heads/groups               | KV/bandwidth savings           |
| Ring/Striped        | parallel     |           O(L^2) | pipeline across devices                     | million‑token context          |
| Cascade/Tree‑aware  | kernel       |           O(L^2) | KV reuse on shared prefixes                 | big wins on shared prompts     |
