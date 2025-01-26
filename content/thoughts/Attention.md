---
id: Attention
tags:
  - technical
abstract: The reason for Attention comparing to LSTM is that its ability to encode additional positional data into the inputs, in which it helps with longer context length and better memory retrieval. Note that most LLMs are decoder-only, given its superior benchmark in zero-shot tasks.
date: "2024-02-07"
description: and posteriori information retrieval.
modified: 2025-01-26 05:35:53 GMT-05:00
title: Attention
---

Attention operates on a sequence of query $Q$, key $K$ and value $V$ vector. Attention matrix of a sequence then computed as [@vaswani2023attentionneed]:

$$
A(Q, K, V) = \text{softmax}(\frac{Q \cdot K^{T}}{\sqrt{d}})V \space \space \text{ for } Q_{L \times d}, K_{L \times d}, V_{L \times d}
$$

> [!note]+ equivalent
>
> We can probably arrange the attention function (composed of multiple [[thoughts/induction heads|attention-heads]]) according to @elhage2021mathematical:
>
> $$
> \text{Attn}^{\vec{l,h}}(X_{\leq i}^{l-1}) = \sum_{j \leq i}a^{l,h}_{i,j} x^{l-1}_j W^{l,h}_{V} W_{O}^{l,h}
> $$
>
> where the ==learn-able== weight matrices $W_{V}^{l,h} \in \mathbb{R}^{d \times d_h}$ and $W_{O}^{l,h} \in \mathbb{R}^{d_h \times d}$, $d_h$ is the dimension per head, are combined OV matrix

## Muti-head Attention

Allows the model to jointly attend to information from different representation subspaces at different positions:

$$
\begin{aligned}
\text{MHA}(Q,K,V) &= \text{concat}(\text{head}_1, \cdots, \text{head}_n) W^O \\
&\text{where } \space \text{head}_i = \text{A}(QW_i^O, KW_i^O, VW_i^O) \\
W^O & \in \mathbb{R}^{hd_v \times d_{\text{model}}}
\end{aligned}
$$

## Group-Query Attention

[@ainslie2023gqatraininggeneralizedmultiquery]

idea: reduce number of KV heads $n_k$ to a fraction $n_k^{'} = \frac{n_q}{k}$ of number of query heads $n_q$ (evenly dividing the query heads into $n_k$ groups with $r$ heads)

## RadixAttention

@zheng2024sglangefficientexecutionstructured proposes this to maintain a LRU eviction policy to maintain relevant [[thoughts/KV compression|KV cache]] for all requests within a [[thoughts/Radix tree|radix tree]], Implemented in https://github.com/sgl-project/sglang

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

![[thoughts/constrained decoding#compressed FSM for jump-ahead tokens.]]

## Multi-head Latent Attention (MLA)

low-rank joint compression for attention ==keys and values== to reduce KV cache during inference [@deepseekai2025deepseekr1incentivizingreasoningcapability{see 2.1.1}]

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
- $W^{KR} \in \mathbb{R}^{d^R_h \times d}$ is the matrix used to produced the duplicate key that carries RoPE
- $\mathrm{RoPE}(.)$ denotes operations for RoPE matrices, and $\mathrm{RoPE}[;]$ denotes ==concatenation==

> [!important] cached generations
>
> Both $\textcolor{blue}{\mathbf{c}_t^{KV}}$ and $\textcolor{blue}{\mathbf{k}_t^{R}}$ should be cached to reduce KV cache while maintaining performance with [[thoughts/Attention#Muti-head Attention|MHA]]

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

## RingAttention

[@liu2023ringattentionblockwisetransformers]

## RazorAttention

[@tang2024razorattentionefficientkvcache]

## Paged Attention

[@kwon2023efficient]

In conjunction with [[thoughts/Continuous batching|continuous batching]], implemented in [[thoughts/vllm|vLLM]]

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
