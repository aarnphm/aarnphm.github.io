---
id: Attention
tags:
  - technical
  - seed
date: "2024-02-07"
modified: "2024-10-13"
title: Attention
---

[@vaswani2023attentionneed]

Attention operates on a sequence of query $Q$, key $K$ and value $V$ vector. Attention matrix of a sequence then computed as:

$$
A(Q, K, V) = \text{softmax}(\frac{Q \cdot K^{T}}{\sqrt{d}})V \space \space \text{ for } Q_{L \times d}, K_{L \times d}, V_{L \times d}
$$

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

by [@ainslie2023gqatraininggeneralizedmultiquery]

idea: reduce number of KV heads $n_k$ to a fraction $n_k^{'} = \frac{n_q}{k}$ of number of query heads $n_q$ (evenly dividing the query heads into $n_k$ groups with $r$ heads)

## RadixAttention

Implemented in [@zheng2024sglangefficientexecutionstructured] where they maintain a LRU eviction policy to maintain relevant [[thoughts/KV compression|KV cache]] for all requests within a [[thoughts/Radix tree|radix tree]]

radix tree setup:

- key: sequence of tokens
- value: KV cache tensor (stored in GPU in a paged layout)

![[thoughts/images/vllm/radix-attention.jpeg]]

_dynamic evolution of the radix tree in response to various requests._

> [!abstract]- explanation of RadixAttention with LRU eviction policy
>
> These requests include two chat sessions, a batch of few-shot learning inquiries, and a self-consistency sampling. Each tree edge carries a label denoting a substring or a sequence of tokens. The nodes are color-coded to reflect different states: green for newly added nodes, blue for cached nodes accessed during the time point, and red for nodes that have been evicted.
>
> [full explanation](https://lmsys.org/blog/2024-01-17-sglang/#backend-automatic-kv-cache-reuse-with-radixattention)

### cache-aware scheduling

define the hit rate as

$$
\text{hit rate} = \frac{\text{number of cached prompt tokens}}{\text{number of prompt tokens}}
$$

_in batch settings: sort requests by matching prefix length and prioritise one with longer matched prefixes instead of FIFO schedule._

```pseudo
\begin{algorithm}
\caption{Cache-Aware Scheduling for RadixAttention with Continuous Batching}
\begin{algorithmic}
\State \textbf{Input:} The radix tree $T$, the memory pool $P$, the current running batch $B$, the waiting queue $Q$.
\State \textbf{Output:} Finished requests and updated system state.
\State $requests \gets Q.\text{get\_all\_requests}()$
\For{$req \in requests$}
    \State $req.\text{prefix\_node}, req.\text{prefix\_len} \gets T.\text{match\_prefix}(req.\text{input\_tokens})$
\EndFor
\State $requests.\text{sort}()$
\State // Select requests for the next batch
\State $available\_size \gets T.\text{evictable\_size}() + P.\text{available\_size}()$
\State $current\_size \gets 0$
\State $new\_batch \gets []$
\For{$req \in requests$}
    \If{$req.\text{size}() + current\_size < available\_size$}
        \State $new\_batch.\text{append}(req)$
        \State $\delta \gets T.\text{increase\_ref\_counter}(req.\text{prefix\_node})$
        \State $available\_size \gets available\_size + \delta$
    \EndIf
\EndFor
\State $Q.\text{remove\_requests}(new\_batch)$
\State // Insert requests into the current running batch
\State $B.\text{merge}(new\_batch)$
\State // Allocate new memory and do eviction if necessary
\State $needed\_size \gets B.\text{needed\_size}()$
\State $success, buffer \gets P.\text{alloc}(needed\_size)$
\If{\textbf{not} success}
    \State $T.\text{evict}(needed\_size)$
    \State $success, buffer \gets P.\text{alloc}(needed\_size)$
\EndIf
\State $B.\text{run}(buffer)$
\State // Process finished requests
\State $finished\_requests \gets B.\text{drop\_finished\_requests}()$
\For{$req \in finished\_requests$}
    \State $T.\text{decrease\_ref\_counter}(req.\text{prefix\_node})$
    \State $T.\text{insert}(req)$
\EndFor
\State \Return $finished\_requests$
\end{algorithmic}
\end{algorithm}
```

## RingAttention

## RazorAttention

by [@tang2024razorattentionefficientkvcache]

## Paged Attention

by [@kwon2023efficient]

Used in conjunction with [[thoughts/Continuous batching|continuous batching]], implemented through [[thoughts/vllm|vLLM]]

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
