---
id: Attention
tags:
  - technical
  - seed
date: "2024-02-07"
modified: "2024-10-13"
title: Attention
---

See also: [Paper](https://arxiv.org/abs/1706.03762) [@vaswani2023attentionneed]

Attention operates on a sequence of query $Q$, key $K$ and value $V$ vector. Attention matrix of a sequence then computed as:

$$
A(Q, K, V) = \text{softmax}(\frac{Q \cdot K^{T}}{\sqrt{d}})V \space \space \text{ for } Q_{L \times d}, K_{L \times d}, V_{L \times d}
$$

See also [[thoughts/Transformers|Transformers]], or [[thoughts/distraction|distraction]]

## Muti-head Attention

Allows the model to jointly attend to information from different representation subspaces at different positions:

$$
\begin{aligned}
\text{MHA}(Q,K,V) &= \text{concat}(\text{head}_1, \cdots, \text{head}_n) W^O \\
&\text{where } \space \text{head}_i = \text{A}(QW_i^O, KW_i^O, VW_i^O) \\
W^O & \in \mathbb{R}^{hd_v \times d_{\text{model}}}
\end{aligned}
$$
## [Group-Query Attention](https://arxiv.org/abs/2305.13245)

by [@ainslie2023gqatraininggeneralizedmultiquery]

idea: reduce number of KV heads $n_k$ to a fraction $n_k^{'} = \frac{n_q}{k}$ of number of query heads $n_q$ (evenly dividing the query heads into $n_k$ groups with $r$ heads)

## RingAttention

## [RazorAttention](https://arxiv.org/abs/2407.15891)

by [@tang2024razorattentionefficientkvcache]

## Paged Attention

[paper](https://arxiv.org/pdf/2309.06180.pdf)

Used in conjunction with [[thoughts/Continuous batching]]

Reduce memory usage of attention mechanism by swapping kv-cache in and out of memory. A block manager is similar to those of *virtual memory* in OS.

Essentially, it's a form of **paging**, such that attention can be stored in contiguous memory.
Partitions the KV cache of each sequence into KV blocks.

Given:
- each block contains KV vectors for fixed number of tokens, denoted as block size $B$.
- Key block $K_j= (k_{(j-1)B+1}, \ldots, k_{jB})$
- Value block $V_j= (v_{(j-1)B+1}, \ldots, v_{jB})$

$$
A_{ij} = \frac{\exp(q_i^T K_j / \sqrt{d})}{\sum_{t=1}^{i//B} \exp(q_i^T K_t / \sqrt{d})}, \quad o_i = \sum_{j=1}^{i//B} V_j A_{ij}^T
$$

where $A_{ij}=(a_{i,(j-1)B+1}, \ldots a_{i,jB})$ is row vector of attention score on j-th KV block.
