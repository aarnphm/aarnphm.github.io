---
id: Attention
tags:
  - technical
  - seed
date: "2024-02-07"
title: Attention
---

Paper: [Attention is all you need](https://arxiv.org/abs/1706.03762)

See also [[thoughts/Transformers]]

### Self-attention

### Muti-head Attention

### Paged Attention

[paper](https://arxiv.org/pdf/2309.06180.pdf)

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
