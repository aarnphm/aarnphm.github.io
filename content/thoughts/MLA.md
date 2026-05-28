---
date: '2026-05-27'
description: low-rank joint compression of attention K,V (and Q), cache only the latent and RoPE-carrying duplicate.
id: attention-mla
modified: 2026-05-28 12:27:47 GMT-04:00
seealso:
  - '[[thoughts/Attention|main stage]]'
  - '[[thoughts/GQA|GQA]]'
  - '[[thoughts/Compression|Compression]]'
  - '[[thoughts/RoPE|RoPE]]'
  - '[[lectures/3/quantisation basics#multi-latent attention|quantization math]]'
tags:
  - ml
  - llm
  - technical
title: Multi-Head Latent Attention
---

```jsx imports={Zoomable,KVCacheVariants}
<Zoomable label="KV cache across attention variants">
  <KVCacheVariants caption="hatched blocks are cache-resident. MHA stores every head's $K,V$; GQA stores one $K,V$ pair per group; MQA stores one shared pair; MLA stores $c_t^{KV}$ and $k_t^R$, then reconstructs $k_{t,i}^C$ and $v_{t,i}^C$ on demand." />
</Zoomable>
```

low-rank joint compression for attention ==keys and values== to reduce KV cache during inference [@deepseekai2025deepseekv3technicalreport, see Section 2.1.1; @deepseekai2024deepseekv2strongeconomicalefficient]

- $d$ denotes the embedding dimension
- $n_h$ denotes number of attention heads
- $d_h$ denotes dimension per head
- $\mathbf{h}_t \in \mathbb{R}^d$ denotes the attention input for the $t$-th token at a given attention layer

$$
\begin{align}
    \boxed{\textcolor{blue}{\mathbf{c}_t^{KV}}} &= W^{DKV} \mathbf{h}_t\quad \quad \tag{1} \\
    [\mathbf{k}_{t,1}^{C}; \mathbf{k}_{t,2}^{C}; \dots; \mathbf{k}_{t, n_h}^{C}] &= \mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}\quad \quad \tag{2} \\
    \boxed{\textcolor{blue}{\mathbf{k}_t^{R}}} &= \mathrm{RoPE}(W^{KR} \mathbf{h}_t)\quad \quad \tag{3} \\
    \mathbf{k}_{i,t} &= [\mathbf{k}_{t,i}^{C}; \mathbf{k}_t^{R}]\quad \quad \tag{4} \\
    [\mathbf{v}_{t,1}^{C}; \mathbf{v}_{t,2}^{C}; \dots; \mathbf{v}_{t,n_h}^{C}] &= \mathbf{v}_t^{C} = W^{UV} \mathbf{c}_t^{KV} \quad \quad \tag{5}
\end{align}
$$

- _where_ $c_{t}^{KV} \in \mathbb{R}^{d_{c}}$ is the compression latent for keys and values
- $d_c \ll d_h n_h$ indicates KV [[thoughts/Compression|compression]] dimension
- $W^{DKV} \in  \mathbb{R}^{d_c \times d}$ denotes down-projection matrix
- $W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$ are the up-projection matrices to keys and values, respectively
- $W^{KR} \in \mathbb{R}^{d^R_h \times d}$ is the matrix used to produce the duplicate key that carries [[thoughts/RoPE|RoPE]]
- $\mathrm{RoPE}(\cdot)$ denotes RoPE application, and $[\,;\,]$ denotes ==concatenation==
- Note that only $\boxed{\textcolor{blue}{\mathbf{c}_t^{KV}}}$ and $\boxed{\textcolor{blue}{\mathbf{k}_t^{R}}}$ need to be cached

```jsx imports={Zoomable,MLALatentPath}
<Zoomable label="MLA latent projection path">
  <MLALatentPath caption="MLA latent projection path. We only cache $c_t^{KV}$ and $k_t^R$ only. reconstruct $k_{t,i}^C$ and $v_{t,i}^C$ from $c_t^{KV}$, concatenate $k_{t,i}^C$ with $k_t^R$, and keep queries on the separate $c_t^Q$ path. slide $d_c$ and $d_h^R$ to move cache size." />
</Zoomable>
```

> [!important] cached generations
>
> Both $\textcolor{blue}{\mathbf{c}_t^{KV}}$ and $\textcolor{blue}{\mathbf{k}_t^{R}}$ should be cached to reduce KV cache while maintaining performance with [[thoughts/Attention#Multi-head Attention|MHA]]

For attention ==queries==, we can perform the same operation:

$$
\begin{aligned}
\mathbf{c}_t^{Q} &= W^{DQ} \mathbf{h}_t \\
[\mathbf{q}_{t,1}^{C}; \mathbf{q}_{t,2}^{C}; \dots; \mathbf{q}_{t, n_h}^{C}] &= \mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^{Q} \\
[\mathbf{q}_{t,1}^{R}; \mathbf{q}_{t,2}^{R}; \dots; \mathbf{q}_{t, n_h}^{R}] &= \mathbf{q}_t^R = \mathrm{RoPE}(W^{QR} \mathbf{c}_t^Q) \\
\mathbf{q}_{t,i} &= [\mathbf{q}_{t,i}^{C}; \mathbf{q}_{t,i}^{R}]
\end{aligned}
$$

- $c_t^Q$ is the compressed latent of queries
- $d_c \ll d_h n_h$ indicates queries compression dimension
- $W^{DQ} \in \mathbb{R}^{d^{'}_c \times d}, W^{UQ} \in \mathbb{R}^{d_h n_h \times d^{'}_c}$ are the up and down [[thoughts/geometric projections|projections]] matrices
- $W^{QR} \in \mathbb{R}^{d_{h}^R n_{h} \times d_{c}^{'}}$ is the matrix that produces queries that carry [[thoughts/RoPE|RoPE]]

> [!abstract] Attention output
>
> The attention output $\mathbf{u}_{t}$ can be calculated with the following:
>
> $$
> \begin{align}
>     \mathbf{o}_{t,i} &= \sum_{j=1}^{t} \operatorname{softmax}_j \left(\frac{\mathbf{q}_{t,i}^{\top}\mathbf{k}_{j,i}}{\sqrt{d_h + d_h^R}}\right) \mathbf{v}_{j,i}^C\quad \quad \tag{10} \\
>     \mathbf{u}_t &= \mathbf{W}^O [\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; \dots; \mathbf{o}_{t, n_h}] \quad \tag{11}
> \end{align}
> $$
