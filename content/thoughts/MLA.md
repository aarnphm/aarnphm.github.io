---
date: '2026-05-27'
description: low-rank joint compression of attention K,V (and Q), cache only the latent and RoPE-carrying duplicate.
id: attention-mla
modified: 2026-05-28 02:06:22 GMT-04:00
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

![[thoughts/images/mla-comparison.webp]]

low-rank joint compression for attention ==keys and values== to reduce KV cache during inference [@deepseekai2025deepseekv3technicalreport, see Section 2.1.1; @deepseekai2024deepseekv2strongeconomicalefficient]

- $d$ denote the embedding dimension
- $n_h$ denotes number of attention heads
- $d_h$ denotes dimension per heads
- $h_t \in \mathbb{R}^d$ denotes the attention input for the $t$-th token at a given attention layer

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
- $W^{KR} \in \mathbb{R}^{d^R_h \times d}$ is the matrix used to produced the duplicate key that carries [[thoughts/RoPE|RoPE]]
- $\mathrm{RoPE}(.)$ denotes operations for RoPE matrices, and $[;]$ denotes ==concatenation==
- Note that only $\boxed{\textcolor{blue}{\mathbf{c}_t^{KV}}}, \boxed{\textcolor{blue}{\mathbf{k}_t^{R}}}$ needs to be cached

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[
  font=\sffamily\small,
  src/.style={draw=black, fill=cyan!20, rounded corners=2pt, minimum width=2cm, minimum height=0.7cm},
  latent/.style={draw=black, fill=orange!35, rounded corners=2pt, minimum width=2cm, minimum height=0.7cm, very thick},
  head/.style={draw=black, fill=gray!10, rounded corners=2pt, minimum width=2.2cm, minimum height=0.55cm},
  rope/.style={draw=black, fill=green!25, rounded corners=2pt, minimum width=2cm, minimum height=0.7cm, very thick},
  arr/.style={->, >=latex, thick},
  cachelbl/.style={font=\sffamily\bfseries\itshape, red!70!black}
]
  \node[src] (h) at (0, 3) {$h_t$};
  \node[latent] (c) at (3.5, 4) {$c_t^{KV}$};
  \node[rope] (kr) at (3.5, 2) {$k_t^R$};

  \node[head] (kc0) at (7.5, 5) {$k_{t,1}^C \dots k_{t,n_h}^C$};
  \node[head] (vc0) at (7.5, 4) {$v_{t,1}^C \dots v_{t,n_h}^C$};

  \draw[arr] (h) -- node[above, font=\sffamily\footnotesize] {$W^{DKV}$} (c);
  \draw[arr] (h) -- node[below, font=\sffamily\footnotesize] {$W^{KR}\!,\,\text{RoPE}$} (kr);
  \draw[arr] (c.east) -- node[above, font=\sffamily\footnotesize] {$W^{UK}$} (kc0.west);
  \draw[arr] (c.east) -- node[below, font=\sffamily\footnotesize] {$W^{UV}$} (vc0.west);

  \node[cachelbl, anchor=west] at (5.2, 1.4) {cache: $\{c_t^{KV},\ k_t^R\}$};
  \draw[red!70!black, dashed, thick, rounded corners] (2.7, 1.4) rectangle (4.3, 4.6);
\end{tikzpicture}
\end{document}
```

```jsx imports={Zoomable,MLALatentPath}
<Zoomable label="MLA latent projection path">
  <MLALatentPath caption="Down-projection caches the joint latent $c_t^{KV}$ and the RoPE duplicate $k_t^R$; per-head $k^C, v^C$ are reconstructed on demand via $W^{UK}, W^{UV}$. Slide $d_c$ and $d_h^R$ to feel the joint low-rank bet collapse the KV cache." />
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
[\mathbf{q}_{t,1}^{R}; \mathbf{q}_{t,2}^{R}; \dots; \mathbf{q}_{t, n_h}^{R}] &= \mathrm{RoPE}(W^{QR} \mathbf{c}_t^Q) \\
\mathbf{q}_{i,t} &= [\mathbf{q}_{t,i}^{C}; \mathbf{q}_t^{R}]
\end{aligned}
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
>     \mathbf{o}_{t,i} &= \sum_{j=1}^{t} \mathrm{Softmax}_j (\frac{q_{t,i}^T k_{j,i}}{\sqrt{d_h + d_h^R}}) v_{j_i}^C\quad \quad \tag{10} \\
>     \mathbf{u}_t &= \mathbf{W}^O [o_{t,1}; o_{t,2}; \dots; o_{t, n_h}] \quad \tag{11}
> \end{align}
> $$
