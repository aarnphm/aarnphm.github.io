---
date: '2026-05-27'
description: reduce KV heads to a fraction of query heads, share K,V across groups for cheaper decode-time cache reuse
id: attention-gqa
modified: 2026-06-05 15:08:05 GMT-04:00
seealso:
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/MLA|MLA]]'
  - '[[thoughts/KV compression]]'
tags:
  - ml
  - llm
  - technical
title: Group-Query Attention
---

idea: reduce number of KV heads $n_k$ to a fraction $n_k^{'} = \frac{n_q}{r}$ of number of query heads $n_q$ (evenly dividing the query heads into $n_k$ groups with $r$ heads)

Let $n_q$ be the number of query heads, $r$ the group size, and $g(i) = \lfloor i / r \rfloor$ the group index. Group-Query Attention keeps per-head query projections but shares the key/value projections within each group:

$$
\begin{aligned}
\text{head}_i &= \operatorname{softmax}\!\left(\frac{Q W_{Q,i} (K W_{K,g(i)})^{\top}}{\sqrt{d_h}} + B_i\right)\, (V W_{V,g(i)}),\\
n_k &= n_q / r,\quad W_{K,g}, W_{V,g} \in \mathbb{R}^{d_{\text{model}} \times d_h}.
\end{aligned}
$$

Each query head keeps its own projection, so the model retains $n_q$ distinct query subspaces, but during decode the runtime reuses the $g(i)$th $K,V$ pair for every head inside the group. Bias term $B_i$ can encode [[thoughts/positional embeddings|positional]] adjustments per query head without duplicating the high-bandwidth value tensors.

> [!math] grouped cache reuse
> For each time step only $n_k$ key/value tiles are fetched from device memory, so the bandwidth term shrinks from $\Theta(h d_h)$ to $\Theta(n_k d_h)$. Choosing $r=2$ halves the cache loads compared with $h=8$ multi-head without sacrificing $Q$-space diversity, matching the decode speedups reported in [@ainslie2023gqatraininggeneralizedmultiquery].

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[
  font=\sffamily\small,
  qhead/.style={draw=black, fill=cyan!20, minimum width=0.5cm, minimum height=0.45cm, inner sep=0pt},
  kvhead/.style={draw=black, fill=orange!30, minimum width=0.5cm, minimum height=0.45cm, inner sep=0pt},
  arrow/.style={->, >=latex, gray!70, thick},
  scope title/.style={font=\sffamily\small\bfseries}
]
  \path[use as bounding box] (-0.5, 1.1) rectangle (15.5, 4.7);

  % --- MHA panel ---
  \node[scope title] at (1.925, 4.2) {MHA};
  \foreach \i in {0,1,2,3,4,5,6,7} {
    \node[qhead] (mq\i) at (\i*0.55, 3.2) {};
    \node[kvhead] (mkv\i) at (\i*0.55, 2.2) {};
    \draw[arrow] (mq\i) -- (mkv\i);
  }
  \node[font=\sffamily\small] at (1.925, 1.5) {8 queries, 8 KV};

  % --- GQA panel ---
  \node[scope title] at (7.425, 4.2) {GQA ($r=2$)};
  \foreach \i in {0,1,2,3,4,5,6,7} {
    \node[qhead] (gq\i) at (\i*0.55 + 5.5, 3.2) {};
  }
  \foreach \j in {0,1,2,3} {
    \node[kvhead] (gkv\j) at (\j*1.1 + 0.275 + 5.5, 2.2) {};
  }
  \foreach \i/\j in {0/0,1/0,2/1,3/1,4/2,5/2,6/3,7/3} {
    \draw[arrow] (gq\i) -- (gkv\j);
  }
  \node[font=\sffamily\small] at (7.425, 1.5) {8 queries, 4 KV};

  % --- MQA panel ---
  \node[scope title] at (12.925, 4.2) {MQA};
  \foreach \i in {0,1,2,3,4,5,6,7} {
    \node[qhead] (sq\i) at (\i*0.55 + 11, 3.2) {};
  }
  \coordinate (skvcenter) at (12.925, 2.2);
  \foreach \i in {0,1,2,3,4,5,6,7} {
    \draw[arrow] (sq\i) -- (skvcenter);
  }
  \node[kvhead] (skv) at (skvcenter) {};
  \node[font=\sffamily\small] at (12.925, 1.5) {8 queries, 1 KV};
\end{tikzpicture}
\end{document}
```

```jsx imports={Zoomable,KVHeadGrouping}
<Zoomable label="KV head grouping">
  <KVHeadGrouping
    caption="Slide the group ratio to watch KV boxes merge and decode cache reads shrink."
    heads={8}
  />
</Zoomable>
```

> [!question]- explore the design space
>
> - [ ] Starting from a vanilla transformer decoder, implement grouped keys/values and benchmark the decode tokens-per-second improvement as context length grows.
> - [ ] Analyse how grouping interacts with rotary or ALiBi positional encodings; does sharing $K/V$ across heads degrade positional resolution?
> - [ ] Reproduce the ablation table from @ainslie2023gqatraininggeneralizedmultiquery to see how aggressively $n_k^{'}$ can be reduced before accuracy drops on your domain.

> [!todo]+ follow-up questions
>
> - Derive how the attention matrix factorises when queries are grouped and quantify the approximation error introduced by shared $K,V$ pairs.
> - Collect empirical results comparing [[thoughts/KV compression|KV cache]] sizes for MHA, Multi-Query, and GQA across popular decoder-only models.
> - Investigate hardware implications: how does GQA interact with tensor parallelism or speculative decoding pipelines?
