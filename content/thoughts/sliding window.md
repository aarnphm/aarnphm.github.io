---
date: '2026-05-27'
description: 'each token attends only inside a fixed radius, local pattern dropping cost from $\mathcal{O}(L^2)$ to $\mathcal{O}(Lw)$.'
id: attention-sliding-window
modified: 2026-05-27 23:17:24 GMT-04:00
seealso:
  - '[[thoughts/Attention|main stage]]'
  - '[[thoughts/flash attention|FlashAttention]]'
  - '[[@beltagy2020longformerlongdocumenttransformer]]'
  - '[[@zaheer2021bigbirdtransformerslonger]]'
tags:
  - ml
  - llm
  - technical
title: sliding window attention
---

Sliding window (or local) attention constrains each token to attend only to neighbours within a fixed radius $w$. The computational cost drops from $\mathcal{O}(L^2)$ to $\mathcal{O}(L \cdot w)$.

Formally, define the binary mask

$$
M_{ij} = \begin{cases}
0 & \text{if } |i-j| \le w\ \text{or } j \in G,\\
-\infty & \text{otherwise},
\end{cases}
$$

where $G$ indexes optional global tokens. A head at position $i$ then evaluates

$$
\text{head}_i = \operatorname{softmax}\!\left(\frac{Q_i W_Q (K W_K)^{\top}}{\sqrt{d_h}} + M_{i,:}\right) (V W_V).
$$

In implementation the KV cache is a circular buffer that keeps only the most recent $2w+|G|$ entries per head; evicted blocks can be recomputed from checkpoints if needed for evaluation.

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[
  font=\sffamily\small,
  cell/.style={draw=gray!40, minimum width=0.32cm, minimum height=0.32cm, inner sep=0pt},
  attended/.style={cell, fill=cyan!50},
  global/.style={cell, fill=orange!60},
  blank/.style={cell, fill=white}
]
  \def\N{12}
  \def\W{2}
  \def\GlobIdx{0}

  \foreach \i in {0,...,11} {
    \foreach \j in {0,...,11} {
      \pgfmathparse{ifthenelse(\i == \GlobIdx || \j == \GlobIdx, 1, 0)}
      \let\isglobal\pgfmathresult
      \pgfmathparse{ifthenelse(abs(\i - \j) <= \W, 1, 0)}
      \let\iswindow\pgfmathresult
      \ifnum\isglobal=1
        \node[global] at (\j*0.35, -\i*0.35) {};
      \else
        \ifnum\iswindow=1
          \node[attended] at (\j*0.35, -\i*0.35) {};
        \else
          \node[blank] at (\j*0.35, -\i*0.35) {};
        \fi
      \fi
    }
  }

  \node[anchor=south, font=\sffamily] at (1.925, 0.45) {key index $j$};
  \node[anchor=east, font=\sffamily, rotate=90] at (-0.5, -1.925) {query index $i$};

  \node[anchor=west, font=\sffamily, fill=cyan!50, draw=gray!40] at (5.0, -0.5) {window $|i-j| \leq w$};
  \node[anchor=west, font=\sffamily, fill=orange!60, draw=gray!40] at (5.0, -1.5) {global token};
\end{tikzpicture}
\end{document}
```

```jsx imports={Zoomable,SlidingWindowMask}
<Zoomable label="sliding window mask matrix">
  <SlidingWindowMask
    caption="Interactive mask: drag w to thicken the band, switch dilation, add global tokens, watch the cost ratio collapse below the diagonal."
    length={24}
  />
</Zoomable>
```

- Motivation: maximise throughput on long-context tasks where relevant information is clustered locally (e.g., speech, DNA sequences).
- Challenge: ensuring important long-range dependencies are not lost, often solved by adding a handful of global tokens or dilation patterns.

> [!todo]+ experiments to run
>
> - Implement a toy [[thoughts/Autoregressive models|autoregressive]] model with sliding window attention and track perplexity as $w$ varies.
> - Document hybrid strategies (e.g., dilated windows, stride patterns) and how they impact the receptive field.
> - Collect references on how models like Longformer or [[thoughts/Transformers|BigBird-style transformers]] mix local and global tokens.
