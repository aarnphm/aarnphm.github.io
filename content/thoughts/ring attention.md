---
date: '2026-05-27'
description: sequence sharded across devices in a ring, K,V blocks circulate so each GPU holds only a slice of the cache.
id: attention-ring
modified: 2026-06-01 00:28:32 GMT-04:00
seealso:
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/tree attention|tree attention]]'
  - '[[thoughts/flash attention|FlashAttention]]'
tags:
  - ml
  - llm
  - technical
title: Ring Attention
---

RingAttention [@liu2023ringattentionblockwisetransformers] shards long contexts across devices in a ring pipeline. Striped Attention [@brandon2023stripedattentionfasterring] improves load balance with alternating shard ownership.

RingAttention shards a long sequence across multiple devices and circulates key/value blocks in a logical ring. Each GPU holds only a slice of the cache, streams neighbouring slices just in time, and discards them after use. The ring topology overlaps communication with computation, so no single device ever needs the full context resident in memory.

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[
  font=\sffamily\small,
  device/.style={draw=black, fill=cyan!20, rounded corners=2pt, minimum width=2cm, minimum height=0.9cm},
  shard/.style={draw=black, fill=orange!30, rounded corners=2pt, minimum width=1.4cm, minimum height=0.5cm, font=\sffamily\footnotesize},
  ring/.style={->, >=latex, thick, gray!70}
]
  \def\R{2.7}
  \foreach \i/\ang in {0/90, 1/0, 2/270, 3/180} {
    \node[device] (d\i) at ({\R*cos(\ang)}, {\R*sin(\ang)}) {device $\i$};
  }
  \foreach \i/\ang in {0/90, 1/0, 2/270, 3/180} {
    \node[shard] at ({(\R + 1.4)*cos(\ang)}, {(\R + 1.4)*sin(\ang)}) {$k_\i, v_\i$};
  }

  \draw[ring] (d0) to[bend right=18] (d1);
  \draw[ring] (d1) to[bend right=18] (d2);
  \draw[ring] (d2) to[bend right=18] (d3);
  \draw[ring] (d3) to[bend right=18] (d0);

  \node[font=\sffamily\itshape, gray] at (0, 0) {KV blocks circulate};
\end{tikzpicture}
\end{document}
```

```jsx imports={Zoomable,RingRotation}
<Zoomable label="ring attention rotation">
  <RingRotation
    caption="step the ring, watch each device accrue all $p$ slices while its own cache stays at $L/p$."
    devices={4}
  />
</Zoomable>
```

> [!question]- experiments
>
> - [ ] Simulate RingAttention with three devices and measure throughput as you vary block size; look for the sweet spot where communication overlaps compute.
> - [ ] Analyse failure cases when network latency spikes; how resilient is the ring schedule compared to fully sharded data parallelism?
> - [ ] Derive how gradient checkpointing interacts with the ring; can we reuse streamed activations during backprop?
