---
date: '2026-05-27'
description: tiled IO-aware attention kernel, recomputes softmax denominators on-the-fly, avoids materialising the full attention matrix.
id: attention-flash
modified: 2026-06-01 20:58:36 GMT-04:00
seealso:
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/tree attention|tree attention]]'
  - '[[thoughts/GPU programming]]'
  - '[[@dao2023flashattention2fasterattentionbetter]]'
  - '[[@shah2024flashattention3fastaccurateattention]]'
tags:
  - ml
  - llm
  - technical
title: Flash Attention
---

FlashAttention [@dao2022flashattentionfastmemoryefficientexact] reframes attention as a tiled matrix multiplication that keeps intermediate results in high-speed SRAM rather than slower GPU DRAM.

Recomputing softmax denominators on-the-fly avoids materialising the full attention matrix.

As sequence lengths $L$ grow, attention becomes more IO-bound than FLOP-bound, so this optimisation yields both speedups and numerical stability (via online normalisation).

FlashAttention partitions the logits $S = QK^{\top}/\sqrt{d_h}$ into $B_m \times B_n$ tiles.

For each tile $t$ the kernel streams $Q_t$ and $K_t$ into SRAM, updates the running maxima $m$ and partition sums $l$, then accumulates the context contribution:

$$
\begin{aligned}
m^{\text{new}}_i &= \max\big(m^{\text{old}}_i, \max_j S_{ij}^{(t)}\big),\\
l^{\text{new}}_i &= e^{m^{\text{old}}_i - m^{\text{new}}_i} l^{\text{old}}_i + \sum_j e^{S_{ij}^{(t)} - m^{\text{new}}_i},\\
O^{\text{new}}_i &= e^{m^{\text{old}}_i - m^{\text{new}}_i} O^{\text{old}}_i + \sum_j e^{S_{ij}^{(t)} - m^{\text{new}}_i} V^{(t)}_j.
\end{aligned}
$$

Only the current tile's $K,V$ blocks ever leave global memory. After processing all tiles the output normalises as $O_i = O^{\text{new}}_i / l^{\text{new}}_i$, matching exact softmax attention while respecting SRAM capacity constraints.

> [!tip] tuning tile shapes
> Choosing $B_m,B_n$ to align with tensor-core fragment sizes (e.g., $64\times64$ for FP16) keeps the kernel compute-bound. FlashAttention-2 further overlaps tiles across heads, while FlashAttention-3 incorporates block-sparse layouts and asynchronous pipeline stages.

```tikz
\usepackage{tikz}
\begin{document}
\definecolor{salmon}{HTML}{FDB2A2}
\definecolor{sage}{HTML}{CDD597}
\definecolor{stone}{HTML}{6F6E69}
\definecolor{paper}{HTML}{FFFCF0}
\begin{tikzpicture}[
  font=\small, >=latex,
  tier/.style={draw=stone, rounded corners=3pt, align=center, minimum width=3.3cm, minimum height=1.15cm},
  mat/.style={draw=stone, fill=paper, rounded corners=2pt, minimum width=1.1cm, minimum height=0.95cm, inner sep=2pt},
  tile/.style={draw=stone, fill=salmon!55, rounded corners=2pt, minimum width=0.95cm, minimum height=0.72cm, inner sep=1pt},
  flow/.style={->, thick, draw=stone},
  bidir/.style={<->, thick, draw=stone}
]
  \path[use as bounding box] (-3.1, 0.3) rectangle (15.4, 7.1);

  % swimlanes: each lane is one level of the GPU memory hierarchy
  \fill[salmon!14, rounded corners=4pt] (-3.0, 4.35) rectangle (15.2, 5.6);
  \fill[sage!16, rounded corners=4pt]   (-3.0, 2.35) rectangle (15.2, 3.6);
  \fill[stone!10, rounded corners=4pt]  (-3.0, 0.5)  rectangle (15.2, 1.75);

  % compute feeds the top of the hierarchy
  \node[draw=stone, fill=white, rounded corners=2pt, minimum width=2.2cm, minimum height=0.62cm] (sm) at (-0.9, 6.45) {SMs (compute)};

  % left column: tiers with bandwidth, capacity (A100, Dao 2022 Fig 1)
  \node[tier, fill=salmon!55] (sram) at (-0.9, 4.95) {SRAM\\{\scriptsize 19 TB/s, 20 MB}};
  \node[tier, fill=sage!55]   (hbm)  at (-0.9, 2.95) {HBM\\{\scriptsize 1.5 TB/s, 40 GB}};
  \node[tier, fill=stone!22]  (dram) at (-0.9, 1.10) {DRAM\\{\scriptsize 12.8 GB/s, 1+ TB}};
  \draw[bidir] (sm) -- (sram);
  \draw[bidir] (sram) -- (hbm);
  \draw[bidir] (hbm) -- (dram);

  % SRAM lane: the on-chip tiles FlashAttention computes on
  \node[tile] (qt) at (5.4, 4.95) {$Q_t$};
  \node[tile] (kt) at (7.5, 4.95) {$K_t$};
  \node[tile] (vt) at (9.6, 4.95) {$V_t$};
  \node[tile, fill=sage!60, minimum width=1.9cm] (st) at (12.3, 4.95) {$m$, $l$, $O_t$};
  \node[font=\scriptsize, text=stone, anchor=west] at (13.45, 4.95) {tile $B_m\times d$};

  % HBM lane: the full token matrices
  \node[mat] (q) at (5.4, 2.95) {$Q$};
  \node[mat] (k) at (7.5, 2.95) {$K$};
  \node[mat] (v) at (9.6, 2.95) {$V$};
  \node[mat] (o) at (11.7, 2.95) {$O$};
  \node[font=\scriptsize, text=stone, anchor=west] at (12.5, 2.95) {full $L\times d$};

  % DRAM lane: what the slow tier holds
  \node[mat, minimum width=1.6cm] (wts) at (5.6, 1.10) {weights};
  \node[mat, minimum width=1.6cm] (cch) at (8.0, 1.10) {cache};

  % FlashAttention dataflow: load tiles up, store result down
  \draw[flow] (q) -- (qt) node[midway, right=1pt, font=\scriptsize, text=stone] {load};
  \draw[flow] (k) -- (kt);
  \draw[flow] (v) -- (vt);
  \draw[flow] (st) -- (o) node[midway, right=1pt, font=\scriptsize, text=stone] {store};
\end{tikzpicture}
\end{document}
```

full $Q, K, V, O$ token matrices live down in the HBM lane while only the active $B_m \times d$ tile sits up in SRAM.

FlashAttention loads tiles up, runs the whole softmax on-chip, and touches HBM again only to store $O$ and stream the next $K, V$ block.

```jsx imports={Zoomable,FlashAttentionTiles}
<Zoomable label="FlashAttention tile streaming">
  <FlashAttentionTiles caption="Step through the tile loop: HBM holds the full $Q, K, V, O$ matrices while SRAM streams in one $(Q_i, K_j, V_j)$ tile pair at a time. Each step updates the running maxima $m_i$, normaliser $l_i$, and partial output $O_i$ via the online softmax recurrence." />
</Zoomable>
```

```jsx imports={Zoomable,FlashDataFlow}
<Zoomable label="FlashAttention data movement">
  <FlashDataFlow />
</Zoomable>
```

- Motivation: eliminate memory bandwidth bottlenecks so that longer contexts fit on commodity GPUs.
- Extension: variants such as FlashAttention-2/3, xFormers, and Triton kernels specialise for [[thoughts/GPU programming|GPU]] architectures and sparse layouts.

## Deriving the online softmax

The $m/\ell/O$ recurrence computes the same softmax as the batched form per tile, so it is beneficial to look into some of the derivation for the online softmax.

With numerically-stable (safe) softmax, for a logit vector $x \in \mathbb{R}^{V}$, exponentiating directly will result in {{sidenotes[overflows]: $e^{x_i}$ exceeds the FP16 ceiling ($\approx 65504$) once $x_i > 11$, and FP32 near $x_i \approx 88$.}}

Subtracting the row maximum $m_V = \max_k x_k$ forces every exponent to be non-positive, so each $e^{(\cdot)} \in (0,1]$:

$$
y_i = \frac{e^{x_i - m_V}}{\sum_{j=1}^{V} e^{x_j - m_V}}, \qquad m_V = \max_{k} x_k
$$

Computed this _requires_ three passes over $x$: one for the maximum $m_V$, one for the denominator $\ell_V = \sum_j e^{x_j - m_V}$, one to normalise. The denominator pass is blocked on the maximum pass, because $\ell_V$ subtracts the _global_ $m_V$, which is unknown until the first pass finishes.

Break it by tracking a _running_ maximum $m_j = \max(m_{j-1}, x_j)$ and defining a surrogate denominator that subtracts $m_j$ rather than $m_V$:

$$
\ell'_j := \sum_{k=1}^{j} e^{x_k - m_j} .
$$

At the end of the sequence $m_V = m_N$, so $\ell'_N = \ell_V$ exactly — the surrogate is a legal drop-in for the true denominator. And $\ell'_j$ admits a one-step recurrence. Split off the last term and factor $e^{-m_j}$ through the historical sum:

$$
\ell'_j = \underbrace{\Big(\sum_{k=1}^{j-1} e^{x_k - m_{j-1}}\Big)}_{=\,\ell'_{j-1}} \, e^{m_{j-1} - m_j} + e^{x_j - m_j}
       = \ell'_{j-1}\, e^{m_{j-1} - m_j} + e^{x_j - m_j} .
$$

The factor $e^{m_{j-1} - m_j}$ is the **correction factor**. When the running maximum holds ($m_j = m_{j-1}$) it is $e^{0}=1$ and the update is plain accumulation; when a new element raises the maximum, it is $< 1$ and retroactively re-bases every previously accumulated term from the old maximum to the new one in $O(1)$, without revisiting any of them. The identity that licenses this is $e^{a-b}e^{b-c} = e^{a-c}$.

Streaming equals batch at _every_ step, by induction on $j$. Base case $j=1$: $m_1 = x_1$ and $\ell'_1 = e^{x_1 - m_1} = 1$, the one-element sum. Inductive step, assuming $\ell'_{j-1} = \sum_{k=1}^{j-1} e^{x_k - m_{j-1}}$:

$$
\ell'_j = \Big(\sum_{k=1}^{j-1} e^{x_k - m_{j-1}}\Big)\,e^{m_{j-1}-m_j} + e^{x_j - m_j}
       = \sum_{k=1}^{j-1} e^{x_k - m_j} + e^{x_j - m_j}
       = \sum_{k=1}^{j} e^{x_k - m_j} .
$$

The correction factor folds into each exponent exactly. The bound $1 \le \ell'_j \le j$ means FP32 carries the denominator for vectors up to $\sim 10^{37}$ elements before it overflows. The pair $(m, \ell)$ is a mergeable summary: it forms a commutative monoid under $[m_a,\ell_a]\oplus[m_b,\ell_b] = [\max(m_a,m_b),\ \ell_a e^{m_a-\max}+\ell_b e^{m_b-\max}]$ with identity $[-\infty, 0]$, so the reduction parallelises across a tile (a warp-shuffle tree), not only down it. That associativity is precisely what makes softmax — and therefore attention — tileable.

One pass now suffices for the _statistics_, yet softmax still needs a second pass to emit $y_i = e^{x_i - m_V}/\ell_V$, because you cannot divide until $\ell_V$ is final. Attention escapes the second pass because its target is $O = \mathrm{softmax}(S)V$, not the probabilities themselves. Apply the surrogate trick a second time, now to the output accumulator. The exact row output $o_i = \sum_{k \le i} (e^{x_k - m_N}/\ell'_N)\,V_k$ depends on the global $m_N, \ell'_N$; define instead

$$
O'_i := \sum_{k=1}^{i} \frac{e^{x_k - m_i}}{\ell'_i}\, V_k , \qquad O'_N = o_N ,
$$

which carries the running statistics and is again exact at the endpoint. Its recurrence inherits the same correction factor on the running weighted sum of value rows:

$$
O'_i = O'_{i-1}\cdot \frac{\ell'_{i-1}\, e^{m_{i-1}-m_i}}{\ell'_i} + \frac{e^{x_i - m_i}}{\ell'_i}\, V_i .
$$

Now $m$, $\ell'$, and $O'$ all advance in a single fused pass. Promote the scalar $x_i$ to a tile of $B_n$ logits $S^{(t)}_{i:}$, take the tile's local maximum, and accumulate the $B_n$ exponentiated contributions at once. Writing the _unnormalised_ output $O^{\text{new}}_i = \ell^{\text{new}}_i \cdot O'_i$ so the $1/\ell$ division can wait until the loop ends, the recurrence collapses to exactly the $m/\ell/O$ update the note states:

$$
\begin{aligned}
m^{\text{new}}_i &= \max\big(m^{\text{old}}_i,\ \textstyle\max_j S^{(t)}_{ij}\big), \\
\ell^{\text{new}}_i &= e^{m^{\text{old}}_i - m^{\text{new}}_i}\,\ell^{\text{old}}_i + \textstyle\sum_j e^{S^{(t)}_{ij} - m^{\text{new}}_i}, \\
O^{\text{new}}_i &= e^{m^{\text{old}}_i - m^{\text{new}}_i}\,O^{\text{old}}_i + \textstyle\sum_j e^{S^{(t)}_{ij} - m^{\text{new}}_i}\,V^{(t)}_j ,
\end{aligned}
$$

with the single normalisation $O_i = O^{\text{new}}_i / \ell^{\text{new}}_i$ after all tiles. The online-softmax statistic is due to Milakov and Gimelshein [@milakov2018onlinenormalizercalculationsoftmax]; the second application to the output accumulator is the step that turns it into single-pass attention.

## IO complexity

The point of FlashAttention is an asymptotic reduction in HBM traffic, and the bound is tight.

Let $N$ be the sequence length, $d$ the head dimension, and $M$ the on-chip SRAM size in elements, in the regime $d \le M \le Nd$. Standard attention (materialise $S=QK^{\top}$, read it back for the softmax, write $P$, re-read for $PV$) moves

$$
\Theta\big(Nd + N^2\big)
$$

bytes through HBM, dominated by the $N^2$ term — the two $N\times N$ writes and reads of the score and probability matrices. FlashAttention never writes $S$ or $P$ to HBM and instead moves

$$
\#\text{HBM}_{\text{flash}} = \Theta\!\Big(\frac{N^2 d^2}{M}\Big) .
$$

The bound follows from the loop structure. Block sizes are set so the resident tiles fit on chip:

- $B_c = \lceil M/4d\rceil$ and $B_r = \min(\lceil M/4d\rceil, d)$
- the factor $4$ reserving room for the four simultaneously-live tiles $Q_i, K_j, V_j, O_i$ plus the $B_r \times B_c$ score tile.
- $K$ and $V$ are each loaded once.
- $Q$ and $O$ are re-streamed once per outer pass over the key blocks, and there are $T_c = \lceil N/B_c\rceil = \Theta(Nd/M)$ such passes.

Total traffic is therefore $\Theta(Nd \cdot T_c) = \Theta(N^2 d^2/M)$.

The ratio against standard attention is $d^2/M$, smaller than one whenever $d^2 < M$

since $d = 64$ gives $d^2 = 4096$ words against an A100's $M \approx 98\,000$ FP16 words, a factor $\approx 24$ on paper.

The measured reduction is up to $9\times$; the gap from $24\times$ is the suppressed constants and the additive $Nd$ term [@dao2022flashattentionfastmemoryefficientexact].

This is optimal in the regime. No exact-attention algorithm achieves $o(N^2 d^2 M^{-1})$ HBM accesses for all $M \in [d, Nd]$: at $M = \Theta(Nd)$ that would mean $o(Nd)$, impossible because the inputs alone occupy $Nd$ words and start in HBM.

> So FlashAttention's IO complexity is asymptotically tight, and every later generation (FA-2, FA-3, FA-4) improves constant factors and hardware utilisation, not the asymptotics.

wrt backward pass, it stores only $O$ and the per-row logsumexp ($O(N)$ memory) instead of $S$ or $P$, then recomputes the score and probability tiles in SRAM during the gradient pass.

The softmax-Jacobian quadratic term collapses to a per-row scalar

$$D_i = \mathrm{rowsum}(dO \circ O)$$

which gives $dS_{ij} = P_{ij}(dP_{ij} - D_i)$ block-locally, so the backward also moves $\Theta(N^2 d^2/M)$ versus the standard $\Theta(Nd + N^2)$.

Recomputation costs more FLOPs, and the kernel runs faster anyway, because attention at these lengths is bounded by HBM bandwidth.

> If the kernel were compute-bound, more FLOPs would be slower; it is faster, which is the direct evidence that bytes, not FLOPs, were the binding constraint.

We should treat the IO model as first-order, in a sense where growing the block size shrinks HBM traffic but eventually makes the kernel compute-bound; past that crossover, further IO reduction stops buying wall-clock time.

The forward kernel runs $K,V$ on the outer loop and $Q$ on the inner loop, so each output block $O_i$ with its statistics $m_i, \ell_i$ is read-modify-written $T_c$ times across HBM—the round-trip that FA-2 later removes by swapping the loop order.

The block sizes $B_c = \lceil M/4d\rceil$, $B_r = \min(\lceil M/4d\rceil, d)$ keep $Q_i, K_j, V_j, O_i$ and the $B_r \times B_c$ score tile co-resident in SRAM

## FlashAttention 2

FlashAttention-2 [@dao2023flashattention2fasterattentionbetter] an improvement from FA1 (I would argue that this was an evolution in design).

- It defers the $1/l_i$ rescaling to the end of the loop instead of every tile, cutting non-matmul FLOPs (which run $\approx 16\times$ slower than matmul on tensor-core hardware).
- It parallelises over the query-sequence dimension, _in addition to batch and heads_, so a single long sequence still saturates the SMs.
- It also partitions work inside a threadblock by splitting $Q$ across warps (split-Q) rather than $K$ (split-K):
  - each warp owns whole output rows and never round-trips partial sums through shared memory for a reduction.

Motivation: FA-1 reached only 30–50% of A100 FP16 peak on the forward pass and 25–35% on the backward, against 80–90% for a well-tuned GEMM, and the loss traced to thread-block and warp work partitioning.

The deferred $1/\ell$ rescaling rewrites the accumulator update to carry only the max-correction $\mathrm{diag}(e^{m^{(j-1)}-m^{(j)}})^{-1}$ inside the loop and apply the single division once at the end, which matters more on A100 given that a non-matmul FP32 FLOP costs $16\times$ a matmul FLOP (19.5 vs 312 TFLOP/s), so even a small count of softmax ops dominates if it sits in the hot path.

It stores one statistic per row, the logsumexp $L = m + \log\ell$, rather than the pair $(m, \ell)$.

Note on _causal masking_: it exploits the block structure via column-blocks entirely above the diagonal are skipped, roughly half the work at long $N$, for a $1.7$–$1.8\times$ gain, and the elementwise mask touches only the $\sim 1$ diagonal block per row.

The loop-order swap (Q outer, K/V inner) is credited to Phil Tillet's Triton fused-attention tutorial.

```tikz
\usepackage{tikz}
\begin{document}
\definecolor{salmon}{HTML}{FDB2A2}
\definecolor{sage}{HTML}{CDD597}
\definecolor{stone}{HTML}{6F6E69}
\definecolor{paper}{HTML}{FFFCF0}
\begin{tikzpicture}[
  font=\small, >=latex,
  wbox/.style={draw=stone, rounded corners=2pt, minimum width=1.0cm, minimum height=0.68cm, inner sep=2pt},
  warp/.style={wbox, fill=salmon!50},
  share/.style={wbox, fill=sage!55, minimum width=1.6cm},
  result/.style={wbox, fill=paper},
  cost/.style={draw=stone, fill=stone!25, rounded corners=2pt, minimum width=3.0cm, minimum height=0.7cm},
  a/.style={->, thick, draw=stone}
]
  \path[use as bounding box] (-0.6, -0.6) rectangle (13.4, 6.2);
  \draw[dashed, draw=stone!45] (6.4, -0.2) -- (6.4, 5.9);

  % left: FlashAttention-1 split-K needs a cross-warp reduction
  \node[text=stone] at (3.0, 5.7) {FA-1 split-K};
  \node[share] (q1) at (3.0, 4.7) {$Q$};
  \node[warp] (k0) at (0.9, 3.4) {$K_0$};
  \node[warp] (k1) at (2.3, 3.4) {$K_1$};
  \node[warp] (k2) at (3.7, 3.4) {$K_2$};
  \node[warp] (k3) at (5.1, 3.4) {$K_3$};
  \draw[a] (q1) -- (k0); \draw[a] (q1) -- (k1); \draw[a] (q1) -- (k2); \draw[a] (q1) -- (k3);
  \node[cost] (red) at (3.0, 1.9) {reduce};
  \draw[a] (k0) -- (red); \draw[a] (k1) -- (red); \draw[a] (k2) -- (red); \draw[a] (k3) -- (red);
  \node[result, fill=sage!55] (oL) at (3.0, 0.6) {$O$};
  \draw[a] (red) -- (oL);

  % right: FlashAttention-2 split-Q keeps warps independent
  \node[text=stone] at (10.0, 5.7) {FA-2 split-Q};
  \node[share] (kv) at (10.0, 4.7) {$K$, $V$};
  \node[warp] (q0) at (7.9, 3.4) {$Q_0$};
  \node[warp] (qq1) at (9.3, 3.4) {$Q_1$};
  \node[warp] (qq2) at (10.7, 3.4) {$Q_2$};
  \node[warp] (qq3) at (12.1, 3.4) {$Q_3$};
  \draw[a] (kv) -- (q0); \draw[a] (kv) -- (qq1); \draw[a] (kv) -- (qq2); \draw[a] (kv) -- (qq3);
  \node[result] (o0) at (7.9, 1.9) {$O_0$};
  \node[result] (o1) at (9.3, 1.9) {$O_1$};
  \node[result] (o2) at (10.7, 1.9) {$O_2$};
  \node[result] (o3) at (12.1, 1.9) {$O_3$};
  \draw[a] (q0) -- (o0); \draw[a] (qq1) -- (o1); \draw[a] (qq2) -- (o2); \draw[a] (qq3) -- (o3);
  \node[font=\scriptsize, text=stone] at (10.0, 0.6) {no reduce};
\end{tikzpicture}
\end{document}
```

## FlashAttention 3

FlashAttention-3 [@shah2024flashattention3fastaccurateattention] targets Hopper (H100) and is built around asynchrony.

Warps inside a CTA are specialised: producer warps do nothing but issue [[thoughts/GPU programming|TMA]] loads of $K, V$ tiles from HBM into shared memory, while consumer warps do nothing but run WGMMA matmuls on the tensor cores.

Because both run asynchronously, the kernel overlaps the load of tile $t{+}1$ with the compute of tile $t$, and ping-pong scheduling further hides the softmax of one warpgroup under the GEMM of another.

They also add FP8 with block quantisation and incoherent (Hadamard) processing for accuracy, and it reaches $\approx 740$ TFLOPs in FP16 (75% of H100 peak) and $\approx 1.2$ PFLOPs in FP8, $1.5$–$2\times$ over FlashAttention-2.

FA-3 leverages three kinds of Hopper asynchrony, on top of FlashAttention2:

- Warp specialisation splits a thread block into a producer warpgroup that issues only TMA async copies HBM$\to$SMEM and consumer warpgroups that run only WGMMA, with `setmaxnreg` handing registers from the register-light producer to the math-heavy consumers and a circular SMEM buffer decoupling load latency from compute.
- softmax running one-to-two orders of magnitude below tensor-core throughput is hidden two ways: inter-warpgroup ping-pong schedules one warpgroup's softmax under the other's GEMMs, and intra-warpgroup two-stage pipelining rotates GEMM0 ($S=QK^\top$ for the next block) against GEMM1 ($O\mathrel{+}=PV$ for the current block) so the current exp runs in the shadow of the next matmul.
- FP8 path adds per-block quantisation (one scale per $B_r\times d$ tile, the dominant accuracy lever) and incoherent processing — left/right-multiplying $Q,K$ by a random Hadamard-times-sign orthogonal $M$, which leaves $QK^\top$ exact yet spreads outliers, applied in $O(d\log d)$ via a fast Walsh–Hadamard transform.

FP16 forward reaches $\approx 740$ TFLOP/s (75% of the 989-TFLOP dense peak), FP8 close to $1.2$ PFLOP/s with RMSE $2.6\times$ below the per-tensor baseline ($9.1\!\times\!10^{-3}$).

> [!IMPORTANT]
>
> FP8 error is still $\sim 48\times$ the FP16 error

```tikz
\usepackage{tikz}
\begin{document}
\definecolor{salmon}{HTML}{FDB2A2}
\definecolor{sage}{HTML}{CDD597}
\definecolor{stone}{HTML}{6F6E69}
\definecolor{paper}{HTML}{FFFCF0}
\begin{tikzpicture}[
  font=\small, >=latex,
  prod/.style={draw=stone, fill=sage!55, rounded corners=2pt, minimum width=2.2cm, minimum height=0.8cm, align=center, inner sep=2pt},
  cons/.style={draw=stone, fill=salmon!55, rounded corners=2pt, minimum width=2.2cm, minimum height=0.8cm, align=center, inner sep=2pt},
  a/.style={->, thick, draw=stone}
]
  \path[use as bounding box] (-3.7, 0.2) rectangle (12.6, 4.6);

  \node[text=stone, anchor=east] at (-0.3, 3.7) {producer (TMA)};
  \node[prod] (p0) at (1.3, 3.7) {load $K_0$, $V_0$};
  \node[prod] (p1) at (4.4, 3.7) {load $K_1$, $V_1$};
  \node[prod] (p2) at (7.5, 3.7) {load $K_2$, $V_2$};

  \node[draw=stone, fill=paper, rounded corners=3pt, minimum width=9.4cm, minimum height=0.55cm] (smem) at (4.4, 2.4) {SMEM};

  \node[text=stone, anchor=east] at (-0.3, 1.1) {consumer (WGMMA)};
  \node[cons] (c0) at (2.85, 1.1) {compute $t_0$};
  \node[cons] (c1) at (5.95, 1.1) {compute $t_1$};
  \node[cons] (c2) at (9.05, 1.1) {compute $t_2$};

  \draw[a] (p0) -- (smem.north -| p0);
  \draw[a] (smem.south -| c0) -- (c0);
  \draw[a] (p1) -- (smem.north -| p1);
  \draw[a] (smem.south -| c1) -- (c1);
  \draw[a] (p2) -- (smem.north -| p2);
  \draw[a] (smem.south -| c2) -- (c2);
\end{tikzpicture}
\end{document}
```

## FlashAttention 4

FlashAttention-4 [@dao2024flashattention4] is the Blackwell (B200) kernel.

Blackwell's MMA is fully asynchronous, so the pipeline splits into scheduler warps that drive async loads and matmul dispatch and compute warps that run softmax, with one tile's matmuls overlapping the next tile's softmax.

> rather than route exponentials through the special-function unit (SFU), it emulates $\exp$ with a polynomial on the FMA units, moving the softmax bottleneck onto the general-purpose compute Blackwell.
>
> It hits $\approx 1605$ TFLOPs in BF16 (71% utilisation), $\approx 2.7\times$ over the Triton kernel.

It currently runs BF16 only; FP4 and 2-CTA matmuls, Blackwell's headline features, are not yet used.

The design premise is asymmetric hardware scaling

- from H100 to B200, BF16 tensor-core throughput grows $\approx 2.25\times$ (1 to 2.25 PFLOP/s) while per-SM special-function-unit count and shared-memory bandwidth stay flat, so the matmul stops being the bottleneck and the SFU-bound exp (forward) and SMEM-bound data movement (backward) become binding.
- The polynomial-exp trick computes $2^x = 2^n \cdot 2^f$ by Cody–Waite range reduction—$n$ is a free exponent-field shift, $2^f$ on $[0,1)$ is a cubic in Horner form, exactly three FMAs (Sollya coefficients $p_0=1.0,\ p_1\!\approx\!0.6951,\ p_2\!\approx\!0.2276,\ p_3\!\approx\!0.0771$)—run on the abundant FP32$\times$2 FMA pipes _in parallel_ with the hardware `MUFU.EX2`, summing their throughput while matching SFU output for BF16.
- Conditional online-softmax rescaling gates the $O$-accumulator correction on $m_j - m_{j-1} > \tau$
  - accumulating against the stale maximum otherwise, which the Hot Chips talk reports cuts corrections $\approx 10\times$
  - a dedicated correction warpgroup performs the deferred rescale off the critical path.
    - The forward kernel is a deep warp-specialised pipeline on Blackwell's fully asynchronous `tcgen05.mma` (issued by a single leader thread, `cta_group::1`), with accumulators $S, P, O$ in tensor memory (TMEM, 256 KB/SM);
    - the backward adopts `cta_group::2`, a $256\times256\times16$ tile split across a CTA pair that halves operand-B SMEM traffic and the $dQ$ atomic reductions.
      - It sustains $\approx 1613$ TFLOP/s in BF16 (71% of the 2.25-PFLOP peak), the first attention kernel past a petaflop on a single GPU, $1.1$–$1.3\times$ over cuDNN 9.13 and $2.1$–$2.7\times$ over Triton.
- FP4 is deliberately unused throughout — it would only speed the matmul, which is no longer the constraint.

```tikz
\usepackage{tikz}
\begin{document}
\definecolor{salmon}{HTML}{FDB2A2}
\definecolor{sage}{HTML}{CDD597}
\definecolor{stone}{HTML}{6F6E69}
\definecolor{paper}{HTML}{FFFCF0}
\begin{tikzpicture}[
  font=\small, >=latex,
  sch/.style={draw=stone, fill=paper, rounded corners=2pt, minimum width=1.9cm, minimum height=0.72cm, align=center, inner sep=2pt},
  mma/.style={draw=stone, fill=salmon!55, rounded corners=2pt, minimum width=1.9cm, minimum height=0.72cm, align=center, inner sep=2pt},
  fma/.style={draw=stone, fill=sage!55, rounded corners=2pt, minimum width=1.9cm, minimum height=0.72cm, align=center, inner sep=2pt},
  a/.style={->, thick, draw=stone}
]
  \path[use as bounding box] (-3.8, 0.3) rectangle (13.4, 5.2);

  \node[text=stone, anchor=east] at (-0.3, 4.0) {scheduler};
  \node[sch] (s0) at (1.2, 4.0) {dispatch};
  \node[sch] (s1) at (3.4, 4.0) {dispatch};
  \node[sch] (s2) at (5.6, 4.0) {dispatch};
  \node[sch] (s3) at (7.8, 4.0) {dispatch};

  \node[text=stone, anchor=east] at (-0.3, 2.6) {tensor cores};
  \node[mma] (m0) at (1.2, 2.6) {MMA $t_0$};
  \node[mma] (m1) at (3.4, 2.6) {MMA $t_1$};
  \node[mma] (m2) at (5.6, 2.6) {MMA $t_2$};
  \node[mma] (m3) at (7.8, 2.6) {MMA $t_3$};

  \node[text=stone, anchor=east] at (-0.3, 1.2) {CUDA cores};
  \node[fma] (f0) at (2.3, 1.2) {softmax $t_0$};
  \node[fma] (f1) at (4.5, 1.2) {softmax $t_1$};
  \node[fma] (f2) at (6.7, 1.2) {softmax $t_2$};

  \node[draw=stone, fill=salmon!28, rounded corners=3pt, align=center, minimum width=2.6cm] (poly) at (11.2, 2.6) {poly-exp\\on FMA};
  \draw[a, dashed] (f2) to[bend right=12] (poly);
\end{tikzpicture}
\end{document}
```

## Throughput across generations

```tikz
\usepackage{tikz}
\begin{document}
\definecolor{salmon}{HTML}{FDB2A2}
\definecolor{sage}{HTML}{CDD597}
\definecolor{stone}{HTML}{6F6E69}
\definecolor{paper}{HTML}{FFFCF0}
\begin{tikzpicture}[font=\small, >=latex]
  \path[use as bounding box] (-1.7, -1.5) rectangle (9.0, 6.6);
  \draw[draw=stone] (0,0) -- (0,5.9);
  \draw[draw=stone] (0,0) -- (8.7,0);
  \node[font=\scriptsize, text=stone, rotate=90, anchor=south] at (-1.15, 2.95) {TFLOP/s};
  \draw[draw=stone!45] (-0.08,1.705) -- (0,1.705); \node[font=\scriptsize, text=stone, anchor=east] at (-0.14,1.705) {500};
  \draw[draw=stone!45] (-0.08,3.41) -- (0,3.41); \node[font=\scriptsize, text=stone, anchor=east] at (-0.14,3.41) {1000};
  \draw[draw=stone!45] (-0.08,5.115) -- (0,5.115); \node[font=\scriptsize, text=stone, anchor=east] at (-0.14,5.115) {1500};
  \node[font=\scriptsize, text=stone, anchor=east] at (-0.14,0) {0};
  \fill[sage!70, draw=stone] (0.35,0) rectangle (1.45,0.341);
  \fill[sage!70, draw=stone] (2.05,0) rectangle (3.15,0.784);
  \fill[salmon!65, draw=stone] (3.75,0) rectangle (4.85,2.524);
  \fill[salmon!65, draw=stone] (5.45,0) rectangle (6.55,4.092);
  \fill[stone!40, draw=stone] (7.15,0) rectangle (8.25,5.5);
  \node[font=\scriptsize, text=stone] at (0.9,0.62) {100};
  \node[font=\scriptsize, text=stone] at (2.6,1.06) {230};
  \node[font=\scriptsize, text=stone] at (4.3,2.80) {740};
  \node[font=\scriptsize, text=stone] at (6.0,4.37) {1200};
  \node[font=\scriptsize, text=stone] at (7.7,5.78) {1613};
  \node[font=\scriptsize, text=stone] at (0.9,-0.32) {FA-1};
  \node[font=\scriptsize, text=stone] at (2.6,-0.32) {FA-2};
  \node[font=\scriptsize, text=stone] at (4.3,-0.32) {FA-3};
  \node[font=\scriptsize, text=stone] at (6.0,-0.32) {FA-3};
  \node[font=\scriptsize, text=stone] at (7.7,-0.32) {FA-4};
  \node[font=\scriptsize, text=stone] at (0.9,-0.72) {FP16};
  \node[font=\scriptsize, text=stone] at (2.6,-0.72) {FP16};
  \node[font=\scriptsize, text=stone] at (4.3,-0.72) {FP16};
  \node[font=\scriptsize, text=stone] at (6.0,-0.72) {FP8};
  \node[font=\scriptsize, text=stone] at (7.7,-0.72) {BF16};
  \fill[sage!70, draw=stone] (0.2,5.6) rectangle (0.5,5.85); \node[font=\scriptsize, text=stone, anchor=west] at (0.55,5.72) {A100};
  \fill[salmon!65, draw=stone] (1.7,5.6) rectangle (2.0,5.85); \node[font=\scriptsize, text=stone, anchor=west] at (2.05,5.72) {H100};
  \fill[stone!40, draw=stone] (3.2,5.6) rectangle (3.5,5.85); \node[font=\scriptsize, text=stone, anchor=west] at (3.55,5.72) {B200};
\end{tikzpicture}
\end{document}
```
