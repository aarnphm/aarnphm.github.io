---
abstract: Attention reads a sequence through input-dependent weighted sums. Multiple heads keep separate reading patterns; faster kernels, sparse patterns, and smaller KV caches address different costs.
date: '2024-02-07'
description: Query-key matching, value mixing, multiple heads, and the costs of attending to a sequence.
id: Attention
modified: 2026-10-14 09:10:00 GMT-04:00
seealso:
  - '[[lectures/2/convexity|emperical finding]]'
  - '[[lectures/2/afp|attention from first principles]]'
socials:
  efficient: https://www.youtube.com/watch?v=Y-o545eYjXM
tags:
  - technical
  - llm
  - ml
title: Attention
transclude:
  title: false
---

Attention computes a weighted sum of values. A query determines the weights by matching against keys. For scaled dot-product attention [@vaswani2023attentionneed],

$$
\begin{aligned}
P &= \operatorname{softmax}_{\text{row}}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right),\\
A(Q,K,V) &= PV,\\
Q &\in \mathbb{R}^{L_q\times d_k},\quad K\in\mathbb{R}^{L_k\times d_k},\quad V\in\mathbb{R}^{L_k\times d_v}.
\end{aligned}
$$

$P$ is the attention matrix; $A$ contains the output vectors. Each row of $P$ sums to one before attention dropout. The mask $M$ is zero for allowed connections and $-\infty$ for forbidden ones. Each query needs at least one allowed key. In causal self-attention, position $i$ can read positions $j\leq i$.

The query/key/value split follows the operation:

- $Q$: what am I looking for?
- $K$: what can each position be matched on?
- $V$: what does that position contribute if I read it?

For one query, weights $(0.7,0.2,0.1)$ produce $0.7v_1+0.2v_2+0.1v_3$. The query-key match determines how much of a value gets through; the value determines what gets through. With independent, zero-mean, unit-variance query and key coordinates, the unscaled dot product has variance $d_k$. Dividing by $\sqrt{d_k}$ keeps that variance at one under those assumptions, which helps avoid an already saturated softmax at initialization.

Attention predates the Transformer. Bahdanau, Cho and Bengio's 2014 translation model learned a soft alignment over encoder states [@bahdanau2016neuralmachinetranslationjointly]. Vaswani et al. introduced the Transformer in 2017, using attention without recurrent or convolutional sequence layers. Its self-attention gives tokens direct paths to other allowed tokens and permits parallel computation across a known input sequence. Autoregressive generation still produces tokens sequentially. Position encodings supply order information that bare, unmasked self-attention lacks [@vaswani2023attentionneed].

> [!note]+ equivalent
>
> A head's contribution to the [[thoughts/mechanistic interpretability#residual stream|residual stream]] can be written as [@elhage2021mathematical]:
>
> $$
> \operatorname{Attn}^{l,h}_i(X^{l-1}) = \sum_{j\leq i} a^{l,h}_{i,j}\,x^{l-1}_j W_V^{l,h}W_O^{l,h}.
> $$
>
> Here $W_V^{l,h}\in\mathbb{R}^{d_{\text{model}}\times d_h}$ and $W_O^{l,h}\in\mathbb{R}^{d_h\times d_{\text{model}}}$. Their product is the head's OV matrix. The QK computation chooses where to read, and OV transforms what is written back. This expression omits biases and treats $X^{l-1}$ as the input to attention, including any preceding normalization. [[thoughts/induction heads|Induction heads]] are one learned use of these operations.

```jsx imports={Zoomable,AttentionCircuits}
<Zoomable label="QK/OV circuit decomposition">
  <AttentionCircuits caption="The same attention layer two ways: textbook softmax on the left, the Anthropic circuit decomposition on the right. Hover a circuit name to see which weights it covers: $QK$ sets where to read, $OV$ sets what to write." />
</Zoomable>
```

## Multi-head Attention

A head has its own query, key and value projections. Its softmax gives it a separate distribution over positions. With equal key and value width $d_h$ per head,

$$
\begin{aligned}
\operatorname{MHA}(Q,K,V) &= \operatorname{concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W_O,\\
\operatorname{head}_r &= \operatorname{softmax}_{\text{row}}\!\left(\frac{(QW_{Q,r})(KW_{K,r})^\top}{\sqrt{d_h}}+M\right)VW_{V,r},\\
W_{Q,r},W_{K,r},W_{V,r} &\in\mathbb{R}^{d_{\text{model}}\times d_h},\qquad W_O\in\mathbb{R}^{hd_h\times d_{\text{model}}}.
\end{aligned}
$$

Here $Q$, $K$ and $V$ denote the inputs before the head projections. In self-attention they come from the same sequence; cross-attention takes queries from one sequence and keys/values from another. The output projection mixes the concatenated head outputs before they enter the [[thoughts/Transformers|transformer]] residual stream.

```jsx imports={Zoomable,MultiHeadAttention}
<Zoomable label="multi-head attention diagram">
  <MultiHeadAttention
    caption="Slide $h$ to divide the fixed $d_m$ width across heads. Each head has its own softmax distribution; the output projection combines their value mixtures."
    heads={4}
  />
</Zoomable>
```

> [!motivation]+ why split the model into heads?
>
> Suppose one output needs information from the preceding token and from a distant noun. Two heads can select those positions separately, then write into different output directions. A single head applies the same positional weights to every value coordinate.
>
> At fixed total width $hd_h=d_{\text{model}}$, splitting into heads preserves the leading matrix-multiplication cost. Each head has less width available to represent its query-key comparisons. Whether a head learns a useful pattern is an empirical question: pruning studies find both specialized heads and heads that can be removed with little effect in the tested models [@voita2019analyzingmha; @michel2019sixteenheads].

> [!example]- two offset heads vs one big head
>
> Consider bidirectional attention at the middle of a three-token sequence. One head favours the next token; another favours the previous token. For $\beta>0$, use finite score rows
>
> $$
> s_+=(0,0,\beta),\qquad s_-=(\beta,0,0),\qquad p_\pm=\operatorname{softmax}(s_\pm).
> $$
>
> Increasing $\beta$ sharpens each distribution. For scalar values $v\in\mathbb{R}^3$, keep the two weighted sums in separate output coordinates:
>
> $$
> y(v)=(p_+v,\ p_-v).
> $$
>
> Adding the score rows first gives $p=\operatorname{softmax}(\beta,0,\beta)$. Its first and last weights are equal. Values $v=(1,0,0)^\top$ and $v'=(0,0,1)^\top$ therefore yield the same single-head mixture $pv=pv'$, while the two-head outputs differ. Once that mixture has discarded which neighbour supplied the value, a post-projection cannot recover it.
>
> ```python shell
> from math import exp, isclose, isfinite
>
> def softmax(scores: list[float]) -> list[float]:
>     largest = max(scores)
>     weights = [exp(score - largest) for score in scores]
>     total = sum(weights)
>     return [weight / total for weight in weights]
>
> def dot(weights: list[float], values: list[float]) -> float:
>     return sum(weight * value for weight, value in zip(weights, values, strict=True))
>
> beta = 10.0  # Larger inverse temperature makes the distribution sharper.
> p_next = softmax([0.0, 0.0, beta])
> p_prev = softmax([beta, 0.0, 0.0])
> p_single = softmax([beta, 0.0, beta])
> left, right = [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]
> y_left = (dot(p_next, left), dot(p_prev, left))
> y_right = (dot(p_next, right), dot(p_prev, right))
>
> assert all(isfinite(x) for row in (p_next, p_prev, p_single) for x in row)
> assert all(isclose(sum(row), 1.0) for row in (p_next, p_prev, p_single))
> assert isclose(dot(p_single, left), dot(p_single, right))
> assert not isclose(y_left[0], y_right[0])
> print("two heads:", y_left, y_right)
> print("one mixture:", dot(p_single, left), dot(p_single, right))
> ```
>
> This is an information-loss example with fixed position-based scores. The next-token connection is allowed because the example is bidirectional; a causal decoder would mask it.

```jsx imports={Zoomable,OffsetHeadsToy}
<Zoomable label="two-head softmax demo">
  <OffsetHeadsToy
    caption="Finite offset scores give two distributions $P^{(+1)}$ and $P^{(-1)}$; a missing boundary neighbour leaves that head's row uniform. The surrogate uses $P=\operatorname{softmax}(S^{(+1)}+S^{(-1)})$. Independent mode samples the output projections. Forced mode fits a ridge-regularized projection to this one sampled $V$, so its residual is a numerical example, with no guarantee about every possible projection or input."
    length={6}
  />
</Zoomable>
```

> [!math] softmax factorisation barrier
>
> The example also shows why an arbitrary fixed single-head routing cannot always replace two heads. Hold two distinct probability rows $p_+$ and $p_-$ fixed, vary scalar values $v$, and require the output $(p_+v,p_-v)$. A single head with probability row $p$ and a linear value/output map $b=(b_1,b_2)$ gives $(pv)b$.
>
> Equality for every $v$ requires
>
> $$
> p_+=b_1p,\qquad p_-=b_2p.
> $$
>
> Summing the coordinates of each probability row gives $b_1=b_2=1$, which would require $p_+=p_-$. That contradicts the chosen rows. The proof concerns one attention operation, fixed routing and linear value/output maps. Additional layers or input-dependent changes to the routing need their own argument. Softmax's non-additivity alone does not establish a general expressivity theorem.

> [!note] Parameter and compute at fixed $d_{\text{model}}$
>
> For ordinary multi-head self-attention with $hd_h=d_{\text{model}}$, ignoring biases:
>
> - **Parameters:** the four packed projections contain $4d_{\text{model}}^2$ weights.
> - **Projection work:** $\Theta(Ld_{\text{model}}^2)$ for a sequence of length $L$.
> - **Attention work:** $\Theta(L^2d_{\text{model}})$ for the score and weighted-value products. A naive implementation also materializes $hL^2$ attention entries. Head count changes softmax work, memory use and kernel efficiency even when the leading matrix-multiplication count stays fixed.
> - **KV cache:** $2d_{\text{model}}$ values per stored token per layer. With $h_{\text{kv}}$ key/value heads, [[thoughts/GQA|GQA]] uses $2h_{\text{kv}}d_h$ values; multi-query attention sets $h_{\text{kv}}=1$ [@ainslie2023gqatraininggeneralizedmultiquery; @shazeer2019fasttransformerdecodingwritehead].
>
> For example, $d_{\text{model}}=4096$ and $h=32$ give $d_h=128$. Ordinary MHA stores $8192$ KV values per token per layer. Eight KV heads store $2048$, one quarter as many. Changing only the query-head count while keeping the total KV width fixed saves no cache space.

```jsx imports={Zoomable,AttentionCostCalculator}
<Zoomable label="attention cost calculator">
  <AttentionCostCalculator caption="This calculator assumes ordinary MHA: $h_{\text{kv}}=h$ and $hd_h=d_{\text{model}}$. For batch $B$, $N$ layers and $L$ cached tokens per sequence, raw KV storage is $2BLNd_{\text{model}}$ times the bytes per value. Lower precision reduces this payload; quantization scales and allocator overhead are excluded. Changing $h$ alone preserves the parameter and cache counts shown here." />
</Zoomable>
```

## optimization

Full attention computes a score for every allowed query-key pair. There are several distinct costs to change: how many pairs exist, how those pairs are computed, how many KV values are stored, and how work is divided across devices. A speedup claim needs to say which cost fell and under what workload.

### Exact, IO-aware kernels

- [[thoughts/flash attention|FlashAttention]] uses tiling and online softmax to avoid writing the full attention matrix to GPU high-bandwidth memory. Tiles can be read more than once; the saving comes from reducing intermediate memory traffic. FlashAttention-2 changes work partitioning. FlashAttention-3 overlaps asynchronous operations on Hopper and includes a lower-precision FP8 path. Here _exact_ means keeping the dense attention operation, subject to floating-point error [@dao2022flashattentionfastmemoryefficientexact; @dao2023flashattention2fasterattentionbetter; @shah2024flashattention3fastaccurateattention].
- [Flash-Decoding](https://pytorch.org/blog/flash-decoding/) splits the cached keys and values into chunks, computes partial attention in parallel and merges the results using their log-sum-exp normalizers. One new query attending to $L$ cached tokens takes $\Theta(L)$ attention work at fixed width. [FlashDecoding++](https://arxiv.org/abs/2311.01282) additionally targets softmax synchronization, small-batch matrix multiplication and kernel selection.

### Sparse/local attention

- Longformer uses a sliding window and selected global tokens. With window width $w$ and $g$ global tokens, attention work scales as $O(L(w+g))$ at fixed width. Linear scaling in $L$ requires bounded $w$ and $g$ [@beltagy2020longformerlongdocumenttransformer].
- BigBird combines local, random and global connections. Fixed connection budgets give linear scaling. Its universal-approximation and Turing-completeness results apply under the paper's assumptions; those results do not promise that a sparse model matches a dense model on a particular task [@zaheer2021bigbirdtransformerslonger].

### Linear/approximate attention

- Reformer hashes queries and keys into buckets so attention can concentrate on likely matches. Its attention cost is $O(L\log L)$ under the paper's hashing/bucket assumptions. Reversible layers separately reduce activation storage [@kitaev2020reformerefficienttransformer].
- Linformer projects keys and values along the sequence dimension to a smaller dimension $k$. The resulting attention work is linear in $L$ for fixed $k$; the projection restricts the representation [@wang2020linformerselfattentionlinearcomplexity].
- Performer approximates the softmax kernel with FAVOR+ random features. Increasing the feature count changes the approximation cost and quality [@choromanski2022rethinkingattentionperformers].
- Nyströmformer constructs an approximation using $m$ landmark representations and a small matrix pseudoinverse. A fixed landmark count permits linear scaling with sequence length; the landmark computation remains part of the cost [@xiong2021nystromformernystrombasedalgorithmapproximating].

### Multi-device and prefix-aware inference

- [[thoughts/ring attention|Ring Attention]] partitions a sequence across devices and circulates KV blocks. Striped Attention redistributes token positions to balance the triangular workload of causal attention. The total dense-attention arithmetic remains quadratic in sequence length [@liu2023ringattentionblockwisetransformers; @brandon2023stripedattentionfasterring].
- [[thoughts/cascade attention|Cascade]] separates shared-prefix attention from request-specific suffix attention, then merges their attention states. A batch can reuse prefix KV loads across its queries. The [FlashInfer description](https://flashinfer.ai/2024/02/02/cascade-inference.html) derives this merge; SGLang's RadixAttention handles reuse of cached prefixes across requests [@zheng2024sglangefficientexecutionstructured].
- [[thoughts/tree attention|Tree Attention]] uses a tree reduction to combine partial attention results across devices during decoding. The tree describes the communication pattern; it does not require shared prompt prefixes [@shyam2025treeattentiontopologyawaredecoding].

## cheatsheet

The costs below concern attention interactions at fixed model width, with projection layers excluded. Prefill processes all $L$ input positions; cached decoding processes one new query against $L$ stored positions. Cache capacity is listed separately because it measures storage.

| Method              | Scope                  | Sequence-dependent work or storage             | Condition                                         |
| ------------------- | ---------------------- | ---------------------------------------------- | ------------------------------------------------- |
| FlashAttention-3    | dense prefill          | $O(L^2)$ work                                  | keeps dense pairs; changes GPU execution          |
| Flash-Decoding / ++ | cached decode          | $O(L)$ work per query                          | fixed head widths and counts                      |
| Longformer          | sparse attention       | $O(L(w+g))$ work                               | window $w$, global-token count $g$                |
| BigBird             | sparse attention       | $O(L)$ work                                    | fixed local, random and global budgets            |
| Reformer            | hashed attention       | $O(L\log L)$ work                              | bounded hash rounds and bucket work               |
| Linformer           | projected attention    | $O(Lk)$ work                                   | projected sequence dimension $k$                  |
| Performer           | kernel approximation   | $O(Lr)$ work                                   | random-feature count $r$                          |
| Nyströmformer       | landmark approximation | $O(Lm^2+m^3)$ work                             | $m$ landmarks; includes small-matrix work         |
| MQA/GQA             | KV storage             | $2h_{\text{kv}}d_h$ values per token per layer | share KV heads across query heads                 |
| Ring/Striped        | distributed prefill    | $O(L^2)$ total work                            | spread across devices; communication also matters |
| Cascade             | shared-prefix decode   | $O(L)$ work per query                          | reuse prefix reads across a batch                 |
| Tree Attention      | distributed decode     | $O(L)$ total work per query                    | tree reduction adds communication                 |
