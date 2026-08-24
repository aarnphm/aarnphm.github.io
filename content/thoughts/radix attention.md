---
date: '2026-05-27'
description: prefix-tree caching of KV pages with LRU eviction, shared request prefixes reuse computed K,V.
id: attention-radix
modified: 2026-06-06 01:39:05 GMT-04:00
seealso:
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/Radix tree|Radix tree]]'
  - '[[thoughts/paged attention|Paged Attention]]'
  - '[[thoughts/KV compression|KV compression]]'
tags:
  - ml
  - llm
  - technical
title: Radix Attention
---

RadixAttention [@zheng2024sglangefficientexecutionstructured] keeps relevant [[thoughts/KV compression|KV cache]] entries for all requests in a [[thoughts/Radix tree|radix tree]], with LRU eviction for cold leaves. SGLang implements it in [sgl-project/sglang](https://github.com/sgl-project/sglang).

Every request has tokens $t_{1:m}$. Let $h$ be the length of the deepest cached prefix. During decoding, the runtime reuses that prefix and computes only the missing tail:

$$
\operatorname{KV}(t_{1:m}) =
\operatorname{KV}_{\mathrm{cache}}(t_{1:h}) \;\Vert\;
\operatorname{KV}_{\mathrm{new}}(t_{h+1:m}).
$$

The LRU policy keeps active prefixes resident on the GPU and frees the coldest leaf pages. Host or file backing belongs to SGLang's optional HiCache layer, not to the base RadixAttention algorithm.

> [!example] request routing pseudo-code
>
> ```python
> def serve(request):
>     prefix, suffix = split_prefix_suffix(request.tokens)
>     node = radix.longest_prefix(prefix)
>     kv_pages = node.cached_kv()
>     for token in suffix:
>         logits, kv_new = transformer.step(token, kv_pages)
>         kv_pages.push(kv_new)
>     radix.touch(node)        # update lru state
>     radix.insert(prefix+suffix, kv_pages)
> ```
>
> Each insert or touch updates the LRU ordering so hot conversations stay resident while cold leaf pages are freed.

```jsx imports={Zoomable,RadixPrefixTree}
<Zoomable label="radix prefix tree">
  <RadixPrefixTree caption="Click a request to walk the deepest matching prefix (coral) and watch newly cached tokens append (olive). Prompt #4 falls back to root and evicts the coldest branch under LRU; the readout reports the per-request hit rate $H = 1 - C / \sum |r|$." />
</Zoomable>
```

_dynamic evolution of the radix tree in response to various requests._

## cache-aware scheduling

We define the hit rate as

$$
\begin{aligned}
\text{hit rate} &= \frac{\sum_{r \in R} \text{number of cached prefill tokens in } r}{\sum_{r \in R} \text{number of prefill tokens in } r} \\[8pt]
&=1 - \frac{C}{\sum_{r \in R} \text{number of prefill tokens}}
\end{aligned}
$$

_in batch settings: sort requests by matching prefix length and prioritise requests with longer matched prefixes instead of FIFO._

```pseudo lineNumber=false
\begin{algorithm}
\caption{Cache-Aware Scheduling}
\begin{algorithmic}
\State \textbf{Input:} Radix tree $T$, Memory pool $P$.
\State \textbf{Input:} current running batch $B$, waiting queue $Q$.
\State \textbf{Output:} Finished requests and updated system state.
\State // Get all requests from the waiting queue
\State requests $\gets Q.\text{get\_all\_requests}()$
\State // Search for prefix matching for all waiting request
\For{req $\in$ requests}
    \State req.prefix\_node, req.prefix\_len $\gets$ T.match\_prefix(req.input\_tokens)
\EndFor
\State // Sort the request according to matched prefix lengths
\State requests.sort()
\State // Select requests for the next batch
\State available\_size $\gets$ T.evictable\_size() + P.available\_size()
\State current\_size $\gets$ 0
\State new\_batch $\gets$ []
\For{req $\in$ requests}
    \If{req.size() + current\_size $\le$ available\_size}
        \State new\_batch.append(req)
        \State $\delta \gets T.\text{increase\_ref\_counter}(req.\text{prefix\_node})$
        \State available\_size $\gets$ available\_size + $\delta$
    \EndIf
\EndFor
\State Q.remove\_requests(new\_batch)
\State // Insert requests into the current running batch
\State B.merge(new\_batch)
\State // Allocate new memory and do eviction if necessary
\State needed\_size $\gets$ B.needed\_size()
\State success, buffer $\gets$ P.alloc(needed\_size)
\If{$\neg \text{success}$}
    \State T.evict(needed\_size)
    \State success, buffer $\gets$ P.alloc(needed\_size)
\EndIf
\State B.run(buffer)
\State // Process finished requests
\State finished\_requests $\gets$ B.drop\_finished\_requests()
\For{req $\in$ finished\_requests}
    \State T.decrease\_ref\_counter(req.prefix\_node)
    \State T.insert(req)
\EndFor
\State \Return finished\_requests
\end{algorithmic}
\end{algorithm}
```

The lower bound for prefill compute is the number of unique token edges in the radix tree:

$$
C \ge \sum_{e \in \operatorname{edges}(T)} |e|.
$$

A DFS schedule reaches this bound when the cache can hold the longest request path. Once an edge $e$ has been computed, every request in the subtree under $e$ can reuse that prefix while the schedule stays inside the subtree. Descendant edges still need their own first computation; no edge is recomputed.

> [!important] cache hit
>
> with cache size $\ge$ the maximum request length, equal to the longest path in the radix tree, edge $e$ **WILL NOT** be evicted during computation of its subtree
> since the common prefix including $e$ of the subtree will be continuously hit.

We can show that longest-shared-prefix-first order is equivalent to DFS order by induction [^proof]

[^proof]: _base_: a random request correspond to node $x \in T$ will be processed.

    - All requests correspond to nodes $\{v_{1}, \ldots, v_{n}\}$ on path $x \gets \text{root}$ doesn't need recomputation.
    - Thus, computation complexity for requests of nodes $\{v_{1}, \ldots, v_{n}, x\}$ is aligned with DFS

    _induction_: assume we visit node $y \in T$, and the visited node align with DFS order. Let $P$ denote _path of_ $y \gets \text{root}$.

    - Each node that has not been visited has the lowest common ancestor with visited nodes on $P$.
    - Since nodes on $P$ are cached, a node $z$ that has yet to be visited with lowest common ancestor on $P$ will have the _longest shared prefix_
    - longest-shared-prefix-first order will select $z$, which is a valid DFS
      $\boxed{}$
