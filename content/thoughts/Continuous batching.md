---
date: '2024-02-08'
description: iteration-level scheduling for LLM serving, why it needs selective batching, and why it leaves KV-cache fragmentation to paging.
id: Continuous batching
modified: 2026-10-13 11:05:00 GMT-04:00
seealso:
  - '[[thoughts/vllm|vLLM]]'
  - '[[thoughts/paged attention|PagedAttention]]'
tags:
  - ml
  - inference
title: Continuous batching
transclude:
  title: false
---

[@280922] replaces request-level batching with iteration-level scheduling: the server rebuilds the batch after every decoding step, so a finished request leaves and a queued one joins without waiting for the rest of its batch. [^paper]

[^paper]:
    The [paper](https://www.usenix.org/conference/osdi22/presentation/yu) and [presentation](https://www.youtube.com/watch?v=Ob9PPLxETYU&ab_channel=USENIX) for the paper. Most notable open source implementation is [[thoughts/vllm|vLLM]].

    p/s: Actually, I think first implemented in [huggingface/tgi](https://github.com/huggingface/text-generation-inference)

## static batching

the naive approach is to form a batch of $N$ requests and run it until **ALL** of them complete. A request that finishes early keeps its slot and produces padding until the longest one finishes.

```
time →    t1  t2  t3  t4  t5  t6  t7  t8 │ t9  t10 ...
         ╔═══════════════════════════════╗
 seq A:  ║ ●   ●   ●   ●   ·   ·   ·   · ║  (waiting in queue)
 seq B:  ║ ●   ●   ●   ●   ●   ●   ●   ● ║  (waiting in queue)
 seq C:  ║ ●   ●   ●   ●   ●   ●   ·   · ║  (waiting in queue)
         ╚═══════════════════════════════╝
         └────── batch 1 locked ─────────┘

 ● = generate token    · = padding (wasted compute)
```

With output lengths $\ell_1,\dots,\ell_N$, the batch runs $N\max_i \ell_i$ token slots and only $\sum_i \ell_i$ of them are useful. Above, $3\cdot 8-(4+8+6)=6$ of $24$ slots are padding, so $25\%$. The waste grows with the spread of output lengths, and a new request also waits for the whole batch even when a slot is idle.

## iteration-level scheduling

_after EACH forward pass (one "iteration"), check which sequences emitted EOS or hit their length limit. Evict them and admit new requests from the queue._

```
time →   t1  t2  t3  t4  t5  t6  t7  t8  t9  t10
         ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
slot 1:  │ A   A   A   A │ D   D   D   D   D │ F ...
slot 2:  │ B   B   B   B   B   B   B   B │ E   E ...
slot 3:  │ C   C   C   C   C   C │ ← (queue empty)
                         ↑       ↑
                         │       └─ C done, no waiting request
                         └─ A done at t4, D admitted at t5
```

no slot produces padding after its request ends. This raises throughput and lowers queueing delay. It does not make decoding compute-bound: a decode step reads all weights and each request's KV cache to produce one token per request, so it is usually limited by memory bandwidth.

### selective batching

Iteration-level scheduling creates a batch whose requests sit at different positions. A request in prefill contributes many tokens and a request in decode contributes one. Dense tensors of shape $[B, \ell, d]$ would need padding again. Orca's answer is _selective batching_: flatten all tokens of all requests into one $[\sum_i n_i, d]$ matrix for the QKV projections, output projection and MLP, because those operations treat each token independently. Attention is split per request, because each token attends only to its own request's keys and values. [@280922]

## single iteration

each iteration runs:

```
┌─────────────────── iteration t ───────────────────┐
│                                                   │
│  1. prefill: new request D (n prompt tokens)      │
│     ┌──────────────────────────────┐              │
│     │ prompt tokens → Q,K,V proj   │  O(n) tokens │
│     │ → causal attention           │  O(n²)       │
│     │ → write K,V to cache         │              │
│     └──────────────────────────────┘              │
│                                                   │
│  2. decode: existing A,B,C                        │
│     ┌──────────────────────────────┐              │
│     │ last token → Q,K,V proj      │  1 token/seq │
│     │ read K,V cache → attention   │  O(seq_len)  │
│     │ → output proj → next token   │              │
│     └──────────────────────────────┘              │
│                                                   │
│  3. sample: argmax/nucleus per sequence           │
│                                                   │
│  4. check: A hit EOS → evict, free KV cache       │
│                                                   │
└───────────────────────────────────────────────────┘
```

mixing a long prefill into the same iteration makes that step slow for every decoding request in it. Splitting long prompts into chunks bounds that stall; see [[thoughts/PD disaggregated serving|PD disaggregated serving]] for the separation of the two phases.

the binding constraint is KV-cache memory. Each token stores a key and a value vector at every layer:

$$
\text{bytes per token} = 2 \cdot L \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot b,
$$

with $L$ layers, $n_{\text{kv}}$ key/value heads, head size $d_{\text{head}}$ and $b$ bytes per element. For OPT-13B in fp16 ($L=40$, $n_{\text{kv}}d_{\text{head}}=5120$, $b=2$) this is $819{,}200$ bytes, about $800$ KB per token, so one 2048-token sequence holds about $1.7$ GB. [@kwon2023efficient] The number of concurrent requests is bounded by the cache memory divided by the per-request footprint.

### [[thoughts/Transformers#KV|KV cache]] fragmentation

Fragmentation comes from how the cache is allocated, independent of the batching policy. Orca and FasterTransformer reserve one contiguous region per request, sized for the request's maximum possible length. [@kwon2023efficient] measured that only $20.4\%$ to $38.2\%$ of the KV-cache memory in such systems held actual token states.

**internal fragmentation**: the region is sized for the maximum length, but the request stops earlier, so the tail of its own region stays unused.

```
GPU memory for KV cache (max_seq_len = 2048 per slot):
┌──────────────────────────────────────────────────────────┐
│ seq A: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] │
│        used: 512        wasted: 1536 (75%)               │
├──────────────────────────────────────────────────────────┤
│ seq B: [████████████████████████████████████████████████]│
│        used: 2048       wasted: 0                        │
├──────────────────────────────────────────────────────────┤
│ seq C: [████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] │
│        used: 768        wasted: 1280 (63%)               │
└──────────────────────────────────────────────────────────┘
```

there is also _reservation_ waste: a running request holds slots for tokens it has not generated yet.

**external fragmentation**: requests reserve regions of different sizes, so freed regions leave gaps of different sizes between live ones.

```
A reserved 512 tokens, C reserved 768, B is still running between them.
After A and C finish, D arrives and needs a 1024-token region:
┌──────────────────────────────────────────────────────────┐
│ [ hole: 512 ][███ seq B running ███][ hole: 768 ][ ... ] │
└──────────────────────────────────────────────────────────┘
  512 + 768 = 1280 ≥ 1024 tokens are free in total, but no
  single hole holds 1024 contiguous tokens, so D must wait
  for B to finish or for the allocator to compact memory
```

[[thoughts/paged attention|PagedAttention]] removes the contiguity requirement. The cache is split into fixed-size blocks (for example 16 tokens), and a per-request block table maps logical positions to physical blocks. A request allocates a new block only when its last block is full, so the waste per request is less than one block and any free block can serve any request. [@kwon2023efficient]

> [!important] continuous batching $\neq$ solving memory fragmentation
>
> iteration-level scheduling removes padding slots, but KV-cache fragmentation persists unless the cache is also paged. [[thoughts/vllm|vLLM]] combines both.

## throughput gains

Orca reported a $36.9\times$ throughput improvement over NVIDIA FasterTransformer at the same latency, on GPT-3 175B. [@280922] That comparison bundles iteration-level scheduling with selective batching and Orca's distributed execution, so it does not isolate the scheduling change. vLLM then reported $2$ to $4\times$ higher throughput than FasterTransformer and Orca at the same latency, which measures what paging adds on top of continuous batching. [@kwon2023efficient] The gain on a given deployment depends on the spread of output lengths, the arrival rate and the memory left for the KV cache.
