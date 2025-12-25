---
date: "2024-02-08"
description: batching strategy for large scale inference deployment
id: Continuous batching
modified: 2025-12-24 23:39:00 GMT-05:00
seealso:
  - "[[thoughts/vllm|vLLM]]"
  - "[[thoughts/Attention#Paged Attention|PagedAttention]]"
tags:
  - ml
  - inference
title: Continuous batching
transclude:
  title: false
---

[@280922] solves the static batching to reduce cost and improve throughput by appending requests continuously into existing KV cache [^paper]

[^paper]:
    The [paper](https://www.usenix.org/conference/osdi22/presentation/yu) and [presentation](https://www.youtube.com/watch?v=Ob9PPLxETYU&ab_channel=USENIX) for the paper. Most notable open source implementation is [[thoughts/vllm|vLLM]].

    p/s: Actually, I think first implemented in [huggingface/tgi](https://github.com/huggingface/text-generation-inference)

## static batching

the naive approach is to form a batch of $N$ requests and process them together until **ALL** complete. Requests that finish early sit idle (padded) while waiting for the longest sequence.

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

inefficiency is $O(\text{batch\_size} \times \text{max\_seq\_len} - \sum \text{actual\_seq\_lens})$. with high variance in output lengths, this wastes 50-70% of compute.

### [[thoughts/Transformers#KV|KV cache]] fragmentation

beyond compute waste, static batching causes memory fragmentation in the KV cache. [@kwon2023efficient] measured 60-80% memory waste from these issues:

**internal fragmentation**: pre-allocate contiguous memory for `max_seq_len` per sequence, but actual length is shorter → unused memory WITHIN the allocated region.

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

**external fragmentation**: sequences complete at different times, leaving holes that are too small to fit new requests.

```
after seq A and C complete, seq D arrives (needs 1800 tokens):
┌──────────────────────────────────────────────────────────┐
│ [     hole: 512     ][████ seq B still running ████████] │
│ [    hole: 768    ][░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] │
└──────────────────────────────────────────────────────────┘
         ↑                              ↑
         └─ seq D can't fit in ─────────┘
            either hole (512 + 768 = 1280 < 1800)
            must wait for B to finish or compact
```

[[thoughts/Attention#Paged Attention|PagedAttention]] solves this by paging KV cache into fixed-size blocks—like OS virtual memory. allocate only what you need, reclaim blocks granularly.

## iteration-level scheduling

_after EACH forward pass (one "iteration"), check which sequences hit EOS. immediately evict completed sequences and admit new ones from the queue._

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

this eliminates COMPUTE fragmentation—no wasted FLOPs on padding. GPU arithmetic utilization approaches 100% when queue depth > 0.

> [!important] continuous batching ≠ solving memory fragmentation
>
> iteration-level scheduling solves compute waste, but KV cache memory fragmentation persists unless you also page the cache. [[thoughts/vllm|vLLM]] combines continuous batching with [[thoughts/Attention#Paged Attention|PagedAttention]] to eliminate both.

## single iteration

each iteration runs:

```
┌─────────────────── iteration t ───────────────────┐
│                                                   │
│  1. prefill: new request D                        │
│     ┌──────────────────────────────┐              │
│     │ prompt tokens → Q,K,V proj   │  O(n²)       │
│     │ → attention → output proj    │              │
│     └──────────────────────────────┘              │
│                                                   │
│  2. decode: existing A,B,C                        │
│     ┌──────────────────────────────┐              │
│     │ last token → Q proj          │  O(1)/seq    │
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

the constraint is KV cache memory: each sequence holds $O(\text{seq\_len} \times L \times d \times 2)$ bytes. max concurrent sequences = GPU memory / per-sequence cache.

## throughput gains

orca reported 2-36× throughput improvement over static batching. the variance depends on output length distribution, request arrival rate, and batch size limits.

typical production numbers falls around 3-8× improvement for conversational workloads where output lengths vary 10-500 tokens.
