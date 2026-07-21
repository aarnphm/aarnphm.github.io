---
date: '2026-07-21'
description: PyTorch and vLLM recall sheet for the Inferact technical loop
id: cheatsheet
modified: 2026-07-21 16:12:05 GMT-04:00
tags:
  - cs
title: Inferact inference cheatsheet
---

# interview sheet

## tensor first

Before any PyTorch implementation, write:

```text
name: semantic axes
shape:
stride:
dtype:
device:
owner:
mutation:
output shape:
```

Rules:

- `view` requires stride compatibility and never hides a copy.
- `reshape` returns a view when possible and may allocate otherwise.
- `transpose` and `permute` change shape and stride without moving storage.
- `contiguous` returns the input when already contiguous and copies otherwise.
- basic indexing usually returns a view; advanced indexing returns a copy.
- broadcasting aligns trailing dimensions and can create stride-zero expanded views.
- writing through an expanded view is unsafe when multiple logical elements share one storage location.
- `gather` requires an index tensor with the same rank as the input and does not broadcast input and index against each other.
- `.item()`, printing a CUDA tensor, and many host-side decisions synchronize the device.

## attention shape ledger

```text
hidden: [B, T, C]
q:      [B, Hq, L, D]
k:      [B, Hkv, S, D]
v:      [B, Hkv, S, Dv]
scores: [B, Hq, L, S]
output: [B, Hq, L, Dv]
logits: [B, T, V]
```

Grouped-query ratio:

$$
G = \frac{H_Q}{H_{KV}}
$$

Each KV head serves `G` query heads. An optimized kernel can reuse K/V across that group instead of materializing repeated heads.

Scaled dot-product attention:

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{D}} + M\right)V
$$

Numerical rules:

- normalize logits in fp32 when low-precision range is risky
- subtract the row maximum before exponentiation
- use `log_softmax` for beam scores
- keep at least one token after top-p filtering
- define the all-masked-row contract
- pass `dropout_p = 0.0` explicitly to SDPA during inference

## decoder block

```text
embedding
  -> RMSNorm
  -> QKV projection
  -> RoPE
  -> attention over current and cached K/V
  -> output projection
  -> residual
  -> RMSNorm
  -> gate and up projection
  -> SiLU(gate) * up
  -> down projection
  -> residual
  -> final norm
  -> LM head
```

Own these variants:

- GQA changes Q-to-KV head ratio and KV size.
- MoE adds router, top-k experts, capacity, token dispatch, expert compute, and weighted combine.
- multimodal adds media processing, encoder execution, placeholder merge, and encoder/cache identity.
- hybrid attention/SSM adds recurrent state with cache semantics distinct from attention KV.
- speculative decoding adds draft proposals, verification, acceptance, rollback, and lookahead cache.

## model-build order

```text
contract and config invariants
  -> module tree and parameter ownership
  -> buffers, request state, and cache state
  -> shape ledger
  -> complete reference forward
  -> randomized, causal, and step-equivalence oracles
  -> state_dict, checkpoint keys, and weight tying
  -> prefill, decode, compile, quantization, and parallelism
  -> vLLM model-runner and loader boundary
```

For a paper fragment, write these before code:

1. every learned tensor and projection shape
2. normalization and residual order
3. position semantics
4. attention, recurrent, media, or denoising state
5. output and loss contract
6. invalid configs
7. a decomposed correctness oracle

Use `nn.ModuleList` or `nn.ModuleDict` for dynamic child modules. Use `register_buffer` for non-parameter tensor state that should follow module device and dtype semantics. Keep request caches outside the model unless the exercise explicitly defines model-owned mutable storage.

## module and inference modes

| mechanism                | changes                                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| `model.eval()`           | module behavior such as dropout and running-stat use                                          |
| `torch.no_grad()`        | autograd recording                                                                            |
| `torch.inference_mode()` | autograd recording plus view tracking and version-counter overhead, with stricter reuse rules |

Parameters are learned persistent state. Buffers are persistent or device-moving model state without gradient optimization. Sequence lengths, block tables, sampling parameters, and request status are runtime state. KV pages are mutable cache storage owned by runtime policy and workers.

## quantization

Affine quantization:

$$
q = \operatorname{clamp}\left(\operatorname{round}\left(\frac{x}{s}\right) + z, q_{min}, q_{max}\right)
$$

Dequantization:

$$
\hat{x} = s(q-z)
$$

Ask:

- Which axis owns each scale?
- Is zero point required?
- Which dtype accumulates the dot product?
- Where does dequantization happen?
- How much payload shrinks after scale overhead?
- What quality evaluation catches silent drift?

Weight, activation, and KV quantization have different error and bandwidth surfaces. Analyze them separately.

## KV cache math

Decoder-only KV bytes per token:

$$
B_{token} = 2 \cdot L \cdot H_{KV} \cdot D \cdot b
$$

Cache-bound token capacity:

$$
T_{capacity} \approx \frac{M_{available}}{B_{token}}
$$

First concurrency estimate at maximum sequence length:

$$
C_{max} \approx \frac{T_{capacity}}{T_{request}}
$$

Correct for:

- tensor or context sharding
- page rounding and partial final blocks
- hybrid cache groups
- quantization scales
- reserved watermark
- beam width
- prefix sharing
- runtime workspaces and graph pools

Paged KV separates logical token order from noncontiguous physical pages. The block table maps logical block indices to physical block IDs. Slot mapping converts token positions into physical cache slots using that block lookup and the within-block offset. Page size trades metadata and kernel locality against internal fragmentation.

## vLLM owner chain

```text
API/frontend
  HTTP, schemas, tokenization, media, detokenization, streaming

EngineCore
  request lifecycle and model-executor coordination

Scheduler
  waiting/running state, per-step token budget, preemption, finish

KV cache manager
  prefix match, logical blocks, allocation, refcounts, reuse, eviction

GPU model runner
  persistent batch, input tensors, slot mapping, graph dispatch, forward, sampling

Attention backend
  metadata, cache write, paged-KV-compatible kernel

Distributed layer
  rank geometry, shards, process groups, collectives, connectors
```

Central scheduler invariant: computed tokens advance toward all currently known prompt, output, and speculative tokens. One token-budget scheduler can therefore express decode, chunked prefill, prefix hits, external KV loads, and lookahead.

## scheduler and cache tradeoffs

### continuous batching

Admit and retire requests at iteration boundaries. It raises utilization and complicates fairness, cancellation, cache ownership, and reproducibility.

### chunked prefill

Decode usually has a bandwidth-heavy token loop. Prefill usually has larger compute-heavy matrix work. Mixing a bounded prefill chunk into remaining token budget can improve utilization while controlling ITL.

Larger token budget tends toward throughput and long-prompt TTFT. Smaller budget tends toward smoother ITL and more steps.

### prefix caching

Hash complete blocks with parent identity, tokens, and identity extras such as LoRA, multimodal content, prompt embeddings, and cache salt. Distinguish a reusable block with zero references from an allocated live block. Prefix locality can conflict with even DP load balance.

### disaggregation

Prefill and decode fleets can optimize separate phases and SLOs. KV transfer latency, bandwidth, placement, expiry, failure, and recompute become part of the request path. Disaggregation does not guarantee higher throughput.

### speculative decoding

Speedup depends on accepted tokens per target verification relative to draft, verification, cache, and scheduling overhead. Measure accepted-token ratio by position, verifier time, draft time, and output tokens per second inside SLO.

## parallelism

| method | split                    | communication                                        | common limiting factor                      |
| ------ | ------------------------ | ---------------------------------------------------- | ------------------------------------------- |
| TP     | tensor dimensions        | row-output all-reduce; optional column-output gather | collective latency and link bandwidth       |
| PP     | layers                   | activation send and receive                          | bubbles and slowest stage                   |
| DP     | requests across replicas | routing and optional synchronization                 | KV locality and load skew                   |
| EP     | MoE experts              | token all-to-all                                     | expert skew and interconnect                |
| CP     | sequence or KV context   | attention-specific exchange/reduction                | long-context communication and kernel shape |

Row-parallel linear layers usually all-reduce their partial outputs. Column-parallel layers usually leave outputs sharded and only all-gather when a full output is requested. Place TP inside a fast-link node when possible. PP can cross weaker links with lower communication frequency and adds stage bubbles. DP replicas have independent KV caches, so routing policy affects prefix hit rate. EP changes MoE communication and load balance.

## compilation and CUDA graphs

```text
eager Python and ATen
  -> Dynamo capture under guards
  -> FX graph and vLLM partition boundaries
  -> Inductor and backend lowering
  -> generated or custom kernels
  -> optional CUDA graph capture
  -> replay at stable shapes and addresses
```

Ask:

- Which shape or value guard can fail?
- Where does Python data-dependent control flow break the graph?
- Which operation needs an opaque custom-op boundary and a fake implementation?
- Which captured sizes cover the serving distribution?
- How much warmup time and graph-pool memory are spent?
- What eager fallback preserves correctness?
- Was compilation excluded from the steady-state benchmark?

## kernel answer pattern

```text
grid
tile
pointer formula
tail and semantic masks
on-chip state
bytes moved
operations
occupancy limiter
correctness oracle
benchmark result
```

Likely bottlenecks:

- tiny decode kernels: launch and CPU dispatch
- long decode attention: HBM bandwidth and available parallel programs
- prefill attention: tensor-core use plus attention IO
- row reductions: register pressure and padded work
- matmul: tensor-core utilization, tiling, shared memory, and register use
- quantization: reduction, packing, scale traffic, and dequant compute

## serving metrics

| metric                      | meaning                                                                                     |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| TTFT                        | arrival to first streamed token                                                             |
| ITL                         | each gap between consecutive output tokens; its distribution exposes jitter and tail stalls |
| TPOT                        | per-request mean of the post-first-token gaps; summarizes average decode cadence            |
| end-to-end latency          | arrival to completed response                                                               |
| goodput                     | throughput for requests that satisfy the latency SLO                                        |
| queue time                  | scheduler admission pressure                                                                |
| KV utilization              | cache-capacity pressure                                                                     |
| preemptions                 | insufficient cache or policy pressure                                                       |
| prefix hit tokens           | reused prefill work                                                                         |
| accepted speculative tokens | useful draft work                                                                           |
| transfer p99 and bytes      | disaggregated or offloaded KV critical path                                                 |
| MFU and bandwidth           | compute and memory efficiency                                                               |

Always segment by prompt length, output length, concurrency, batch composition, model, accelerator, precision, and cold or warm state.

See the official [vLLM performance-metrics definitions](https://docs.vllm.ai/projects/spyre/en/latest/user_guide/performance.html) for the ITL and TPOT distinction.

## system-design order

1. workload and arrival distribution
2. model, context, precision, and quality
3. TTFT, ITL, end-to-end, and goodput targets
4. weight, KV, compute, and communication estimates
5. request and state ownership
6. batching, cache, routing, and parallelism
7. overload, cancellation, retry, and failure scope
8. observability and rollback
9. two experiments that decide the unresolved tradeoff

## deep-dive answer pattern

```text
Given [workload and SLO], [metric] was dominated by [measured mechanism].
I changed [specific boundary] through [mechanism].
This changed [resource or correctness ratio] and produced [measured result].
The cost was [regression or risk], bounded by [test, ablation, rollout, or alert].
```

Bring:

- workload distributions and benchmark metadata
- architecture and request-lifecycle diagrams
- profiler trace
- before and after with variance
- ablation
- correctness matrix
- failed approaches
- rollout and rollback
- residual risk
- next falsifying experiment

## source shortcuts

- [[hinterland/prep/inferact/core|core map]]
- [[hinterland/prep/inferact/model-builds|PyTorch model builds]]
- [[hinterland/prep/inferact/role-drills|role drills]]
- [[hinterland/prep/inferact/study|study route]]
- [[hinterland/prep/inferact/mocks|timed mocks]]
- [[thoughts/vllm|vLLM]]
- [[thoughts/paged attention|PagedAttention]]
- [[thoughts/Continuous batching|continuous batching]]
- [[thoughts/Speculative decoding|speculative decoding]]
- [[thoughts/PD disaggregated serving|P/D serving]]
- [[thoughts/distributed inference|distributed inference]]
- [[thoughts/GPU programming|GPU programming]]
- [[thoughts/flash attention|FlashAttention]]
- [[thoughts/quantization|quantization]]
- [[hinterland/prep/nv/cheatsheet|general coding and systems sheet]]
