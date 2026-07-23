---
date: '2026-07-21'
description: implementation-first PyTorch model construction for the Inferact inference loop
id: model-builds
modified: 2026-07-23 03:45:29 GMT-04:00
tags:
  - cs
title: Inferact PyTorch model builds
---

# PyTorch model builds

This lane trains the task hiding inside "implement a model from a paper, diagram, or config." The output is a correct `nn.Module` hierarchy with explicit tensor, state, serialization, and inference contracts. The smaller operations in [[hinterland/prep/inferact/role-drills|role drills]] are repair tools for these builds.

The default route covers M01 through M05 and M10. It takes ten and a half focused hours before re-solves, matching one quarter of the forty-two-hour plan. M06 through M09 are architecture or runtime stretches chosen from recruiter signal and baseline misses.

## construction protocol

Before implementing the artifact, produce this page:

1. input and output contract, including shapes, dtype, device, invalid input, and mutation
2. config invariants such as head divisibility, vocabulary size, patch size, and layer schedule
3. module tree with every parameter-owning child
4. persistent buffers and their `state_dict` behavior
5. per-call state, per-request state, and mutable inference state, including cache, scheduler, or loader state when applicable
6. shape ledger through the complete forward, denoising-step, or load path
7. reference implementation and numerical-stability choices
8. hidden-test matrix
9. artifact-specific inference follow-ups: prefill and decode for autoregressive models, denoising and scheduler state for M08, loader mapping and ownership for M09, plus compilation, quantization, and parallelism where applicable
10. vLLM boundary stressed by the architecture

Use `torch`, `torch.nn`, and `torch.nn.functional`. Avoid NumPy, `transformers`, `einops`, and copied model code during the timed build. CPU correctness comes first. CUDA, compilation, fused kernels, and vLLM integration enter after the reference path passes.

## shared config vocabulary

Use a typed config instead of a loose dictionary:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderConfig:
  vocab_size: int
  hidden_size: int
  num_layers: int
  num_query_heads: int
  num_kv_heads: int
  head_dim: int
  intermediate_size: int
  max_sequence_length: int
  rope_theta: float = 10_000.0
  dropout_probability: float = 0.0
  tie_word_embeddings: bool = True
```

Validate at construction:

- every size and count is positive
- `hidden_size == num_query_heads * head_dim`
- `num_query_heads % num_kv_heads == 0`
- dropout is in `[0, 1)`
- model-specific layer schedules and patch sizes are valid

The interviewer may change field names. Preserve the invariants.

## universal hidden tests

Every model build should survive:

| surface          | tests                                                                                                                 |
| ---------------- | --------------------------------------------------------------------------------------------------------------------- |
| registration     | all learned tensors appear in `named_parameters`; buffers move with `.to()`; plain tensor attributes are deliberate   |
| serialization    | strict `state_dict` round trip; tied weights remain tied when promised                                                |
| shapes           | batch and sequence size one, multiple batches, odd valid lengths, invalid divisibility                                |
| dtype and device | fp32 CPU reference; one lower-precision forward where supported; no accidental CPU tensor creation                    |
| modes            | `train()` and `eval()` behavior is explicit; the module passes `dropout_p=0.0` to SDPA during evaluation              |
| numerics         | fp32 reductions where required, finite masked outputs, stable loss and routing probabilities                          |
| equivalence      | vectorized result against a small loop or decomposed reference                                                        |
| inference        | artifact-appropriate inference or load path, no unexpected mutation, and cache, scheduler, or loader ownership stated |
| compilation      | top-level compile boundary and likely guards identified, even when compilation is not run                             |

## M01: minimal decoder-only language model

**Priority:** core. **Time:** 75 minutes.

Use tiers 0 and 1 of [[hinterland/prep/inferact/gpt-lab|the tiny GPT lab]] as the executable acceptance harness. Run its checked-in reference canary first, then point the same construction, causal, loss, registration, tying, serialization, and backward tests at the blank-editor implementation. Cache and generation enter with M03.

Implement a complete small causal language model using PyTorch primitives:

```python
class TinyGPT(nn.Module):
  def __init__(self, config: GPTConfig) -> None: ...

  def forward(
    self, input_ids: Tensor, labels: Tensor | None = None
  ) -> CausalLMOutput: ...
```

The lab's `GPTConfig` is the executable M01 specialization of the shared `DecoderConfig` vocabulary. Use its field names for this route so one implementation can continue into the optional simple M03 path. The advanced path translates the same invariants to M02's config and module tree.

The model contains token embeddings, two or more pre-norm decoder layers, causal self-attention through `scaled_dot_product_attention`, a feed-forward sublayer, final normalization, and an LM head. Use ordinary multi-head attention, LayerNorm, and GELU for this first build. Return logits shaped `[B, T, V]` and optional shifted next-token cross-entropy.

Acceptance:

- all layers live in `nn.ModuleList`
- the causal contract is verified by changing a future token and checking earlier logits
- labels use `-100` as the ignore index
- evaluation is deterministic when dropout is configured
- tied embeddings share the same `Parameter` object
- a strict `state_dict` round trip preserves output
- parameter count is derived from the config and checked against the module

Probes:

- Why does an ordinary Python list lose child-module registration?
- Where must the label and logit shift occur?
- Why must SDPA receive zero dropout during evaluation?
- Which tensors are parameters, persistent buffers, and per-call intermediates?
- What part of this implementation will dominate prefill and decode separately?

## M02: Llama-style GQA model from config

**Priority:** core. **Time:** 120 minutes in two slices.

Upgrade M01 to a Llama-shaped architecture:

- RMSNorm
- bias-free Q, K, V, output, and MLP projections
- grouped-query attention
- rotary position embeddings
- SwiGLU with fused gate and up projection allowed
- pre-norm residual blocks
- final RMSNorm and optional tied LM head

The first slice ends after one decoder layer matches a decomposed reference. The second slice builds the full `ModuleList`, logits path, initialization, and tests.

Acceptance:

- Q uses `num_query_heads`; K and V use `num_kv_heads`
- RoPE rotates Q and K at the supplied positions and leaves V unchanged
- GQA ratios one, four, and eight match an explicit repeated-KV reference
- RMSNorm reduces in fp32 and returns the input dtype
- the model handles nonzero starting positions
- the complete model matches a manual one-layer calculation on a tiny deterministic config
- module names form stable checkpoint paths

Probes:

- Which projection dimensions change when moving from MHA to GQA?
- Which model state can be recomputed and which belongs in a buffer?
- Why does RoPE position choice differ between prefill and decode?
- Which projections become column-parallel and row-parallel in a vLLM port?
- How would fused QKV and gate-up checkpoints map into this module tree?

## M03: cache-aware prefill and decode

**Priority:** core. **Time:** 105 minutes across two full-route slices, or a seventy-five-minute scaffolded delta on the compressed route.

The checked-in cache tier in [[hinterland/prep/inferact/gpt-lab|the tiny GPT lab]] supplies the simple functional canary. Its advanced adapter suite supplies executable mixed-prefix, metadata, ownership, and generation acceptance without constraining the M02-shaped module tree. On the full and intermediate routes, spend this 105-minute block on one path selected before day 5 by the frozen baseline P19 artifact. Choose the blank simple homogeneous-cache GPT when P19 fails padded-length exclusion, full-versus-final-token parity, or caller-owned cache immutability. Choose the M02 extension when all three pass, point the advanced adapter factories at that artifact, and record the branch decision. The simple branch spends day 5's thirty-minute M03 slice on cache construction and day 6's seventy-five-minute block on positions, parity, and generation. The compressed route is a declared scaffolded exception: use its supplied working M02 artifact, implement the advanced cache delta in one uninterrupted seventy-five-minute block, then grade it in the separate fifteen-minute test block.

Extend M02 with an explicit functional KV-cache contract:

```python
LayerKV = tuple[Tensor, Tensor]


class CacheAwareCausalLM(nn.Module):
  def forward(
    self,
    input_ids: Tensor,
    positions: Tensor,
    past_key_values: tuple[LayerKV, ...] | None = None,
    cache_lengths: Tensor | None = None,
    cache_start_positions: Tensor | None = None,
    use_cache: bool = False,
  ) -> CausalLMOutput: ...

  def generate(
    self,
    input_ids: Tensor,
    positions: Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
  ) -> Tensor: ...
```

Use cache tensors shaped `[B, Hkv, S, D]`, cache lengths shaped `[B]`, and optional absolute cache-start positions shaped `[B]`. Row `b` contains valid past keys in `[0, cache_lengths[b])`; their absolute positions are contiguous from `cache_start_positions[b]`. The reference implementation returns new cache tensors and lengths and does not mutate caller-owned storage. Treat in-place fixed-capacity cache updates as a follow-up design.

Acceptance:

- one full prefill and token-by-token decode produce matching logits
- chunked prefill produces the same final cache and logits as full prefill
- causal validity uses absolute query and key positions
- empty cache, one-token cache, and nonzero position offsets work
- mixed prefix lengths mask padded cache entries per row
- cache length and layer count are validated
- `use_cache=False` avoids returning cache state
- caller-owned cache inputs remain unchanged
- cached greedy generation matches full-prefix recomputation, waits for every batch row to finish, and restores the prior module mode
- the advanced adapter suite passes through factories that instantiate the actual M02-shaped artifact

Probes:

- Why is `is_causal=True` insufficient for every rectangular decode shape?
- What changes when the cache is preallocated and updated in place?
- Which cache layout supports contiguous attention and which supports paged serving?
- How would sliding-window or hybrid layers alter the cache contract?
- Which dimensions are sharded under tensor or context parallelism?

## runtime extension: cross-request prefix caching

Begin with [[hinterland/prep/inferact/gpt-lab#runtime prefix-cache extension|the executable exact-prefix snapshot cache]] over a passing `CacheAwareCausalLM`. It owns block-aligned tuple lookup, cloned full-prefix K/V snapshots, entry-level LRU, and suffix-prefill parity. The model remains a functional K/V producer for one request.

After that implementation is clean, describe how a serving runtime replaces duplicated snapshots with physical block storage, chained lookup, leases, reference counts, and page-table-aware eviction.

Partition the returned per-layer K/V tensors into immutable cloned full blocks shaped `[1, Hkv, B, D]`. Publish only complete blocks. Keep the last partial block request-private until later tokens complete it. Hash each block with SHA-256 over its parent digest, exact block token IDs, model identity, cache-schema identity, adapter identity, media hashes, tenant salt, and absolute block start. A digest selects candidates; exact canonical identity verification decides reuse.

Lookup must return the longest contiguous block-aligned prefix and leave at least one prompt token uncached, so the cap is $B\lfloor(T-1)/B\rfloor$. Lease every hit before exposing storage, increment its reference count, and release it after suffix prefill. Evict the least-recently-used zero-reference chain leaf so a resident child is never stranded behind an evicted ancestor. If every leaf is owned or protected by the current publication, bypass the remaining blocks and let inference complete.

Production follow-up:

- shared prompt branches reuse one physical common block
- the same tokens under a different model, cache schema, adapter, media input, tenant salt, or absolute start miss
- a forced SHA-256 collision cannot bypass exact identity verification
- source K/V tensors, returned materializations, and concurrent leases cannot mutate stored blocks
- partial tails stay private and become reusable only after sealing a full block
- live leases prevent eviction, and suffix-first LRU pressure preserves a reachable resident chain
- cached suffix logits and final K/V state match full `CacheAwareCausalLM` recomputation

Run the straightforward executable reference:

```bash
uv run --project content/thoughts/tsfm/lecture-3-exercise pytest content/hinterland/prep/inferact/labs/test_prefix_cache.py -q
```

Probes:

- Why is the hash a lookup accelerator rather than the complete identity contract?
- Why does an exactly block-aligned prompt still need one recomputed block in this runner?
- Which identity fields change for RoPE, multimodal inputs, or LoRA?
- What breaks when eviction ignores request ownership?
- How would the physical block table replace concatenated model-facing K/V without changing the parity oracle?

## M04: vLLM-shaped model port

**Priority:** core. **Time:** 120 minutes in two slices.

Port the M02 architecture toward the model-runtime boundary without importing vLLM. Preserve ordinary PyTorch modules while adopting the important serving contract:

```python
class RuntimeCausalLM(nn.Module):
  def __init__(self, config: DecoderConfig, prefix: str = '') -> None: ...

  def embed_input_ids(self, input_ids: Tensor) -> Tensor: ...

  def forward(
    self,
    input_ids: Tensor | None,
    positions: Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: Tensor | None = None,
  ) -> Tensor | IntermediateTensors: ...

  def compute_logits(self, hidden_states: Tensor) -> Tensor: ...

  def load_weights(
    self, weights: Iterable[tuple[str, Tensor]]
  ) -> set[str]: ...
```

Inputs are flattened token rows `[N]` rather than padded `[B, T]`. The harness supplies `IntermediateTensors` as a typed mapping of pipeline-boundary names to tensors. On a single rank or first pipeline stage, exactly one of `input_ids` and `inputs_embeds` is supplied. A later pipeline stage consumes `intermediate_tensors` and may receive neither. A non-final stage returns `IntermediateTensors`; the final stage returns hidden states. Every nested module receives a unique state-dictionary prefix. The exercise uses a simple attention context supplied by the harness; explain where a real vLLM attention operator and runtime cache write would replace the reference path.

Acceptance:

- flattened and padded reference paths agree after packing and unpacking
- nested prefixes are unique and stable
- `embed_input_ids` and `inputs_embeds` paths agree
- a simulated two-stage split passes intermediate tensors without re-embedding tokens
- separated Q, K, and V checkpoint tensors load into one packed QKV parameter
- separated gate and up tensors load into one merged parameter
- missing, duplicate, unexpected, and shape-mismatched weights fail deliberately
- `compute_logits` stays outside the core hidden-state forward

Probes:

- Why does an inference runtime flatten scheduled tokens?
- Why does each model submodule need a full prefix?
- Where do tensor-parallel shard loaders change the checkpoint mapping?
- Which training-only paths should disappear from a serving model?
- Where would pipeline-parallel intermediate tensors enter the signature?
- Which model operation should become an opaque custom-op boundary for compilation?

Read the official [vLLM basic-model implementation guide](https://docs.vllm.ai/en/v0.17.0/contributing/model/basic/) after the first attempt, then compare against local `vllm/model_executor/models/llama.py`.

## M05: top-two MoE causal language model

**Priority:** core. **Time:** 105 minutes.

Replace selected M02 feed-forward layers with a sparse mixture of experts:

```python
class SparseMoE(nn.Module):
  def __init__(
    self,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    experts_per_token: int,
  ) -> None: ...

  def forward(self, hidden_states: Tensor) -> MoEOutput: ...
```

Start with a reference implementation using `nn.ModuleList` and explicit expert loops. Route flattened tokens, select top-k experts, normalize selected probabilities, run the chosen SwiGLU experts, and combine outputs back into original token order. Return router logits and expert counts beside the hidden states.

Acceptance:

- `experts_per_token` one and two work
- top-k ties have a documented deterministic rule
- selected weights sum to one per token
- empty experts are valid
- output ordering matches the original `[B, T, D]` input
- a vectorized dispatch path matches the loop reference
- one backward smoke test reaches router and selected expert parameters
- dense-equivalent experts reproduce a dense MLP reference

Probes:

- Which temporary tensors scale with tokens, experts, and capacity?
- What changes when expert capacity drops or pads tokens?
- Why does expert parallelism introduce all-to-all?
- How does load skew damage both latency and utilization?
- Which weights should remain fp32 during routing?
- How would quantized expert weights change dispatch and kernel choice?

## M06: multimodal causal language model

**Priority:** stretch. **Time:** 90 minutes.

Build a small vision-language model with a text decoder, patch encoder, modality projector, and placeholder merge:

```python
class TinyVisionLanguageModel(nn.Module):
  def forward(
    self,
    input_ids: Tensor,
    pixel_values: Tensor | None = None,
    image_token_mask: Tensor | None = None,
    image_counts: Tensor | None = None,
    media_ids: tuple[str, ...] | None = None,
  ) -> CausalLMOutput: ...
```

Use `nn.Conv2d` to patchify flattened images shaped `[Nimg, C, H, W]`. `image_counts` is shaped `[B]`, sums to `Nimg`, and assigns consecutive images to each request. Project patch features to the language hidden size and replace the exact number of placeholder-token embeddings per request. This reference contract uses one configured spatial size; variable spatial sizes require a padded-media tensor and validity mask or an item-wise encoder path. The text-only path must avoid invoking the media encoder.

Acceptance:

- patchification rejects nondivisible spatial dimensions
- placeholder count must equal `image_counts[b]` times patches per image for each request
- text-only and multimodal batches preserve request boundaries
- zero images, one image, and multiple images per request are explicit contracts
- media IDs align one-to-one with flattened images and make encoder-cache identity testable
- scatter or indexed-copy output matches a loop reference
- the media projector participates in dtype and device conversion
- repeated calls expose enough identity information to design an encoder cache

Probes:

- Which work belongs before scheduling and which belongs in the model runner?
- What identifies reusable encoder output?
- How do visual-token counts affect TTFT and token budgets?
- Where can variable image sizes create compilation guards?
- How would disaggregated encoding change ownership and failure behavior?

## M07: hybrid attention and recurrent-state model

**Priority:** stretch. **Time:** 105 minutes.

Alternate ordinary attention layers with a simplified gated recurrent layer defined by the supplied recurrence:

$$
s_t = g_t \odot s_{t-1} + (1 - g_t) \odot u_t
$$

$$
y_t = o_t \odot s_t
$$

Return recurrent state per hybrid layer beside ordinary K/V state per attention layer.

Acceptance:

- full-sequence and token-step recurrence agree
- reset masks isolate sequences in a packed batch
- recurrent state is updated rather than appended
- attention and recurrent layer schedules validate against the config
- cache/state tuples retain stable layer identity
- mixed layer outputs preserve dtype and shape

Probes:

- Which update and validity semantics differ from append-only KV even when one allocator stores both state families?
- Which prefix-cache boundary is valid for a mutable recurrent state?
- How should hybrid cache groups size and align their physical storage?
- Which recurrence should become an opaque custom operator?

## M08: diffusion transformer denoiser

**Priority:** stretch. **Time:** 120 minutes.

Implement a small diffusion-transformer denoiser:

```python
class TinyDiT(nn.Module):
  def forward(
    self, noisy_samples: Tensor, timesteps: Tensor, conditioning: Tensor
  ) -> Tensor: ...
```

Patchify with `nn.Conv2d`, add position state, embed the timestep, project conditioning, apply adaptive-normalization transformer blocks, and unpatchify to the input spatial shape. The denoising scheduler remains outside the model.

Acceptance:

- patchify and unpatchify are inverse in shape
- nondivisible spatial dimensions fail clearly
- timestep and conditioning broadcast only across intended axes
- position data moves with the module
- classifier-free guidance can be expressed through batched conditional and unconditional inputs
- repeated denoising calls reuse model parameters without storing scheduler state in the module

Probes:

- Which state belongs to the denoising loop rather than the model?
- Why does diffusion serving produce a different batching problem from autoregressive decode?
- Which dimensions stay stable enough for graph capture?
- How do guidance and variable image sizes change effective batch shape?

## M09: checkpoint adapter and weight tying

**Priority:** stretch. **Time:** 75 minutes.

Given a source checkpoint naming scheme and the M02 target model, implement a strict streaming loader that:

- maps source names to target names
- packs Q, K, and V into one target parameter
- packs gate and up projections
- transposes only when source and target conventions require it
- loads tensor slices without retaining the full checkpoint
- preserves embedding and LM-head tying
- reports loaded, missing, duplicate, unexpected, and mismatched tensors

Acceptance:

- randomized source weights produce the same logits before and after conversion
- every target parameter is accounted for exactly once unless explicitly tied
- wrong shapes and duplicate shards fail
- dtype conversion is deliberate
- the loader explains where tensor-parallel rank slicing occurs

Probes:

- Why can a successful `load_state_dict(strict=False)` still produce a broken model?
- How do fused projections alter shard dimensions?
- Which tensor owns the tied storage after loading?
- What metadata must accompany quantized weights?

## M10: paper-fragment capstone

**Priority:** core capstone. **Time:** 105 minutes, then a 75-minute re-solve.

Implement this architecture from the fragment alone:

```text
token embedding
  -> N x {
       RMSNorm
       RoPE GQA with optional cache
       residual
       RMSNorm
       top-two MoE
       residual
     }
  -> final RMSNorm
  -> tied logits projection
```

```python
class PaperFragmentLM(nn.Module):
  def forward(
    self,
    input_ids: Tensor,
    positions: Tensor,
    past_key_values: tuple[LayerKV, ...] | None = None,
    cache_lengths: Tensor | None = None,
    cache_start_positions: Tensor | None = None,
    use_cache: bool = False,
    return_router_state: bool = False,
  ) -> PaperFragmentOutput: ...
```

The output contains logits, optional updated cache, and optional router logits and expert counts. State the simplifications chosen to finish the reference path inside the time box.

Acceptance:

- M02 GQA/RoPE/RMSNorm behavior remains correct
- M03 prefill and token-step decode agree
- M05 selected expert weights normalize and token order survives dispatch
- router telemetry is absent unless requested
- invalid head, expert, position, and cache configurations fail before forward mutation
- state-dictionary names are stable enough for the M09 loader
- the entire model runs under `torch.inference_mode()` from a tiny randomized config

Probes:

- Which simplification preserves the architecture while reducing code volume?
- Which state belongs to the model, request, cache manager, and sampler?
- Which vLLM boundaries need changes: registry, loader, attention, cache spec, model runner, or sampler?
- Which model operations become tensor-parallel or expert-parallel?
- Which benchmark and oracle decide whether the first production optimization is safe?

## runtime extension: compile-friendly static cache

After M10 passes, wrap M03 in a fixed-capacity cache interface with preallocated K/V buffers, tensor positions, and no data-dependent Python loop over active tokens.

Check that eager functional cache and static in-place cache produce matching logits, storage addresses remain stable, invalid positions fail before mutation, and `model.compile()` preserves supported outputs. Record graph breaks, guards, recompilations, and steady-state timing with compilation excluded.

## repair map

| build failure                                           | owning repair drills |
| ------------------------------------------------------- | -------------------- |
| shapes, views, GQA, and masks                           | P01 through P06      |
| logits, penalties, filtering, and generation            | P07 through P13      |
| normalization, RoPE, MLP, module state, and attention   | P14 through P18      |
| decode attention, KV append, paging, and batch mutation | P19 through P22      |
| quantization, MoE routing, and multimodal merge         | P23 through P26      |
| compilation and distributed execution                   | P27 and P28          |

Return to one drill that owns the first failed invariant, then resume the model build from the same boundary. Reimplementing six neighboring primitives is penance without information gain.

After the owning repair or M-series boundary passes, draw one connected case from [[hinterland/prep/inferact/pytorch-practice|the PyTorch practice bank]]. A failed build never skips directly to transfer testing. When ragged runtime tensors, cache reordering, speculative state, tensor-parallel layers, or custom operators have no narrower P-series owner, repair the named M-series owner through its acceptance tests before drawing the PT case.

## model-construction mocks

### model mock A: architecture from prose

**Time:** 60 minutes.

Receive a one-page architecture description. Produce the config invariants, module tree, one complete decoder layer, full model composition, and six hidden tests. The partner changes one detail at minute thirty: MHA becomes GQA, LayerNorm becomes RMSNorm, or absolute positions become RoPE.

### model mock B: add incremental decoding

**Time:** 60 minutes.

Receive a correct dense model without cache support. Add positions and a functional per-layer KV cache. Prove full-prefill and token-step equivalence. Explain the transition to a paged runtime cache.

### model mock C: port a model into an inference runtime

**Time:** 75 minutes.

Receive a training-oriented `nn.Module` and checkpoint map. Remove training-only paths, flatten scheduled tokens, separate hidden-state forward from logits, add strict weight loading, and mark tensor-parallel and custom-op boundaries.

### model mock D: unfamiliar architecture fragment

**Time:** 60 minutes.

Choose MoE, multimodal, hybrid recurrent/attention, or DiT. The partner supplies only equations and shapes. Produce a reference `nn.Module`, state ownership table, randomized oracle, and serving-integration analysis.

## model-build rubric

Score each dimension from zero to four:

| dimension                  | four-point evidence                                                                                                                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| contract and config        | shapes, errors, mutation, modes, and config invariants are fixed before code                                                                                                                            |
| module and state ownership | applicable parameters, buffers, children, request state, and cache, scheduler, or loader state have correct owners                                                                                      |
| tensor mechanics           | every reshape, broadcast, gather, scatter, and allocation follows the shape ledger                                                                                                                      |
| numerical correctness      | masks, normalization, reductions, loss, and degenerate rows are deliberate                                                                                                                              |
| complete forward path      | the module composes end to end and produces the promised output structure                                                                                                                               |
| tests and serialization    | artifact-specific randomized oracle, state round trip, dtype, and edge cases pass; causality and tying are tested when promised                                                                         |
| inference reasoning        | autoregressive prefill, decode, and cache; M08 denoising and scheduler state; or M09 loader mapping and ownership are explained with applicable compile, quantization, parallelism, and vLLM boundaries |

Award zero when the dimension is absent or wrong, one for fragments that cannot reach a working result without rescue, two for a workable baseline with a material gap, three when one bounded gap prevents the listed four-point evidence, and four only when that evidence survives the probes. Use integer scores only.

A build is ready at 24 out of 28 with every dimension at least 2. A model that emits correctly shaped nonsense scores at most 12. Shape cosplay has had a good run.

## paper-to-code answer pattern

```text
The model owns [parameters and persistent buffers].
The request owns [positions, modality identity, and sampling state].
The runtime owns [KV or recurrent cache storage].

Input [shape] moves through [module sequence] to output [shape].
The reference path uses [PyTorch primitives and numerical choices].
Correctness is decided by [oracle, equivalence, and state tests].
Serving stresses [model runner, attention, cache, loader, compiler, or scheduler boundary].
```

## source spine

- [PyTorch module notes](https://docs.pytorch.org/docs/stable/notes/modules.html)
- [PyTorch transformer building blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)
- [PyTorch SDPA tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- [where to apply `torch.compile`](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.where_to_apply_compile.html)
- [vLLM basic-model implementation](https://docs.vllm.ai/en/v0.17.0/contributing/model/basic/)
- [vLLM model registration](https://docs.vllm.ai/en/v0.17.0/contributing/model/registration/)
- local `vllm/model_executor/models/llama.py`
- local `vllm/model_executor/models/qwen3_moe.py`
- local multimodal and hybrid implementations selected from the registry
