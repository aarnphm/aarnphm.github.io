---
date: '2026-07-21'
description: adaptive study and re-solve plan for the Inferact technical loop
id: study
modified: 2026-07-21 16:12:05 GMT-04:00
tags:
  - cs
title: Inferact study route
---

# study route

The default route is fourteen days at three focused hours per day. It yields forty-two hours with 45% assigned to PyTorch model construction and mechanism repair. The baseline can compress this to seven days. Each day names a primary implementation block that fits inside seventy-five minutes. Longer model builds continue across named slices. Stretch drills enter the re-solve queue only after the exit test passes.

Every standalone drill, first model-build slice, mock, and named clean reconstruction begins with a blank editor. Continuation slices resume the previous build artifact. Every session ends with a written first-wrong-decision log. Passive reading is capped at one third of the block.

## daily loop

| minutes | action                                                                              |
| ------: | ----------------------------------------------------------------------------------- |
|      15 | recall ten cards and state yesterday's weakest invariant                            |
|      75 | implement one primary model-build slice, PyTorch repair drill, or Triton drill      |
|      35 | read one owning code path or official design document, then draw its state boundary |
|      35 | rehearse one system-design slice or one deep-dive slice aloud                       |
|      15 | test, profile, and answer Socratic probes                                           |
|       5 | record the first wrong decision and schedule the re-solve                           |

On mock days, replace the middle 145 minutes with the timed mock and repair pass.

## baseline diagnostic

Take this before studying. Use CPU PyTorch, a blank editor, documentation only for signature lookup, and a real timer.

| time | task                                                                                | signal                                            |
| ---: | ----------------------------------------------------------------------------------- | ------------------------------------------------- |
|  20m | P03, split and merge attention heads                                                | shape, stride, view, and copy fluency             |
|  35m | P11, deterministic sampler with temperature, top-k, and top-p                       | stable logits, sort, scatter, batching, RNG       |
|  45m | P19, single-token decode attention                                                  | attention shapes, masks, GQA, cache semantics     |
|  25m | derive KV bytes per token and maximum cache-bound concurrency for a supplied config | arithmetic, assumptions, sharding, block rounding |
|  25m | draw a request through vLLM from API arrival to streamed token                      | state ownership and technical vocabulary          |

Score each dimension from zero to three:

| score | meaning                                                                   |
| ----: | ------------------------------------------------------------------------- |
|     0 | cannot produce a working mechanism                                        |
|     1 | needs substantial hints or violates the contract                          |
|     2 | correct after repair, explanation has one important gap                   |
|     3 | correct within time, tests are deliberate, follow-ups reach the mechanism |

Dimensions:

- tensor semantics
- numerical correctness
- model and attention mechanics
- cache and runtime ownership
- communication under pressure

Route from the total:

- 0 to 7: full fourteen-day route
- 8 to 11: use days 1 through 10, then days 12 through 14
- 12 to 15: use the seven-day compression route

Any zero keeps its full study day regardless of total score.

## fourteen-day route

### day 1: tensor metadata and vectorization

Primary block: P01 through P03. Stretch: P06.

Read:

- [PyTorch tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [`Tensor.view`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html)
- [broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
- [`torch.gather`](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)

Exit test: given any shape and stride, explain whether flattening can remain a view and where the next operation allocates.

### day 2: logits, sampling, and numerical stability

Primary block: P10 and P11. Stretch: P07 through P09.

Read the vLLM sampler files from [[hinterland/prep/inferact/core#current vLLM source tour|source-tour stop 9]]. Compare the implementation strategy with `torch.multinomial` without copying vLLM code into the drill.

Exit test: implement a complete batched sampler in forty minutes, including the all-masked and temperature-zero contracts.

### day 3: first complete causal LM

Primary block: M01 from [[hinterland/prep/inferact/model-builds|the model-build lane]]. Use P03, P05, P17, or P18 only when the build exposes a failed shape, mask, module-state, or attention invariant.

Read:

- [module notes](https://docs.pytorch.org/docs/stable/notes/modules.html)
- [autograd mechanics and grad modes](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- the Llama MLP and decoder-layer classes in local `vllm/model_executor/models/llama.py`

Exit test: reconstruct the module tree, produce logits from input IDs, prove causality, round-trip the state dictionary, and explain parameter, buffer, request-state, and cache ownership without mixing them.

### day 4: Llama-style model, first slice

Primary block: the first seventy-five minutes of M02. Finish the typed config, RMSNorm, RoPE, GQA, SwiGLU, and one decoder layer. Use P04, P05, P14, P15, and P18 as targeted repairs.

Read:

- [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- local `vllm/model_executor/models/llama.py`
- local `vllm/model_executor/layers/attention/attention.py`

Draw the dense decoder block and label every tensor shape. Exit after one Llama-style decoder layer matches the decomposed reference over randomized small shapes.

### day 5: Llama completion and cached decode start

Primary block: finish the remaining forty-five minutes of M02, then spend thirty minutes defining M03's cache, position, and output contracts. Use P19, P20, and P21 only when decode attention, cache append, or logical-to-physical lookup fails.

Read:

- [[thoughts/paged attention|paged attention]]
- [vLLM prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- local `vllm/v1/kv_cache_interface.py`
- local `vllm/v1/core/kv_cache_manager.py`

Exit test: the complete Llama-style model produces logits and stable checkpoint names, and M03 has an executable one-layer prefill/decode test.

### day 6: cache-aware model completion

Primary block: finish M03. Use P19 through P22 as cache and batching repairs. Use P23 and P24 as quantization follow-ups after cache parity passes.

Read:

- local `vllm/v1/worker/gpu_model_runner.py` state-update and input-preparation sections
- local block-table and attention-metadata preparation
- [[thoughts/quantization|quantization]] after the reference cache path passes

Exit test: full prefill, chunked prefill, and token-step decode produce matching logits and cache state. Then explain which mutation and storage choices permit graph replay.

### day 7: vLLM-shaped model port, first slice

Primary block: the first seventy-five minutes of M04. Convert the padded training model into a flattened-token hidden-state model with `embed_input_ids`, explicit positions, unique prefixes, and a separate logits path.

Read the official vLLM basic-model guide and local Llama model. Draw the ownership boundary among model code, attention operator, model runner, loader, and cache manager.

### day 8: model port completion and compiler boundary

Primary block: finish M04's remaining forty-five minutes, including packed-weight loading, then spend thirty minutes on P27. Stretch: T01 or T03 when a Triton environment is available.

Read:

- [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [custom operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- local vLLM compilation decorators, backend, and CUDA-graph dispatcher

Exit test: the flattened-token and padded references agree, checkpoint packing is strict, and the model's eager, compiled, custom-op, and graph-replay boundaries can be drawn with their guards.

### day 9: architecture variant

Primary block: the first seventy-five minutes of M05. Use the first thirty minutes of the system/deep-dive slice to finish its acceptance tests, then use the remaining five minutes to state the serving consequence. P25 is the repair drill. M06 and P26 are stretch work when the recruiter signals multimodal implementation.

Read source-tour stops 4 through 8 from [[hinterland/prep/inferact/core|the core map]]. Trace how the chosen architecture changes model construction, cache or encoder state, model-runner inputs, scheduling, parallelism, and weight loading.

### day 10: paper-fragment capstone

Use the 75-minute implementation block and thirty minutes of the system slice for the first M10 attempt. Preserve the final five minutes of the system slice for the first wrong architecture decision. P28 remains a distributed follow-up rather than the primary implementation.

Read:

- [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html)
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [[thoughts/distributed inference|distributed inference]]
- local `vllm/distributed/parallel_state.py`

Exit test: the paper-fragment model reaches end-to-end logits, cached decode parity, and deterministic top-two routing. Grade it with the model-build rubric as the route's targeted model mock. Then draw the TP and EP rank geometry and place every collective on an edge.

### day 11: Triton performance lane

Primary block: T04. Stretch: T02, T03, and T05. If the interview is confirmed PyTorch-only, replace the primary kernel with a P18 or P21 re-solve plus profiler analysis.

Read the [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/) and [[thoughts/GPU programming|GPU programming]].

Exit test: for one kernel, state the grid, tile, pointer formula, masks, bytes, operations, occupancy limit, and benchmark evidence in ninety seconds.

### day 12: deep-dive construction

Choose the project using this rubric, scored zero to three per row:

| axis                | three-point evidence                                                      |
| ------------------- | ------------------------------------------------------------------------- |
| inference relevance | scheduler, cache, kernel, compiler, distributed serving, or model runtime |
| ownership           | designed, implemented, debugged, and operated the mechanism               |
| measurement         | trace, benchmark method, before and after, variance, and ablation         |
| mechanism           | can derive the resource or correctness effect                             |
| failure evidence    | rejected paths, regression, rollback, or incident                         |
| Socratic durability | can redesign under model, hardware, workload, and SLO counterfactuals     |

Build:

- one-sentence claim
- architecture and request-lifecycle diagrams
- profiler trace
- benchmark and ablation tables
- correctness matrix
- failure ledger
- rollout and rollback timeline
- residual-risk slide

Use the deep-dive probes in [[hinterland/prep/inferact/role-drills#deep-dive hostile probes|role drills]].

### day 13: complete mock loop

Run the seventy-five-minute M10 re-solve, one forty-five-minute system mock, and a twenty-five-minute deep-dive from [[hinterland/prep/inferact/mocks|the mock set]]. Use the ordinary fifteen-minute test block for grading and put every miss into the next day's repair queue.

Grade each round before reviewing notes. Rewrite only the deep-dive slide that failed under questioning.

### day 14: repair and taper

Run a seventy-five-minute M01 reconstruction from an empty editor. If M01 is already clean, use the block for the weakest model-build or coding-mock repair. Run one twenty-minute system-design outline, one twenty-minute kernel explanation from grid through benchmark, and review the recall deck once.

Stop adding topics. Finish three hours before sleep. The marginal value of one more blog post is approximately dust.

## seven-day compression

This route assumes a baseline score of twelve or higher. M02 through M04 use supplied working starter models and test only the named architecture, cache, or runtime delta. They do not claim the full-route reconstruction gates.

### day 1

Primary block: P03 and P11. Stretch: P06, P08, and P14. Read tensor views and the sampler path.

### day 2

Primary block: M01. Use P15 and P18 as repairs. Draw the complete module tree and attention owner chain.

### day 3

Primary block: add the M02 Llama-style architecture delta to a supplied working M01 model. Use P04, P05, P14, P15, P16, and P18 as targeted repairs.

### day 4

Primary block: add the M03 cache contract to a supplied working M02 model. Trace the vLLM scheduler, KV cache manager, model runner, and sampler. Primary system slice: S03.

### day 5

Primary block: port a supplied working M02 model through the M04 flattened-token and loader contract. Primary system slice: S06. Use the deep-dive slice to build the claim, architecture, and evidence spine.

### day 6

Run coding mock 4 with the supplied executable baseline, system mock 1, and a twenty-five-minute hostile deep-dive. Cap grading at fifteen minutes and queue repairs for day 7.

### day 7

Run clean re-solves, one recall pass, one twenty-minute design outline, and one twenty-minute kernel explanation from grid through benchmark, then stop.

## one-day emergency route

This route trains survival, not coverage.

1. P03 in fifteen minutes.
2. P11 in thirty-five minutes.
3. P19 in forty-five minutes.
4. P21 in fifty minutes.
5. Draw the vLLM request lifecycle and derive KV bytes per token in thirty minutes.
6. Run S05 in twenty minutes.
7. Rehearse the project claim, trace, intervention, failure, result, and residual risk in forty minutes.
8. Review [[hinterland/prep/inferact/cheatsheet|the interview sheet]] once.

If the baseline model-and-attention dimension is zero, replace P03 and P19 with a sixty-minute scaffolded M01 attempt. The emergency route keeps the same total time.

## redo schedule

For every miss:

- first clean re-solve: later the same day after an unrelated task
- second clean re-solve: the next day
- third clean re-solve: three days later
- final check: one week later or the day before the interview

If the third solve still needs a hint, return to the tensor or state invariant. More random prompts will add entropy.

## error log

```text
drill:
date:
miss: contract | shape | aliasing | dtype | numerics | algorithm | API | testing | communication
first wrong decision:
owning invariant:
allocation and synchronization surprise:
next re-solve:
clean from blank editor: yes | no
```

## readiness gates

### full fourteen-day route

The package is ready when:

- P03 and P11 have clean later-day re-solves
- every P-series mechanism that scored zero or broke a model build has a clean later-day re-solve
- M01 has one clean reconstruction from an empty editor
- M03 cached next-token logits match uncached final-position logits across mixed prefix lengths
- M04 preserves the model's logits while changing its serving contract and checkpoint mapping
- M05 passes its full hidden-test matrix; M06 replaces day 11's Triton block and its fifteen-minute test block when multimodal implementation is confirmed
- M10 reaches end-to-end logits in seventy-five minutes and names exact cache, loader, parallelism, and runtime consequences
- one Triton kernel can be explained from pointer math to benchmark, even if the inference track never asks for code
- the vLLM request lifecycle can be drawn in five minutes with state owners
- KV capacity can be estimated from model config and corrected for sharding and blocks
- three system designs finish inside forty-five minutes with capacity and failure analysis
- the deep-dive survives thirty hostile questions without inventing data
- one complete coding, system, and deep-dive loop reaches the relevant rubrics' ready thresholds
- one additional targeted mock reaches 24 out of 28, with every dimension at least 2 out of 4

### seven-day compression

The compressed route is ready when:

- M01 has one clean build from an empty editor
- the scaffolded M02 architecture delta passes its reference oracle
- the scaffolded M03 cache path has cached-versus-uncached parity across mixed prefix lengths
- the scaffolded M04 port preserves logits and validates checkpoint packing
- coding mock 4 reaches 24 out of 28 with every dimension at least 2
- one complete coding, system, and deep-dive loop reaches the relevant rubrics' ready thresholds
- the shared vLLM lifecycle, KV arithmetic, system-design, deep-dive, and kernel-explanation gates above hold

The one-day route is triage. It exposes failure modes and does not claim the package-level readiness gate.
