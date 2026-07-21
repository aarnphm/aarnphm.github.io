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

The default route is fourteen days at three focused hours per day. It yields forty-two hours split roughly 19:8:6:5:4 across PyTorch, vLLM, system design, deep dive, and Triton. The baseline can compress this to seven days. Each day names a primary implementation block that fits inside seventy-five minutes. Stretch drills enter the re-solve queue only after the exit test passes.

Every session begins with a blank editor and ends with a written first-wrong-decision log. Passive reading is capped at one third of the block.

## daily loop

| minutes | action                                                                              |
| ------: | ----------------------------------------------------------------------------------- |
|      15 | recall ten cards and state yesterday's weakest invariant                            |
|      75 | implement one primary PyTorch or Triton drill from an empty file                    |
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
- 8 to 11: use days 1 through 10, then days 13 and 14
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

### day 3: beam state, normalization, and modules

Primary block: P14 and P17. Stretch: P12, P13, and P16.

Read:

- [module notes](https://docs.pytorch.org/docs/stable/notes/modules.html)
- [autograd mechanics and grad modes](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- the Llama MLP and decoder-layer classes in local `vllm/model_executor/models/llama.py`

Exit test: explain parameter, buffer, request-state, and cache ownership without mixing them.

### day 4: RoPE and reference attention

Primary block: P18. Stretch: P04, P05, and P15.

Read:

- [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- local `vllm/model_executor/models/llama.py`
- local `vllm/model_executor/layers/attention/attention.py`

Draw the dense decoder block and label every tensor shape. Exit only after the PyTorch reference path matches the manual path over randomized small shapes.

### day 5: decode and KV storage

Primary block: P21. Stretch: P19 and P20. The baseline already exercised P19, so schedule it as a later-day re-solve when it scored below three.

Read:

- [[thoughts/paged attention|paged attention]]
- [vLLM prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- local `vllm/v1/kv_cache_interface.py`
- local `vllm/v1/core/kv_cache_manager.py`

Exit test: derive logical token to physical block and slot, then explain the policy/physical-storage boundary.

### day 6: persistent batching and quantization

Primary block: P23 and P24. Stretch: P22.

Read:

- [[thoughts/quantization|quantization]]
- local `vllm/v1/worker/gpu_model_runner.py` state-update and input-preparation sections
- local quantization base interfaces and one FP8 method

Exit test: explain what batch state can mutate without invalidating a captured graph and which quantization scales broadcast along each axis.

### day 7: model variants

Primary block: P25. Stretch: P26. Use the architecture slice to sketch a tiny dense decoder from config with the mechanisms from P14 through P18.

For a paper-to-code rehearsal, take one model diagram and write:

1. persistent parameters
2. per-request state
3. cache state
4. tensor shapes
5. reference forward
6. vLLM interfaces stressed by the architecture

Read the live role again. Explain dense, GQA, MoE, multimodal, hybrid attention/SSM, and diffusion serving at mechanism depth.

### day 8: compiler and CUDA execution

Primary block: P27 and T01 when a Triton environment is available. Stretch: T03.

Read:

- [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [custom operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- local vLLM compilation decorators, backend, and CUDA-graph dispatcher

Exit test: draw eager execution, compiled execution, a custom-op boundary, and graph replay. Name the guard and stable-address assumptions.

### day 9: vLLM scheduler and request lifecycle

Read source-tour stops 6 through 9 from [[hinterland/prep/inferact/core|the core map]]. Trace one request through:

```text
waiting -> scheduled tokens -> cache allocation -> model execution -> sampling -> output update -> finish and free
```

Implement one small scheduler simulation or reuse R07 and R15 from [[hinterland/prep/nv/role-drills|the NVIDIA role drills]].

Run system prompts S03 and S05 for twenty minutes each.

### day 10: distributed inference

Implement P28 or simulate collectives in one process when the environment cannot launch ranks.

Read:

- [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html)
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [[thoughts/distributed inference|distributed inference]]
- local `vllm/distributed/parallel_state.py`

Primary system slice: S06. Stretch: S10 and S11. For each attempted prompt, draw rank geometry and place every collective on an edge.

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

Run coding mock 2, system mock 1, and a twenty-five-minute deep-dive from [[hinterland/prep/inferact/mocks|the mock set]]. Cap grading at fifteen minutes and put every miss into the next day's repair queue.

Grade each round before reviewing notes. Rewrite only the deep-dive slide that failed under questioning.

### day 14: repair and taper

Run one forty-five-minute coding mock chosen from the weakest lane. Run one twenty-minute system-design outline. Review the recall deck once.

Stop adding topics. Finish three hours before sleep. The marginal value of one more blog post is approximately dust.

## seven-day compression

### day 1

Primary block: P03 and P11. Stretch: P06, P08, and P14. Read tensor views and the sampler path.

### day 2

Primary block: P18. Stretch: P15, P19, and P20. Draw the dense decoder and attention owner chain.

### day 3

Primary block: P21. Stretch: P23 and P25. Use the architecture slice for one paper-to-code model sketch.

### day 4

Trace the vLLM scheduler, KV cache manager, model runner, and sampler. Primary system slice: S03. Stretch: S05.

### day 5

Primary block: P27 or T03. Primary system slice: S06. Stretch: S10. Use the deep-dive slice to build the claim, architecture, and evidence spine.

### day 6

Run coding mock 2, system mock 1, and a twenty-five-minute hostile deep-dive. Cap grading at fifteen minutes and queue repairs for day 7.

### day 7

Clean re-solves only, one recall pass, one twenty-minute design outline, then stop.

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

The package is ready when:

- P03, P11, P18, P19, P20, P21, P23, P25, and P27 have clean later-day re-solves
- one Triton kernel can be explained from pointer math to benchmark, even if the inference track never asks for code
- the vLLM request lifecycle can be drawn in five minutes with state owners
- KV capacity can be estimated from model config and corrected for sharding and blocks
- three system designs finish inside forty-five minutes with capacity and failure analysis
- the deep-dive survives thirty hostile questions without inventing data
- two complete mocks reach the relevant rubric's ready threshold of 24 out of 28, with every dimension at least 2 out of 4
