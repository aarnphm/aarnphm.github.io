---
date: '2026-07-21'
description: adaptive study and re-solve plan for the Inferact technical loop
id: study
modified: 2026-07-23 03:45:29 GMT-04:00
tags:
  - cs
title: Inferact study route
---

# study route

The default route is fourteen days at three focused hours per day. It yields forty-two scheduled hours with 45% assigned to PyTorch model construction and mechanism repair. The baseline can compress this to seven days. Each day names a primary implementation block that fits inside seventy-five minutes. Longer model builds continue across named slices. [[hinterland/prep/inferact/pytorch-practice|The PyTorch practice bank]] replaces a repair or mock slot only after its canonical owner is clean, so its inventory adds transfer coverage without adding required hours. [[hinterland/prep/inferact/programming-practice|The programming bank]] also consumes replacement slots and remains capped at one general case per three completed timed PyTorch coding rounds after its two uncapped calibrations. [[hinterland/prep/inferact/gpt-lab|The tiny GPT lab]] is the executable M01 canary and the simple M03 alternative. Miss-driven clean re-solves are additional unless a later stretch, reading, or practice block is explicitly replaced.

Every standalone drill, first non-scaffolded model-build slice, non-scaffolded coding mock, and named clean reconstruction begins with a blank editor. Continuation slices resume the previous build artifact. The compressed M02 through M04 deltas and coding mock 4 start from their supplied working models. System-design and deep-dive rounds start from their stated prompt and evidence artifacts. Every session ends with a written first-wrong-decision log. Passive reading is capped at one third of the block.

## daily loop

| minutes | action                                                                                            |
| ------: | ------------------------------------------------------------------------------------------------- |
|      15 | alternate ten recall cards with eight PQ questions from four lanes                                |
|      75 | implement one model slice, GPT lab tier, canonical repair, sampled PT or GP case, or Triton drill |
|      35 | read one owning code path or official design document, then draw its state boundary               |
|      35 | rehearse one system-design slice or one deep-dive slice aloud                                     |
|      15 | test, profile, and answer Socratic probes                                                         |
|       5 | record the first wrong decision and schedule the re-solve                                         |

An ordinary sixty-minute mock and its repair pass replace the middle 145-minute region. A named full-loop rehearsal replaces the complete 180-minute daily loop. Reference canaries and candidate grading use the final fifteen-minute test block, preserving the seventy-five-minute implementation block.

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
- 8 to 11, or 12 to 15 with any zero dimension: use days 1 through 10, then days 12 through 14
- 12 to 15 with no zero dimension: use the seven-day compression route

A zero forbids compression and preserves its owning block: tensor semantics maps to day 1, numerical correctness to day 2, model and attention mechanics to days 3 and 4, cache and runtime ownership to days 5 through 8, and communication under pressure to days 12 and 13.

For the full and intermediate routes, freeze P19 at its forty-five-minute baseline deadline and record the M03 branch before day 5. Select the simple GPT branch when the artifact fails padded-length exclusion, full-versus-final-token parity, or caller-owned cache immutability. Select the advanced M02-shaped branch only when all three pass. Later repair changes owner readiness and does not rewrite the recorded branch or its time allocation. The compressed route overrides this selector and always uses its supplied M02-shaped artifact for advanced M03.

## route calibrations

The general-programming calibration has a separate score from the fifteen-point PyTorch and runtime baseline. Run its two cases inside the day 1 and day 2 replacement slices described below:

|   time | task                                                                                   | signal                                          |
| -----: | -------------------------------------------------------------------------------------- | ----------------------------------------------- |
|    25m | GP01, stable active-row compaction                                                     | arrays, maps, inverse invariants, invalid input |
| 40–45m | choose GP04, GP10, or GP19 for cache, graph, or concurrency according to weakest prior | ownership, state, tests, and failure semantics  |

Grade both with [[hinterland/prep/inferact/programming-practice#general-programming rubric|the general-programming rubric]]. Any score below 24, or any dimension below 2, activates at most one GP replacement slot per three completed timed PyTorch coding rounds until a different sibling reaches the ready threshold. Spend it only when every PyTorch or GPT owner scheduled before or inside the replacement block is clean. A score from 19 through 23 with every dimension at least 2 goes directly to a different sibling. A lower score, or any dimension below 2, first returns to the governing pattern and then tests a sibling. The two calibration rounds sit outside this cap. These scores do not change the fifteen-point PyTorch route selection.

The calibrations occupy replacement time inside days 1 and 2, so they do not extend the forty-two-hour route. GP01 uses twenty-five minutes of day 1's system and deep-dive slice. The second case may use up to day 2's thirty-five-minute system and deep-dive slice plus ten minutes of its source-reading slice. GP10 ends after forty minutes and uses the remaining five minutes to state its atomic rejection invariant.

## fourteen-day route

### day 1: tensor metadata and vectorization

Primary block: P01 through P03. Stretch: P06.

Programming calibration: run GP01 in twenty-five minutes, grade it, and record the first wrong decision. Use the remaining ten minutes of the replaced slice to explain its stable permutation and inverse invariant.

Read:

- [PyTorch tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [`Tensor.view`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html)
- [broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
- [`torch.gather`](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)

Exit test: given any shape and stride, explain whether flattening can remain a view and where the next operation allocates.

### day 2: logits, sampling, and numerical stability

Primary block: P10 and P11. Stretch: P07 through P09.

Programming calibration: choose GP04, GP10, or GP19 from the weakest prior surface and run its declared forty-five, forty, or forty-five-minute contract using the replacement time defined above.

Read the vLLM sampler files from [[hinterland/prep/inferact/core#current vLLM source tour|source-tour stop 9]]. Compare the implementation strategy with `torch.multinomial` without copying vLLM code into the drill.

Exit test: implement a complete batched sampler in forty minutes, including the all-masked and temperature-zero contracts.

### day 3: first complete causal LM

Primary block: M01 from [[hinterland/prep/inferact/model-builds|the model-build lane]]. Use P03, P05, P17, or P18 only when the build exposes a failed shape, mask, module-state, or attention invariant.

Use the final fifteen-minute test block to run the reference canary and the non-cache tiers from [[hinterland/prep/inferact/gpt-lab|the tiny GPT lab]], then grade the blank-editor M01 build against the same contracts.

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

Primary block: finish the remaining forty-five minutes of M02, then spend thirty minutes on the recorded M03 branch. For the simple branch, complete stage 4 of the GPT lab from the M01 artifact. For the advanced branch, define M03's cache, position, and output contracts on the M02 artifact and build its one-layer prefill/decode test. Use P19, P20, and P21 only when decode attention, cache append, or logical-to-physical lookup fails.

Read:

- [[thoughts/paged attention|paged attention]]
- [vLLM prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- local `vllm/v1/kv_cache_interface.py`
- local `vllm/v1/core/kv_cache_manager.py`

Exit test: the complete Llama-style model produces logits and stable checkpoint names. On the simple branch, one cache pair per layer validates and functional append leaves caller-owned cache tensors unchanged. On the advanced branch, the executable one-layer prefill/decode test covers cache-input immutability, rectangular absolute-position masking, and full-versus-cached last-token parity. A failed exit enters the redo schedule without changing the baseline-selected branch.

### day 6: cache-aware model completion

Primary block: finish M03. Use P19 through P22 as cache and batching repairs. Use P23 and P24 as quantization follow-ups after cache parity passes.

Use the final fifteen-minute test block for the checked-in simple canary and candidate grading. Spend the uninterrupted M03 implementation block on the branch recorded after baseline P19. The simple branch continues with stages 5 and 6 of the homogeneous-cache GPT. The advanced branch continues the M02 artifact through mixed prefix lengths, explicit absolute positions, and the adapter suite. A failed day-5 exit enters the redo schedule without switching branches.

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

### day 11: prefix cache or Triton performance lane

For the PyTorch-inference track, use the first fifty minutes of the implementation block for [[hinterland/prep/inferact/gpt-lab#runtime prefix-cache extension|the exact-prefix snapshot cache]] over the passing `CacheAwareCausalLM`. Use the remaining twenty-five minutes to map that code onto physical blocks, chained hashes, leases, and page tables. Keep M03's recorded simple-or-advanced branch unchanged. Use the final fifteen-minute test block to run `labs/test_prefix_cache.py`, inspect the first failed owner, and rerun the narrow case. For a role that confirms kernel coding, use T04 as the primary block and T02, T03, and T05 as stretches. When a programming calibration remains below threshold, the weakest GP family may replace the branch not selected for the interview after every required PyTorch owner is clean.

For the PyTorch-inference track, read the [vLLM automatic prefix-caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/) and trace [[hinterland/prep/inferact/core#priority 5: vLLM request lifecycle|the runtime owner chain]]. For the kernel track, read the [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/) and [[thoughts/GPU programming|GPU programming]].

PyTorch-inference exit test: the executable cache finds a branched prompt's exact common prefix, partial tails create no entry, entry-level LRU is deterministic, and cached suffix logits plus final K/V match full recomputation. Then explain how physical blocks, hash collision checks, live leases, and page-table eviction replace the snapshot representation in production. Kernel-track exit test: state the grid, tile, pointer formula, masks, bytes, operations, occupancy limit, and benchmark evidence in ninety seconds.

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

### day 13: full-loop rehearsal

Run the exact 180-minute full loop from [[hinterland/prep/inferact/mocks#full-loop rehearsal|the mock set]]: sixty minutes of coding, two ten-minute breaks, a forty-minute deep dive, a forty-five-minute system design, and a fifteen-minute written repair plan. Use coding mock 4 by default. Use coding mock 8 when a general calibration missed, or another fixed mock when its owner is weaker.

Grade each round before reviewing notes. Day 14 owns the first clean repair for a miss from this loop.

### day 14: repair and taper

Run a seventy-five-minute M01 reconstruction from an empty editor. If M01 is already clean, complete any outstanding required M02 one-layer or selected-M03 cache reconstruction before using the block for another model-build, GPT-lab, GP-family, or coding-mock repair. If every required reconstruction and repair is clean, draw one unseen case from [[hinterland/prep/inferact/pytorch-practice|the PyTorch practice bank]] in the weakest clean family. Run one twenty-minute system-design outline, one twenty-minute kernel explanation from grid through benchmark, and review the recall deck once.

Stop adding topics. Finish three hours before sleep. The marginal value of one more blog post is approximately dust.

## seven-day compression

This route assumes a baseline score of twelve or higher with no zero dimension. M02 through M04 use supplied working starter models and test only the named architecture, cache, or runtime delta. They do not claim the full-route reconstruction gates.

### day 1

Primary block: P03 and P11. Stretch: P06, P08, and P14. Read tensor views and the sampler path. Run GP01 for twenty-five minutes in the system and deep-dive replacement slice.

### day 2

Primary block: M01. Use P15 and P18 as repairs. Draw the complete module tree and attention owner chain. Run GP04, GP10, or GP19 for its declared forty-five, forty, or forty-five minutes in the same replacement slices used by the full route.

### day 3

Primary block: add the M02 Llama-style architecture delta to a supplied working M01 model. Use P04, P05, P14, P15, P16, and P18 as targeted repairs.

### day 4

Primary block: use the compressed route's declared scaffolded exception to add the advanced M03 cache contract to a supplied working M02 model in seventy-five minutes. Use the separate fifteen-minute test block to grade it through the executable adapter contract. Trace the vLLM scheduler, KV cache manager, model runner, and sampler. Primary system slice: S03.

### day 5

Primary block: port a supplied working M02 model through the M04 flattened-token and loader contract. Primary system slice: S06. Use the deep-dive slice to build the claim, architecture, and evidence spine.

### day 6

Run the exact 180-minute full-loop rehearsal with coding mock 4, the primary-project deep dive, and system mock 1. Use its fifteen-minute repair plan to queue clean re-solves for day 7.

### day 7

Run clean re-solves, one recall pass, one twenty-minute design outline, and one twenty-minute kernel explanation from grid through benchmark. A fresh PT-series case may replace one re-solve only when every required owner is already clean. When a programming calibration missed, solve a different unseen sibling from that family at least one day later before any supplementary PT case. A score below 19 or any dimension below 2 requires family repair first. Then stop.

## one-day emergency route

This route trains survival, not coverage. The numbered work takes 235 minutes. Cap the final interview-sheet review at ten minutes, for a 245-minute route.

1. P03 in fifteen minutes.
2. P11 in thirty-five minutes.
3. P19 in forty-five minutes.
4. P21 in fifty minutes.
5. Draw the vLLM request lifecycle and derive KV bytes per token in thirty minutes.
6. Run S05 in twenty minutes.
7. Rehearse the project claim, trace, intervention, failure, result, and residual risk in forty minutes.
8. Review [[hinterland/prep/inferact/cheatsheet|the interview sheet]] once.

If the baseline model-and-attention dimension is zero, replace P03 and P19 with a sixty-minute partial M01 build from an empty editor: implement the validated config, embeddings, one pre-norm attention-and-MLP block, final normalization, and logits. Grade causality and shape only; cache, loss, and serialization remain outside this triage attempt. The emergency route keeps the same total time and does not claim the M01 readiness gate.

The emergency route does not draw from the supplementary practice bank. It diagnoses canonical gaps.

## redo schedule

For a canonical P-series or M-series miss, use later scheduled blocks where possible and keep the checks in this order:

- first clean re-solve: later the same day after an unrelated task, or the next day after a full-loop rehearsal
- second clean re-solve: the next day
- retention check: three days later
- final retention check: one week later

The full route also has two named model reconstructions even when the first attempts pass. Rebuild one M02 Llama-style decoder layer from an empty file in forty-five minutes on a later day and require randomized equality with the decomposed RMSNorm, RoPE, GQA, and SwiGLU oracle. Rebuild the selected M03 cache slice from an empty file in forty-five minutes on a later day: homogeneous cache plus rectangular positions for the simple branch, or mixed prefix lengths plus explicit cache metadata for the advanced branch. Require cached-versus-full parity and cache-input immutability. These checks may replace a later stretch, reading, or day-14 repair block after M01 is clean; otherwise they are additional work.

When the interview arrives before a future retention interval, use the day before as the last available check after all earlier completed checks. Do not reverse the sequence to manufacture spacing.

After the owner is clean, replace its next repetition with a fresh PT-series case from the same family. If that case fails, repair the owner, re-solve it once, and test transfer with a different PT case three days later. Repeating the same supplementary prompt proves recall of the prompt.

For a GP-series miss from 19 through 23 with every dimension at least 2, solve one different unseen sibling on a later day. For a lower score or any dimension below 2, first repair the named algorithm, representation, or lifecycle family later the same day and then solve the sibling. Require 24 out of 28 with every dimension at least 2 for readiness. A three-days-later sibling is a retention check rather than a second readiness gate. A general case never substitutes for an unresolved P-series or M-series owner.

For a system-design miss, produce a corrected one-page artifact later the same day with the workload, SLO, capacity arithmetic, state owners, overload and failure policy, rejected alternative, and deciding experiment. On the next available day, run a fresh forty-five-minute prompt from a different S-series family and require 24 out of 28 with every dimension at least 2. A three-days-later twenty-minute outline checks retention when time remains.

For a deep-dive miss, repair the same project's evidence sheet later the same day: claim and personal ownership, workload and baseline, causal mechanism, implementation, correctness evidence, failure, rollout, and residual risk. Remove or source every invented number. On the next available day, rerun a forty-minute hostile round on the same project with unseen probes and counterfactuals and require 24 out of 28 with every dimension at least 2. A second complete 180-minute loop is optional after every failed constituent round has passed its own clean re-solve.

## error log

```text
drill:
canonical_owner:
source_owner:
family:
language:
mode: canonical | transfer | mock
contract_delta:
previously_seen: yes | no
date:
miss: contract | shape | aliasing | dtype | numerics | algorithm | API | testing | communication
first wrong decision:
owning invariant:
failure_or_cancellation_invariant:
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
- the tiny-GPT construction and M01 numerical tiers pass against that reconstruction, and the checked-in reference passes the complete suite
- M02's complete Llama-style model passes its randomized decomposed oracle and stable checkpoint schema, then one decoder layer passes the named forty-five-minute later-day reconstruction
- M03 follows one timed path: a blank reconstruction of the simple GPT cache and generation tiers with homogeneous-prefix parity, or the advanced mixed-prefix extension with all ten adapter tests passing against the actual M02-shaped artifact
- the selected M03 cache slice passes its named forty-five-minute later-day reconstruction with cache-input immutability and cached-versus-full parity
- when day 11 uses the PyTorch-inference lane, the separate prefix-cache suite passes physical full-block sharing, exact post-hash identity, private tails, lease-safe leaf eviction, and cached-versus-full suffix logits and K/V parity
- M04 preserves the model's logits while changing its serving contract and checkpoint mapping
- M05 passes its full hidden-test matrix; M06 replaces day 11's Triton block and its fifteen-minute test block when multimodal implementation is confirmed
- M10 reaches end-to-end logits in seventy-five minutes and names exact cache, loader, parallelism, and runtime consequences
- one Triton kernel can be explained from pointer math to benchmark, even if the inference track never asks for code
- the vLLM request lifecycle can be drawn in five minutes with state owners
- KV capacity can be estimated from model config and corrected for sharding and blocks
- three system designs finish inside forty-five minutes with capacity and failure analysis
- the deep-dive survives thirty hostile questions without inventing data
- one complete coding, system, and deep-dive loop reaches the relevant rubrics' ready thresholds
- one additional targeted mock or unseen mock-eligible PT-series case reaches 24 out of 28, with every dimension at least 2 out of 4
- both general-programming calibrations are complete; after a score from 19 through 23 with every dimension at least 2, one different later-day GP sibling reaches 24 out of 28 with every dimension at least 2; after a lower score or any dimension below 2, the governing family is repaired before that sibling

### seven-day compression

The compressed route is ready when:

- M01 has one clean build from an empty editor
- the checked-in tiny-GPT reference passes every tier, the blank M01 passes its construction and numerical tiers, and the scaffolded advanced M03 path passes its executable adapter suite
- the scaffolded M02 architecture delta passes its reference oracle
- the scaffolded advanced M03 path has cached-versus-uncached parity across mixed prefix lengths
- the scaffolded M04 port preserves logits and validates checkpoint packing
- coding mock 4 reaches 24 out of 28 with every dimension at least 2
- one complete coding, system, and deep-dive loop reaches the relevant rubrics' ready thresholds
- the shared vLLM lifecycle, KV arithmetic, system-design, deep-dive, and kernel-explanation gates above hold
- when either programming calibration misses, one different later-day sibling reaches 24 out of 28 with every dimension at least 2; a score below 19 or any dimension below 2 requires governing-family repair first

### intermediate route

The baseline score of eight through eleven uses days 1 through 10 and days 12 through 14. It inherits the full-route gates for every scheduled P-series and M-series owner, the tiny-GPT tiers, and any GP-family repair activated by calibration. It moves the omitted day 11 kernel explanation into day 14 and may use an unseen mock-eligible PT-series case only as the additional targeted mock after all required owners are clean.

The one-day route is triage. It exposes failure modes and does not claim the package-level readiness gate.
