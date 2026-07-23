---
date: '2026-07-21'
description: timed coding, system-design, and technical deep-dive mocks for Inferact
id: mocks
modified: 2026-07-23 03:45:29 GMT-04:00
tags:
  - cs
title: Inferact interview mocks
---

# timed mocks

Use a real timer and no agents. Start from a blank editor unless the mock explicitly supplies an executable scaffold; coding mock 4 is the named scaffolded exception. The guide tests independent coding despite Inferact's agent-heavy daily workflow.

Alternate mechanism rounds with complete model construction. The four implementation mocks in [[hinterland/prep/inferact/model-builds#model-construction mocks|the model-build lane]] cover architecture-from-prose, cache conversion, runtime porting, and an unfamiliar architecture fragment. [[hinterland/prep/inferact/gpt-lab|The tiny GPT lab]] supplies an executable model and cache canary. [[hinterland/prep/inferact/pytorch-practice|The PyTorch practice bank]] supplies unseen tensor and runtime transfer rounds after canonical owners are clean. [[hinterland/prep/inferact/programming-practice|The programming bank]] supplies general algorithms and stateful-code insurance.

Keep an auditable model-construction ratio in the mock log. Let `n` be all completed PyTorch coding rehearsals selected from fixed mocks 1 through 6, a sampled P-series or PT-series case, or model mock A through D. Model mock C counts once at its full seventy-five minutes. Exclude Triton mock 7, GP mock 8, and route drills that were not run as a mock. Let `m` be the rounds in `n` whose prompt required an `nn.Module` and whose frozen submission executed an end-to-end module path. Coding mock 4 and model mocks A through D can enter `m`; a standalone tensor function cannot. Require `m >= ceil(n / 2)` by interview day.

## coding-round structure

|   minute | action                                                                         |
| -------: | ------------------------------------------------------------------------------ |
|   0 to 5 | restate the contract, write shapes, reject invalid input, hand-run one example |
|  5 to 10 | choose operations, state aliasing and numerical-stability decisions            |
| 10 to 40 | implement the reference-quality solution                                       |
| 40 to 50 | test ordinary, degenerate, tail, dtype, and invalid cases                      |
| 50 to 58 | answer performance or runtime follow-up                                        |
| 58 to 60 | summarize costs, synchronization, and remaining risk                           |

## coding-mock sampler

Keep the eight fixed coding mocks as calibration rounds. Sample supplementary rounds this way:

1. run the canonical P-series drill or fixed mock while its owner is unresolved
2. otherwise alternate a model-construction round with one mock-eligible PT-series case from the weakest clean family
3. exclude the previous two PT IDs, owner families, and contract deltas
4. for a P-series or PT-series round, run the declared case time inside sixty minutes, use the remainder for hostile probes, and grade the artifact frozen at the declared deadline with the twenty-eight-point coding rubric
5. for model mocks A, B, or D, run the full sixty minutes and grade with the model-build rubric; run model mock C only in its full seventy-five-minute implementation block and never truncate it into this sixty-minute shell
6. record the prompt ID, prompt kind, canonical owner, duration, frozen-deadline score, and first wrong decision in the error log
7. return a failed PT case to its canonical owner and use a fresh sibling for the later transfer check; return a failed model mock to the first failed M-series or P-series owner

After the two uncapped programming calibrations, admit at most one GP-series round per earned replacement slot under [[hinterland/prep/inferact/index#round contracts|the counted-round rule]] when a calibration, mock, or recruiter signal exposes a general weakness and every PyTorch or GPT owner scheduled before or inside that replacement block is clean. An active remediation uses a different unseen sibling from the failed family and overrides the ordinary family exclusions. Otherwise select the weakest general family and exclude the previous two GP IDs and contract shapes. Grade with [[hinterland/prep/inferact/programming-practice#general-programming rubric|the separate general-programming rubric]]. A score from 19 through 23 with every dimension at least 2 goes directly to the sibling. A lower score or any dimension below 2 returns to its named algorithm or lifecycle family first.

## coding mock 1: batched generation primitives

**Main, 45 minutes:** implement P11, deterministic sampler.

**Extension, 15 minutes:** add per-row sampling parameters and a vocabulary-sharded top-k design sketch.

Probes:

- fp16 overflow
- all-masked rows
- top-p crossing token
- tie behavior
- RNG state under changing batch composition
- synchronization from scalar extraction

## coding mock 2: decode path

**Main, 35 minutes:** implement P19, single-token decode attention.

**Extension, 25 minutes:** append new K/V in place and support grouped-query attention without concatenating cache tensors.

Probes:

- shape ledger
- padded cache masks
- SDPA reference semantics
- cache data-pointer stability
- bandwidth-bound decode
- repeated K/V reads under GQA

## coding mock 3: paged KV

**Main, 45 minutes:** implement P21, paged-cache lookup.

**Extension, 15 minutes:** calculate capacity and discuss prefix-shared physical blocks with reference counts.

Probes:

- logical-to-physical mapping
- boundary tokens
- invalid block ids
- page size
- aliasing shared prefixes
- eviction policy versus storage layout

### prefix-cache runtime variation

Use this variation instead of coding mock 3's ordinary main and extension. The scaffold supplies the executable `CacheAwareCausalLM` with functional prefill and decode. Do not modify its M03 model contract.

**Main, 45 minutes:** implement a runtime `PrefixCache` and prefill runner over fixed-size physical blocks. Match the longest valid block-aligned prefix, run only the remaining suffix through the model, publish newly complete blocks, and reconstruct the returned per-layer cache.

**Acceptance, inside the same 45 minutes:**

- cache only complete immutable blocks; keep every partial tail request-private
- chain each block hash from parent hash, exact block tokens, and a namespace covering model and cache revision, position origin, adapters, media or prompt embeddings, and tenant salt
- cap a prompt of `T` tokens at `max(0, floor((T - 1) / B) * B)` reused tokens so a nonempty request still produces next-token logits
- verify exact tokens and namespace after a digest hit so a forced hash collision cannot alias state
- pin every acquired physical block until lease release; a released zero-reference block may remain resident
- enforce capacity in physical blocks with deterministic LRU eviction restricted to zero-reference chain leaves
- prove that two branched prompts share their common physical blocks

**Final 15 minutes:** add boundary tests at `0`, `B - 1`, `B`, `B + 1`, and `2B`; force a digest collision; hold two leases through capacity pressure; then prove cached-suffix logits and every layer's reconstructed K/V match full uncached prefill.

Probes:

- why an exactly block-aligned prompt recomputes its final block
- why the parent hash is part of semantic identity
- ownership versus residency
- block capacity versus prefix-entry capacity
- lock scope and lease cleanup
- cache invalidation after a model, adapter, media, salt, or position change

## coding mock 4: model from a paper fragment

The mock partner supplies this architecture fragment:

```text
normalize hidden state
project Q with Hq heads and K/V with Hkv heads
apply rotary position embeddings
attend over cached K/V
project output and add residual
route each token to two of E experts
combine expert outputs by normalized router weight
```

**Main, 45 minutes:** implement the architecture delta inside a supplied executable M01-style model. Preserve its config and tensor contracts and prove an end-to-end logits path.

**Extension, 15 minutes:** add the hidden tests, then explain cached-versus-uncached behavior, model loading, tensor parallelism, quantization, and vLLM integration points.

Probes:

- parameter and buffer registration
- GQA divisibility
- position semantics
- expert capacity
- checkpoint weight mapping
- baseline recovery when all experts are identical
- cached-versus-uncached parity
- exact model, cache, loader, runner, and sampler boundaries
- compile and custom-op boundaries

## coding mock 5: persistent batch mutation

**Main, 40 minutes:** implement P22, live-request compaction over a dictionary of tensor fields.

**Extension, 20 minutes:** admit new requests into freed rows while preserving request-to-row identity.

Probes:

- atomicity across fields
- stale-row leakage
- CPU and GPU bookkeeping
- graph capture
- cancellation
- deterministic ordering

## coding mock 6: quantized model component

**Main, 40 minutes:** implement P23 and P24 for int8 per-output-channel weights.

**Extension, 20 minutes:** design a fused dequantization and matmul kernel and state the accuracy evaluation.

Probes:

- qmin and qmax
- zero point
- scale axis
- accumulation dtype
- memory reduction
- why KV quantization needs separate analysis

## coding mock 7: Triton softmax

**Main, 45 minutes:** implement T03, fused stable softmax.

**Extension, 15 minutes:** add ragged row lengths and state when one row stops fitting on chip.

Probes:

- power-of-two padding
- safety and semantic masks
- negative infinity
- approximate exponentiation
- register pressure
- benchmark warmup

## coding mock 8: general distributed programming

When this round replaces an unfinished calibration, implement the exact outstanding case: GP01 in twenty-five minutes, GP10 in forty minutes, or the previously selected GP04 or GP19 in forty-five minutes. Use Python and apply the extension contract below for the remaining thirty-five, twenty, or fifteen minutes. It counts as that calibration rather than a supplementary mock.

When both calibrations are complete and remediation is active, implement the exact unseen same-family sibling named in the error log for its declared time and use the remainder of sixty minutes for the extension. This round spends the earned GP slot.

**Main after both calibrations and with no active remediation:** implement one core case from [[hinterland/prep/inferact/programming-practice|the programming bank]] for that case's declared twenty-five-to-forty-five-minute timebox. Follow the bank's language rule.

**Extension after both calibrations:** use the remainder of sixty minutes. For a stateful or concurrent case, add cancellation, close semantics, and one distributed failure scenario. For a sequence, graph, or parser case, add an online constraint plus an inference-runtime ownership and failure sketch.

Choose the four relevant probes:

- fairness
- starvation
- byte and token limits
- ownership
- backpressure
- deterministic tests

Freeze the base artifact at its declared deadline and grade that snapshot with the general-programming rubric rather than the tensor and numerics rubric below. The remaining mock time is an unscored extension for probes, lifecycle variants, and diagnosis. Post-deadline repairs cannot change the calibration, remediation, or readiness score.

## system-design structure

|   minute | action                                                               |
| -------: | -------------------------------------------------------------------- |
|   0 to 7 | define model, workload distribution, hardware, SLOs, and quality     |
|  7 to 15 | estimate weights, KV, concurrency, compute phases, and communication |
| 15 to 25 | draw request and state owners                                        |
| 25 to 35 | choose cache, batching, routing, parallelism, and overload policies  |
| 35 to 42 | add failure recovery, observability, and rollback                    |
| 42 to 45 | name the two experiments that choose among designs                   |

## system mock 1: agentic chat fleet

Design a multi-tenant vLLM service for long-horizon agents with a large shared tool prefix, bursty parallel tool calls, mixed short and long turns, and strict streaming SLOs.

Required decisions:

- prefix-aware routing versus load balance
- DP replica count and model parallelism inside a replica
- continuous batching and chunked-prefill token budget
- cache salt and tenant isolation
- cancellation after tool calls
- autoscaling under warmup, graph capture, and cold KV caches

Required metrics:

- p95 TTFT
- p99 ITL
- goodput
- queue wait
- prefix hit tokens divided by query tokens
- KV utilization and preemptions
- cancellation waste

## system mock 2: prefill and decode disaggregation

Design separate prefill and decode fleets for long prompts and short interactive outputs.

Required decisions:

- P:D replica ratio
- KV connector and transport
- placement relative to storage and interconnect
- request and cache ownership
- transfer retry, expiration, and recompute fallback
- independent scaling signals

Derive KV transfer bytes for one request and compare the transfer time with the TTFT or ITL budget.

## system mock 3: MoE on heterogeneous accelerators

Serve a large MoE model across two fast GPU nodes and one slower spillover pool.

Required decisions:

- TP, PP, DP, and EP geometry
- expert placement and replication
- all-to-all topology
- routing skew and stragglers
- admission policy for the slower pool
- correctness and performance gates across backends

## system mock 4: hybrid multimodal model

Serve a model with text attention layers, recurrent state-space layers, and a vision encoder.

Required decisions:

- separate cache specs for KV and recurrent state
- block alignment and prefix validity
- media processing and encoder-cache identity
- scheduler budgets for text tokens and encoder work
- cache invalidation after model or media changes
- overload and cancellation behavior

## system mock 5: production regression

After a release, output throughput remains flat while p99 ITL regresses 35% and only mixed-length batches are affected.

Design the investigation:

1. verify workload and metric parity
2. separate scheduler, CPU gap, attention backend, graph replay, cache, and collective hypotheses
3. choose traces, counters, and controlled ablations
4. define rollback threshold
5. design the regression test and benchmark gate

## deep-dive mock 1: twenty-five-minute primary project

Use this shape:

| time | section                                          |
| ---: | ------------------------------------------------ |
|   1m | claim, workload, result, and residual risk       |
|   3m | problem, baseline, and SLO                       |
|   4m | architecture and personal boundary               |
|   5m | trace or benchmark evidence and causal diagnosis |
|   5m | design alternatives and implementation           |
|   3m | correctness and quality                          |
|   2m | deployment, incident, or rollback                |
|   2m | failure, lessons, and next experiment            |

The mock partner interrupts at least every three minutes with a probe from [[hinterland/prep/inferact/role-drills#deep-dive hostile probes|the hostile list]].

## deep-dive mock 2: counterfactual attack

Present the project in ten minutes. Spend thirty minutes redesigning it under four counterfactuals chosen by the mock partner:

- ten times longer context
- one tenth the request rate
- a different accelerator backend
- MoE instead of a dense model
- multimodal inputs
- strict determinism
- half the memory
- disaggregated prefill and decode
- one failed rank
- a tighter p99 SLO

Answers must say which assumption changed, which mechanism stops holding, what design moves, and which measurement decides the new choice.

## coding rubric

Give each dimension zero to four points:

| dimension      | zero                               | two                                 | four                                                             |
| -------------- | ---------------------------------- | ----------------------------------- | ---------------------------------------------------------------- |
| contract       | solves a different problem         | main behavior found after prompting | shapes, errors, mutation, ties, dtype, and device stated first   |
| tensor model   | axes or broadcasting are wrong     | correct after repair                | shapes, strides, aliasing, and allocation stay explicit          |
| numerics       | unstable or undefined invalid rows | ordinary values work                | accumulation dtype, masks, and degenerate rows are deliberate    |
| implementation | does not run                       | mostly works with repair            | clear PyTorch that passes systematic tests                       |
| testing        | no useful tests                    | happy path and one edge             | randomized reference, tails, low precision, invalid input        |
| performance    | claims speed without a mechanism   | gives broad complexity              | names bytes, operations, synchronization, allocation, bottleneck |
| communication  | interviewer loses the state        | understandable with gaps            | every change stays attached to an invariant                      |

Award one when some relevant evidence exists but the two-point behavior is not yet workable. Award three only when the two-point behavior is fully satisfied and one bounded gap prevents the four-point description. Use integer scores only.

Interpret the total out of 28:

- 24 to 28 with every dimension at least 2: ready for this shape
- 19 to 23: one clean re-solve
- 13 to 18: return to the owning invariant
- 0 to 12: learn the mechanism before another mock

## system-design rubric

Give each dimension zero to four:

- workload and SLO definition
- capacity arithmetic
- state ownership
- scheduling, cache, and parallelism choices
- tradeoff and rejected alternative
- failure and overload behavior
- observability and deciding experiments

Use the same integer scale for every dimension: zero is absent or wrong; one is a fragment that needs rescue; two is a coherent baseline with a material gap; three is independently correct with one bounded gap; four survives the named probes with grounded arithmetic and explicit tradeoffs.

A design is ready at 24 out of 28 with every dimension at least 2. Missing arithmetic caps the score at 20 because the boxes remain ungrounded.

## deep-dive rubric

Give each dimension zero to four:

- claim and personal ownership
- workload and baseline evidence
- causal diagnosis
- implementation mechanism
- correctness and quality
- failure, deployment, and residual risk
- counterfactual durability

Use the same integer scale for every dimension: zero is absent, invented, or wrong; one is an unsupported fragment; two is a coherent account with a material evidence gap; three is independently defensible with one bounded gap; four survives hostile probes with ownership and evidence intact.

A deep dive is ready at 24 out of 28 with every dimension at least 2. Any invented metric, hidden co-owner, or causal claim without evidence resets the relevant dimension to zero.

## full-loop rehearsal

Run this once while at least one later repair day remains. Three or more days before the interview is preferable:

1. one coding mock, 60 minutes
2. break, 10 minutes
3. primary-project deep dive, 40 minutes including interruption
4. break, 10 minutes
5. one system design, 45 minutes
6. written repair plan, 15 minutes

The loop totals 180 minutes. Run a second loop with different prompts only after the repair tasks have clean re-solves.
