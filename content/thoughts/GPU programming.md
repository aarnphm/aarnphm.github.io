---
id: GPU programming
aliases:
  - gpus
tags:
  - seed
  - ml
date: "2025-09-08"
modified: 2025-09-18 18:10:41 GMT-04:00
noindex: true
title: GPU
---

> [!abstract]
> modal glossary entry: https://modal.com/gpu-glossary/device-hardware/cuda-device-architecture

> [!tip]
> uccl project: https://github.com/uccl-project/uccl

> [!note]
> mostly covers nvidia, but the architectural patterns apply to amd and intel discrete gpus with similar simt front-ends

## Architecture Overview

See [[lectures/420/notes|comprehensive GPU architecture notes]] for detailed coverage of GPU fundamentals, CUDA programming model, and optimization techniques.

> [!info] core terminology
>
> | concept                       | summary                                                                                                                                                                                              | jump-off                                                                  |
> | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
> | gpu vs cpu                    | throughput-optimized accelerator (270k+ resident threads on [[lectures/420/notes#thread count comparison: epyc vs h100 \| hopper]]) vs latency-optimized cpu (≈192 hardware threads on 96-core epyc) | [[lectures/420/notes#cpu vs gpu: philosophical differences]]              |
> | sm (streaming multiprocessor) | scheduling + execution quad containing cuda cores, tensor cores, shared memory partitions                                                                                                            | [[lectures/420/notes#hopper/blackwell sm topology]]                       |
> | simt                          | warp-level (32 thread) execution model issuing one instruction per warp                                                                                                                              | [[lectures/420/notes#thread execution model: simt]]                       |
> | memory hierarchy              | registers (≈1 cycle) → shared/l1 (20–30) → l2 (≈200) → hbm (≈400) → nvlink fabric                                                                                                                    | [[lectures/420/notes#hopper h100 memory hierarchy (per sm unless noted)]] |
> | latency hiding                | 64 resident warps per sm swap on stall to cover ≈400-cycle hbm accesses                                                                                                                              | [[lectures/420/notes#hopper concurrency multiplier]]                      |

## Execution Units

See [[lectures/420/notes#SM execution units: a detailed breakdown|execution unit breakdown]] for detailed comparison.

> [!table] sm execution roles (hopper/blackwell)
>
> | unit                      | capacity per sm                           | responsibility                        | deeper dive                                             |
> | ------------------------- | ----------------------------------------- | ------------------------------------- | ------------------------------------------------------- |
> | warp schedulers           | 4 quads, 16 warp issue slots/cycle        | pick ready warps, arbitrate pipelines | [[lectures/420/notes#hopper/blackwell sm topology]]     |
> | cuda cores                | 128 fp32/int32 alus                       | scalar and vector integer/fp work     | [[lectures/420/notes#cuda cores vs tensor cores]]       |
> | tensor cores              | 4 mma pipelines (fp16/bf16/tf32/fp8/fp4)  | matrix multiply-accumulate via wgmma  | [[lectures/420/notes#wgmma warp-group matrix multiply]] |
> | load/store units          | 64b sector loads, cp.async, tma front-end | coalesced global/shared traffic       | [[lectures/420/notes#memory coalescing]]                |
> | special function units    | exp, sin, cos, rsqrt throughput           | transcendental evaluation             | [[lectures/420/notes#special function unit (sfu)]]      |
> | tensor memory accelerator | descriptor-driven dma (sm90+)             | async tensor copies, multicast        | [[lectures/420/notes#tensor memory accelerator (tma)]]  |

## NVIDIA Architectures

### Hopper (H100, 2022)

> [!example] hopper quick facts
>
> | metric       | value                                                                                      |
> | ------------ | ------------------------------------------------------------------------------------------ |
> | sm count     | 132 (144 on sxm)                                                                           |
> | cuda cores   | 16,896                                                                                     |
> | tensor cores | 528 (4th gen)                                                                              |
> | hbm          | 80 gb hbm3 @ 3.35 tb/s                                                                     |
> | fp16 peak    | 1,979 tflop/s                                                                              |
> | fp8 peak     | 3,958 tflop/s                                                                              |
> | reference    | [[lectures/420/notes#hopper h100 memory hierarchy (per sm unless noted)]] · modal glossary |

> [!tip] hopper standout features
>
> - [[lectures/420/notes#thread block clusters and distributed shared memory|thread block clusters]] + dsme fabric
> - [[lectures/420/notes#tensor memory accelerator (tma)|tensor memory accelerator]] for descriptor-driven async copies
> - [[lectures/420/notes#wgmma warp-group matrix multiply|wgmma]] instructions and mbarrier synchronization
> - fp8 (e4m3, e5m2) execution paths highlighted in section 11 of the jax scaling book: https://jax-ml.github.io/scaling-book/

### Blackwell (B200, 2024)

> [!example] blackwell quick facts
>
> | metric       | value                                                                                           |
> | ------------ | ----------------------------------------------------------------------------------------------- |
> | sm count     | 192                                                                                             |
> | cuda cores   | 24,576                                                                                          |
> | tensor cores | 768 (5th gen)                                                                                   |
> | hbm          | 192 gb hbm3e @ 8 tb/s                                                                           |
> | fp16 peak    | 2,250 tflop/s                                                                                   |
> | fp4 peak     | 20,000 tflop/s                                                                                  |
> | reference    | [[lectures/420/notes#nvidia architectures \| architecture table]] · jax scaling book section 12 |

> [!tip] blackwell enhancements
>
> - [[lectures/420/notes#mixed precision arithmetic|nvfp4 + mxfp8]] pipelines (tensor core fp4)
> - second-generation tma with multicast/reduction shortcuts
> - nvls fabric for multi-gpu topologies, complements nvlink
> - guidance in "how to scale your model" (jax scaling book): https://jax-ml.github.io/scaling-book/

## Programming Models

### CUTLASS and CUTe DSL

See [[lectures/420/notes#cutlass and the cute dsl|CUTLASS and CUTe DSL section]] for comprehensive coverage.

> [!table] templated gemm stack
>
> | component                         | focus                                                             | follow-up                                                    |
> | --------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------ |
> | cutlass                           | template-based gemm primitives, epilogue fusion, cute integration | [[lectures/420/notes#cutlass stages]]                        |
> | cute dsl                          | layout algebra for compile-time tiling/swizzling                  | [[lectures/420/notes#layout algebra]]                        |
> | cute swizzle ops                  | xor, block-raked layouts to kill bank conflicts                   | [[lectures/420/notes#swizzling and bank conflict avoidance]] |
> | cute local_tile / local_partition | hierarchical decompositions down to register tiles                | [[lectures/420/notes#local_tile and local_partition]]        |

### Triton Linear Layout

See also: [post](https://www.lei.chat/posts/triton-linear-layout-concept/) · modal glossary on tensor cores: https://modal.com/gpu-glossary/device-hardware/tensor-core

Triton's layout system provides high-level abstractions similar to CUTe but with Python-based programming model.

## Performance Analysis

### Roofline Model

See [[lectures/420/notes#roofline model and performance bounds|Roofline model section]].

> [!abstract] roofline checklist
>
> - arithmetic intensity (flop/byte) classifies memory- vs compute-bound behavior
> - ridge point on h100: i ≈ 592 flop/byte (1979 tflop/s ÷ 3.35 tb/s)
> - optimization target: increase reuse via tiling so intensity crosses the ridge point

### Profiling Tools

> [!info] profiling toolkit
>
> - nsys for system-wide timeline capture and stream overlap analysis
> - ncu (nsight compute) for kernel metrics + roofline overlays
> - critical metrics: occupancy, bank conflicts, dram throughput, tensor core utilization

## Resources

> [!reference] curated resources
>
> | theme                  | resource                                                                                                 | notes                                                                    |
> | ---------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
> | fundamentals           | [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)                 | end-to-end walkthrough of gpu perf stack                                 |
> | official docs          | [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)                         | baseline runtime + api semantics                                         |
> | templated gemm         | [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)                                               | hierarchical tiling + cute dsl references                                |
> | architecture deep dive | [Hopper Architecture Whitepaper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) | latency/bandwidth tables, sm diagrams [@nvidia2022hopper]                |
> | modal glossary         | https://modal.com/gpu-glossary/device-hardware/cuda-device-architecture                                  | concise recap of sm→gpc→gpc topology, updated for hopper                 |
> | scaling text           | https://jax-ml.github.io/scaling-book/                                                                   | sections 11–12 cover hopper/blackwell perf envelopes [@jax2025blackwell] |

> [!todo] open threads
>
> - derive cutlass 4.2 cute dsl paged-attention microbenchmark here once implementation stabilizes
> - add amd cdna 3 summary + translation to hip/kokkos analogues
