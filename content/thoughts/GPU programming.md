---
aliases:
  - gpus
date: '2025-09-08'
description: CUDA execution, memory traffic, and precision-specific performance limits
id: GPU programming
modified: 2026-06-07 01:19:33 GMT-04:00
permalinks:
  - /gpus
socials:
  glossary: https://modal.com/gpu-glossary
  history: https://fabiensanglard.net/cuda/
tags:
  - ml
  - hardware
title: GPU
---

A GPU runs many threads so that some can make progress while others wait for data or earlier instructions. Kernel performance depends on the work each thread does and the data it moves. A peak FLOP rate alone leaves out the second part.

## arithmetic bandwidth

Arithmetic throughput is the number of arithmetic operations completed per second. A quoted peak needs a data type and an instruction path. Tensor Core matrix multiplication and scalar FP32 instructions have different ceilings on the same chip. For supported sparse patterns, the quoted effective Tensor Core rate can be twice the dense rate.

The [Modal glossary](https://modal.com/gpu-glossary/perf/arithmetic-bandwidth) defines the term.

## architecture overview

CUDA groups threads into blocks. An SM, or streaming multiprocessor, executes resident blocks in warps of $32$ threads. A warp instruction applies to its active threads; divergence can leave some threads inactive for part of the execution.

Registers hold per-thread values. Shared memory belongs to a block, with cluster access available on supported hardware. L1 is local to an SM and L2 is shared across the GPU. HBM holds device memory. NVLink carries traffic between devices. Access latency depends on the instruction and access pattern. See the [CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html).

H100 SXM has an upper residency limit of

$$
132\ \text{SMs}\times64\ \text{warps/SM}\times32\ \text{threads/warp}
=270{,}336\ \text{threads}.
$$

Actual occupancy depends on block size and the registers and shared memory each block needs. Ready warps can issue while other warps wait, so residency helps hide latency when there is independent work available. The [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy) gives the resource limits.

## execution units

The following counts describe Hopper. NVIDIA's [architecture account][hopper] and [scheduler presentation](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/CUDA-Programming-and-Optimization.pdf#page=27) give the details.

| unit                      | Hopper role                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------------- |
| warp scheduler            | Four per SM. Each holds up to $16$ resident warps and can issue at most one warp instruction per cycle. |
| arithmetic units          | $128$ FP32 units and $64$ INT32 units per SM. Instruction throughput depends on the operation.          |
| Tensor Cores              | Four per SM. Matrix multiply-accumulate instructions include Hopper's warp-group MMA path.              |
| load/store units          | Execute memory instructions. Coalescing combines a warp's global accesses into memory transactions.     |
| special-function units    | Execute supported special-function instructions. A library function can require several instructions.   |
| Tensor Memory Accelerator | TMA moves tensor tiles asynchronously using descriptors and supports cluster multicast.                 |

For coalesced global access, an aligned warp reading $32$ consecutive FP32 values requests $128$ bytes in four $32$-byte transactions. Scattered addresses can require more transactions for the same useful data. These transaction sizes differ from the width of one thread's load instruction. See [CUDA's coalescing example](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#coalesced-global-memory-access).

## AMD

The reading pointers for [[thoughts/PD disaggregated serving|disaggregated serving]] are RCCL and SGLang, with NIXL and UCX for the transfer path. The AMD CDNA 3 architecture pass is still open.

## NVIDIA

### cuda

The earlier Tesla architecture is described in [@lindholm2008nvidia]. Keep architecture-specific counts tied to the chip they describe.

### hopper

These are [H100 SXM specifications](https://www.nvidia.com/en-us/data-center/h100/). The full GH100 design contains $144$ SMs; the H100 SXM product exposes $132$ [@nvidia2022hopper].

| metric                | H100 SXM                                                                                           |
| --------------------- | -------------------------------------------------------------------------------------------------- |
| SMs                   | $132$                                                                                              |
| FP32 CUDA cores       | $16{,}896$                                                                                         |
| Tensor Cores          | $528$                                                                                              |
| HBM                   | $80\ \mathrm{GB}$ HBM3 at $3.35\ \mathrm{TB/s}$                                                    |
| FP16 Tensor Core peak | approximately $989.5\ \mathrm{TFLOP/s}$ dense, $1{,}979\ \mathrm{TFLOP/s}$ with supported sparsity |
| FP8 Tensor Core peak  | $1{,}979\ \mathrm{TFLOP/s}$ dense, $3{,}958\ \mathrm{TFLOP/s}$ with supported sparsity             |

Hopper supports thread-block clusters and distributed shared memory. TMA handles descriptor-based tensor copies, and `wgmma` provides asynchronous warp-group matrix operations. The synchronization rules are covered in the [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) for TMA and clusters, and the [warp-group MMA guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/warpgroup_programming.html) for WGMMA.

### blackwell

The B200 specifications below use the current [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#high-bandwidth-memory-hbm3-subsystem) and [HGX B200 product table](https://www.nvidia.com/en-us/data-center/hgx/). The arithmetic figures are per GPU, obtained by dividing the eight-GPU HGX totals by $8$.

| metric                | B200                                                                   |
| --------------------- | ---------------------------------------------------------------------- |
| HBM                   | $180\ \mathrm{GB}$ HBM3e at approximately $8\ \mathrm{TB/s}$           |
| FP16 Tensor Core peak | $2{,}250\ \mathrm{TFLOP/s}$ dense, $4{,}500\ \mathrm{TFLOP/s}$ sparse  |
| FP4 Tensor Core peak  | $9{,}000\ \mathrm{TFLOP/s}$ dense, $18{,}000\ \mathrm{TFLOP/s}$ sparse |

Blackwell SM100 adds `tcgen05.mma` and FP4 matrix operations. Its instruction forms and data layouts are described in the [CUTLASS Blackwell documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html). Hopper's WGMMA description does not specify this newer path.

TMA multicast and reductions already exist on Hopper. NVLink SHARP, called NVLS in NCCL, offloads supported collective operations to NVSwitch on compatible systems, as described in the [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable).

### cutlass and cute dsl

[CUTLASS](https://docs.nvidia.com/cutlass/latest/) provides GPU linear-algebra kernels and components for building them. CuTe expresses tensor layouts and the partitioning of work across threads. CuTe DSL provides a Python interface for writing and compiling GPU kernels.

The local [[lectures/420#CuTe DSL mental model|CuTe notes]] include layout examples. Consult the installed version's API when turning those sketches into runnable code.

### triton linear layout

Lei Zhang's [linear-layout note](https://www.lei.chat/posts/triton-linear-layout-concept/) explains how Triton represents mappings between logical tensor indices and hardware locations. That mapping determines which values each thread holds and how values move between layouts.

## roofline

For a kernel with $F$ floating-point operations and $Q$ bytes transferred through HBM, define arithmetic intensity as

$$
I=\frac{F}{Q}.
$$

With compute ceiling $P_{\max}$ and HBM bandwidth $B$, the corresponding roofline bound is

$$
P\le\min(P_{\max},BI),
\qquad
I_{\mathrm{ridge}}=\frac{P_{\max}}{B}.
$$

Using the dense FP16 Tensor Core ceiling for H100 SXM gives

$$
I_{\mathrm{ridge}}
=\frac{989.5\times10^{12}\ \mathrm{FLOP/s}}
{3.35\times10^{12}\ \mathrm{byte/s}}
\approx295\ \mathrm{FLOP/byte}.
$$

Using the sparse peak would double this ceiling and require matching sparsity assumptions. Tiling can increase reuse and reduce $Q$, but larger tiles also consume more registers and shared memory. Measure runtime after the change. Crossing the ridge point alone does not establish a speedup. NVIDIA's [occupancy guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics) explains the resource tradeoff.

### profiling tools

Use Nsight Systems (`nsys`) to inspect the application timeline and stream overlap. Use Nsight Compute (`ncu`) to inspect an individual kernel's memory traffic and instruction use. NVIDIA documents the distinction in its [profiling migration guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#migrating-to-nsight-tools-from-visual-profiler-and-nvprof).

## resources

- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) walks through GPU performance calculations.
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) defines the programming model and API contracts.
- [UCCL](https://github.com/uccl-project/uccl) is a communication project to follow.

> [!todo] open threads
>
> - derive a CuTe DSL paged-attention microbenchmark with recorded inputs and measurements
> - add an AMD CDNA 3 summary and the corresponding HIP/Kokkos notes

[hopper]: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
