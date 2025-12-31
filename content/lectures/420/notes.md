---
date: "2025-09-30"
description: GPUs, CUTLASS, and CuTe
id: notes
modified: 2025-12-30 19:06:25 GMT-05:00
slides: true
tags:
  - workshop
  - gpu
title: supplement to 0.420
transclude:
  title: false
---

see also: [[thoughts/GPU programming|GPU]], [[thoughts/XLA]], [[thoughts/LLMs]]

> [!tip] Runnable Code Examples
>
> All CUDA code snippets in these notes have been collected into verifiable, runnable examples:
>
> - **Location**: [[lectures/420/examples|code]]
> - **Examples**: Vector addition, vectorized loads, matrix multiplication, memory coalescing, parallel reduction
> - **Quick start**: `cd examples && make && make test`
> - **Target hardware**: Optimized for H200 (Hopper SM 9.0)
>
> Each example includes timing, verification, and performance metrics demonstrating concepts from the lecture.

> [!abstract] readings
>
> GPU architecture and CUDA fundamentals
>
> - CUDA architecture: uniform hardware design enabling general-purpose parallel computation
> - Hopper architecture: wgmma instructions, tensor memory accelerator, thread block clusters
>
> Matrix multiplication optimization
>
> - matmul tiling strategies: block tiling, memory coalescing, tensor core utilization
> - Flash Attention 4: warp specialization, persistent kernels, asynchronous pipelines
>
> CuTe DSL and CUTLASS
>
> - layout algebra: composition, complement, division, product operations
> - CUTLASS 3.0: hierarchical tiling with local_tile and local_partition

## matmul

> [!note] Matmul Workloads
>
> - Transformers: Attention mechanisms compute $QK^T$ and multiply by $V$
> - Neural networks: Every layer performs $y = Wx + b$
> - Inference: Billions of matmul operations per forward pass
> - Training: Forward pass, backpropagation, and optimizer updates

> [!warning] Performance Constraints
>
> - GPT-4 scale: $\approx 10^{24}$ FLOPs for training
> - Real-time inference: <100ms latency, >1000 tokens/sec throughput
> - Memory bandwidth: hundreds of GB/s to multiple TB/s
> - Mixed precision: fp16, bf16, fp8, fp4 to stay within power envelopes
> - Central challenge: Matrix multiplication is memory-bound on modern GPUs, so peak throughput requires deep hardware-awareness (i.e: arthmetic intensity)

## what is a GPU?

### CPU vs GPU: philosophical differences

> [!info] Latency-centric CPU design vs throughput-centric GPU design
>
> | Aspect                 | CPU                                         | GPU                                                  |
> | ---------------------- | ------------------------------------------- | ---------------------------------------------------- |
> | Design Goal            | Low-latency sequential execution            | High-throughput parallel execution                   |
> | Execution Model        | Complex out-of-order with branch prediction | Simple in-order with no speculation                  |
> | Cache Hierarchy        | Large L1/L2/L3 caches (tens of MB)          | Minimal caching, programmer-managed shared memory    |
> | Core Count             | Few cores (4-64 on server CPUs)             | Hundreds of SMs running tens of thousands of threads |
> | Per-Thread Performance | High (sophisticated ILP, speculation)       | Low (simple in-order execution)                      |
> | Aggregate Throughput   | Optimized for single-thread speed           | Optimized for total work completed per second        |
> | Context Switching      | Microseconds (expensive OS operation)       | Single clock cycle (zero-overhead warp switching)    |
> | Clock Frequency        | 3-5 GHz                                     | 1.4 GHz                                              |
> | Memory Model           | Sequential consistency guaranteed           | Relaxed consistency, explicit synchronization        |
> | Power per Core         | 10-20W per complex core                     | <1W per simple execution unit                        |

> [!note] Clarification: "cycle" terminology
>
> When we refer to **"cycles"** throughout this document, we mean **clock cycles** of the GPU core clock (not memory clock or boost clock).
>
> - **H100 core clock**: 1.41 GHz → 1 cycle ≈ 0.71 nanoseconds
> - **EPYC 9654 base clock**: 2.4 GHz → 1 cycle ≈ 0.42 nanoseconds (3.7 GHz boost → 0.27 ns)
>
> **Latency examples**:
>
> - Global memory (HBM3): ~400 cycles ≈ 280 ns
> - L2 cache: ~200 cycles ≈ 140 ns
> - Shared memory/L1: ~20 cycles ≈ 14 ns
> - Register access: ~1 cycle ≈ 0.71 ns
>
> When we say "context switching costs 1,000 CPU cycles," that's ~420 ns on EPYC at base clock. GPU warp switching is literally 0 cycles—happens instantaneously in hardware without stalling the pipeline.

> [!important] tradeoff
>
> CPUs optimize for latency: "How fast can I complete this single task?"
>
> GPUs optimize for throughput: "How many tasks can I complete per second?"

Example: A single CPU core might execute a 1000-instruction sequence in 200 nanoseconds (5 instructions/ns via ILP). A GPU thread might take 2000 nanoseconds for the same sequence, but the GPU executes 100,000 such threads concurrently, achieving 500× higher total throughput.

### GPU architectural hierarchy

Modern NVIDIA GPUs (Hopper H100, Blackwell B200) organize computation hierarchically:

> [!example] GPU hierarchy from top to bottom
>
> ```
> ┌─────────────────────────────────────────────────────────────────┐
> │                        GPU Chip (Die)                           │
> │  ┌──────────────────────────────────────────────────────────┐   │
> │  │   Graphics Processing Cluster (GPC) 0                    │   │
> │  │  ┌────────────────────────────────────────────┐          │   │
> │  │  │ Texture Processing Cluster (TPC) 0         │          │   │
> │  │  │  ┌──────────────────────────────────────┐  │          │   │
> │  │  │  │ Streaming Multiprocessor (SM) 0      │  │          │   │
> │  │  │  │  ┌────────────────────────────────┐  │  │          │   │
> │  │  │  │  │ Warp Scheduler 0               │  │  │          │   │
> │  │  │  │  │  Warp 0  [T0 T1 ... T31]       │  │  │          │   │
> │  │  │  │  │  Warp 1  [T0 T1 ... T31]       │  │  │          │   │
> │  │  │  │  │  ...                           │  │  │          │   │
> │  │  │  │  ├────────────────────────────────┤  │  │          │   │
> │  │  │  │  │ Execution Units                │  │  │          │   │
> │  │  │  │  │  - 128 CUDA Cores (FP32/INT32) │  │  │          │   │
> │  │  │  │  │  - 4 Tensor Cores (Matrix ops) │  │  │          │   │
> │  │  │  │  │  - 64 FP64 Cores               │  │  │          │   │
> │  │  │  │  ├────────────────────────────────┤  │  │          │   │
> │  │  │  │  │ Memory                         │  │  │          │   │
> │  │  │  │  │  - 256 KB Registers            │  │  │          │   │
> │  │  │  │  │  - 256 KB Shared Memory / L1   │  │  │          │   │
> │  │  │  │  └────────────────────────────────┘  │  │          │   │
> │  │  │  │  SM 1, SM 2, ...                     │  │          │   │
> │  │  │  └──────────────────────────────────────┘  │          │   │
> │  │  │  TPC 1, TPC 2, ...                         │          │   │
> │  │  └────────────────────────────────────────────┘          │   │
> │  │  GPC 1, GPC 2, ...                                       │   │
> │  └──────────────────────────────────────────────────────────┘   │
> │                                                                 │
> │  ┌────────────────────────────────────┐                         │
> │  │  L2 Cache (Unified, 60 MB H100)    │                         │
> │  └────────────────────────────────────┘                         │
> │  ┌────────────────────────────────────┐                         │
> │  │  HBM3 Memory (80 GB @ 3.35 TB/s)   │                         │
> │  └────────────────────────────────────┘                         │
> └─────────────────────────────────────────────────────────────────┘
> ```

#### Level 1: the GPU chip

The GPU chip integrates all components. Here's a comparison of NVIDIA's latest datacenter GPUs:

> [!info] NVIDIA GPU architecture comparison
>
> | Component                 | Hopper H100 (2022)      | Blackwell B200 (2024)  | Improvement |
> | ------------------------- | ----------------------- | ---------------------- | ----------- |
> | Architecture              | Hopper (4th gen)        | Blackwell (5th gen)    | Next-gen    |
> | Process node              | TSMC 4N                 | TSMC 4NP               | Refined 4nm |
> | Streaming Multiprocessors | 132 SMs (144 SXM)       | 192 SMs                | 1.45×       |
> | GPC topology              | 8 GPCs × 9 TPCs × 2 SMs | 8 GPCs (estimated)     | —           |
> | CUDA cores                | 16,896 (128/SM)         | 24,576 (128/SM)        | 1.45×       |
> | Tensor cores              | 528 (4th gen, 4/SM)     | 768 (5th gen, 4/SM)    | 1.45×       |
> | FP64 cores                | 8,448 (64/SM)           | 12,288 (64/SM)         | 1.45×       |
> | L2 cache                  | 60 MB                   | 96 MB                  | 1.6×        |
> | Memory                    | 80 GB HBM3              | 192 GB HBM3e           | 2.4×        |
> | Memory bandwidth          | 3.35 TB/s               | 8 TB/s                 | 2.39×       |
> | Peak FP64                 | 34 TFLOP/s              | 45 TFLOP/s             | 1.32×       |
> | Peak FP32                 | 67 TFLOP/s              | 90 TFLOP/s             | 1.34×       |
> | Peak FP16 (Tensor)        | 1,979 TFLOP/s           | 2,250 TFLOP/s          | 1.14×       |
> | Peak FP8 (Tensor)         | 3,958 TFLOP/s           | 4,500 TFLOP/s          | 1.14×       |
> | Peak FP4 (Tensor)         | —                       | 20,000 TFLOP/s         | New         |
> | TDP                       | 700W                    | 1,000W                 | 1.43×       |
> | Key features              | WGMMA, TMA, clusters    | FP4, 2nd-gen TMA, NVLS | Enhanced    |

#### Thread count comparison: EPYC vs H100

To illustrate the scale difference, compare a high-end server CPU with H100:

> [!example] AMD EPYC 9654 (Zen 4) vs NVIDIA H100 thread counts
>
> | Metric                    | AMD EPYC 9654             | NVIDIA H100              | Ratio  |
> | ------------------------- | ------------------------- | ------------------------ | ------ |
> | Physical cores            | 96 cores                  | 132 SMs                  | 1.4×   |
> | Hardware threads per core | 2 (SMT)                   | 2,048 (64 warps × 32)    | 1,024× |
> | Total concurrent threads  | 192                       | 270,336                  | 1,408× |
> | Context switch overhead   | ~1,000 cycles (µs)        | 0 cycles                 | ∞×     |
> | Clock frequency           | 3.7 GHz boost             | 1.41 GHz                 | 0.38×  |
> | Thread execution          | Out-of-order, speculative | In-order, no speculation | —      |
> | L3 Cache                  | 384 MB                    | 60 MB L2 (no L3)         | —      |
> | Memory bandwidth          | 460 GB/s (DDR5)           | 3,350 GB/s (HBM3)        | 7.3×   |
> | TDP                       | 360W                      | 700W                     | 1.9×   |

> [!note] Hopper concurrency multiplier
>
> H100 runs 1,408× more threads concurrently than a 96-core EPYC CPU. This massive thread count enables:
>
> - Latency hiding: While 64 warps wait on memory (400 cycles), the remaining ~63 warps keep execution units busy
> - Throughput optimization: Even if each thread is 10× slower than a CPU thread, 1,408× more threads = 140× higher aggregate throughput
> - Memory bandwidth utilization: 270K threads can saturate 3.35 TB/s HBM3 bandwidth; 192 CPU threads cannot saturate 460 GB/s DDR5

- EPYC 9654: 96 physical cores × 2-way SMT = 192 hardware threads
  - Each core runs 2 threads via simultaneous multithreading (SMT)
  - Context switching between threads takes ~1,000 cycles (OS scheduler overhead)
  - Total: 192 threads across entire CPU

- H100: 132 SMs × 64 warps/SM × 32 threads/warp = 270,336 concurrent threads
  - Each SM runs 64 warps (2,048 threads) simultaneously
  - Warp scheduler switches between warps in 0 cycles (hardware-managed)
  - Each warp executes 32 threads in lockstep (SIMT)
  - Total: 270,336 threads across entire GPU

> [!important] why is it important for matmul?
>
> A 4096×4096 matrix has 16,777,216 elements. Assigning one element per thread:
>
> - EPYC: 192 threads → 87,381 iterations per thread (serial bottleneck)
> - H100: 270,336 threads → 62 iterations per thread (parallel throughout)

With tensor cores computing 16×16 tiles, H100 needs only 256×256 = 65,536 tiles, easily mapped to 270K threads with massive occupancy.

#### Level 2: graphics processing cluster (GPC)

A GPC groups multiple TPCs and provides:

- Shared L2 cache access
- Raster engines (for graphics workloads)
- High-bandwidth interconnect between SMs

Hopper's 8 GPCs enable hierarchical scheduling: thread block clusters (introduced in Hopper) preferentially schedule within GPC boundaries to minimize DSMEM access latency.

#### Level 3: texture processing cluster (TPC) and streaming multiprocessor (SM)

Each TPC contains 2 SMs. The SM is the fundamental execution unit—the GPU equivalent of a CPU core, but simpler and massively parallel.

An SM can execute up to 2,048 threads concurrently (64 warps × 32 threads/warp). Compare this to a single CPU core running 1-2 threads!

#### Level 4: warps and threads

A warp is a group of 32 threads that execute in lockstep (SIMT execution):

- All 32 threads execute the same instruction simultaneously
- Each thread operates on different data (SIMT = Single Instruction, Multiple Thread)
- Warp scheduling is hardware-managed with zero-overhead context switching
- Divergence (threads taking different control-flow paths) causes serialization

```cuda
// Each of 32 threads in a warp computes one element
__global__ void vector_add(float* a, float* b, float* c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];  // 32 additions execute in parallel per warp
    }
}
```

> [!note] Why 32 threads per warp?
>
> Historical constraint tied to memory system design:
>
> - Shared memory has 32 banks (4-byte words)
> - 32 threads enable single-cycle coalesced memory access
> - Register file ports and execution unit groups sized for warp-level parallelism

### latency hiding through massive parallelism

The GPU's performance model relies on latency hiding: while one warp waits for memory, the scheduler executes other warps on the same SM.

> [!example] Latency hiding in action
>
> Consider an SM with 64 resident warps:
>
> ```
> Cycle 0:  Warp 0  issues memory load (400 cycles latency)
> Cycle 1:  Warp 1  issues memory load
> Cycle 2:  Warp 2  issues memory load
> ...
> Cycle 63: Warp 63 issues memory load
> Cycle 64: Warp 0  data arrives! Execute computation
> Cycle 65: Warp 1  data arrives! Execute computation
> ...
> ```
>
> The SM never stalls because it has 64 warps in flight. By the time Warp 0's memory request completes, all other warps have issued their requests. This requires 400 / 64 ≈ 6.25 warps per cycle issue rate, well within the SM's capability.

Occupancy measures how many warps are resident on an SM:

$$\text{Occupancy} = \frac{\text{Active warps per SM}}{\text{Maximum warps per SM}} = \frac{\text{Active warps}}{64}$$

Higher occupancy → more latency hiding → higher throughput.

> [!warning] CPU architectural constraints
>
> 1. Cache coherence overhead: With 100K+ threads, maintaining cache coherence across cores would require prohibitive bandwidth and complexity
> 2. Branch prediction complexity: Speculative execution for thousands of threads with divergent control flow is intractable
> 3. Power density: Complex out-of-order cores consume 10-20W each; 10,000 such cores would require megawatts
> 4. Memory model: CPUs guarantee sequential consistency; GPUs use relaxed memory models with explicit synchronization

GPUs sidestep these constraints by:

- Eliminating cache coherence (programmer-managed shared memory)
- Using simple in-order execution (no speculation)
- Running threads at lower frequency (1.4 GHz vs 3-5 GHz for CPUs)
- Exposing explicit memory hierarchy and synchronization primitives

The result: GPUs achieve 10-100× higher compute throughput than CPUs for data-parallel workloads, at the cost of requiring algorithm redesign and careful memory management.

## gpu architecture fundamentals

### cuda design philosophy

> [!note] Historical context
> Before CUDA (pre-2006), GPUs exposed fixed-function stages that required shader-specific toolchains. CUDA unified the execution model to make every SM programmable in C-like kernels, enabling GPGPU. [@lindholm2008nvidia]

> [!tip] Uniform architecture principle
> Replace specialized, fixed-function units with flexible computational blocks that run general-purpose kernels and graphics shaders interchangeably. [@lindholm2008nvidia]

This design enables general-purpose GPU computing (GPGPU) while maintaining high performance for graphics workloads.

### streaming multiprocessor (SM) architecture

The streaming multiprocessor (SM) is the fundamental execution tile in NVIDIA GPUs.

An SM contains multiple specialized execution units, each serving distinct purposes. Understanding their differences is critical for performance optimization.

> [!example] SM execution unit dataflow
>
> ```
>                         ┌─────────────────────────────────┐
>                         │     Warp Scheduler (x4)         │
>                         │  - Selects ready warp           │
>                         │  - Decodes instruction          │
>                         │  - Dispatches to execution unit │
>                         └─────────┬───────────────────────┘
>                                   │ Issue instruction
>                    ┌──────────────┼───────────────┬─────────────┐
>                    │              │               │             │
>         ┌──────────▼──────┐  ┌────▼─────┐  ┌──────▼──────┐  ┌───▼────────┐
>         │  CUDA Cores     │  │ Tensor   │  │  Load/Store │  │    SFU     │
>         │    (x32/quad)   │  │  Cores   │  │   Units     │  │   (x8)     │
>         │                 │  │  (x2)    │  │   (x32)     │  │            │
>         │  FP32, INT32    │  │  Matrix  │  │   Memory    │  │ Transcen.  │
>         │  FMA, ADD, MUL  │  │  MMA ops │  │   Ops       │  │ sin,cos,√  │
>         └────────┬────────┘  └────┬─────┘  └──────┬──────┘  └─────┬──────┘
>                  │                │               │               │
>                  │                │               │               │
>                  └────────────────┴───────────────┴───────────────┘
>                                   │
>                           ┌───────▼────────┐
>                           │ Register File  │
>                           │   (256 KB)     │
>                           └────────────────┘
>
>       Separate from execution units:
>
>         ┌──────────────────────────────────────┐
>         │  Tensor Memory Accelerator (TMA)     │
>         │  - Operates independently of threads │
>         │  - DMA engine: Global → Shared       │
>         │  - Single descriptor, bulk transfer  │
>         └──────────────────────────────────────┘
> ```

#### 1. Warp Scheduler

_function_: Control unit that selects which warp executes next and dispatches instructions to execution units.

> [!info] Warp scheduler responsibilities
>
> - Maintains scoreboard of warp states (ready, stalled on memory, stalled on dependency)
> - Selects eligible warp each cycle using round-robin or priority policy
> - Decodes instruction and routes to appropriate execution unit (CUDA core, tensor core, LSU, SFU)
> - Handles warp divergence: issues predicated instructions for each execution path
> - Zero-overhead context switching: switching between warps costs zero cycles

Example: If Warp 0 issues a memory load and stalls, the scheduler immediately switches to Warp 1 without any penalty.

Hopper has 4 warp schedulers per SM, each managing 16 warps (64 total warps per SM).

#### 2. CUDA Core

_function_: General-purpose arithmetic logic unit (ALU) for scalar operations.

Operations: FP32 (single-precision), INT32 arithmetic, logical operations

> [!info] CUDA core capabilities (Hopper)
>
> - FP32 FMA: `a * b + c` in single precision
> - INT32 operations: addition, multiplication, bit shifts, logical AND/OR/XOR
> - FP32 comparison and conversion operations
> - Throughput: 1 FP32 FMA per core per cycle
> - 128 CUDA cores per SM = 128 FP32 operations/cycle/SM

Example instructions:

```cuda
float result = a * b + c;        // FP32 FMA → CUDA core
int idx = threadIdx.x * 4 + 2;   // INT32 arithmetic → CUDA core
bool cond = (x > threshold);     // Comparison → CUDA core
```

> you can also think of it as a scalar floating-point/integer execution unit. Analogous to a single ALU in a CPU core, but much simpler (in-order, no speculation).

#### 3. Tensor Core

_function_: Specialized matrix multiply-accumulate (MMA) unit operating on small tiles (e.g., 16×16 matrices).

Operations: Matrix multiplication in mixed precision (FP16, BF16, FP8, INT8, FP4)

> [!info] Tensor core capabilities (Hopper 4th-gen)
>
> - Computes $D = A \times B + C$ where $A, B$ are 16×16 tiles
> - Input types: FP16, BF16, TF32, FP8 (E4M3, E5M2), INT8, INT4
> - Accumulator: FP32 or FP16
> - Throughput: 4 Tensor cores per SM × 256 FP16 FMA/tensor core/cycle = 1,024 FP16 operations/cycle/SM
> - 20× faster than CUDA cores for dense matrix ops

Example instructions:

```cuda
// WMMA (Warp-level Matrix Multiply Accumulate) - Ampere/Ada
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // 16×16×16 → Tensor core

// WGMMA (Warp-group Matrix Multiply Accumulate) - Hopper
asm("wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16"
    "{%0, %1, ...}, {%16, %17, ...}, {%32, %33, ...};"
    : outputs : inputs);  // 64×256 matrix op → Tensor core
```

> [!important] discrepancy with CUDA Cores
>
> Tensor Cores compute entire matrix tiles per instruction, not scalar operations. They are not programmable—only accessible via specific matrix instructions (WMMA, WGMMA).

#### 4. Load/Store Unit (LSU)

_function_: Memory access unit that issues load and store operations to the memory hierarchy.

Operations: Reads and writes to registers, shared memory, L1 cache, L2 cache, and global memory (HBM)

> [!info] LSU capabilities
>
> - Global memory loads/stores: `ld.global`, `st.global`
> - Shared memory loads/stores: `ld.shared`, `st.shared`
> - Coalescing: Merges 32 thread accesses into fewer transactions
> - Asynchronous copies: `cp.async` (Ampere+), `cp.async.bulk` (Hopper+)
> - 32 LSUs per warp scheduler (total varies by architecture)
> - Throughput: 32 bytes/cycle per LSU (for coalesced access)

Example instructions:

```cuda
float value = global_ptr[threadIdx.x];     // ld.global → LSU
smem[threadIdx.x] = value;                 // st.shared → LSU
__pipeline_memcpy_async(smem, gmem, 128);  // cp.async → LSU
```

#### 5. Special Function Unit (SFU)

_function_: Hardware accelerator for transcendental and special mathematical functions.

Operations: Reciprocal, square root, logarithm, exponential, trigonometric functions

> [!info] SFU capabilities
>
> - Transcendentals: `sin`, `cos`, `exp`, `log`, `rsqrt` (reciprocal square root)
> - Lower precision than CUDA cores: ~1-2 ULP (units in last place) error
> - 8-16 SFUs per SM (shared across warp schedulers)
> - Throughput: 1 operation per thread per cycle (for 32-thread warp = 32 ops/cycle)
> - Much faster than computing via CUDA core polynomial approximations

Example instructions:

```cuda
float result = expf(x);          // Exponential → SFU
float inv_sqrt = rsqrtf(x);      // Reciprocal sqrt → SFU
float angle = sinf(theta);       // Sine → SFU
```

> [!important]
> Transcendentals require complex hardware (lookup tables, iterative algorithms).
> Dedicating silicon to SFUs is more efficient than implementing in general-purpose CUDA cores.

> [!warning] Precision tradeoff
> SFU functions are faster but less precise than CUDA core implementations:
>
> | Function  | SFU (`__sinf`) | CUDA Core (`sin`) | Speedup |
> | --------- | -------------- | ----------------- | ------- |
> | `sin(x)`  | ~20 cycles     | ~100-200 cycles   | 5-10×   |
> | Precision | 1-2 ULP error  | 1 ULP error       | Lower   |
>
> Use intrinsics (`__sinf`, `__expf`) for SFU, standard functions (`sin`, `exp`) for CUDA cores.

#### 6. Tensor Memory Accelerator (TMA)

**Function**: Asynchronous DMA (Direct Memory Access) engine for bulk tensor transfers between global and shared memory.

**Operations**: Descriptor-based multi-dimensional memory copies

> [!info] TMA capabilities (Hopper-exclusive)
>
> - Hardware unit **separate from thread execution**
> - Transfers entire tensors (1D/2D/3D/4D/5D) with single instruction
> - Address generation offloaded from threads to hardware
> - Supports swizzling, padding, and L2 promotion in descriptor
> - Multicast: One transfer → multiple SMs' shared memories
> - Reduction: Aggregate data from multiple SMs
> - Synchronized via `mbarrier` primitives

> [!tip] Detailed TMA documentation
> For comprehensive coverage including descriptor creation, PTX instructions, mbarrier synchronization, multicast/reduction operations, and performance comparisons, see the [[lectures/420/notes#tensor memory accelerator (TMA)|dedicated TMA section]].

**Basic example**:

```cuda
// Traditional load (LSU-based, per-thread)
for (int i = threadIdx.x; i < 128; i += blockDim.x) {
    smem[i] = global_ptr[i];  // Each thread issues load → LSU
}

// TMA load (single thread issues, hardware executes)
if (threadIdx.x == 0) {
    asm("cp.async.bulk.tensor.2d.shared::cluster.global"
        "[%0], [%1], [%2];"  // One instruction, entire 2D tile → TMA
        :: "r"(smem), "l"(&tma_desc), "r"(&mbarrier));
}
// Other threads continue computing while TMA executes!
```

Key difference from LSU:

- LSU: Per-thread loads/stores, threads compute addresses, threads stall waiting for data
- TMA: Single-instruction bulk transfer, hardware computes addresses, threads execute concurrently

#### 7. "Core" terminology clarification

The term "core" is overloaded in GPU terminology:

> [!important] Different meanings of "core"
>
> | Term                | Meaning                         | Analogy                                   |
> | ------------------- | ------------------------------- | ----------------------------------------- |
> | GPU "Core" (casual) | Streaming Multiprocessor (SM)   | Like a CPU core                           |
> | CUDA Core           | FP32/INT32 ALU within an SM     | Like a single execution unit in a CPU ALU |
> | Tensor Core         | Matrix multiply-accumulate unit | Like a vector unit, but for matrices      |
> | SM Core Count       | Number of execution units in SM | Total ALU count                           |
>
> When NVIDIA says "H100 has 16,896 CUDA cores," they mean 132 SMs × 128 FP32 ALUs = 16,896 scalar execution units.

Comparison to CPU: A CPU core is a complete execution pipeline (fetch, decode, execute, writeback) with ALUs, branch predictor, caches, etc. A CUDA core is just the ALU—the warp scheduler acts as the shared fetch/decode unit for 128 CUDA cores.

#### Putting it together: instruction dispatch

When a warp executes, the warp scheduler dispatches each instruction to the appropriate unit:

```cuda
__global__ void example_kernel(float* A, float* B, half* C_tensor) {
    int tid = threadIdx.x;

    // 1. Load/Store Unit: Memory access
    float a = A[tid];                          // ld.global → LSU
    float b = B[tid];                          // ld.global → LSU

    // 2. CUDA Core: Scalar arithmetic
    float sum = a + b;                         // fadd → CUDA Core
    float prod = sum * 2.0f;                   // fmul → CUDA Core

    // 3. Special Function Unit: Transcendental
    float result = expf(prod);                 // exp → SFU

    // 4. Tensor Core: Matrix operation
    wmma::fragment<...> a_frag, b_frag, c_frag;
    wmma::load_matrix_sync(a_frag, C_tensor, 16);  // ld.shared → LSU
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); // mma → Tensor Core

    // 5. TMA: Bulk transfer (if using Hopper)
    if (tid == 0) {
        asm("cp.async.bulk.tensor..." :: ...); // TMA operation
    }

    // Back to LSU for store
    A[tid] = result;                           // st.global → LSU
}
```

The warp scheduler automatically routes each operation to the correct execution unit based on the instruction opcode. Programmers don't explicitly control this—it happens transparently based on what operations you write in the kernel.

#### Kernel launch: from host to device

CUDA kernels are launched from host (CPU) code using the `<<<...>>>` syntax, specifying the grid and block dimensions.

**Basic launch syntax**:

```cpp
// Kernel definition
__global__ void vector_add(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host code
int main() {
    int N = 1 << 20;  // 1M elements
    float *d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Copy data to device (omitted for brevity)
    // ...

    // Launch configuration
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // Launch kernel: <<<blocks, threads>>>
    vector_add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back (omitted)
    // ...

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
```

**Launch configuration parameters**:

```cpp
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);
```

- `gridDim`: Number of blocks (can be 1D, 2D, or 3D: `dim3(x, y, z)`)
- `blockDim`: Threads per block (also `dim3(x, y, z)`)
- `sharedMemBytes` (optional): Dynamic shared memory per block in bytes
- `stream` (optional): CUDA stream for asynchronous execution

**Multi-dimensional launch example**:

```cpp
// 2D matrix addition kernel
__global__ void matrix_add(float* A, float* B, float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

// Host launch
int M = 4096, N = 4096;
dim3 threads_per_block(32, 32);  // 1024 threads per block
dim3 blocks_per_grid(
    (N + threads_per_block.x - 1) / threads_per_block.x,  // x dimension
    (M + threads_per_block.y - 1) / threads_per_block.y   // y dimension
);

matrix_add<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, M, N);
```

**Dynamic shared memory example**:

```cpp
__global__ void reduce_sum(float* input, float* output, int N) {
    extern __shared__ float smem[];  // Dynamic shared memory

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    smem[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// Host launch with dynamic shared memory
int threads = 256;
int blocks = (N + threads - 1) / threads;
size_t shared_mem_size = threads * sizeof(float);

reduce_sum<<<blocks, threads, shared_mem_size>>>(d_input, d_output, N);
```

**Hopper-specific launch with thread block clusters**:

```cpp
__global__ void __cluster_dims__(2, 2, 1)  // 2×2 cluster attribute
cluster_kernel(float* data) {
    __shared__ float smem[256];

    // Cluster API
    cluster_group cluster = this_cluster();
    int cluster_rank = cluster.block_rank();

    // Access distributed shared memory
    if (cluster_rank == 0) {
        float* neighbor_smem = cluster.map_shared_rank(smem, 1);
        // Use neighbor's shared memory...
    }

    cluster.sync();
}

// Host launch (requires extended API for clusters)
cudaLaunchConfig_t config = {0};
config.gridDim = dim3(num_blocks_x, num_blocks_y, 1);
config.blockDim = dim3(threads_x, threads_y, 1);

cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim.x = 2;  // 2×2 cluster
attrs[0].val.clusterDim.y = 2;
attrs[0].val.clusterDim.z = 1;
config.attrs = attrs;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, cluster_kernel, data);
```

**Asynchronous launch with streams**:

```cpp
// Create streams
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Launch kernels on different streams (run concurrently)
kernel1<<<blocks, threads, 0, stream1>>>(d_data1);
kernel2<<<blocks, threads, 0, stream2>>>(d_data2);

// Wait for specific stream
cudaStreamSynchronize(stream1);

// Or wait for all
cudaDeviceSynchronize();

// Cleanup
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

> [!important] Launch bounds optimization
>
> Use `__launch_bounds__` to hint occupancy to the compiler:
>
> ```cpp
> // Guarantee at least 2 blocks per SM
> __global__ void __launch_bounds__(256, 2)
> optimized_kernel(float* data) {
>     // maxThreadsPerBlock = 256
>     // minBlocksPerMultiprocessor = 2
>     // Compiler optimizes register usage to achieve this occupancy
> }
> ```
>
> This helps the compiler limit register usage to ensure the desired occupancy, trading off register availability for higher thread count.

## gpu architecture fundamentals

> [!note] Hopper/Blackwell SM topology
>
> ```
> ┌────────────────────────────────────────────────────────────┐
> │                SM (Quad View)                              │
> ├──────────────┬──────────────┬──────────────┬───────────────┤
> │ Quad 0       │ Quad 1       │ Quad 2       │ Quad 3        │
> │ Warp Sched   │ Warp Sched   │ Warp Sched   │ Warp Sched    │
> │ + 2 Tensor   │ + 2 Tensor   │ + 2 Tensor   │ + 2 Tensor    │
> │ Core Pairs   │ Core Pairs   │ Core Pairs   │ Core Pairs    │
> │ + 32 CUDA    │ + 32 CUDA    │ + 32 CUDA    │ + 32 CUDA     │
> │ Cores        │ Cores        │ Cores        │ Cores         │
> ├──────────────┴──────────────┴──────────────┴───────────────┤
> │ Shared/L1 (256 KB logical) split into 4 partitions         │
> ├────────────────────────────────────────────────────────────┤
> │ Register File (256 KB) logically partitioned per quad      │
> └────────────────────────────────────────────────────────────┘
> ```
>
> - Hopper and Blackwell SMs expose four warp-scheduler quads, each feeding a pair of tensor-core pipelines and 32 FP/INT ALUs, enabling 16 warp issue slots per SM cycle. [@nvidia2022hopper; @scalingbook]
> - Each quad accesses its slice of the 256 KB shared-memory/L1 pool and its register file segment; Hopper's thread block clusters let quads collaborate via distributed shared memory fabric. [@nvidia2022hopper]
> - Blackwell retains the quad layout but doubles tensor-core throughput via FP4/MX formats and adds enhanced asynchronous transaction engines feeding the quads. [@nvidia2024nvfp4; @nvidia2025cutlass42]
> - Warp-specialized instructions (WGMMA) target warp-group scheduling across quads, while the tensor memory accelerator (TMA) streams tiles from HBM to shared memory with single-issue descriptors. [@nvidia2022hopper; @cutlass2025cute41]

### thread execution model: SIMT

> [!note] SIMT hierarchy
> Grid → Cooperative Thread Arrays (thread blocks) → Warps (32 threads) → Threads

> [!note] Execution semantics
>
> - A thread owns its registers and program counter.
> - A warp executes one instruction per cycle across 32 threads; divergence serializes paths.
> - Thread blocks share an SM's shared memory and can synchronize with `__syncthreads()`.
> - Grids aggregate blocks scheduled by the GPU runtime across SMs. [@lindholm2008nvidia]

### SIMT vs SIMD: architectural distinction

While both SIMT (Single Instruction Multiple Thread) and SIMD (Single Instruction Multiple Data) exploit data parallelism, their execution models differ fundamentally:

> [!important] Key differences between SIMT and SIMD
>
> | Aspect            | SIMD (CPU Vector Units)                      | SIMT (GPU Warps)                                  |
> | ----------------- | -------------------------------------------- | ------------------------------------------------- |
> | Programming Model | Explicit vector operations (`_mm256_add_ps`) | Scalar code executed by multiple threads          |
> | Program Counter   | Single PC for all lanes                      | Independent PC per thread (hardware consolidates) |
> | Control Flow      | All lanes must take same path                | Threads can diverge, hardware serializes          |
> | Register Model    | Vector registers (e.g., 512-bit ZMM)         | Scalar registers per thread                       |
> | Synchronization   | Implicit (lockstep execution)                | Explicit (`__syncwarp()`, `__syncthreads()`)      |
> | Memory Access     | Contiguous vector load/store                 | Per-thread addresses, hardware coalesces          |

> [!example] SIMD code (AVX-512)
>
> ```cpp
> __m512 a = _mm512_load_ps(ptr_a);      // Load 16 floats
> __m512 b = _mm512_load_ps(ptr_b);      // Load 16 floats
> __m512 c = _mm512_add_ps(a, b);        // Add 16 floats
> _mm512_store_ps(ptr_c, c);             // Store 16 floats
> ```
>
> Programmer explicitly manages vector width; all 16 lanes execute identically.

> [!example] SIMT code (CUDA)
>
> ```cuda
> int tid = threadIdx.x;                  // Thread 0-31 in warp
> float a = ptr_a[tid];                   // Each thread loads its element
> float b = ptr_b[tid];                   // 32 independent loads (coalesced)
> float c = a + b;                        // Scalar addition per thread
> ptr_c[tid] = c;                         // 32 independent stores (coalesced)
> ```
>
> Programmer writes scalar code; hardware executes 32 threads in lockstep when paths converge.

The critical advantage of SIMT: divergence tolerance. When threads diverge (e.g., `if (tid < 16)`), hardware serializes execution paths with predication. SIMD requires manual masking or explicit branching at vector granularity.

> [!note] Hardware implementation convergence
> Modern CPU SIMD (e.g., AVX-512) added masked operations (`_mm512_mask_add_ps`) to support conditional execution, narrowing the gap. Conversely, GPUs optimize for warp uniformity through compiler transformations and warp-wide operations (e.g., `__shfl_sync()`, `__ballot_sync()`) that assume SIMD-like behavior.

Example kernel for $C = AB$ illustrates SIMT mapping of one element per thread:

```cuda
// Grid: (M/32, N/32) thread blocks
// Block: (32, 32) threads
__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

Each thread computes one element of $C$. Threads in the same warp compute adjacent elements (enabling memory coalescing).

### warp divergence and predication

> [!warning] Warp divergence
> Divergence happens when threads inside a warp follow different control-flow paths, forcing serialized execution of each path with masked-off threads.

```cuda
if (threadIdx.x < 16) {
    // Path A: threads 0-15
    compute_A();
} else {
    // Path B: threads 16-31
    compute_B();
}
```

> [!example] Serialized execution cost
>
> 1. Execute Path A with threads 16-31 masked (inactive)
> 2. Execute Path B with threads 0-15 masked (inactive)
> 3. Wall-clock cycles sum across both paths.

> [!tip] Predication
> Hardware can predicate simple conditionals, evaluating both sides but committing results selectively, eliminating divergent branches when the divergent work per thread is minimal.

```cuda
int result = (condition) ? value_A : value_B;
// Compiles to predicated instructions (no divergence)
```

> [!success] Divergence mitigation
>
> - Minimize divergent branches within warps.
> - Use warp-level primitives (`__shfl_*`, `__ballot_sync`, `__any_sync`, `__all_sync`).
> - Align data layout so each warp processes contiguous regions.

## memory hierarchy: the performance bottleneck

Modern GPU programming is fundamentally about managing memory hierarchy. Compute is abundant; bandwidth is scarce.

### memory levels: capacity vs latency

```
┌────────────────────────────────────────────────────────────┐
│ Registers       ~1 cycle    256KB/SM      thread-private   │
├────────────────────────────────────────────────────────────┤
│ Shared Memory   ~20 cycles  256KB/SM      block-shared     │
├────────────────────────────────────────────────────────────┤
│ L1 Cache        ~30 cycles  128KB/SM      transparent      │
├────────────────────────────────────────────────────────────┤
│ L2 Cache        ~200 cycles 50MB          GPU-wide         │
├────────────────────────────────────────────────────────────┤
│ Global (HBM)    ~400 cycles 80GB          3.35 TB/s        │
└────────────────────────────────────────────────────────────┘
```

Arithmetic intensity determines achievable performance:

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Loaded from Memory}}
$$

For matmul $C_{M\times N} = A_{M\times K} B_{K\times N}$:

- Compute: $2MNK$ FLOPs
- Memory (naive): $(MK + KN + MN) \times 4$ bytes
- Arithmetic intensity: $\frac{2MNK}{4(MK + KN + MN)}$

For large $K$, this approaches $\frac{MN}{2(M+N)}$ FLOPs per byte. With square tiles where $M=N$, this simplifies to $\frac{M}{4}$. Tiling achieves much higher intensity by reusing data in shared memory.

> [!example] Memory hierarchy data flow [^data-flow]
>
> ```
> Global Memory (HBM)     [████████████] 80GB @ 3.35 TB/s
>     ↓ ~400 cycles            ↑ writeback
>     ↓                        ↑
> L2 Cache                [████] 50MB shared
>     ↓ ~200 cycles            ↑
>     ↓                        ↑
> L1/Shared Memory        [██] 256KB per SM
>     ↓ ~20 cycles             ↑
>     ↓                        ↑
> Registers               [█] 256KB per SM, ~1 cycle
>     ↓                        ↑
>     └────────────────────────┘
>          Compute Units
>     [Tensor Cores | CUDA Cores]
> ```

[^data-flow]: Data flows down from global memory through the cache hierarchy to registers where computation happens. Results flow back up. The key optimization is keeping data in lower levels (registers, shared memory) as long as possible to avoid expensive trips to HBM. Each level trades capacity for latency—registers are tiny but instant, HBM is massive but slow.

> [!info] Hopper H100 memory hierarchy (per SM unless noted)
>
> | Level                            | Capacity               | Approx latency   | Notes                                                        |
> | -------------------------------- | ---------------------- | ---------------- | ------------------------------------------------------------ |
> | Registers                        | 256 KB                 | ~1 cycle         | Allocated per warp, spilled to L1 on pressure                |
> | Shared/L1 cache                  | 256 KB (configurable)  | 20–30 cycles     | Supports 64/128/256 KB split, sector-based replacement       |
> | Tensor Memory Accelerator queues | 32 descriptors         | <10 cycles issue | Streams multidimensional tiles without register address math |
> | L2 cache (device)                | 50 MB                  | ~200 cycles      | 16 channels, sector caches feed SM quads                     |
> | HBM3 global mem                  | 80 GB @ 3.35 TB/s      | 400–500 cycles   | Four stacks per SXM, 64-byte access granularity              |
> | NVLink/NVSwitch                  | 900 GB/s bidirectional | microseconds     | Fabric for scale-out multi-GPU                               |
>
> Latency and bandwidth derived from architectural specs and microbenchmark studies.

### memory coalescing

> [!tip] Coalesced access
> Threads in a warp should read contiguous addresses so the memory subsystem collapses the warp request into a single 32-byte, 64-byte, or 128-byte transaction.

Reading a matrix row demonstrates coalesced vs uncoalesced patterns:

```cuda
// Coalesced: each thread reads consecutive element
float value = A[row * N + threadIdx.x];  // ✓ Good

// Uncoalesced: threads read strided elements
float value = A[threadIdx.x * N + col];  // ✗ Bad (if N > 32)
```

> [!warning] Performance impact
>
> - Coalesced: one transaction per warp.
> - Uncoalesced: up to 32 transactions per warp (32× slower in the worst case).

> [!note] Vectorized loads
> Hopper issues 128-bit `LDG.128` or `LDGSTS.128` instructions when data is 16-byte aligned, reducing instruction count by 4×.

When accessing aligned data, GPUs can issue wide vectorized load instructions that fetch multiple elements atomically in a single instruction.

**Basic example**:

```cuda
// Scalar loads (4 instructions)
float a0 = A[idx + 0];
float a1 = A[idx + 1];
float a2 = A[idx + 2];
float a3 = A[idx + 3];

// Vectorized load (1 instruction)
float4 data = *reinterpret_cast<float4*>(&A[idx]);
// Equivalent to: float4 data; data.x = A[idx]; data.y = A[idx+1]; ...
```

**Requirements for vectorized loads**:

- Address must be 16-byte aligned (for `float4`, `int4`, `double2`)
- Data must be contiguous in memory
- Loads 128 bits atomically
- **4× fewer memory instructions** compared to scalar loads
- Available for: `float4`, `float2`, `int4`, `int2`, `double2`, `half8`, etc.

> [!success] Performance benefits
>
> | Load Type      | Instructions | Bytes per Instruction | Efficiency |
> | -------------- | ------------ | --------------------- | ---------- |
> | Scalar `float` | 4            | 4 bytes               | 1×         |
> | `float2`       | 2            | 8 bytes               | 2×         |
> | `float4`       | 1            | 16 bytes              | 4×         |
>
> For a warp loading 128 bytes total:
>
> - Scalar loads: 32 threads × 4 bytes = 32 instructions per warp
> - Vectorized `float4`: 32 threads × 16 bytes = 8 `float4` loads = 8 instructions per warp
> - **4× instruction reduction** → higher memory throughput, lower register pressure

**PTX assembly**:

```ptx
// Scalar load
ld.global.f32 %f0, [%r0];           // Load 4 bytes

// Vectorized load (128-bit)
ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%r0];  // Load 16 bytes atomically
```

**Practical usage in matrix operations**:

```cuda
// Loading row of matrix A (assuming 16-byte alignment)
__global__ void matmul_vectorized(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Process 4 elements at a time using vectorized loads
    for (int k = 0; k < N; k += 4) {
        // Load 4 consecutive elements from A and B
        float4 a_vec = *reinterpret_cast<float4*>(&A[row * N + k]);
        float4 b_vec = *reinterpret_cast<float4*>(&B[k * N + col]);

        // Accumulate
        sum += a_vec.x * b_vec.x;
        sum += a_vec.y * b_vec.y;
        sum += a_vec.z * b_vec.z;
        sum += a_vec.w * b_vec.w;
    }

    C[row * N + col] = sum;
}
```

> [!warning] Alignment requirements
>
> Misaligned access forces the hardware to split the load into multiple transactions:
>
> ```cuda
> // Aligned (good)
> float4* ptr_aligned = (float4*)&A[0];   // A[0] at 16-byte boundary
> float4 data = *ptr_aligned;             // Single 128-bit load
>
> // Misaligned (bad)
> float4* ptr_misaligned = (float4*)&A[1]; // A[1] not at 16-byte boundary
> float4 data = *ptr_misaligned;           // Splits into 2 transactions!
> ```
>
> Ensure arrays are allocated with proper alignment:
>
> ```cuda
> // Host allocation with alignment
> float* A;
> cudaMalloc(&A, N * sizeof(float));  // cudaMalloc returns 256-byte aligned
>
> // Shared memory alignment
> __shared__ __align__(16) float smem[128];  // Force 16-byte alignment
> ```

> [!example] Warp-level thread mapping and coalescing [^warp]
>
> ```
> Thread Block (32×32 = 1024 threads)
> ┌────────────────────────────────────────┐
> │ Warp 0  [T0 T1 ... T31] ─┐             │
> │ Warp 1  [T0 T1 ... T31]  │← 32 Warps   │
> │ Warp 2  [T0 T1 ... T31]  │             │
> │   ...                    │             │
> │ Warp 31 [T0 T1 ... T31] ─┘             │
> └────────────────────────────────────────┘
>            ↓ Memory Access
>     [A0][A1][A2]...[A31]  ← Coalesced!
>     └─ 128-byte transaction ─┘
>
>     vs.
>
>     [A0]  [A32] [A64] ... [A992]  ← Strided
>      ↑     ↑     ↑         ↑
>     32 separate transactions! ✗
> ```

[^warp]: Warps of 32 threads execute in lockstep (SIMT). When they access consecutive memory addresses (coalesced), the hardware issues a single wide transaction—one 128-byte load serves the entire warp. Strided access forces 32 separate transactions, killing bandwidth. Always structure access patterns so adjacent threads read adjacent addresses. This is why column-major access in row-major layouts is disastrous.

### shared memory and bank conflicts

> [!note] Shared memory organization
>
> - 32 banks on Hopper/Blackwell, each 4 bytes wide (8 bytes in 64-bit mode).
> - Successive 32-bit words map to successive banks; the 33rd word aliases bank 0 again.

> [!warning] Bank conflicts
> Multiple threads hitting the same bank force serialization; pad or swizzle to stagger accesses.

```cuda
__shared__ float smem[32][32];

// Conflict-free: each thread accesses different bank
float val = smem[threadIdx.x][col];  // ✓ Good

// 32-way conflict: all threads access bank 0
float val = smem[col][0];  // ✗ Bad
```

> [!tip] Swizzling

```cuda
__shared__ float smem[32][33];  // Add padding column
// Now adjacent threads access different banks even for column access
```

> [!note] Broadcast
> Hardware detects when all threads read the same address and performs a single fetch with broadcast.

### tensor memory accelerator (TMA)

> Hopper introduces the tensor memory accelerator to issue multidimensional copies from HBM into shared memory with a single descriptor, enabling asynchronous pipelines that free registers from manual address math.

> [!important] TMA architectural components
>
> 1. TMA Descriptors: Host-side created metadata encoding tensor shape, strides, element size, swizzling pattern
> 2. Global-to-Shared Engine: Hardware unit that executes bulk transfers without thread participation
> 3. mbarrier Synchronization: Phase-based barriers coordinating producer/consumer pipelines
> 4. Multicast/Reduction: SM-to-SM communication via thread block clusters

#### TMA descriptor creation

TMA descriptors live in global memory and encode the entire tensor layout:

```cpp
// Host-side descriptor creation
CUtensorMap tma_desc;
cuTensorMapEncodeTiled(
    &tma_desc,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    /*rank=*/2,                        // 2D tensor
    global_ptr,                        // Global memory base address
    /*dims=*/{M, N},                   // Tensor dimensions
    /*strides=*/{N * sizeof(float), sizeof(float)},  // Row-major strides
    /*box_dims=*/{TILE_M, TILE_N},     // Tile dimensions
    /*elem_strides=*/{1, 1},           // Contiguous elements
    CU_TENSOR_MAP_SWIZZLE_128B,        // Swizzle mode for bank conflict avoidance
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

The descriptor replaces hundreds of per-thread address computations with a single hardware table lookup.

#### TMA PTX instructions

Hopper's `cp.async.bulk.tensor` PTX instructions transfer entire tiles:

```cuda
// TMA load: Global → Shared memory
asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    " [%0], [%1, {%2, %3}], [%4];"
    :: "r"(smem_ptr),                  // Destination: Shared memory
       "l"(&tma_desc),                 // TMA descriptor
       "r"(tile_coord_m),              // Tile coordinate along M
       "r"(tile_coord_n),              // Tile coordinate along N
       "r"(mbarrier_ptr)               // Arrival barrier
);

// TMA store: Shared → Global memory
asm volatile(
    "cp.async.bulk.tensor.2d.global.shared::cta"
    " [%0, {%1, %2}], [%3];"
    :: "l"(&tma_desc),                 // TMA descriptor
       "r"(tile_coord_m),              // Tile coordinate
       "r"(tile_coord_n),
       "r"(smem_ptr)                   // Source: Shared memory
);
```

> [!note] Zero-thread overhead
> TMA copies execute independently of thread execution. Only one thread in the block issues the TMA instruction; all 1024 threads remain free to execute other work.

#### mbarrier synchronization

TMA uses mbarrier (arrival/wait barrier) for producer-consumer synchronization:

```cuda
__shared__ __align__(8) uint64_t mbarrier;

// Initialize barrier with expected transaction count
if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 :: "r"(&mbarrier), "r"(1));  // Expect 1 TMA transaction
}
__syncthreads();

// Producer: Issue TMA load (thread 0 only)
if (threadIdx.x == 0) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(smem_ptr), "l"(&tma_desc), "r"(0), "r"(0), "r"(&mbarrier));
}

// All threads wait for TMA completion
asm volatile("mbarrier.try_wait.parity.shared.b64 %0, [%1], %2;"
             : "=r"(waitComplete)
             : "r"(&mbarrier), "r"(phase));

// Consume data from shared memory
float value = smem[threadIdx.y][threadIdx.x];
```

The `mbarrier::complete_tx::bytes` semantics ensure the barrier triggers only after the full tensor transfer completes.

#### TMA multicast and reduction

Thread block clusters enable TMA to multicast data to multiple SMs simultaneously:

```cuda
// Multicast TMA load: One descriptor → Multiple SMs' shared memories
asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
    " [%0], [%1, {%2, %3}], [%4], %5;"
    :: "r"(smem_ptr),                  // Destination in each SM's shared memory
       "l"(&tma_desc),
       "r"(tile_m), "r"(tile_n),
       "r"(mbarrier_ptr),
       "h"(cluster_mask)               // Bit mask: which SMs receive data
);
```

TMA reduction aggregates data across cluster SMs:

```cuda
// Reduction: Aggregate partial results from distributed shared memory
asm volatile(
    "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group"
    " [%0, {%1, %2}], [%3];"
    :: "l"(&tma_desc), "r"(tile_m), "r"(tile_n), "r"(smem_ptr)
);
```

Supported reduction operations: `add`, `min`, `max`, `and`, `or`, `xor`.

#### Performance impact

> [!success] TMA performance benefits
>
> | Metric                 | Traditional `cp.async`               | TMA                           |
> | ---------------------- | ------------------------------------ | ----------------------------- |
> | Register pressure      | High (address arithmetic per thread) | Minimal (descriptor lookup)   |
> | Instruction overhead   | ~10-20 instructions per thread       | 1 instruction per block       |
> | Software pipelining    | Manual loop unrolling required       | Hardware-managed via mbarrier |
> | Multi-SM coordination  | Requires explicit synchronization    | Native multicast support      |
> | Bank conflict handling | Manual swizzling in code             | Encoded in descriptor         |

> [!example] Measured speedup in Flash Attention 4
> TMA-based implementation achieves 15-20% higher occupancy and 1.18× end-to-end speedup vs hand-tuned `cp.async` due to:
>
> - Freed registers enabling more concurrent thread blocks
> - Elimination of address computation overhead
> - Overlap of TMA transfers with WGMMA computation

The canonical TMA pattern in CUTLASS 3.x:

```cpp
using TmaLoad = SM90_TMA_LOAD;
auto tma_load = make_tma_copy(TmaLoad{}, gmem_layout, smem_layout, tile_shape, cluster_shape);

// In kernel: Single thread issues copy
if (cute::elect_one_sync()) {
    cute::copy(tma_load, gmem_tensor(tile_coord), smem_tensor);
}
cute::cp_async_wait<0>();  // Wait for all pending TMA operations
__syncthreads();
```

### thread block clusters and distributed shared memory

Hopper introduces thread block clusters: a new programming hierarchy level that groups multiple thread blocks executing on different SMs to enable distributed shared memory (DSMEM) and collective operations.

> [!important] CUDA programming hierarchy evolution
>
> ```
> Ampere/Ada:           Hopper/Blackwell:
> Grid                  Grid
>  └─ Thread Block       └─ Thread Block Cluster (NEW)
>      └─ Warp                └─ Thread Block
>          └─ Thread               └─ Warp
>                                      └─ Thread
> ```
>
> Clusters introduce locality: blocks within a cluster are guaranteed to co-schedule on nearby SMs, enabling low-latency SM-to-SM communication.

#### Distributed Shared Memory (DSMEM)

Traditionally, shared memory is private to a thread block. DSMEM allows threads to read and write shared memory on neighboring SMs within the cluster:

```cuda
__global__ void __cluster_dims__(2, 2, 1)  // 2×2 cluster = 4 blocks
cluster_kernel(float* data) {
    __shared__ float smem[256];

    // Each block initializes its own shared memory
    int tid = threadIdx.x;
    smem[tid] = blockIdx.x * 1000 + tid;
    __syncthreads();

    // Get cluster dimensions and block rank within cluster
    cluster_group cluster = this_cluster();
    int cluster_rank = cluster.block_rank();  // 0-3
    int num_blocks = cluster.num_blocks();    // 4

    // Access shared memory from neighboring block
    // Block 0 reads from Block 1's shared memory
    if (cluster_rank == 0) {
        float neighbor_value = cluster.map_shared_rank(smem, 1)[tid];
        // neighbor_value contains Block 1's data via DSMEM!
    }

    cluster.sync();  // Synchronize all blocks in cluster
}
```

> [!success] DSMEM performance characteristics
>
> - 7× faster than global memory roundtrip for SM-to-SM data exchange
> - Direct memory access: No explicit copy instructions needed
> - Hardware-managed coherence: L1 cache ensures consistency across SMs
> - Low latency: ~20 cycles for intra-cluster access vs ~400 for global memory

#### Cluster API

CUDA exposes clusters via the cooperative groups API:

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(Cluster_X, Cluster_Y, Cluster_Z)
my_kernel() {
    // Query cluster properties
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int cluster_size = cluster.num_blocks();           // Total blocks in cluster
    unsigned int cluster_rank = cluster.block_rank();           // This block's index (0..size-1)
    dim3 cluster_idx = cluster.dim_blocks();                    // Cluster dimensions
    dim3 block_in_cluster = cg::block_rank_in_cluster();        // This block's 3D coordinate

    // Distributed barrier: Wait for all blocks in cluster
    cluster.sync();

    // Access remote shared memory
    __shared__ int smem[128];
    int* remote_smem = cluster.map_shared_rank(smem, target_rank);
    int value = remote_smem[threadIdx.x];
}
```

Launch configuration extends grid dimensions:

```cuda
dim3 grid(Bx, By, Bz);
dim3 block(Tx, Ty, Tz);

// Launch with cluster dimensions
cudaLaunchKernelEx(&config, my_kernel, args...);

// Or use extended launch attributes
cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim.x = Cluster_X;
attrs[0].val.clusterDim.y = Cluster_Y;
attrs[0].val.clusterDim.z = Cluster_Z;
cudaLaunchKernelEx(&config, my_kernel, args...);
```

#### Use cases: overlap and multicast

Use case 1: Cooperative matmul with data reuse

```cuda
// 2×2 cluster: Each block computes different output tiles, but shares input tiles
__global__ void __cluster_dims__(2, 2, 1)
cluster_matmul(float* A, float* B, float* C) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    cluster_group cluster = this_cluster();

    // Block 0 loads tile of A via TMA multicast to all 4 blocks
    if (cluster.block_rank() == 0) {
        // TMA multicast: one load → 4 SMs' shared memories
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global"
                     ".mbarrier::complete_tx::bytes.multicast::cluster"
                     " [%0], [%1, {%2, %3}], [%4], %5;"
                     :: "r"(As), "l"(&tma_desc_A), "r"(tile_m), "r"(k),
                        "r"(mbarrier), "h"(0xF));  // 0xF = all 4 blocks
    }

    // Each block loads its own slice of B independently
    // ... load Bs ...

    cluster.sync();

    // All blocks now have shared A via DSMEM, compute local outputs
    matmul_tile(As, Bs, C);
}
```

Use case 2: Distributed reduction across SMs

```cpp
__global__ void __cluster_dims__(4, 1, 1)
cluster_reduce(float* input, float* output) {
    __shared__ float smem[256];

    // Each block computes partial sum
    smem[threadIdx.x] = input[blockIdx.x * 256 + threadIdx.x];
    __syncthreads();

    float partial = block_reduce_sum(smem);

    cluster_group cluster = this_cluster();

    // Write partial result to Block 0's shared memory via DSMEM
    if (threadIdx.x == 0) {
        float* block0_smem = cluster.map_shared_rank(smem, 0);
        block0_smem[cluster.block_rank()] = partial;
    }

    cluster.sync();

    // Block 0 aggregates from all 4 blocks
    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < 4; ++i) {
            total += smem[i];
        }
        output[blockIdx.x / 4] = total;
    }
}
```

#### Hardware constraints

> [!warning] Cluster size limitations
>
> - Maximum cluster size: 8 thread blocks on Hopper (SM90)
> - Blocks in a cluster must fit on adjacent SMs with available resources
> - If resources insufficient, driver falls back to non-clustered execution
> - Optimal cluster sizes: 2, 4, or 8 blocks depending on workload

> [!note] GPC topology awareness
> Hopper H100 SMs are organized into Graphics Processing Clusters (GPCs):
>
> - 8 GPCs × 18 SMs/GPC = 144 total SMs
> - Clusters scheduled within GPC boundaries for minimal latency
> - Cross-GPC DSMEM access incurs higher latency (~50 vs 20 cycles)

The combination of TMA multicast + thread block clusters + DSMEM enables warp-specialized persistent kernels where producer warps stream data via TMA while consumer warps continuously compute, achieving near-peak utilization by hiding memory latency.

## matrix multiplication: from naive to optimized

### GEMM

General Matrix Multiply (GEMM) computes $D = \alpha AB + \beta C$ where $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, $C \in \mathbb{R}^{M \times N}$, and $\alpha, \beta$ are scalars.

GEMM accounts for 80-90% of compute in transformer training and inference. Optimizing GEMM is optimizing AI.

> [!info] BLAS hierarchy and naming conventions
>
> BLAS (Basic Linear Algebra Subprograms) defines three levels:
>
> - Level 1: Vector-vector operations (AXPY: $y = \alpha x + y$)
> - Level 2: Matrix-vector operations (GEMV: $y = \alpha Ax + \beta y$)
> - Level 3: Matrix-matrix operations (GEMM: $C = \alpha AB + \beta C$)
>
> Naming: `{precision}{operation}{modifiers}`
>
> - SGEMM: Single-precision (FP32) GEMM
> - DGEMM: Double-precision (FP64) GEMM
> - HGEMM: Half-precision (FP16) GEMM
> - Modifiers: `T` for transpose (e.g., `GEMM_NT`: A normal, B transposed)
>
> cuBLAS provides optimized BLAS implementations for NVIDIA GPUs. CUTLASS provides template-based building blocks to create custom GEMM kernels.

> [!example] GEMM operation breakdown
>
> For $C_{M \times N} = A_{M \times K} B_{K \times N}$:
>
> ```
> C[i,j] = Σ(k=0 to K-1) A[i,k] * B[k,j]
>
> Computational complexity:
> - FLOPs: 2MNK (K multiplies + K-1 adds per output element)
> - Memory (naive): (MK + KN + MN) elements
> - Arithmetic intensity (naive): 2MNK / 2(MK+KN+MN) bytes
>
> For square matrices (M=N=K):
> - FLOPs: 2N³
> - Memory: 3N² elements = 6N² bytes (FP16)
> - Arithmetic intensity: 2N³ / 6N² = N/3 FLOPs/byte
>
> Example: N=4096
> - FLOPs: ~137 GFLOP
> - Memory: ~192 MB
> - Arithmetic intensity: ~683 FLOPs/byte (highly compute-bound with tiling!)
> ```

We build intuition by progressively optimizing matmul, measuring performance at each stage.

### baseline: naive implementation

```cuda
__global__ void matmul_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

Performance analysis:

- Each thread loads $2K$ values from global memory (two loads per MAC)
- Computes $2K$ FLOPs
- Arithmetic intensity: $\frac{2K}{2K \times 2 \text{ bytes}} = \frac{1}{2}$ FLOPs/byte (0.5 FLOP/byte)
- Roofline bound on H100: $0.5 \times 3.35\,\text{TB/s} = 1.68\,\text{TFLOP/s}$
- Utilization vs 1,979 TFLOP/s peak: $1.68 / 1,979 \approx 0.08\%$

> [!calculation] 4096×4096×4096 FP16 matmul — naive kernel roofline
>
> - Total FLOPs: $2N^3 = 2 \times 4096^3 = 1.3744 \times 10^{11}$ (137.4 GFLOP)
> - Global traffic: each MAC issues two 2-byte loads ⇒ $2N^3 \times 2 = 2.75 \times 10^{11}$ bytes plus $N^2 \times 2 = 3.4 \times 10^7$ bytes for the store → $2.75\times10^{11}$ bytes (256 GiB)
> - Arithmetic intensity: $1.3744\times10^{11} / 2.75\times10^{11} = 0.5$ FLOP/byte
> - Peak sustained throughput bounded by HBM3 bandwidth (3.35 TB/s): $0.5 \times 3.35 = 1.68$ TFLOP/s
> - Runtime lower bound: $2.75\times10^{11} / 3.35\times10^{12} = 8.2\times10^{-2}$ s ⇒ $\ge 82$ ms for the naive kernel at scale 4096

Bottleneck: Memory bandwidth. Each value of $A$ and $B$ is loaded multiple times from global memory.

### tiled matmul with shared memory

trick: Load tiles into shared memory, reuse across multiple thread computations.

Tiling strategy:

```
A (M×K)  ×  B (K×N)  =  C (M×N)

Partition into tiles:
- Thread block tile: 128×128 output elements
- Shared memory tile: 128×K_tile (typically K_tile=32)
```

> [!example] matmul tiling
>
> visualization [^matmul-tiling]
>
> ```
> Matrix A (M×K)              Matrix B (K×N)
> ┌─────────────────┐        ┌──────────────────────┐
> │ ┌───┐           │        │ ┌───┐ ┌───┐ ┌───┐    │
> │ │128│           │        │ │ 3 │ │   │ │   │    │
> │ │ × │  Thread   │        │ │ 2 │ │   │ │   │    │
> │ │ 3 │  Block    │   ×    │ └─┬─┘ └───┘ └───┘    │
> │ │ 2 │  Tile     │        │   │                  │
> │ └───┘           │        │   │ K=32             │
> │                 │        │   ↓                  │
> └─────────────────┘        └──────────────────────┘
>                                     ↓
>                           Matrix C (M×N)
>                     ┌──────────────────────┐
>                     │ ┌─────┐ ┌───┐ ┌───┐  │
>                     │ │ 128 │ │   │ │   │  │
>                     │ │  ×  │ │   │ │   │  │
>                     │ │ 128 │ │   │ │   │  │
>                     │ └─────┘ └───┘ └───┘  │
>                     │                      │
>                     └──────────────────────┘
> ```

> [!calculation] 128×128×32 tile bandwidth math (FP16 on H100)
>
> - Per $K$-slice we load $128\times32$ elements from $A$ and $B$ each ⇒ $8192$ halfs = 16 KB
> - With $K = 4096$ there are 128 such slices, so an output tile pulls $128 \times 16$ KB = 2.10 MB from HBM and writes a 128×128 tile (128 KB)
> - Summed over all tiles of a 4096³ GEMM, global traffic drops to 2.18 GB (down from 256 GB in the naive kernel)
> - Arithmetic intensity climbs to $2N^3 / 2.18\,\text{GB} = 63$ FLOP/byte ⇒ roofline cap $63 \times 3.35 = 211$ TFLOP/s
> - Bandwidth-limited runtime floor: $2.18\,\text{GB} / 3.35\,\text{TB/s} = 0.65$ ms, a 126× reduction in bytes moved compared to the naive loop

[^matmul-tiling]: Each thread block computes a 128×128 tile of output C. Input tiles are loaded into shared memory (128×32 slices from A and B), reused across all 128×128=16,384 threads. The K dimension is tiled into chunks of 32; the outer loop steps through K/32 iterations. This transforms $O(MNK)$ global memory accesses into $O(MN + NK)$ with an $O(K)$ reuse factor, boosting arithmetic intensity by >120×. With FP16 inputs (2-byte elements) the same blocking yields $I \approx 64$ FLOP/byte; further hierarchy-aware staging (L2 residency, tensor memory accelerator) can push toward the $I \approx 10^3$ regime needed to become compute-bound on Hopper.

Algorithm:

```cuda
__global__ void matmul_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[128][32];
    __shared__ float Bs[32][128];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * 128 + ty;
    int col = blockIdx.x * 128 + tx;

    float sum = 0.0f;

    // Loop over K dimension in tiles
    for (int tile = 0; tile < K; tile += 32) {
        // Cooperatively load tile into shared memory
        As[ty][tx] = A[row * K + (tile + tx)];
        Bs[ty][tx] = B[(tile + ty) * N + col];
        __syncthreads();

        // Compute using shared memory
        for (int k = 0; k < 32; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}
```

Performance analysis:

- Each tile loaded once from global memory
- Reused across 128 thread computations
- Arithmetic intensity: $\frac{2 \times 128 \times 128 \times 32}{2 \times 128 \times 32 \times 2} \approx 64$ FLOPs/byte
- Performance: ~300 TFLOP/s
- Utilization: 15%

Improvement: 10× faster than naive implementation

Remaining issues:

- Not using tensor cores (matrix multiply accelerators)
- Bank conflicts in shared memory
- Register spilling (too many live values)

## tensor cores and mixed precision

### tensor core architecture

Tensor cores are specialized hardware units for accelerating matrix multiplication, introduced in Volta (2017) and evolved through Turing, Ampere, Ada Lovelace, Hopper, and Blackwell architectures.

Evolution across generations:

| Architecture        | Year | Tensor Core Gen | FP16 TFLOP/s | New Features                  |
| ------------------- | ---- | --------------- | ------------ | ----------------------------- |
| Volta (V100)        | 2017 | 1st gen         | 125          | Mixed precision (FP16)        |
| Turing (T4)         | 2018 | 2nd gen         | 65           | INT8, INT4 support            |
| Ampere (A100)       | 2020 | 3rd gen         | 312          | TF32, BF16, FP64 tensor cores |
| Ada Lovelace (L40S) | 2022 | 4th gen         | 362          | FP8 support                   |
| Hopper (H100)       | 2022 | 4th gen         | 1979         | FP8, wgmma instructions, TMA  |
| Blackwell (B200)    | 2024 | 5th gen         | 4500         | FP4, dynamic precision        |

### tensor core operation

Fundamental operation: $D = A \times B + C$

Where:

- $A$: $M \times K$ matrix
- $B$: $K \times N$ matrix
- $C$: $M \times N$ matrix (accumulator)
- $D$: $M \times N$ matrix (result)

Warp-level matrix multiply-accumulate (WMMA):

```cuda
// Ampere/Ada: wmma namespace
wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

wmma::load_matrix_sync(a_frag, a_ptr, lda);
wmma::load_matrix_sync(b_frag, b_ptr, ldb);
wmma::fill_fragment(c_frag, 0.0f);

wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

wmma::store_matrix_sync(d_ptr, c_frag, ldd, wmma::mem_row_major);
```

Warp group matrix multiply-accumulate (WGMMA) - Hopper:

```cuda
// Hopper: wgmma instructions operate on warp groups (128 threads)
// Compiled to GMMA SASS instructions
// Asynchronous execution with explicit synchronization

wgmma::mma_async<
    MMA_M, MMA_N, MMA_K,
    float,      // D type
    __nv_bfloat16,  // A type
    __nv_bfloat16,  // B type
    float       // C type
>(d_frag, a_frag, b_frag, c_frag);

wgmma::commit_group();
wgmma::wait_group<0>();  // Wait for all preceding wgmma operations
```

Differences:

- WMMA: Warp-level (32 threads), synchronous
- WGMMA: Warp group-level (128 threads), asynchronous, higher throughput

> [!example] Tensor core operation flow [^tensor-core]
>
> ```
> Warp Registers                     Shared Memory
> ┌──────────────┐                  ┌──────────────┐
> │ A_frag       │                  │   A_tile     │
> │ [16×16 BF16] │ ←── load_sync ── │   [128×32]   │
> └──────────────┘                  └──────────────┘
> ┌──────────────┐                  ┌──────────────┐
> │ B_frag       │                  │   B_tile     │
> │ [16×16 BF16] │ ←── load_sync ── │   [32×128]   │
> └──────────────┘                  └──────────────┘
>         ↓                                 ↓
>         └──────── mma_sync ───────────────┘
>                     ↓
>         ┌──────────────────────┐
>         │  C_frag (accumulator)│
>         │  [16×16 FP32]        │
>         │  D = A×B + C         │
>         └──────────────────────┘
>                     ↓
>         Repeat for K dimension →
> ```

[^tensor-core]: Tensor cores operate on 16×16 matrix tiles. Each warp loads fragments from shared memory into registers via load_matrix_sync, issues mma_sync to the tensor core hardware, accumulates into FP32. The outer loop tiles over K, so C_frag accumulates across multiple K-tiles (e.g., 32 iterations for K=512). One mma_sync on Hopper performs 2×16×16×16 = 8,192 FLOPs in a single instruction—this achieves 20× speedup vs CUDA cores because tensor cores are massive matrix-multiply ALUs.

### mixed precision arithmetic

Motivation: Reduce memory bandwidth and increase throughput by using lower precision for compute-intensive operations.

Precision formats:

```text
┌──────────┬──────────┬────────────┬──────────────┬─────────────┐
│ Format   │ Bits     │ Range      │ Precision    │ Use Case    │
├──────────┼──────────┼────────────┼──────────────┼─────────────┤
│ FP32     │ 32       │ ±3.4e38    │ 7 decimal    │ Reference   │
│ TF32     │ 19*      │ ±3.4e38    │ 3 decimal    │ Training    │
│ FP16     │ 16       │ ±65504     │ 3 decimal    │ Training    │
│ BF16     │ 16       │ ±3.4e38    │ 2 decimal    │ Training    │
│ FP8-E5M2 │ 8        │ ±57344     │ 1 decimal    │ Inference   │
│ FP8-E4M3 │ 8        │ ±448       │ 2 decimal    │ Inference   │
│ MXFP8    │ 8+scale  │ Dynamic    │ 1-2 decimal  │ Inference   │
│ NVFP4    │ 4        │ [-7,7]     │ ~1 decimal   │ Inference   │
│ INT8     │ 8        │ [-128,127] │ Quantized    │ Inference   │
│ INT4     │ 4        │ [-8,7]     │ Quantized    │ Inference   │
└──────────┴──────────┴────────────┴──────────────┴─────────────┘
* TF32 uses 32-bit container but 19 bits precision (8 exp + 10 mantissa + 1 sign)
```

BF16 (Brain Float 16):

- Same exponent range as FP32 (8 bits)
- Reduced mantissa (7 bits vs 23 in FP32)
- Advantage: Easy conversion from FP32 (truncate mantissa)
- Widely used in training (better gradient dynamics than FP16)

FP8 formats:

- E5M2: 5-bit exponent, 2-bit mantissa (wider range, less precision)
- E4M3: 4-bit exponent, 3-bit mantissa (narrower range, more precision)
- Used with block scaling: Group of values share a common scale factor

MXFP8 (Microscaling FP8):

- Groups of 32-128 values share a scaling factor
- Enables dynamic range adjustment
- Better accuracy than naive FP8

INT4/NVFP4:

- 4-bit quantization for extreme compression
- Blackwell architecture introduces native FP4 tensor core support
- Enables 2× memory bandwidth savings vs FP8

### quantization strategies

Static quantization:

```python
# Quantize weights offline
scale = max(abs(W)) / 127
W_int8 = round(W / scale).clip(-128, 127)

# Dequantize during inference
W_approx = W_int8 * scale
```

Dynamic quantization:

- Compute scale factors at runtime
- Better accuracy for activations with dynamic range
- Higher overhead (compute scale per batch)

Per-channel vs per-tensor quantization:

- Per-tensor: Single scale for entire weight matrix
- Per-channel: Different scale for each output channel
- Per-channel: Better accuracy, more memory for scale storage

Quantization-aware training (QAT):

- Simulate quantization during training
- Learn scale factors and quantized weights jointly
- Best accuracy but requires retraining

### tensor core matmul with mixed precision

Typical pattern: FP16/BF16 input, FP32 accumulation

```cuda
__global__ void matmul_wmma_bf16(
    __nv_bfloat16* A,
    __nv_bfloat16* B,
    float* C,
    int M, int N, int K
) {
    using namespace nvcuda::wmma;

    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // Declare fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    fill_fragment(c_frag, 0.0f);

    // Loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        load_matrix_sync(a_frag, A + aRow * K + aCol, K);
        load_matrix_sync(b_frag, B + bRow * N + bCol, N);

        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    store_matrix_sync(C + warpM * WMMA_M * N + warpN * WMMA_N, c_frag, N, mem_row_major);
}
```

Performance: ~650 TFLOP/s (BF16 input, FP32 accumulation on H100)

Improvement: 20× faster than tiled FP32 implementation

## CUTLASS

### overview and architecture

> [!info] CUTLASS building blocks
>
> - Template-specialized GEMM, convolution, and rank-$k$ kernels tuned for each compute capability.
> - Hierarchical tiling primitives anchored in the CuTe DSL for layouts, tensors, and tilers.
> - Math operator classes spanning SIMT, WMMA, and WGMMA tensor core paths with FP64→FP4 precision coverage.
> - Extensible epilogues and fusion hooks for bias, activation, quantization, and paged-attention style reductions.

> [!tip] Design philosophy
>
> - Compose kernels from reusable layout algebra, copy atoms, and math atoms instead of monolithic CUDA.
> - Keep performance portable by specializing minimal policy structs per architecture while reusing CuTe tilers.
> - Enable user-defined epilogues and layout transforms so kernels fuse per-application epilogue math without touching the mainloop.

### CUTLASS architecture layers

```
┌─────────────────────────────────────────────────────┐
│ High-Level Operations (GEMM, Conv, Rank-K, etc.)    │
├─────────────────────────────────────────────────────┤
│ CuTe DSL (Layouts, Tensors, Tilers)                 │ ← Abstraction layer
├─────────────────────────────────────────────────────┤
│ Thread Block Tiles (CTAs)                           │
│  - Mainloop: Load tiles, compute MMA                │
│  - Epilogue: Store results, apply operations        │
├─────────────────────────────────────────────────────┤
│ Warp Tiles                                          │
│  - MMA atoms: Tensor core operations                │
│  - Copy atoms: Memory movement patterns             │
├─────────────────────────────────────────────────────┤
│ Thread Layout and Register Allocation               │
└─────────────────────────────────────────────────────┘
```

## CuTe layout algebra

### layout definition and properties

A layout $\mathcal{L}$ is formally defined as an ordered pair:

$$
\mathcal{L} = (S, D)
$$

where:

- $S = (s_0, s_1, \ldots, s_{r-1})$ is the shape tuple (positive integers)
- $D = (d_0, d_1, \ldots, d_{r-1})$ is the stride tuple (positive integers)
- $r$ is the rank (number of dimensions)

Layout function $f_{\mathcal{L}}: \mathbb{N}^r \to \mathbb{N}$:

$$
f_{\mathcal{L}}(x_0, x_1, \ldots, x_{r-1}) = \sum_{i=0}^{r-1} x_i \cdot d_i
$$

where $0 \le x_i < s_i$ for all $i$.

Layout size:

$$
\text{size}(\mathcal{L}) = \prod_{i=0}^{r-1} s_i
$$

Layout cosize:

$$
\text{cosize}(\mathcal{L}) = \max_{x \in \text{domain}} f_{\mathcal{L}}(x) + 1
$$

Example:

```cpp
// Layout (8, 4):(1, 8) represents 8×4 column-major matrix
// Shape: (8, 4) → 32 elements
// Stride: (1, 8) → column-major ordering
// f(row, col) = row * 1 + col * 8
// cosize = f(7, 3) + 1 = 7 + 24 + 1 = 32
```

> [!example] Layout mapping visualization
>
> ```
> Layout L = (4, 3):(1, 4)  [column-major 4×3 matrix]
>
> Logical View (coordinates):        Physical Memory (linear addresses):
>
>     col: 0   1   2                  addr:  0  1  2  3  4  5  6  7  8  9  10 11
> row:  ┌───┬───┬───┐                        ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
>   0   │ 0 │ 4 │ 8 │                      [00 01 02 03 04 05 06 07 08 09 10 11]
>       ├───┼───┼───┤
>   1   │ 1 │ 5 │ 9 │                 Mapping function:
>       ├───┼───┼───┤                 f(r, c) = r × 1 + c × 4
>   2   │ 2 │ 6 │10 │
>       ├───┼───┼───┤                 Examples:
>   3   │ 3 │ 7 │11 │                 (0,0) → 0×1 + 0×4 = 0
>       └───┴───┴───┘                 (1,2) → 1×1 + 2×4 = 9
>                                      (3,1) → 3×1 + 1×4 = 7
>
> Column-major: consecutive rows stored contiguously (stride 1 in row dimension)
> ```

Implementation follows CuTe 4.1 layout/partition semantics for Hopper/Blackwell targeted kernels.

Theorem (Layout Function Properties).
Let $\mathcal{L} = (S, D)$ be a layout. Then:

1. Bijection (if cosize = size): $f_{\mathcal{L}}$ is bijective on its domain
2. Linearity: $f_{\mathcal{L}}(c_1 x_1 + c_2 x_2) = c_1 f_{\mathcal{L}}(x_1) + c_2 f_{\mathcal{L}}(x_2)$ for coordinates
3. Mode Independence: $f_{\mathcal{L}}$ factorizes as sum of per-mode contributions

_Proof._ (1) If cosize = size, then $f_{\mathcal{L}}$ maps size distinct coordinates to size distinct addresses, hence bijective. (2) Follows from linearity of dot product. (3) Follows from definition. $\square$

### fundamental layout operations

#### composition

Definition. Let $\mathcal{L}_A = (S_A, D_A)$ and $\mathcal{L}_B = (S_B, D_B)$ be layouts. Their composition $\mathcal{L}_A \circ \mathcal{L}_B$ is defined mode-wise:

$$
f_{\mathcal{L}_A \circ \mathcal{L}_B}(x) = f_{\mathcal{L}_A}(f_{\mathcal{L}_B}(x))
$$

Intuition: $\mathcal{L}_B$ selects elements from $\mathcal{L}_A$, creating a new access pattern.

Example:

```cpp
// A: (8, 4):(1, 8) - column-major 8×4 matrix
// B: (2, 2):(0, 1) - select 2×2 tile
// A ∘ B: First 2 rows, first 2 columns
auto tiled = composition(layout_A, layout_B);
```

> [!example] Layout composition visualization
>
> ```
> Given:
>   Layout A = (8, 4):(1, 8)  [8×4 column-major, 32 elements]
>   Layout B = (2, 2):(0, 1)  [2×2 tile selector]
>
> Step 1: Layout A maps coordinates to memory addresses
>
>   A's logical view:              A's memory layout:
>      0   8  16  24               [0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15,
>      1   9  17  25                16,17,18,19,20,21,22,23, 24,25,26,27,28,29,30,31]
>      2  10  18  26
>      3  11  19  27               f_A(r,c) = r×1 + c×8
>      4  12  20  28
>      5  13  21  29
>      6  14  22  30
>      7  15  23  31
>
> Step 2: Layout B = (2,2):(0,1) selects a 2×2 region
>
>   B's logical view:              B maps to indices in A:
>      (0,0) (0,1)                   0  1
>      (1,0) (1,1)                   0  1
>                                  (stride 0 in row, stride 1 in col)
>
> Step 3: Composition R = A ∘ B applies B's coordinates through A's mapping
>
>   R(i,j) = A(B(i,j))
>
>   R(0,0) = A(B(0,0)) = A(0,0) = 0×1 + 0×8 = 0
>   R(0,1) = A(B(0,1)) = A(0,1) = 0×1 + 1×8 = 8
>   R(1,0) = A(B(1,0)) = A(0,0) = 0×1 + 0×8 = 0
>   R(1,1) = A(B(1,1)) = A(0,1) = 0×1 + 1×8 = 8
>
>   Result R = (2,2):(0,8) - selects first row of A, columns 0 and 1
>                             (broadcasts across rows due to stride 0)
>
>   R's view:
>      0  8     ← same as A[0,0] and A[0,1]
>      0  8     ← broadcast (stride 0 in row dimension)
> ```

Theorem (Composition Associativity).
Composition of layouts is associative: $(\mathcal{L}_A \circ \mathcal{L}_B) \circ \mathcal{L}_C = \mathcal{L}_A \circ (\mathcal{L}_B \circ \mathcal{L}_C)$.

_Proof._ Direct computation using function composition associativity. $\square$

#### logical division

Definition. Let $\mathcal{L} = (S, D)$ be a layout and $T = (t_0, t_1, \ldots, t_{r-1})$ be a shape tuple. Logical division $\mathcal{L} / T$ partitions $\mathcal{L}$ into tiles of shape $T$.

Result: Hierarchical layout with shape $(T, S/T)$ and reorganized strides.

Two strategies:

1. Inner partition: Contiguous tiles
2. Outer partition: Strided tiles

Example:

```cpp
// Layout (64, 64):(1, 64) partitioned by (8, 8)
// Inner: ((8, 8), (8, 8)) - 8×8 tiles, 8×8 tile grid
// Outer: ((8, 8), (8, 8)) with different strides
auto tiled = logical_divide(layout, Shape<_8, _8>{});
```

> [!example] Logical division (tiling) visualization
>
> ```
> Original Layout: L = (16, 16):(1, 16)  [16×16 column-major]
> Tile Shape: T = (4, 4)
> Result: L / T creates hierarchical layout ((4,4), (4,4))
>
> Original 16×16 matrix:                 After logical_divide by (4,4):
>
> ┌─────────────────────────┐           ┌─────┬─────┬─────┬─────┐
> │  0  16  32  48 ... 240  │           │Tile │Tile │Tile │Tile │
> │  1  17  33  49 ... 241  │           │ 0,0 │ 0,1 │ 0,2 │ 0,3 │
> │  2  18  34  50 ... 242  │           │     │     │     │     │
> │  3  19  35  51 ... 243  │           │ 4×4 │ 4×4 │ 4×4 │ 4×4 │
> │  4  20  36  52 ... 244  │           ├─────┼─────┼─────┼─────┤
> │  5  21  37  53 ... 245  │           │Tile │Tile │Tile │Tile │
> │  6  22  38  54 ... 246  │           │ 1,0 │ 1,1 │ 1,2 │ 1,3 │
> │  7  23  39  55 ... 247  │           │     │     │     │     │
> │  8  24  40  56 ... 248  │           │ 4×4 │ 4×4 │ 4×4 │ 4×4 │
> │  9  25  41  57 ... 249  │           ├─────┼─────┼─────┼─────┤
> │ 10  26  42  58 ... 250  │           │Tile │Tile │Tile │Tile │
> │ 11  27  43  59 ... 251  │           │ 2,0 │ 2,1 │ 2,2 │ 2,3 │
> │ 12  28  44  60 ... 252  │           │     │     │     │     │
> │ 13  29  45  61 ... 253  │           │ 4×4 │ 4×4 │ 4×4 │ 4×4 │
> │ 14  30  46  62 ... 254  │           ├─────┼─────┼─────┼─────┤
> │ 15  31  47  63 ... 255  │           │Tile │Tile │Tile │Tile │
> └─────────────────────────┘           │ 3,0 │ 3,1 │ 3,2 │ 3,3 │
>                                       │     │     │     │     │
>                                       │ 4×4 │ 4×4 │ 4×4 │ 4×4 │
>                                       └─────┴─────┴─────┴─────┘
>
> Hierarchical shape: ((4, 4), (4, 4))
>                      └─────┘  └─────┘
>                      tile     tile
>                      dims     grid
>
> Indexing:
>   tiled[tile_y, tile_x, elem_y, elem_x]
>
> Example: Access element at global (5, 10)
>   tile_y = 5 / 4 = 1, elem_y = 5 % 4 = 1
>   tile_x = 10 / 4 = 2, elem_x = 10 % 4 = 2
>   → tiled[1, 2, 1, 2] = original[5, 10]
>
> Benefits:
>   - Enables hierarchical algorithms (block-level then thread-level)
>   - Natural mapping to GPU grid/block/thread hierarchy
>   - Compose with other layout operations (swizzle, partition)
> ```

#### logical product

Definition. Let $\mathcal{L}_A$ and $\mathcal{L}_B$ be layouts. Their logical product $\mathcal{L}_A \times \mathcal{L}_B$ tiles $\mathcal{L}_A$ across dimensions of $\mathcal{L}_B$.

Blocked product:

```cpp
template <class BlockLayout, class TilerLayout>
auto blocked_product(BlockLayout block, TilerLayout tiler) {
    auto prod = logical_product(block, tiler);
    return zip(get<0>(prod), get<1>(prod));
}
```

Raked product: Similar but interleaves blocks.

Use case: Creating thread-value (TV) layouts where each thread processes multiple values.

> [!example] Thread-value layouts: Blocked vs Raked
>
> ```
> Given: 4 threads (T0, T1, T2, T3), each processing 4 values
> Total: 16 elements arranged in 4×4 grid
>
> BLOCKED PRODUCT - Contiguous blocks per thread:
> ═══════════════════════════════════════════════
>
>   Thread layout: (2, 2):(2, 1)        Value layout: (2, 2):(1, 2)
>   ┌────┬────┐                         Each thread gets 2×2 block
>   │ T0 │ T1 │
>   ├────┼────┤
>   │ T2 │ T3 │
>   └────┴────┘
>
> Result - Each thread owns a contiguous block:
>
>    0   1   2   3        Memory Layout:
>  ┌───┬───┬───┬───┐
> 0│T0 │T0 │T1 │T1 │     [T0:0, T0:1, T0:2, T0:3, T1:0, T1:1, T1:2, T1:3,
>  │ 0 │ 1 │ 0 │ 1 │      T2:0, T2:1, T2:2, T2:3, T3:0, T3:1, T3:2, T3:3]
>  ├───┼───┼───┼───┤
> 1│T0 │T0 │T1 │T1 │     Good for:
>  │ 2 │ 3 │ 2 │ 3 │     - Coalesced loads (threads load adjacent addresses)
>  ├───┼───┼───┼───┤     - Matrix tiling
> 2│T2 │T2 │T3 │T3 │     - Tensor core layouts
>  │ 0 │ 1 │ 0 │ 1 │
>  ├───┼───┼───┼───┤
> 3│T2 │T2 │T3 │T3 │
>  │ 2 │ 3 │ 2 │ 3 │
>  └───┴───┴───┴───┘
>
> RAKED PRODUCT - Interleaved values per thread:
> ══════════════════════════════════════════════
>
> Result - Threads interleaved across space:
>
>    0   1   2   3        Memory Layout:
>  ┌───┬───┬───┬───┐
> 0│T0 │T1 │T0 │T1 │     [T0:0, T1:0, T0:1, T1:1, T2:0, T3:0, T2:1, T3:1,
>  │ 0 │ 0 │ 1 │ 1 │      T0:2, T1:2, T0:3, T1:3, T2:2, T3:2, T2:3, T3:3]
>  ├───┼───┼───┼───┤
> 1│T2 │T3 │T2 │T3 │     Good for:
>  │ 0 │ 0 │ 1 │ 1 │     - Warp-level reductions
>  ├───┼───┼───┼───┤     - Strided memory patterns
> 2│T0 │T1 │T0 │T1 │     - SIMD-style operations
>  │ 2 │ 2 │ 3 │ 3 │
>  ├───┼───┼───┼───┤
> 3│T2 │T3 │T2 │T3 │
>  │ 2 │ 2 │ 3 │ 3 │
>  └───┴───┴───┴───┘
>
> CuTe code example:
> ──────────────────
> // Blocked: Each thread gets contiguous 2×2 tile
> auto thr_layout = make_layout(Shape<_2,_2>{}, Stride<_2,_1>{});
> auto val_layout = make_layout(Shape<_2,_2>{}, Stride<_1,_2>{});
> auto blocked = blocked_product(thr_layout, val_layout);
>
> // Raked: Threads interleaved with stride
> auto raked = raked_product(thr_layout, val_layout);
>
> important: Blocked for spatial locality, Raked for warp-level parallelism
> ```

### hierarchical tiling with local_tile

`local_tile` partitions tensors at the thread block level:

```cpp
// Partition global tensor for thread block
Tensor global_tensor = make_tensor(ptr, Shape<_M, _N>{});
Tensor block_tensor = local_tile(
    global_tensor,
    Shape<_128, _128>{},                    // Tile shape
    make_coord(blockIdx.x, blockIdx.y)      // Block coordinate
);
// Result: 128×128 tensor for this thread block
```

$$
\text{local\_tile}(T, S_{\text{tile}}, c) = T[\{c_i \cdot s_i : (c_i+1) \cdot s_i\}_{i=0}^{r-1}]
$$

Slices tensor $T$ at tile coordinate $c$ with tile shape $S_{\text{tile}}$.

### thread-level partitioning with local_partition

`local_partition` distributes tile across threads within a block:

```cpp
// Partition block tile for threads
Tensor thread_fragment = local_partition(
    block_tensor,
    thread_layout,        // e.g., (32, 4) = 128 threads
    thread_idx,           // Linear thread index within block
    Step<_1, _1, X>{}    // Which modes to partition
);
```

Step parameter semantics:

- `_1`: Partition this dimension (threads get different slices)
- `X`: Broadcast this dimension (threads see same values)

Example: GEMM thread mapping

```cpp
// Matrix A: Broadcast across columns, partition across rows
auto thread_A = local_partition(block_A, thr_layout, thr_idx, Step<_1, X>{});

// Matrix B: Broadcast across rows, partition across columns
auto thread_B = local_partition(block_B, thr_layout, thr_idx, Step<X, _1>{});

// Matrix C: Partition across both dimensions
auto thread_C = local_partition(block_C, thr_layout, thr_idx, Step<_1, _1>{});
```

This pattern enables efficient matrix multiplication where:

- Each thread reads unique portions of $A$ (rows)
- Each thread reads unique portions of $B$ (columns)
- Each thread writes unique portions of $C$ (output elements)

> [!example] CuTe hierarchical tiling [^cute]
>
> ```
> Global Tensor (M×N)
> ┌─────────────────────────────────────┐
> │  ┌──────────┐  ┌──────────┐         │
> │  │ Block 0  │  │ Block 1  │         │  ← local_tile
> │  │ (128×128)│  │ (128×128)│   ...   │    (CTA level)
> │  └──────────┘  └──────────┘         │
> │       ↓                             │
> │  ┌────────────────────────────┐     │
> │  │ ┌───┐ ┌───┐ ┌───┐ ┌───┐    │     │
> │  │ │T0 │ │T1 │ │T2 │ │T3 │    │     │  ← local_partition
> │  │ │4×8│ │4×8│ │4×8│ │4×8│... │     │    (Thread level)
> │  │ └───┘ └───┘ └───┘ └───┘    │     │
> │  │                            │     │
> │  │ Step<_1,_1>: Each thread   │     │
> │  │ gets different slice       │     │
> │  │                            │     │
> │  │ vs.                        │     │
> │  │                            │     │
> │  │ Step<_1,X>: Broadcast      │     │
> │  │ across dimension 1         │     │
> │  └────────────────────────────┘     │
> └─────────────────────────────────────┘
> ```

[^cute]: CuTe's hierarchical tiling cleanly separates concerns. local_tile partitions the global tensor across thread blocks (CTAs) using block coordinates (blockIdx). local_partition then distributes each block's tile across threads using a thread layout and Step parameter. Step<\_1,\_1> means partition both dimensions (each thread gets unique slice), Step<\_1,X> means partition dimension 0, broadcast dimension 1 (all threads see same values in dimension 1). This matches the GEMM pattern: matrix A broadcasts across columns, matrix B broadcasts across rows, matrix C is fully partitioned.

### swizzling and bank conflict avoidance

Swizzling permutes memory layout to avoid shared memory bank conflicts:

#### Bank conflict fundamentals

Shared memory on NVIDIA GPUs is organized into 32 banks of 4-byte words. Simultaneous accesses to the same bank by different threads in a warp serialize, reducing effective bandwidth by up to 32×.

> [!important] Bank conflict rules
>
> - No conflict: All threads access different banks → 1 transaction
> - Broadcast: All threads access same address → 1 transaction (hardware multicast)
> - N-way conflict: N threads access different addresses in same bank → N transactions (serialized)
> - Worst case: 32-way conflict when accessing column in row-major layout

Problem: Column access in row-major layout causes bank conflicts:

```cpp
__shared__ float smem[128][128];  // Row-major: smem[row][col]
float val = smem[threadIdx.x][col];  // All threads hit same bank
```

> [!note] Why this conflicts
> Address of `smem[i][col]` is `base + (i * 128 + col) * sizeof(float)`.
> Bank = `(address / 4) % 32 = (i * 128 + col) % 32`.
> When `col` is constant, all threads map to bank `col % 32` → 32-way conflict.

> [!example] Bank conflict visualization
>
> ```
> Shared memory organized into 32 banks (4-byte words)
>
> Without Swizzle - Column Access (32-way conflict):
> ═══════════════════════════════════════════════════
> Threads 0-31 all read column 0:
>
>   Thread  Row  Address        Bank                  ┌──────────────────────────┐
>    T0  →  0   base + 0×128    0                     │ Bank 0: T0,T1,T2...T31   │ ← 32 accesses!
>    T1  →  1   base + 1×128    0                     │        (serialized)      │
>    T2  →  2   base + 2×128    0                     │                          │
>   ...                                               │ Bank 1: [idle]           │
>   T31  → 31   base + 31×128   0                     │ Bank 2: [idle]           │
>                                                     │  ...                     │
> Result: 32 transactions (1 per thread, serialized)  │ Bank 31: [idle]          │
>                                                     └──────────────────────────┘
>
> With XOR Swizzle - Bank Spreading (conflict-free):
> ══════════════════════════════════════════════════
> swizzled_col = col ^ ((row >> shift) & mask)
>
>   Thread  Row  Swizzle          Bank  ┌─────────────────────────┐
>    T0  →  0   0 ^ (0>>5 & 7)=0    0   │ Bank 0:  T0             │ ← 1 access
>    T1  →  1   0 ^ (1>>5 & 7)=0    0   │ Bank 1:  [idle]         │
>   ...                                 │  ...                    │
>   T31  → 31   0 ^ (31>>5 & 7)=0   0   │ Bank 31: [idle]         │
>                                       └─────────────────────────┘
>
> Better example with row >> 3 (for 8 rows per bank rotation):
>
>   T0  → row 0:  col ^ (0>>3 & 7) = col ^ 0  → Bank col%32
>   T8  → row 8:  col ^ (8>>3 & 7) = col ^ 1  → Bank (col^1)%32
>   T16 → row 16: col ^ (16>>3 & 7) = col ^ 2 → Bank (col^2)%32
>   T24 → row 24: col ^ (24>>3 & 7) = col ^ 3 → Bank (col^3)%32
>
> When accessing same column across rows, XOR rotates bank assignment:
>
>   Row 0-7:   Banks 0,1,2,3,...31     (no rotation)
>   Row 8-15:  Banks 1,0,3,2,...30     (XOR with 1)
>   Row 16-23: Banks 2,3,0,1,...29     (XOR with 2)
>   Row 24-31: Banks 3,2,1,0,...28     (XOR with 3)
>
> Result: 1 transaction for warp (all threads access different banks)
> Speedup: 32× fewer transactions!
> ```

#### Swizzle function design

A swizzle function $S: (\text{row}, \text{col}) \to \text{address}$ permutes the linearized address to spread consecutive rows across different banks.

XOR-based swizzle:

$$S(\text{row}, \text{col}) = \text{row} \cdot W + (\text{col} \oplus f(\text{row}))$$

where $\oplus$ is bitwise XOR and $f$ is a shift/mask function.

Common pattern: $f(\text{row}) = (\text{row} \gg \text{shift}) \& \text{mask}$

Solution: Apply swizzle function to permute addresses:

```cpp
// XOR-based swizzle with shift=5, mask=7 (for 128-byte swizzle)
constexpr int SHIFT = 5;  // log2(32)
constexpr int MASK = 7;   // For 8-element rotation
int swizzled_col = col ^ ((row >> SHIFT) & MASK);
float val = smem[row][swizzled_col];  // Conflict-free
```

#### CuTe swizzle atoms

CuTe encodes swizzle patterns as layout transformations via `Swizzle<B, M, S>` atoms:

- B: Base-2 logarithm of swizzle width (e.g., 3 for 8-byte, 7 for 128-byte)
- M: Mask bits controlling XOR pattern
- S: Shift bits controlling row contribution

```cpp
// 128-byte swizzle for FP16 matrix (most common for tensor cores)
using Swizzle128 = Swizzle<3, 3, 3>;  // B=3 (8-byte), M=3 (8 elements), S=3

// Layout composition
using SmemLayoutRowMajor = Layout<Shape<_128, _128>, Stride<_128, _1>>;
using SmemLayoutSwizzled = decltype(composition(Swizzle128{}, SmemLayoutRowMajor{}));

__shared__ alignas(128) half_t smem[decltype(shape(SmemLayoutSwizzled{}))];
auto smem_tensor = make_tensor(make_smem_ptr(smem), SmemLayoutSwizzled{});
```

#### Swizzle modes by data width

Different swizzle patterns optimize for different element sizes and tensor core operand layouts:

> [!info] NVIDIA swizzle modes
>
> | Mode               | Swizzle Width | Element Size     | Typical Use Case        |
> | ------------------ | ------------- | ---------------- | ----------------------- |
> | `Swizzle<2, 0, 3>` | 32B           | 4B (FP32, INT32) | CUDA core matmul        |
> | `Swizzle<3, 2, 3>` | 64B           | 2B (FP16, BF16)  | Ampere tensor cores     |
> | `Swizzle<3, 3, 3>` | 128B          | 2B (FP16, BF16)  | Hopper WGMMA            |
> | `Swizzle<4, 3, 3>` | 128B          | 1B (FP8, INT8)   | Hopper FP8 tensor cores |

128-byte swizzle pattern (most important for Hopper):

```cpp
// Explicit swizzle for 128×128 FP16 tile
template <int ROW, int COL>
constexpr int swizzle_128B(int row, int col) {
    // Swizzle<3,3,3>: XOR across 16-byte groups
    int col_major = col >> 3;  // Divide by 8 (16 bytes / 2 bytes per FP16)
    int row_contrib = (row >> 3) & 0x7;  // Take bits [5:3] of row
    int swizzled_major = col_major ^ row_contrib;
    return (swizzled_major << 3) | (col & 0x7);
}
```

#### Swizzle layout algebra

CuTe's swizzle atoms compose with other layout transformations:

```cpp
// Step 1: Row-major layout
auto base_layout = make_layout(make_shape(_128, _128), make_stride(_128, _1));

// Step 2: Apply 128B swizzle
auto swizzled = composition(Swizzle<3,3,3>{}, base_layout);

// Step 3: Tile into 16×16 tensor core fragments
auto tiled = logical_divide(swizzled, make_tile(_16, _16));

// Step 4: Partition among threads
auto thread_layout = make_layout(make_shape(_4, _8), make_stride(_8, _1));
auto thread_data = local_partition(tiled, thread_layout, thread_idx);
```

The composed layout automatically generates swizzled addresses when indexed.

#### Performance verification

To verify conflict-free access, profile with `ncu`:

```bash
ncu --metrics l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    --launch-skip 10 --launch-count 1 ./matmul_kernel
```

> [!success] Measured impact
>
> Matrix transpose benchmark (128×128 FP16):
>
> | Configuration          | Bandwidth | Bank Conflicts/Warp | Speedup |
> | ---------------------- | --------- | ------------------- | ------- |
> | No swizzle             | 450 GB/s  | 31.2                | 1.0×    |
> | Padding (`[128][129]`) | 680 GB/s  | 0.0                 | 1.51×   |
> | 128B swizzle           | 710 GB/s  | 0.0                 | 1.58×   |
>
> Swizzling achieves better bandwidth than padding by maintaining cache line alignment and avoiding wasted shared memory.

> [!example] Shared memory bank conflicts and swizzling [^swizzling]
>
> ```
> Without Swizzling (32×32 array):
> ┌─────────────────────────────────┐
> │ Col0  Col1  Col2  ...  Col31    │
> ├─────────────────────────────────┤
> │ B0    B1    B2    ...  B31      │  ← Row 0
> │ B0    B1    B2    ...  B31      │  ← Row 1 (same banks!)
> │ B0    B1    B2    ...  B31      │  ← Row 2 (same banks!)
> │ ...                             │
> └─────────────────────────────────┘
>
> Column Access: All threads → Bank 0 = 32-way conflict!
>
> With Swizzling (XOR pattern):
> ┌─────────────────────────────────┐
> │ Col0  Col1  Col2  ...  Col31    │
> ├─────────────────────────────────┤
> │ B0    B1    B2    ...  B31      │  ← Row 0
> │ B1    B2    B3    ...  B0       │  ← Row 1 (shifted!)
> │ B2    B3    B4    ...  B1       │  ← Row 2 (shifted!)
> │ ...                             │
> └─────────────────────────────────┘
>
> Column Access: Threads → Different banks = Conflict-free!
>
> XOR Function: col_swizzled = col ^ (row >> shift)
> ```

[^swizzling]: Without swizzling, shared memory banks align with columns. When a warp reads down a column (threads 0-31 each reading row i, column 0), all threads hit bank 0—a 32-way conflict serializing the access. Swizzling applies an XOR permutation that rotates the bank assignment per row. Now threads reading column 0 hit banks 0,1,2,...,31 across different rows, eliminating conflicts. The XOR pattern is cheap (one instruction) and mathematically proven conflict-free for power-of-2 dimensions. CuTe's Swizzle<3,3,3> atoms encode this transformation in the layout type system, so the compiler generates swizzled addresses automatically.

Theorem (Swizzle Conflict-Free Access).
Let $S: \mathbb{N} \to \mathbb{N}$ be a swizzle function such that $S(i) \equiv i \pmod{32}$ has bijection within warp-stride. Then column access via $S$ is bank-conflict-free.

_Proof._ Bank assignment depends on address $\bmod 32$. Bijection ensures threads map to distinct banks. $\square$

## advanced optimizations

### flash attention

Problem: Standard attention $\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d})V$ requires materializing $QK^T$ in global memory:

- Memory: $O(N^2)$ for sequence length $N$
- Bandwidth bottleneck for long sequences

> idea: fuse attention computation and apply tiling + online softmax to keep intermediate results in shared memory.

> [!abstract] why fuse attention?
>
> - naïve attention materializes the $QK^T$ scores in HBM: $O(N^2)$ bytes dominate runtime at long sequence length $N$
> - flash attention tiles the $Q$, $K$, and $V$ matrices so intermediate tensors stay in shared memory, shrinking external traffic toward $O(N)$

> [!note] online softmax recap
> streaming the softmax keeps a running maximum $m$ and partial sum $\ell$ per row:
>
> $$
> m' = \max(m, m_{\text{block}}), \qquad \ell' = e^{m - m'} \ell + e^{m_{\text{block}} - m'} \ell_{\text{block}}
> $$
>
> and the output accumulator updates as
>
> $$
> O' = e^{m - m'} O + e^{m_{\text{block}} - m'} \tilde O_{\text{block}}
> $$
>
> the rescaling factors guarantee the streamed result matches the two-pass softmax. proof follows by induction on the number of processed tiles.

> [!math] theorem (Online Softmax Correctness).
>
> The streaming softmax algorithm produces identical results to the standard two-pass algorithm.

_Proof._ By induction on the number of blocks. Base case (one block) is trivial. For inductive step, show that the correction factors exactly compensate for the difference between local and global normalization constants. The key observation is that $e^{m^{(old)} - m^{(updated)}}$ rescales old outputs to account for the new global maximum. $\square$

### flash attention 4: warp specialization

> [!table] throughput evolution (h100 hopper)
>
> | release           | core idea                                                | measured tflop/s |
> | ----------------- | -------------------------------------------------------- | ---------------- |
> | flash attention 1 | tiled attention + online softmax                         | ≈300             |
> | flash attention 2 | improved parallelization + block design                  | ≈500             |
> | flash attention 3 | fp8 friendly tiles, hopper-specific tuning               | ≈650             |
> | flash attention 4 | warp specialization, persistent kernels, async pipelines | 764              |

> [!info] fa4 upgrades
>
> - warp-specialized thread blocks assign dedicated warps to loading, mma, softmax, correction, and epilogue paths so each pipeline stays saturated
> - persistent kernels process work queues in-place to amortize launch latency and keep l2 warm between tiles
> - tensor memory accelerator (tma) and wgmma issue overlapping pipelines that stream q/k/v tiles directly into shared memory, minimizing register pressure
> - polynomial exp approximations cut sfu usage by an order of magnitude with dynamic fallbacks for extreme logits

#### warp specialization

Assign different roles to different warps within a thread block:

```
Thread Block (128-384 threads)
├─ Load Warp (1 warp)
│  └─ Dedicated to TMA memory operations
├─ MMA Warps (4 warps)
│  └─ Matrix multiply-accumulate using wgmma
├─ Softmax Warps (8 warps)
│  └─ Parallel softmax computation
├─ Correction Warps (4 warps)
│  └─ Numerical stability corrections
└─ Epilogue Warps (1-2 warps)
    └─ Final output processing and stores
```

Benefits:

- Each warp optimized for specific operation
- Enables asynchronous execution (load while compute)
- Improves instruction-level parallelism
- Reduces warp divergence

#### persistent kernels

Standard kernel launch:

```cuda
for (int i = 0; i < num_batches; ++i) {
    matmul_kernel<<<grid, block>>>(A[i], B[i], C[i]);
}
```

Problems:

- Kernel launch overhead (~5-10 μs per launch)
- Cold caches on each launch
- No data reuse across batches

Persistent kernel:

```cuda
__global__ void persistent_matmul(WorkQueue* queue) {
    while (true) {
        WorkItem item = queue->get_next();
        if (item.done) break;

        // Process this work item
        compute_tile(item.A, item.B, item.C);
    }
}

// Single launch processes all batches
persistent_matmul<<<grid, block>>>(queue);
```

Benefits:

- Amortize kernel launch overhead
- Keep L2 cache warm
- Enable software pipelining across work items
- Performance: 317 TFLOP/s → 660 TFLOP/s (persistent kernels)

#### asynchronous pipelines

Goal: Overlap memory movement with computation using TMA and wgmma asynchronous instructions.

Pipeline stages:

```
Stage:     | 0 | 1 | 2 | 3 | 4 | 5 | 6 | ...
─────────────────────────────────────────────────
TMA Load:  | L0| L1| L2| L3| L4| L5| L6|
Compute:   |   | C0| C1| C2| C3| C4| C5|
Store:     |   |   | S0| S1| S2| S3| S4|
```

Implementation pattern:

```cuda
// Producer: Load warp
for (int stage = 0; stage < num_stages; ++stage) {
    cute::copy(tma_load, gmem_tile[stage], smem_tile[stage]);
    pipeline_commit();
}

// Consumer: MMA warps
for (int stage = 0; stage < num_stages; ++stage) {
    pipeline_wait(stage);  // Wait for load completion

    wgmma::mma_async(acc, smem_A[stage], smem_B[stage]);
    wgmma::commit_group();

    pipeline_release(stage);  // Signal load warp
}

wgmma::wait_group<0>();  // Wait for all MMA completion
```

Synchronization primitives:

- `pipeline_commit()`: Signal load completion
- `pipeline_wait(stage)`: Wait for specific stage
- `wgmma::commit_group()`: Commit warpgroup MMA operations
- `wgmma::wait_group<N>()`: Wait for N most recent wgmma operations

Performance impact: 660 TFLOP/s → 764 TFLOP/s (asynchronous pipelines)

> [!example] Flash Attention pipeline stages [^fa]
>
> ```
> time →
> stage:  0    1    2    3    4    5
> ──────────────────────────────────
> load   [Q]  [K0] [K1] [K2] [K3] ...
> compute     [QK0][QK1][QK2][QK3] ...
> softmax          [S0] [S1] [S2] ...
> attn·V               [AV0][AV1] ...
> write                      [O0] [O1]
>
> Memory Usage:
> ┌─────────────────────────────┐
> │ Shared Memory (per block)   │
> │ ┌─────┐ ┌─────┐ ┌─────┐     │
> │ │  Q  │ │  K  │ │  V  │     │
> │ │ tile│ │ tile│ │ tile│     │
> │ └─────┘ └─────┘ └─────┘     │
> │                             │
> │ Registers (per thread)      │
> │ ┌───┐ ┌───┐ ┌───┐           │
> │ │max│ │sum│ │out│           │
> │ └───┘ └───┘ └───┘           │
> └─────────────────────────────┘
> shared memory holds q/k/v tiles; registers retain {max, sum, out} accumulators per row.
> ```

[^fa]: flash attention keeps the $QK^T$ tiles in shared memory and streams softmax statistics so load, compute, softmax, and value accumulation stages overlap. the modal reverse-engineering traces show the asynchronous pipeline reaching 764 tflop/s on h100, a 2.5× improvement over naïve attention.

#### Cubic polynomial exponential approximation

Standard: $e^x$ requires expensive transcendental operation

Approximation: $e^x \approx a_3 x^3 + a_2 x^2 + a_1 x + a_0$

Coefficients chosen to minimize error over typical softmax input range:

- $a_3 \approx 0.0139$
- $a_2 \approx 0.0878$
- $a_1 \approx 0.5 $
- $a_0 \approx 1.0$

Dynamic switching: Use exact $e^x$ for extreme values (very large/small), approximation for typical range.

Performance: 10× fewer exponential operations, ~15% speedup overall.

## paged attention, CuTe implementation

### motivation

Problem: LLM inference with KV cache:

- Attention requires access to key/value cache from all previous tokens
- Cache grows linearly with sequence length: $O(N \cdot d \cdot L)$
- Monolithic allocation wastes memory (padding, fragmentation)

[@kwon2023efficient]

- Store KV cache in fixed-size pages (e.g., 16 tokens per page)
- Pages allocated non-contiguously (like virtual memory)
- Attention kernel reads from paged storage

Benefits:

- Reduced memory fragmentation
- Dynamic allocation (grow cache as needed)
- Efficient batching (share pages across requests)

> [!example] Paged attention memory layout (virtual memory analogy) [^paged-attention]
>
> ```
> ┌─────────────────────────────────────────────────────────────────────┐
> │ VIRTUAL ADDRESS SPACE (Logical KV Cache per Sequence)               │
> └─────────────────────────────────────────────────────────────────────┘
>
> Sequence 0 (48 tokens, 3 pages):        Sequence 1 (32 tokens, 2 pages):
> ┌──────────────────────────────┐       ┌──────────────────────────────┐
> │ Virtual Page 0  [tok 0..15]  │       │ Virtual Page 0  [tok 0..15]  │
> │ Virtual Page 1  [tok 16..31] │       │ Virtual Page 1  [tok 16..31] │
> │ Virtual Page 2  [tok 32..47] │       └──────────────────────────────┘
> └──────────────────────────────┘               │          │
>        │          │          │                 │          │
>        │          │          └─────────┐       │          │
>        │          └──────────┐         │       │          └────────┐
>        │                     │         │       │                   │
>        ▼                     ▼         ▼       ▼                   ▼
> ┌─────────────────────────────────────────────────────────────────────┐
> │ PHYSICAL MEMORY (GPU Global Memory - Non-contiguous Pages)          │
> └─────────────────────────────────────────────────────────────────────┘
>
> ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
> │ P0  │ P1  │ P2  │ P3  │ P4  │ P5  │ P6  │ P7  │ P8  │ P9  │ P10 │ P11 │
> │Seq0 │     │     │     │     │Seq1 │     │Seq0 │     │Seq1 │     │     │
> │16tok│     │     │     │     │16tok│     │16tok│     │16tok│     │     │
> │tok  │     │     │     │     │tok  │     │tok  │     │tok  │     │     │
> │0-15 │     │     │     │     │0-15 │     │16-31│     │16-31│     │     │
> └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
>   ▲                               ▲           ▲           ▲
>   │                               │           │           │
>   Seq0,VP0                     Seq1,VP0    Seq0,VP1   Seq1,VP1
>
> ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬───────┐
> │ P12 │ P13 │ P14 │ P15 │ ...                                             │
> │Seq0 │     │     │     │                                                 │
> │16tok│     │     │     │                                                 │
> │tok  │     │     │     │                                                 │
> │32-47│     │     │     │                                                 │
> └─────┴─────┴─────┴─────┴─────────────────────────────────────────────────┘
>   ▲
>   │
>   Seq0,VP2
>
> ┌─────────────────────────────────────────────────────────────────────┐
> │ BLOCK TABLE (Virtual Page → Physical Page Translation)              │
> └─────────────────────────────────────────────────────────────────────┘
>
>        Virtual Page Index:     0    1    2    3    ...
>                               ┌────┬────┬────┬────┬────┐
> Sequence 0 (len=48) →         │ 0  │ 7  │ 12 │ -1 │ -1 │
>                               └────┴────┴────┴────┴────┘
>                                 ↓    ↓    ↓
>                           Phys  P0   P7   P12  (non-contiguous!)
>
>                               ┌────┬────┬────┬────┬────┐
> Sequence 1 (len=32) →         │ 5  │ 9  │ -1 │ -1 │ -1 │
>                               └────┴────┴────┴────┴────┘
>                                 ↓    ↓
>                           Phys  P5   P9
>
>
> ─────────────
> 1. Virtual address space: Each sequence sees contiguous logical pages [0,1,2,...]
> 2. Physical pages: Scattered across GPU memory (like OS virtual memory)
> 3. Block table: Translation lookaside buffer (TLB) for page mapping
> 4. Attention kernel: Translates virtual page → physical page per access
> 5. Benefits: Zero fragmentation, dynamic growth, page sharing for prefixes
>
> Memory Access Pattern in Kernel:
> ────────────────────────────────
> for page_idx in range(num_pages_for_seq):
>     physical_page = block_table[seq_id, page_idx]  # Virtual → Physical
>     keys = key_cache[physical_page]                # Load from physical memory
>     scores = query @ keys.T
>     ...
> ```

[^paged-attention]: Instead of allocating contiguous memory for each sequence's KV cache, paged attention uses fixed-size pages (e.g., 16 tokens) that can be scattered in memory. A block table maps logical sequence positions to physical pages. Seq 0 tokens 0-15 live in page 0, tokens 16-31 in page 7 (non-contiguous!), tokens 32-47 in page 12. This eliminates fragmentation from varying sequence lengths, enables dynamic growth without reallocation, and allows page sharing across sequences for prefix caching (multiple prompts sharing the same prefix reuse the same physical pages).

### paged attention algorithm

Given:

- Query: $Q \in \mathbb{R}^{d}$ (current token)
- Key cache: $\{K_0, K_1, \ldots, K_{N-1}\}$ stored in pages
- Value cache: $\{V_0, V_1, \ldots, V_{N-1}\}$ stored in pages
- Block table: Maps logical page index to physical page address

Compute: $O = \text{softmax}(Q K^T / \sqrt{d}) V$.

Algorithm:

```
1. Initialize: max_score = -inf, sum = 0, output = 0
2. For each page p:
   a. Load page p keys into shared memory
   b. Compute scores: s_i = Q · K_i / sqrt(d)
   c. Update online softmax statistics (max, sum)
   d. Load page p values into shared memory
   e. Accumulate output with correction
3. Final rescale: output = output / sum
```

### CuTe implementation

```cpp
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

template <int HEAD_DIM, int PAGE_SIZE, int BLOCK_SIZE>
__global__ void paged_attention_kernel(
    float* __restrict__ out,                    // [num_seqs, num_heads, head_dim]
    const float* __restrict__ query,            // [num_seqs, num_heads, head_dim]
    const float* __restrict__ key_cache,        // [num_pages, num_heads, page_size, head_dim]
    const float* __restrict__ value_cache,      // [num_pages, num_heads, page_size, head_dim]
    const int* __restrict__ block_tables,       // [num_seqs, max_num_pages]
    const int* __restrict__ context_lens,       // [num_seqs]
    int num_seqs,
    int num_heads,
    int max_num_pages
) {
    // Thread block processes one query (sequence, head)
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (seq_idx >= num_seqs || head_idx >= num_heads) return;

    // Shared memory for keys and values
    __shared__ float smem_keys[PAGE_SIZE][HEAD_DIM];
    __shared__ float smem_values[PAGE_SIZE][HEAD_DIM];

    // Thread layout: BLOCK_SIZE threads process HEAD_DIM elements
    int tid = threadIdx.x;

    // Load query into registers using CUTe
    auto query_ptr = query + seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    auto query_layout = make_layout(Shape<Int<HEAD_DIM>>{}, Stride<_1>{});
    auto query_tensor = make_tensor(make_gmem_ptr(query_ptr), query_layout);

    // Create register tensor for query
    auto q_reg = make_tensor<float>(Shape<Int<HEAD_DIM>>{});

    // Partition query for this thread
    auto q_thread = local_partition(query_tensor,
                                     Layout<Shape<Int<BLOCK_SIZE>>>{},
                                     tid,
                                     Step<_1>{});
    auto q_reg_thread = local_partition(q_reg,
                                         Layout<Shape<Int<BLOCK_SIZE>>>{},
                                         tid,
                                         Step<_1>{});

    // Copy query to registers
    copy(q_thread, q_reg_thread);

    // Online softmax state
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Output accumulator
    auto out_reg = make_tensor<float>(Shape<Int<HEAD_DIM>>{});
    clear(out_reg);

    auto out_reg_thread = local_partition(out_reg,
                                           Layout<Shape<Int<BLOCK_SIZE>>>{},
                                           tid,
                                           Step<_1>{});

    // Get context length for this sequence
    int context_len = context_lens[seq_idx];
    int num_pages = (context_len + PAGE_SIZE - 1) / PAGE_SIZE;

    // Iterate over pages
    for (int page_idx = 0; page_idx < num_pages; ++page_idx) {
        // Get physical page number from block table
        int physical_page = block_tables[seq_idx * max_num_pages + page_idx];

        // Calculate number of valid tokens in this page
        int tokens_in_page = min(PAGE_SIZE, context_len - page_idx * PAGE_SIZE);

        // Load keys from global to shared memory using CUTe
        auto key_page_ptr = key_cache +
                           physical_page * num_heads * PAGE_SIZE * HEAD_DIM +
                           head_idx * PAGE_SIZE * HEAD_DIM;
        auto key_page_layout = make_layout(
            Shape<Int<PAGE_SIZE>, Int<HEAD_DIM>>{},
            Stride<Int<HEAD_DIM>, _1>{}
        );
        auto key_page_tensor = make_tensor(make_gmem_ptr(key_page_ptr), key_page_layout);

        // Shared memory tensor
        auto smem_key_layout = make_layout(
            Shape<Int<PAGE_SIZE>, Int<HEAD_DIM>>{},
            Stride<Int<HEAD_DIM>, _1>{}
        );
        auto smem_key_tensor = make_tensor(make_smem_ptr(&smem_keys[0][0]), smem_key_layout);

        // Cooperatively load keys
        auto key_copy_thr = local_partition(
            key_page_tensor,
            Layout<Shape<Int<BLOCK_SIZE>>>{},
            tid,
            Step<_1, _1>{}
        );
        auto smem_key_thr = local_partition(
            smem_key_tensor,
            Layout<Shape<Int<BLOCK_SIZE>>>{},
            tid,
            Step<_1, _1>{}
        );
        copy(key_copy_thr, smem_key_thr);
        __syncthreads();

        // Compute attention scores for this page
        float scores[PAGE_SIZE];
        for (int i = 0; i < tokens_in_page; ++i) {
            float score = 0.0f;
            // Dot product: Q · K_i
            for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
                score += q_reg(d) * smem_keys[i][d];
            }
            // Warp-level reduction
            for (int offset = 16; offset > 0; offset /= 2) {
                score += __shfl_down_sync(0xffffffff, score, offset);
            }
            // Broadcast score to all threads
            score = __shfl_sync(0xffffffff, score, 0);
            score /= sqrtf((float)HEAD_DIM);
            scores[i] = score;
        }

        // Update online softmax statistics
        float page_max = -INFINITY;
        for (int i = 0; i < tokens_in_page; ++i) {
            page_max = fmaxf(page_max, scores[i]);
        }

        float old_max = max_score;
        max_score = fmaxf(max_score, page_max);

        // Correction factor for previous output
        float correction = expf(old_max - max_score);

        // Scale previous output and sum
        for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
            out_reg(d) *= correction;
        }
        sum_exp *= correction;

        // Load values from global to shared memory
        auto val_page_ptr = value_cache +
                           physical_page * num_heads * PAGE_SIZE * HEAD_DIM +
                           head_idx * PAGE_SIZE * HEAD_DIM;
        auto val_page_tensor = make_tensor(make_gmem_ptr(val_page_ptr), key_page_layout);
        auto smem_val_tensor = make_tensor(make_smem_ptr(&smem_values[0][0]), smem_key_layout);

        auto val_copy_thr = local_partition(val_page_tensor,
                                             Layout<Shape<Int<BLOCK_SIZE>>>{},
                                             tid,
                                             Step<_1, _1>{});
        auto smem_val_thr = local_partition(smem_val_tensor,
                                             Layout<Shape<Int<BLOCK_SIZE>>>{},
                                             tid,
                                             Step<_1, _1>{});
        copy(val_copy_thr, smem_val_thr);
        __syncthreads();

        // Accumulate output: O += exp(scores) * Values
        for (int i = 0; i < tokens_in_page; ++i) {
            float weight = expf(scores[i] - max_score);
            sum_exp += weight;

            for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
                out_reg(d) += weight * smem_values[i][d];
            }
        }
        __syncthreads();
    }

    // Final rescale
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        out_reg(d) /= sum_exp;
    }

    // Write output to global memory
    auto out_ptr = out + seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    auto out_layout = make_layout(Shape<Int<HEAD_DIM>>{}, Stride<_1>{});
    auto out_tensor = make_tensor(make_gmem_ptr(out_ptr), out_layout);

    auto out_global_thr = local_partition(out_tensor,
                                           Layout<Shape<Int<BLOCK_SIZE>>>{},
                                           tid,
                                           Step<_1>{});

    copy(out_reg_thread, out_global_thr);
}
```

> [!note] Key CuTe concepts exercised
>
> 1. Layout algebra: define multi-dimensional access patterns.
> 2. Tensors: couple pointers and layouts for type-safe copies.
> 3. `local_partition`: distribute work across threads.
> 4. `copy`: issue coalesced transfers via policy specializations.

### performance analysis

Memory access pattern:

- Each page loaded once into shared memory
- Reused across all threads computing attention
- Coalesced loads using CuTe partitioning
- TMA could be used for further optimization

Arithmetic intensity:
For $P$ pages with $T$ tokens per page, head dimension $d$:

- Compute: $O(PTd)$ FLOPs (dot products and exponentials)
- Memory: $O(PT d)$ bytes (load keys and values once)
- Arithmetic intensity: $O(1)$ FLOPs/byte

Optimization opportunities:

1. Use wgmma for Q·K^T computation (requires layout transformation).
2. TMA for asynchronous page loading.
3. Persistent kernel for batched sequences following FlashAttention 4 scheduling.
4. FP8/FP16/FP4 quantization with block scaling via NVFP4 tensor core paths.

## performance profiling and optimization tools

### NVIDIA profiling tools

#### nsys (Nsight Systems)

Purpose: System-wide performance profiling

Usage:

```bash
nsys profile --trace=cuda,nvtx -o profile ./matmul_benchmark
```

Capabilities:

- Timeline view of GPU/CPU activity
- Kernel launch overhead analysis
- Memory copy overhead
- CUDA API calls
- NVTX markers for custom regions

Key metrics:

- Kernel duration
- SM utilization
- Memory bandwidth utilization
- CPU-GPU synchronization overhead

#### ncu (Nsight Compute)

Purpose: Detailed kernel-level profiling

Usage:

```bash
ncu --set full -o kernel_profile ./matmul_benchmark
```

Capabilities:

- Instruction-level analysis
- Memory access patterns (coalescing, cache hit rates)
- Compute throughput (FLOP/s, tensor core utilization)
- Occupancy analysis
- Roofline model visualization

Key metrics:

- SM Efficiency: Active cycles / total cycles
- Achieved Occupancy: Active warps / max warps
- Memory Throughput: GB/s for global/shared/L2
- Compute Throughput: TFLOP/s, tensor core utilization
- Bank Conflicts: Shared memory bank conflict rate
- Branch Divergence: Warp divergence percentage

### optimization workflow

Step-by-step process:

1. Profile with nsys: Identify bottleneck kernels

   ```bash
   nsys profile ./benchmark
   # Look for longest-running kernels, synchronization overhead
   ```

2. Detailed analysis with ncu:

   ```bash
   ncu --set full --kernel-name matmul_kernel ./benchmark
   # Focus on:
   # - Memory bandwidth vs compute throughput
   # - Occupancy limiter (registers, shared memory, blocks)
   # - Cache hit rates
   ```

3. Identify bottleneck type:
   - Memory-bound: Low compute throughput, high memory utilization
     - Solution: Increase tiling, use shared memory, vectorize loads
   - Compute-bound: High compute throughput, low memory utilization
     - Solution: Use tensor cores, mixed precision, reduce redundant compute
   - Latency-bound: Low occupancy, many stalls
     - Solution: Increase parallelism, reduce register usage, overlap with ILP

4. Apply targeted optimizations:
   - Memory-bound → Tiling, coalescing, shared memory
   - Compute-bound → Tensor cores, mixed precision, kernel fusion
   - Latency-bound → Occupancy tuning, asynchronous operations

5. Measure and iterate:
   ```bash
   ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
                 dram__throughput.avg.pct_of_peak_sustained_elapsed,\
                 l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
                 l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
       ./benchmark
   ```

### roofline model

Roofline model visualizes performance limits based on arithmetic intensity:

> [!example] Roofline model for H100 [^roofline]
>
> ```
> Performance (TFLOP/s)
> │
> │ 1979 ──────────────────────────────  ← Tensor core peak (FP16/BF16)
> │        │                     ↑
> │        │               Near-perfect GEMM (I ≳ 1000)
> │        │                     │
> │  214   │      ╱──────────────┘  ← FP16 tiled (I ≈ 64)
> │        │     ╱
> │        │    ╱
> │        │   ╱
> │ 1.68   │  ╱    ← Naive (I ≈ 0.5)
> │        │ ╱
> │────────┴────────────────────────────→
> 0      0.5      64      600     1000+  Arithmetic Intensity (FLOP/byte)
>                   ↑ Ridge point (≈ 592)
> ```

[^roofline]:
    The ridge point sits at $I = P_{\text{peak}} / B \approx 1979 / 3.35 \approx 592$ FLOP/byte for H100.

    - Kernels below that intensity are memory-bound with performance capped by $I \times 3.35\,\text{TB/s}$.
    - The naive FP16 GEMM (I ≈ 0.5) therefore tops out at 1.68 TFLOP/s.
    - A 128×128×32 FP16 tiled kernel raises intensity to ≈64, pushing the roofline bound to ~214 TFLOP/s.
    - Only when the program approaches perfect data reuse—each matrix element fetched once, giving $I \gtrsim 10^3$—does the kernel enter the compute-bound regime and approach the 1,979 TFLOP/s tensor-core ceiling.

For a given arithmetic intensity $I$ (FLOP/byte):

- If $I \times B < P_{\text{peak}}$: Memory-bound (performance = $I \times B$)
- If $I \times B \ge P_{\text{peak}}$: Compute-bound (performance = $P_{\text{peak}}$)

Where $B$ is memory bandwidth and $P_{\text{peak}}$ is peak compute throughput.

Example for matmul on H100:

- Memory bandwidth: 3.35 TB/s
- Peak FP16 tensor core: 1979 TFLOP/s
- Ridge point: $I_{\text{ridge}} = 1979 / 3350 \approx 0.59$ FLOP/byte

For $I < 0.59$: memory-bound, for $I > 0.59$: compute-bound.

Tiled matmul with blocking achieves $I \approx 64$ FLOP/byte → compute-bound.

### occupancy tuning

Occupancy: Ratio of active warps to maximum possible warps per SM.

Limits:

- Registers per thread
- Shared memory per block
- Thread blocks per SM
- Threads per block

CUDA occupancy calculator:

```bash
cuda-occupancy-calculator \
    --threads-per-block 256 \
    --registers-per-thread 64 \
    --shared-memory-per-block 49152 \
    --compute-capability 90  # Hopper
```

Trade-offs:

- High occupancy: More parallelism, better latency hiding
- Low occupancy: More resources per thread, potential for higher ILP

Optimal: Typically 50-75% occupancy sufficient if instructions hide latency.

## summary and key takeaways

### hardware architecture insights

1. Memory hierarchy dominates: Most GPU optimizations focus on data movement, not compute
2. SIMT execution: Warp-level thinking essential (coalescing, divergence, synchronization)
3. Tensor cores: 10-30× speedup for matmul with mixed precision
4. Specialization: Hopper/Blackwell add domain-specific hardware (TMA, wgmma, FP8/FP4)

### software abstraction levels

```
┌──────────────────────────────────────────┐
│ High-level: PyTorch, JAX, TensorFlow     │
│  - Automatic differentiation             │
│  - Kernel fusion (TorchInductor, XLA)    │
├──────────────────────────────────────────┤
│ Mid-level: CUTLASS, CuTe DSL             │
│  - Layout algebra abstractions           │
│  - Hierarchical tiling                   │
│  - Performance portability               │
├──────────────────────────────────────────┤
│ Low-level: CUDA C++, PTX                 │
│  - Explicit control                      │
│  - Hand-tuned kernels                    │
│  - Inline assembly for wgmma, TMA        │
└──────────────────────────────────────────┘
```

When to use each:

- High-level: Rapid prototyping, standard operations
- Mid-level (CUTLASS/CuTe): Custom operations, library development
- Low-level: Bleeding-edge features, final 10-20% optimization

### CuTe DSL mental model

Think of CuTe as a type system for memory:

- Layouts describe how data is organized
- Operations (composition, division, product) transform layouts
- Compiler optimizes layout operations at compile-time
- Runtime performs only data movement and compute

Key concepts:

1. Separate logical (shape) from physical (stride)
2. Hierarchical tiling (grid → block → warp → thread)
3. Composable abstractions (layouts, tensors, partitions)

### CuTe DSL Python examples

NVIDIA's CuTe DSL is available as a Python library in the CUTLASS package, allowing rapid prototyping and exploration of CuTe concepts before implementing in C++. The Python API mirrors the C++ API closely, making it an excellent learning tool.

#### setup and basics

Install CUTLASS with Python support:

```bash
pip install nvidia-cutlass
```

Basic imports and tensor creation:

```python
import cutlass
import cutlass.cute as cute


@cute.jit
def create_tensor_from_ptr(ptr: cute.Pointer):
  # Create layout: 8×5 column-major
  layout = cute.make_layout((8, 5), stride=(1, 8))
  tensor = cute.make_tensor(ptr, layout)
  return tensor
```

A tensor in CuTe is mathematically defined as: `T(c) = *(E + L(c))` where:

- `E` is the engine (pointer-like object for memory access)
- `L` is the layout (coordinate-to-offset mapping)
- `c` is a coordinate in the logical space

#### data types

CuTe provides GPU-specific numeric types for runtime dynamic values:

```python
# Integer types: Int8, Int16, Int32, Int64
x = cutlass.Int32(42)

# Floating point types: Float16, Float32, Float64, BFloat16, TFloat32
y = cutlass.Float32(3.14159)

# 8-bit float formats for Hopper/Ada
fp8_e4m3 = cutlass.Float8E4M3(1.5)  # E4M3: 4 exponent bits, 3 mantissa bits
fp8_e5m2 = cutlass.Float8E5M2(2.0)  # E5M2: 5 exponent bits, 2 mantissa bits


# Type conversion and arithmetic
@cute.jit
def type_operations():
  a = cutlass.Int32(10)
  b = cutlass.Int32(3)
  x = cutlass.Float32(5.5)

  sum_int = a + b  # Int32
  mixed = a + x  # Promotes to Float32

  return sum_int, mixed
```

These types are essential for mixed-precision computations (FP8/FP16 input, FP32 accumulation) on modern GPUs.

#### layout algebra operations

##### coalesce

The `coalesce` operation simplifies a layout by flattening modes while preserving the coordinate-to-offset mapping:

```python
@cute.jit
def coalesce_example():
  # Hierarchical layout: (2, (1, 6)) with complex strides
  layout = cute.make_layout((2, (1, 6)), stride=(1, (cutlass.Int32(6), 2)))
  result = cute.coalesce(layout)

  print('Original:', layout)  # ((2,(1,6)), (1,(6,2)))
  print('Coalesced:', result)  # (12, 1) - simplified to rank-1

  # Preserves mapping: layout(i, j, k) == result(linearized_index)
  assert layout.size() == result.size()  # Both have 12 elements
```

Use case: Simplifying layout expressions before performing partitioning or composition.

##### composition

Layout composition creates a new access pattern by chaining layouts: `R(c) = A(B(c))`

```python
@cute.jit
def composition_example():
  # A: 6×2 layout with stride (8, 2)
  A = cute.make_layout((6, 2), stride=(cutlass.Int32(8), 2))

  # B: 4×3 layout with stride (3, 1)
  B = cute.make_layout((4, 3), stride=(3, 1))

  # R = A ∘ B: B's coordinates index into A's domain
  R = cute.composition(A, B)

  print('Layout A:', A)
  print('Layout B:', B)
  print('Composition R = A ∘ B:', R)

  # R selects elements from A using B's pattern
  # Example: R(1, 2) = A(B(1, 2)) = A(1*3 + 2*1) = A(5)
```

Use case: Implementing tiling or selecting subsets of a tensor.

##### logical divide (tiling)

Partition a layout into tiles of a specified shape:

```python
@cute.jit
def logical_divide_example():
  # 64×64 column-major layout
  layout = cute.make_layout((64, 64), stride=(1, 64))

  # Partition into 8×8 tiles
  tiled = cute.logical_divide(layout, (8, 8))

  print('Original shape:', layout.shape)  # (64, 64)
  print('Tiled shape:', tiled.shape)  # ((8, 8), (8, 8))
  # Outer tuple: tile dimensions
  # Inner tuple: number of tiles per dimension

  # Access tile (2, 3): tiled[2, 3, :, :]
  # Access element (5, 7) within tile (2, 3): tiled[2, 3, 5, 7]
```

This operation is fundamental to CuTe's hierarchical tiling model: global → block → thread.

#### tensor operations

##### tensor creation from DLPack

Interoperate with PyTorch, NumPy, JAX:

```python
import torch
from cutlass.cute.runtime import from_dlpack


def torch_to_cute():
  # PyTorch tensor
  a = torch.randn(8, 5, dtype=torch.float32).cuda()

  # Convert to CuTe tensor
  @cute.jit
  def process_tensor(src: cute.Tensor):
    cute.print_tensor(src)
    # Perform CuTe operations...

  process_tensor(from_dlpack(a))
```

##### coordinate tensors

Coordinate tensors map coordinates to themselves, useful for understanding layout transformations:

```python
@cute.jit
def coordinate_tensor_example():
  layout = cute.make_layout((4, 3), stride=(1, 4))
  coord_tensor = cute.make_identity_tensor(layout.shape)

  # Prints:
  # [[(0,0) (0,1) (0,2)]
  #  [(1,0) (1,1) (1,2)]
  #  [(2,0) (2,1) (2,2)]
  #  [(3,0) (3,1) (3,2)]]
  cute.print_tensor(coord_tensor)

  # After applying a layout transformation:
  transformed = cute.composition(layout, coord_tensor)
  # Shows how coordinates map through the layout
```

##### tensor partitioning

Partition tensors across threads (Python equivalent of `local_partition`):

```python
@cute.jit
def partition_example():
  # Global tensor: 128×128
  global_layout = cute.make_layout((128, 128), stride=(1, 128))
  global_tensor = cute.make_tensor(ptr, global_layout)

  # Thread layout: 32×4 = 128 threads
  thread_layout = cute.make_layout((32, 4))

  # Partition for thread 0
  thread_tensor = cute.local_partition(global_tensor, thread_layout, 0)

  print('Global shape:', global_tensor.shape)  # (128, 128)
  print('Thread shape:', thread_tensor.shape)  # (4, 32) per thread
```

#### elementwise addition kernel

Complete example demonstrating CuTe kernel authoring with proper launch pattern:

```python
# Step 1: Define device kernel
@cute.kernel
def elementwise_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
  # Thread indexing
  tidx, _, _ = cute.arch.thread_idx()
  bidx, _, _ = cute.arch.block_idx()
  bdim, _, _ = cute.arch.block_dim()

  thread_idx = bidx * bdim + tidx

  # Compute global coordinate
  m, n = gA.shape
  ni = thread_idx % n
  mi = thread_idx // n

  # Bounds check
  if mi < m and ni < n:
    # Load, compute, store
    a_val = gA[mi, ni]
    b_val = gB[mi, ni]
    gC[mi, ni] = a_val + b_val


# Step 2: Define host wrapper
@cute.jit
def elementwise_add_launch(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
  num_threads_per_block = 256
  m, n = mA.shape
  num_blocks = (m * n + num_threads_per_block - 1) // num_threads_per_block

  elementwise_add_kernel(mA, mB, mC).launch(
    grid=(num_blocks, 1, 1), block=(num_threads_per_block, 1, 1)
  )


# Step 3: Prepare and convert tensors
M, N = 1024, 1024
a = torch.randn(M, N, device='cuda', dtype=torch.float16)
b = torch.randn(M, N, device='cuda', dtype=torch.float16)
c = torch.zeros(M, N, device='cuda', dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

# Step 4: Compile
compiled_add = cute.compile(elementwise_add_launch, a_, b_, c_)

# Step 5: Execute
compiled_add(a_, b_, c_)

# Verify correctness
torch.testing.assert_close(c, a + b)
```

Vectorized version using CuTe's tiling and layout algebra:

```python
@cute.kernel
def vectorized_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
  tidx, _, _ = cute.arch.thread_idx()
  bidx, _, _ = cute.arch.block_idx()

  # Slice for thread-block level view
  blk_coord = ((None, None), bidx)
  tA = gA[blk_coord]
  tB = gB[blk_coord]
  tC = gC[blk_coord]

  # Slice for thread level view (each thread gets a fragment)
  thr_A = tA[(None, tidx)]
  thr_B = tB[(None, tidx)]
  thr_C = tC[(None, tidx)]

  # Vectorized operations on fragments
  for i in range(cute.size(thr_A)):
    thr_C[i] = thr_A[i] + thr_B[i]


@cute.jit
def vectorized_add_launch(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
  # Create thread and value layouts
  thr_layout = cute.make_layout((4, 32), stride=(32, 1))  # 128 threads
  val_layout = cute.make_layout((4, 8), stride=(8, 1))  # 32 values/thread

  # Generate tiler
  tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

  # Divide tensors into tiles
  gA = cute.zipped_divide(mA, tiler_mn)
  gB = cute.zipped_divide(mB, tiler_mn)
  gC = cute.zipped_divide(mC, tiler_mn)

  # Launch kernel
  vectorized_add_kernel(gA, gB, gC).launch(
    grid=(cute.size(gC, mode=[1]), 1, 1),
    block=(cute.size(tv_layout, mode=[0]), 1, 1),
  )


# Use same compilation pattern as above
compiled_vec = cute.compile(vectorized_add_launch, a_, b_, c_)
compiled_vec(a_, b_, c_)
```

Key advantages of CuTe's tiling approach:

- Automatic coalesced memory access through layout algebra
- Vectorized loads via proper stride patterns (128-bit transactions)
- Compile-time optimization of memory access patterns
- Reduced memory latency through tiling

#### practical workflow

1. Prototype in Python using `cute.jit`:
   - Experiment with layout transformations
   - Verify partitioning logic
   - Test kernel logic on small inputs

2. Translate to C++ for production:
   - Python syntax maps directly to C++ CuTe API
   - Replace `@cute.jit` with template functions
   - Add compile-time optimizations (e.g., `cute::Int<128>` for static dimensions)

3. Profile and iterate:
   - Use `nsys` or `ncu` to identify bottlenecks
   - Refine layout choices based on bank conflict analysis
   - Optimize thread-to-data mappings

Example notebooks:

- [CuTe Layout Algebra](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb)
- [Data Types](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/data_types.ipynb)
- [Elementwise Add](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/elementwise_add.ipynb)
- [Tensor Operations](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/tensor.ipynb)

#### common pitfalls and solutions

##### kernel launch pattern

The proper pattern requires three components:

1. Device kernel with `@cute.kernel`
2. Host wrapper with `@cute.jit` that calls `.launch()`
3. Compilation with `cute.compile()`

Correct pattern:

```python
# Step 1: Define device kernel
@cute.kernel
def my_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
  # Device-side kernel code
  tidx, _, _ = cute.arch.thread_idx()
  bidx, _, _ = cute.arch.block_idx()
  # ... kernel implementation ...
  c[tidx] = a[tidx] + b[tidx]


# Step 2: Define host wrapper with @cute.jit
@cute.jit
def launch_wrapper(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
  # Optional: perform layout transformations
  tiled_a = cute.zipped_divide(a, tile_shape)
  tiled_b = cute.zipped_divide(b, tile_shape)

  # Launch kernel with grid/block configuration
  my_kernel(tiled_a, tiled_b, c).launch(
    grid=(num_blocks, 1, 1), block=(256, 1, 1)
  )


# Step 3: Convert tensors
a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

# Step 4: Compile the wrapper
compiled_fn = cute.compile(launch_wrapper, a_, b_, c_)

# Step 5: Execute (can call multiple times)
for _ in range(100):
  compiled_fn(a_, b_, c_)
```

Common mistakes:

```python
# ❌ Wrong: Calling kernel directly without @cute.jit wrapper
tensor_ = from_dlpack(tensor, assumed_align=16)
my_kernel(tensor_).launch(grid=..., block=...)  # Missing MLIR context!


# ❌ Wrong: Converting inside @cute.jit
@cute.jit
def bad_wrapper():
  a_ = from_dlpack(a, assumed_align=16)  # Convert outside!
  my_kernel(a_).launch(...)


# ❌ Wrong: Trying to compile @cute.kernel directly
compiled = cute.compile(my_kernel, a_)  # Compile the @cute.jit wrapper!
```

##### type annotation errors

Ensure proper type annotations for kernel arguments:

```python
@cute.kernel
def my_kernel(
  output: cute.Tensor,  # Runtime dynamic
  input: cute.Tensor,  # Runtime dynamic
  scale: cutlass.Float32,  # Runtime dynamic
  tile_size: cutlass.Constexpr,  # Compile-time constant
):
  # Scale must be wrapped: cutlass.Float32(value)
  # tile_size can be raw int (compile-time known)
  pass


# Call with proper types
my_kernel(
  output_,
  input_,
  cutlass.Float32(0.125),  # Wrap runtime values
  64,  # Compile-time constant
).launch(grid=..., block=...)
```

##### tensor lifetime issues

CuTe tensors reference original PyTorch memory - ensure tensors stay alive:

```python
# ❌ Tensor goes out of scope before kernel executes
def bad_example():
  temp = torch.randn(M, N, device='cuda')
  temp_ = from_dlpack(temp, assumed_align=16)
  return temp_  # temp deleted, temp_ now invalid!


result = bad_example()
my_kernel(result).launch(...)  # Undefined behavior!


# ✅ Keep original tensor alive
def good_example():
  tensor = torch.randn(M, N, device='cuda')
  tensor_ = from_dlpack(tensor, assumed_align=16)
  my_kernel(tensor_).launch(...)
  torch.cuda.synchronize()  # Wait for kernel completion
  return tensor  # Original tensor stays valid
```

##### grid/block configuration

Grid and block dimensions must be tuples of 3 integers:

```python
# ❌ Wrong dimensions
my_kernel(tensor_).launch(grid=256, block=128)

# ✅ Correct - always 3D tuples
my_kernel(tensor_).launch(
  grid=(num_blocks, 1, 1), block=(threads_per_block, 1, 1)
)

# Using CuTe size helpers
my_kernel(tiled_tensor).launch(
  grid=(cute.size(tiled_tensor, mode=[1]), 1, 1),  # Number of tiles
  block=(cute.size(thread_layout, mode=[0]), 1, 1),  # Threads per block
)
```

### optimization checklist

Memory:

- [ ] Coalesced global memory access
- [ ] Shared memory tiling for data reuse
- [ ] Avoid bank conflicts (padding, swizzling)
- [ ] Vectorized loads (128-bit when possible)
- [ ] TMA for asynchronous transfers (Hopper+)

Compute:

- [ ] Tensor cores for matmul (WMMA or wgmma)
- [ ] Mixed precision (BF16/FP8 input, FP32 accumulation)
- [ ] Minimize warp divergence
- [ ] Fuse operations in epilogue (avoid separate kernels)

Parallelism:

- [ ] Sufficient occupancy (50-75% typically optimal)
- [ ] Minimize register usage (avoid spilling)
- [ ] Overlap memory and compute (async pipelines)
- [ ] Persistent kernels for batched workloads

Advanced:

- [ ] Warp specialization (producer-consumer patterns)
- [ ] Online algorithms (streaming softmax)
- [ ] Approximate computation (cubic polynomial exp)
- [ ] Graph optimization (CUDA graphs, avoid launch overhead)

### further resources

Documentation:

- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CuTe Tutorials](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial)
- [NVIDIA Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### next lectures

Hands-on exercises:

1. Implement tiled matmul and measure performance progression
2. Add tensor core support using WMMA
3. Convert to CuTe DSL and explore layout transformations
4. Implement paged attention kernel and profile with ncu
5. Experiment with wgmma on Hopper

Extended topics:

- Conv2D optimization (im2col, implicit GEMM)
- Sparse matrix multiplication (structured sparsity)
- Multi-GPU programming (NCCL, NVLink)
- Kernel fusion with TorchInductor/XLA
- Custom CUDA graphs for inference serving
