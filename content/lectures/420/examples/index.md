---
date: '2025-09-30'
description: teaching kernels for checking CUDA indexing, memory access, reductions, CuTe layouts, and paged attention.
id: index
modified: 2026-06-05 15:08:04 GMT-04:00
tags:
  - seed
  - ml
  - gpu
title: CUDA programming examples
---

These are teaching kernels. `verify.sh` checks the basic build and numerical outputs on the local GPU. performance claims require a profiler trace from the same device, clock state, compiler, and input shape.

## Prerequisites

- NVIDIA GPU with compute capability ≥ 7.0 (Volta or newer)
- CUDA Toolkit 11.0+ installed
- `nvcc` compiler in PATH
- For best results: Ampere (SM 8.x), Ada (SM 8.9), or Hopper (SM 9.0) architecture

## Build Instructions

### quick start

```bash
# Build basic CUDA examples (01-05)
make

# Run basic examples
make test

# Build CuTe examples (requires CUTLASS - see CuTe Setup below)
make CUTLASS_DIR=~/cutlass cute

# Run CuTe examples
make test_cute

# Run Python example
make test_python

# Run all tests
make test_all

# Clean build artifacts
make clean

# Build specific example
make 01_vector_add
./01_vector_add
```

### basic cuda examples (01-05)

Core GPU programming concepts using standard CUDA.

---

#### 01_vector_add.cu

**Basic Vector Addition**

Demonstrates:

- Basic CUDA kernel launch syntax `<<<blocks, threads>>>`
- Thread indexing: `blockIdx.x * blockDim.x + threadIdx.x`
- Memory management: `cudaMalloc`, `cudaMemcpy`, `cudaFree`
- Host-device synchronization
- Error checking and timing

**Key concepts from lecture**:

- [[lectures/420#kernel-launch-from-host-to-device|Kernel launch: from host to device]]
- [[lectures/420#level-4-warps-and-threads|Level 4: warps and threads]]

representative output shape:

```
Launch config: 4096 blocks × 256 threads = 1048576 total threads
Verification: PASSED
```

---

#### 02_vectorized_loads.cu

**Vectorized Memory Access**

Demonstrates:

- Scalar vs vectorized loads (`float` vs `float4`)
- per-lane vector load instructions
- alignment and memory coalescing

**Key concepts from lecture**:

- [[lectures/420#memory-coalescing|Vectorized loads]]
- instruction-count comparison using `float4`

the vector version moves more values per lane instruction. inspect SASS and DRAM-sector metrics because coalescing, memory traffic, and instruction count are separate quantities.

---

#### 03_matmul.cu

**Matrix Multiplication Optimization**

Demonstrates:

- Naive implementation (global memory only)
- Tiled implementation with shared memory
- Memory reuse through tiling

**Key concepts from lecture**:

- [[lectures/420#matrix-multiplication-from-naive-to-optimized|matrix multiplication: from naive to optimized]]
- [[lectures/420#shared-memory-and-bank-conflicts|shared memory]]
- Arithmetic intensity improvement

the timing output compares these two kernels on the current device. neither kernel is a tensor-core performance reference.

**Formula**:

- FLOPs: $2 \times M \times N \times K$ (multiply-add)
- Memory footprint: $O(MK + KN + MN)$
- Tiled memory complexity: $O(\frac{MNK}{B_{\text{tile}}})$

---

#### 04_coalescing.cu

**Memory Coalescing**

Demonstrates:

- Coalesced vs uncoalesced memory access
- Shared memory usage to enable coalescing
- Matrix transpose as canonical example

**Key concepts from lecture**:

- [[lectures/420#memory-coalescing|memory coalescing]]
- [[lectures/420#memory-coalescing|Warp-level thread mapping]]

Access patterns:

```
Coalesced:   [T0][T1][T2]...[T31] → Single 128-byte transaction
Uncoalesced: [T0]....[T0+32N]...   → 32 separate transactions!
```

---

#### 05_reduction.cu

**Parallel Reduction**

Demonstrates:

- Dynamic shared memory allocation
- Warp-level synchronization optimization
- Bank conflict avoidance

**Key concepts from lecture**:

- [[lectures/420#shared-memory-and-bank-conflicts|shared memory]]
- [[lectures/420#thread-execution-model-simt|__syncthreads()]]
- [[lectures/420#warp-divergence-and-predication|warp divergence]]

Optimizations:

1. **Basic**: Simple shared memory reduction
2. **Optimized**: First-level reduction during global load, unrolled last warp

both paths are checked against the CPU result. compare elapsed time only after confirming that the byte-count formula and synchronization path match.

---

### CuTe DSL Examples (06-10)

Advanced GPU programming using NVIDIA's CuTe domain-specific language from CUTLASS 3.x+.

---

#### 06_cute_basics.cu

**CuTe DSL Fundamentals**

Demonstrates:

- Layout algebra: `make_shape`, `make_stride`, `make_layout`
- Tensor creation: `make_tensor` with global/shared memory
- Logical operations: `logical_divide`, `composition`
- 2D tensor indexing and strided layouts

**Key concepts from lecture**:

- [[lectures/420#cute-layout-algebra|CuTe layout algebra]]
- [[lectures/420#composition|Layout composition]]

**Prerequisites**: CUTLASS 3.x+ headers (see [CuTe Setup](#cute-setup) below)

Expected output:

```
Layout: (8):(1)
Tensor indexing: A(0)=... A(1)=...
2D layout: (4,8):(8,1)
```

---

#### 07_cute_swizzling.cu

**Bank Conflict Avoidance with Swizzling**

Demonstrates:

- Matrix transpose without swizzling (bank conflicts)
- Matrix transpose with a composed `Swizzle<3,3,3>` layout
- Swizzle pattern visualization
- Performance comparison

**Key concepts from lecture**:

- [[lectures/420#shared-memory-and-bank-conflicts|shared memory bank conflicts]]
- [[lectures/420#shared-memory-and-bank-conflicts|Swizzle patterns]]

Swizzle pattern:

```cpp
using SmemLayout = decltype(composition(
    Swizzle<3, 3, 3>{},
    Layout<Shape<Int<TILE_M>, Int<TILE_N>>, Stride<Int<TILE_N>, Int<1>>>{}));
```

---

#### 08_cute_tiling.cu

**Hierarchical Tiling**

Demonstrates:

- Grid → CTA → Thread decomposition
- `local_tile`: Extract CTA tile from global tensor
- `local_partition`: Distribute tile across threads
- `Step<_1,X,_1>`: Control partitioning vs broadcasting
- GEMM with CuTe tiling

**Key concepts from lecture**:

- [[lectures/420#hierarchical-tiling-with-local_tile|Hierarchical tiling]]
- [[lectures/420#hierarchical-tiling-with-local_tile|CuTe tiling primitives]]

this example exercises layout composition. its scalar compute path is not a CUTLASS collective or a tensor-core performance target.

Tiling hierarchy:

```cpp
auto cta_tile = make_tile(Int<TILE_M>{}, Int<TILE_N>{}, Int<TILE_K>{});
auto gA_cta = local_tile(A, cta_tile, cta_coord, Step<_1, X, _1>{});
auto sA_thr = local_partition(sA, thr_layout, tid);
```

---

#### 09_cute_paged_attention.cu

**Paged Attention with CuTe**

Demonstrates:

- KV cache paging with fixed-size pages
- Block tables: logical-to-physical page mapping
- Online softmax: streaming computation with running max/sum
- Memory efficiency vs pre-allocated caches

**Key concepts from lecture**:

- [[lectures/420#paged-attention-cute-implementation|Paged attention]]
- [[lectures/420#flash-attention|Online softmax]]

Key benefits:

1. **Bounded internal waste**: At most one partially filled final page per sequence
2. **Dynamic growth**: Allocate pages as needed
3. **Page sharing**: Common prefixes share pages
4. **Memory efficiency**: Capacity grows in page-size increments

Architecture:

```
Sequence 0: [logical pages] → Block table[0] → [physical pages 0,1,7]
Sequence 1: [logical pages] → Block table[1] → [physical pages 2,5,9]
```

---

#### 10_cute_paged_attention.py

**Python CuTe DSL: Paged Attention**

Demonstrates:

- Python-native GPU kernel development with `@cute.kernel`
- JIT compilation with `@cute.jit`
- PyTorch reference implementation for verification
- Memory efficiency analysis

**Prerequisites**:

```bash
pip install nvidia-cutlass-dsl torch
```

**Key concepts from lecture**:

- [[lectures/420#cute-dsl-python-examples|CuTe Python DSL]]
- [[lectures/420#paged-attention-cute-implementation|Paged attention]]

Usage:

```bash
# Falls back to PyTorch reference if CuTe DSL unavailable
python 10_cute_paged_attention.py
```

Kernel definition:

```python
@cute.kernel
def paged_attention_cute_kernel(output, query, key_cache, ...):
    seq_idx = cute.blockIdx.y
    head_idx = cute.blockIdx.x
    # Online softmax with CuTe tensors

@cute.jit
def launch():
    paged_attention_cute_kernel[grid, block](...)
```

---

## CuTe Setup

The CuTe examples (06-10) require **CUTLASS 3.x+** headers.

### Installation

```bash
# Clone CUTLASS
git clone https://github.com/NVIDIA/cutlass.git ~/cutlass
cd ~/cutlass
git checkout v3.5.0  # or latest 3.x release

# Build CuTe examples
cd /path/to/examples
make CUTLASS_DIR=~/cutlass cute

# Or set environment variable
export CUTLASS_DIR=~/cutlass
make cute
```

### Python CuTe DSL (Optional)

For example 10:

```bash
pip install nvidia-cutlass-dsl torch

# Verify installation
python -c "import nvidia.cutlass.dsl as cute; print('CuTe DSL available')"
```

**Note**: CuTe Python DSL requires:

- CUDA 12.0+
- Python 3.8+
- PyTorch with CUDA support
- Currently in beta (API may change)

### Testing CuTe Examples

```bash
# Test all CuTe examples
make test_cute

# Test Python example
make test_python

# Test everything (basic + CuTe + Python)
make test_all
```

---

## File Structure

```
examples/
├── common.cuh                        # Shared utilities (timing, error checking, verification)
│
├── 01_vector_add.cu                  # Basic CUDA kernel
├── 02_vectorized_loads.cu            # Memory bandwidth optimization
├── 03_matmul.cu                      # Tiling and shared memory
├── 04_coalescing.cu                  # Memory access patterns
├── 05_reduction.cu                   # Parallel primitives
│
├── 06_cute_basics.cu                 # CuTe layout algebra
├── 07_cute_swizzling.cu              # Bank conflict avoidance
├── 08_cute_tiling.cu                 # Hierarchical tiling
├── 09_cute_paged_attention.cu        # Paged attention (C++)
├── 10_cute_paged_attention.py        # Paged attention (Python, CuTe, PyTorch)
│
├── Makefile                          # Build system with CuTe support
├── verify.sh                         # Automated testing script
└── index.md                          # This file
```

## Common Utilities (`common.cuh`)

**Macros:**

- `CUDA_CHECK(call)`: Checks CUDA errors with file/line info

**Classes:**

- `GpuTimer`: CUDA event-based timing
  ```cpp
  GpuTimer timer;
  timer.start();
  kernel<<<...>>>(...);
  timer.stop();
  float ms = timer.elapsed();
  ```

**Functions:**

- `init_array<T>(T* arr, size_t n, T max_val)`: Random initialization
- `verify_results<T>(result, expected, n, tolerance)`: Correctness check
- `print_device_info()`: Display GPU properties

## Building for Different Architectures

Edit `Makefile` to change the target architecture:

```makefile
# Volta (V100): sm_70
ARCH = -arch=sm_70

# Turing (RTX 20xx): sm_75
ARCH = -arch=sm_75

# Ampere (A100): sm_80
ARCH = -arch=sm_80

# Ada (RTX 40xx): sm_89
ARCH = -arch=sm_89

# Hopper (H100): sm_90
ARCH = -arch=sm_90
```

Or specify at build time:

```bash
make ARCH="-arch=sm_90" 01_vector_add
```

## performance records

record GPU model, application clocks, CUDA and compiler versions, input shape, warm-up count, median kernel time, bytes transferred, and profiler metrics. a hardware name by itself does not define an expected range for these teaching kernels.

## Profiling with Nsight Compute

Profile any example with `ncu`:

```bash
# Basic metrics
ncu --set full ./01_vector_add

# Specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
              dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./03_matmul

# Roofline analysis
ncu --set roofline ./03_matmul
```

## Troubleshooting

**Error: `no kernel image available`**

- Your GPU architecture is newer than the compiled code
- Rebuild with correct `-arch` flag

**Error: `out of memory`**

- Reduce problem size in the example source
- Check available GPU memory: `nvidia-smi`

**Low performance**

- Check GPU isn't throttling: `nvidia-smi dmon`
- Ensure no other processes using GPU
- Verify correct architecture flag was used

## Extending the Examples

To add your own kernel:

1. Create `06_mykernel.cu`
2. Include `common.cuh`
3. Add to `TARGETS` in `Makefile`
4. Follow the pattern:

   ```cpp
   #include "common.cuh"

   __global__ void my_kernel(...) { ... }

   int main() {
       print_device_info();
       // Allocate, copy, launch, verify
       return 0;
   }
   ```

## References

- Full lecture notes: [[lectures/420/index]]
- GPU architecture overview: [[lectures/420#what-is-a-gpu|what is a GPU?]]
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Nsight Compute: https://developer.nvidia.com/nsight-compute

## License

These examples are educational materials for Lecture 0.420. Free to use and modify.
