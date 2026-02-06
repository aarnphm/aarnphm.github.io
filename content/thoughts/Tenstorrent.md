---
date: '2025-10-05'
description: RISC-V based AI accelerators with programmable architecture
id: Tenstorrent
modified: 2025-11-10 08:39:32 GMT-05:00
tags:
  - ml
  - asic
  - hardware
title: Tenstorrent
---

Jim Keller's company (legendary chip designer, AMD, Tesla, Intel), the man need no introduction.

RISC-V cores, explicit SRAM management, Ethernet fabric. no automatic caching, no hidden schedulers [@pizzini2025tenstorrent; @vasiljevic2024blackhole].

see also: [[thoughts/GPU programming]], [[thoughts/MLIR]], [[thoughts/Compiler]], [[thoughts/pdfs/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf|Jasmina's talk at HotChip]]

## execution model comparison

| dimension    | NVIDIA GPU              | Tenstorrent                               |
| ------------ | ----------------------- | ----------------------------------------- |
| parallelism  | SIMT warps (32 threads) | 5 RISC-V cores per Tensix                 |
| scheduling   | hardware warp scheduler | software pipeline (reader/compute/writer) |
| memory model | automatic L1/L2 caching | explicit SRAM circular buffers            |
| programming  | CUDA kernels            | three separate kernels per Tensix         |
| interconnect | NVLink (900 GB/s)       | Ethernet (100G-400G)                      |

## architecture

### wormhole (second gen, production)

see also: https://docs.tenstorrent.com/aibs/wormhole/specifications.html

**specifications:** [@tenstorrent2024wormhole; @semianalysis2021wormhole]

- 12nm GlobalFoundries, 670mm² die
- 12GB GDDR6, 192-bit bus
- 16×100G Ethernet (chip-level)
- 466 TFLOPS FP8 (n300 dual chip)
- 160W single chip, 300W dual chip

**products:**

- n150: 72 Tensix cores, 288 GB/s bandwidth, $999
- n300: 128 Tensix cores (dual chip), 384 GB/s per chip, $1,399

compare H100 PCIe: 80GB HBM3 @ 2TB/s, 1979 TFLOPS FP16, $30,000.

raw compute: H100 has 4.2× throughput. raw price: H100 costs 21× more.

### blackhole (third gen, sampling)

**specifications:** [@tenstorrent2024blackhole; @vasiljevic2024blackhole]

- 140 Tensix cores (p150a), 6nm TSMC
- 32GB GDDR6, 256-bit bus, 512 GB/s bandwidth
- 4×800G QSFP-DD Ethernet ports
- 745 TFLOPS FP8, 372 TFLOPS FP16
- 16 "big" RISC-V cores (SiFive X280, 64-bit dual-issue)
- 752 "baby" RISC-V cores (5 per Tensix)

**products:**

- p150a: single chip
- Galaxy: 32 chips, 4×8 mesh, 23.8 PFLOPS FP8

big RISC-V cores handle control plane, offload host CPU. baby cores manage data movement within Tensix [@pizzini2025tenstorrent].

### tensix core architecture

each Tensix core runs five pipelines concurrently [@pizzini2025tenstorrent]:

```
┌──────────────────────────────────────────────┐
│                 Tensix Core                  │
├──────────────────────────────────────────────┤
│ RISC-V 0 (reader)    → fetch DRAM/NoC → CB   │
│ RISC-V 1 (unpack)    → CB → compute engines  │
│ RISC-V 2,3 (compute) → math/SIMD execution   │
│ RISC-V 4 (writer)    → CB → DRAM/NoC         │
├──────────────────────────────────────────────┤
│ Math Engine: MAC arrays (32×32 tiles)        │
│ SIMD Engine: FP32/FP16/BF16/FP8/INT8         │
│ SRAM: 1.5MB (Wormhole/Blackhole)             │
│   - organized as circular buffers (CBs)      │
│   - explicit management, no automatic cache  │
└──────────────────────────────────────────────┘
```

**data flow:**

1. reader fetches tiles from DRAM or remote cores via NoC
2. tiles land in circular buffer (CB) in SRAM
3. unpacker moves tiles from CB to compute engines
4. math/SIMD engines execute on tiles
5. packer writes results to output CB
6. writer sends tiles to DRAM or remote cores

all five stages run asynchronously. circular buffers decouple stages. synchronization via producer/consumer counters.

**contrast with CUDA:**

CUDA: write one kernel, hardware scheduler issues warps, L1/L2 cache data automatically.

Tenstorrent: write three kernels (reader, compute, writer), explicitly manage SRAM, coordinate via circular buffers.

## memory hierarchy

### latency and bandwidth

| level     | NVIDIA H100      | Tenstorrent Wormhole             | Tenstorrent Blackhole             |
| --------- | ---------------- | -------------------------------- | --------------------------------- |
| registers | ~1 cycle         | ~1 cycle                         | ~1 cycle                          |
| L1/shared | 20-30 cycles     | N/A (explicit SRAM)              | N/A (explicit SRAM)               |
| SRAM      | N/A              | 1.5MB [@tenstorrent2024wormhole] | 1.5MB [@tenstorrent2024blackhole] |
| L2 cache  | ~200 cycles      | N/A                              | N/A                               |
| HBM/GDDR  | 300-500 ns       | GDDR6: 288-384 GB/s              | GDDR6: 512 GB/s                   |
| fabric    | NVLink: 900 GB/s | 16×100G = 200 GB/s               | 4×800G = 3.2 TB/s                 |

Tenstorrent has no automatic caching. SRAM is explicitly managed circular buffer space [@thuning2024attention].

CUDA kernel issues load, hardware decides L1/L2/HBM path. Tenstorrent kernel explicitly specifies SRAM buffer, transfer size, source/destination.

### circular buffers (CBs)

SRAM allocated as ring buffers. producer writes, consumer reads [@tenstorrent2024metalium]. example:

```cpp
// Allocate circular buffer 0: 4 tiles of 32×32 FP16
cb_reserve_back(cb_id=0, num_tiles=4);
cb_push_back(cb_id=0, num_tiles=4);

// Consumer waits for tiles
cb_wait_front(cb_id=0, num_tiles=1);
// process tile
cb_pop_front(cb_id=0, num_tiles=1);
```

producer can write ahead (up to buffer size). consumer blocks if empty. double/triple buffering overlaps compute and data movement.

NVIDIA equivalent: none. closest analogue is manual shared memory management in CUDA, but even that has automatic L1 caching underneath.

## network-on-chip (NoC)

2D mesh connecting Tensix cores, DRAM controllers, Ethernet ports.

**topology:** grid layout, each core has router, bidirectional links to neighbors.

**programming:** explicit tile transfers, source/destination coordinates.

```cpp
// NoC read: fetch tile from remote core
noc_async_read(src_addr, dst_local_addr, size, src_noc_x, src_noc_y);

// NoC write: send tile to remote core
noc_async_write(src_local_addr, dst_addr, size, dst_noc_x, dst_noc_y);

// Multicast: send to multiple destinations
noc_async_write_multicast(src_addr, dst_addr, size, start_x, start_y, end_x, end_y);
```

NoC is exposed to programmers. you control data movement. you control bandwidth utilization.

CUDA: memory copies handled by driver, DMA engines. you call `cudaMemcpy`, hardware does routing.

Tenstorrent: you specify NoC coordinates, manage congestion, orchestrate multicast.

## ethernet fabric

Wormhole: 16×100G. Blackhole: 12×400G.

commodity switches, standard cables. contrast NVLink requiring proprietary switches (NVSwitch costs $5k+).

**topology example (Blackhole Galaxy):**

32 chips in 4×8 mesh. each chip connects to 4 neighbors (north, south, east, west). forms 2D torus.

```
[0]─[1]─[2]─[3]─[4]─[5]─[6]─[7]
 │   │   │   │   │   │   │   │
[8]─[9]─[10]─[11]─[12]─[13]─[14]─[15]
 │   │   │   │   │   │   │   │
[16]─[17]─[18]─[19]─[20]─[21]─[22]─[23]
 │   │   │   │   │   │   │   │
[24]─[25]─[26]─[27]─[28]─[29]─[30]─[31]
```

**programming:**

Ethernet exposed as remote memory. send/receive semantics integrated with NoC addressing.

```cpp
// Send tensor to remote chip via Ethernet
ethernet_send(chip_id=1, buffer_addr, size);

// Receive from remote chip
ethernet_recv(chip_id=0, buffer_addr, size);
```

**tradeoffs:**

Ethernet latency: microseconds. NVLink latency: nanoseconds.

data-parallel training with frequent all-reduce hits latency wall. pipeline parallelism works better (coarse-grained transfers).

cost: commodity 400G switch costs \$USD10k for 32 ports. NVSwitch costs \$USD150k+ for equivalent connectivity.

## software stack

### tt-metalium (low-level runtime)

bare-metal programming model [@tenstorrent2024metalium; @ttmetal2024]. three kernel types per Tensix core:

**reader kernel (RISC-V 0):**

```cpp
// reader.cpp
void MAIN {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t num_tiles = get_arg_val<uint32_t>(1);

  constexpr uint32_t cb_id = 0;
  const uint32_t tile_bytes = get_tile_size(cb_id);

  for (uint32_t i = 0; i < num_tiles; ++i) {
    cb_reserve_back(cb_id, 1);
    uint32_t write_addr = get_write_ptr(cb_id);

    // NoC read from DRAM to CB
    noc_async_read(input_addr, write_addr, tile_bytes);
    noc_async_read_barrier();

    cb_push_back(cb_id, 1);
    input_addr += tile_bytes;
  }
}
```

**compute kernel (RISC-V 1-3):**

```cpp
// compute.cpp
void MAIN {
  constexpr uint32_t cb_in = 0;
  constexpr uint32_t cb_out = 1;

  uint32_t num_tiles = get_arg_val<uint32_t>(0);

  for (uint32_t i = 0; i < num_tiles; ++i) {
    cb_wait_front(cb_in, 1);

    acquire_dst();

    // Unpack tile from CB to math engine
    unpack_tilize_init_short(cb_in);
    unpack_tilize(cb_in, 0);

    // Math operation (e.g., GELU)
    gelu_tile_init();
    gelu_tile(0);

    // Pack result back to CB
    pack_tile(0, cb_out);

    release_dst();

    cb_pop_front(cb_in, 1);
    cb_push_back(cb_out, 1);
  }
}
```

**writer kernel (RISC-V 4):**

```cpp
// writer.cpp
void MAIN {
  uint32_t output_addr = get_arg_val<uint32_t>(0);
  uint32_t num_tiles = get_arg_val<uint32_t>(1);

  constexpr uint32_t cb_id = 1;
  const uint32_t tile_bytes = get_tile_size(cb_id);

  for (uint32_t i = 0; i < num_tiles; ++i) {
    cb_wait_front(cb_id, 1);
    uint32_t read_addr = get_read_ptr(cb_id);

    // NoC write from CB to DRAM
    noc_async_write(read_addr, output_addr, tile_bytes);
    noc_async_write_barrier();

    cb_pop_front(cb_id, 1);
    output_addr += tile_bytes;
  }
}
```

**host code:**

```cpp
// Launch kernels on device
Device *device = CreateDevice(0);
CommandQueue queue(device);

Program program = CreateProgram();
program.SetRuntimeArgs(CoreCoord{0, 0}, {input_addr, num_tiles});

// Compile and run
EnqueueProgram(queue, program);
Finish(queue);
```

contrast CUDA:

```cuda
// CUDA: single kernel
__global__ void gelu_kernel(float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = input[idx];
    output[idx] = 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
  }
}

// Launch
gelu_kernel<<<blocks, threads>>>(d_input, d_output, n);
```

CUDA: one kernel, hardware schedules warps, automatic memory fetching.

Tenstorrent: three kernels, explicit pipeline, manual SRAM management.

### tt-nn (operation library)

higher-level API with pre-optimized operations [@ttmetal2024].

```python
import ttnn

device = ttnn.open_device(device_id=0)

# Create tensors on device
x = ttnn.from_torch(torch_tensor, device=device, layout=ttnn.TILE_LAYOUT)

# Operations
y = ttnn.gelu(x)
z = ttnn.matmul(x, weight)
out = ttnn.add(y, z)

# Back to torch
result = ttnn.to_torch(out)
```

**tile layout:**

operations work on 32×32 tiles. tensors must be tile-aligned.

```python
# Convert to tile layout (pad to multiples of 32)
x_tiled = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

# Matmul requires tile layout
# [M, K] @ [K, N] where M, K, N are multiples of 32
result = ttnn.matmul(a_tiled, b_tiled)
```

**available operations:**

- linear algebra: `matmul`, `bmm`, `conv2d`, `linear`
- activations: `gelu`, `relu`, `silu`, `softmax`
- normalization: `layernorm`, `rmsnorm`, `groupnorm`
- attention: `scaled_dot_product_attention`

each operation maps to optimized Metalium kernels. users avoid writing reader/compute/writer by hand.

### tt-mlir (compiler infrastructure)

MLIR-based compilation stack [@ttmlir2024]. ingests high-level models, lowers to Metalium kernels.

**dialects:**

- **ttir (Tenstorrent IR)**: high-level tensor operations
- **ttnn**: maps to tt-nn library calls
- **ttmetal**: maps to Metalium kernels
- **ttkernel**: RISC-V code generation

**compilation flow:**

```
StableHLO/Torch → TTIR → TTNN → TTMetal → TTKernel → Binary
```

**example lowering:**

```mlir
// StableHLO input
func.func @matmul(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1,
    contracting_dims = [1] x [0] : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
  return %0 : tensor<128x512xf32>
}

// After convert-to-ttir
func.func @matmul(%arg0: !tt.tensor<128x256xf32>, %arg1: !tt.tensor<256x512xf32>) -> !tt.tensor<128x512xf32> {
  %0 = ttir.matmul %arg0, %arg1 : (!tt.tensor<128x256xf32>, !tt.tensor<256x512xf32>) -> !tt.tensor<128x512xf32>
  return %0 : !tt.tensor<128x512xf32>
}

// After lower-to-ttnn
func.func @matmul(%arg0: !tt.tensor<128x256xf32>, %arg1: !tt.tensor<256x512xf32>) -> !tt.tensor<128x512xf32> {
  %empty = ttnn.empty [128, 512] : !tt.tensor<128x512xf32>
  %0 = ttnn.matmul %arg0, %arg1, %empty : (!tt.tensor<128x256xf32>, !tt.tensor<256x512xf32>, !tt.tensor<128x512xf32>) -> !tt.tensor<128x512xf32>
  return %0 : !tt.tensor<128x512xf32>
}

// After lower-to-ttmetal
// Generates reader/compute/writer kernels, allocates CBs, computes tiling
```

**key passes:**

- **tiling pass**: partition operations into tiles fitting in SRAM
- **layout pass**: convert to tile layout (32×32 blocks)
- **memory planning**: allocate circular buffers, schedule transfers
- **kernel generation**: emit reader/compute/writer kernels

similar to IREE's compilation flow (StableHLO → Linalg → HAL). Tenstorrent's MLIR stack is newer, less mature than XLA or IREE.

### tt-xla (XLA backend)

XLA custom backend. JAX/TensorFlow models compile via XLA → Tenstorrent.

```python
import jax
import jax.numpy as jnp

# Force compilation to Tenstorrent via XLA
with jax.default_device(jax.devices('tt')[0]):
  x = jnp.ones((128, 256))
  y = jnp.ones((256, 512))
  z = jnp.matmul(x, y)  # Compiles to Tenstorrent
```

XLA emits HLO, Tenstorrent backend lowers HLO → Metalium kernels.

**lowering path:**

```
JAX → HLO → Tenstorrent custom calls → Metalium kernels
```

**kernel generation:**

HLO operations (dot, convolution, etc.) map to templates. backend generates tiled reader/compute/writer based on tensor shapes.

## programming model

### tiling strategies

operations exceed SRAM (1.5MB). tiling: break into chunks.

**matmul example:**

```
C[M, N] = A[M, K] @ B[K, N]
SRAM holds: ~1.5MB ≈ 384K FP32 elements ≈ 192 tiles (32×32 FP32)

Tile sizes: M_tile=64, N_tile=64, K_tile=64
Each iteration processes:
  A_tile: 64×64 = 4K elements = 16KB
  B_tile: 64×64 = 4K elements = 16KB
  C_tile: 64×64 = 4K elements = 16KB
  Total: 48KB per tile triplet, fits comfortably in 1.5MB SRAM

Loop:
for i in range(0, M, 64):
  for j in range(0, N, 64):
    C_tile = zeros(64, 64)
    for k in range(0, K, 64):
      A_tile = load_tile(A, i, k)
      B_tile = load_tile(B, k, j)
      C_tile += matmul_tile(A_tile, B_tile)
    store_tile(C, i, j, C_tile)
```

**double buffering:**

overlap compute and data movement. while computing tile N, prefetch tile N+1.

```cpp
// Allocate 2 buffers per CB
cb_reserve_back(cb_a, 2);  // Buffers 0, 1

for (int i = 0; i < num_tiles; ++i) {
  uint32_t buf = i % 2;  // Alternate buffers

  // Prefetch next tile to buffer (i+1)%2
  if (i + 1 < num_tiles) {
    noc_async_read(src_addr + (i+1)*tile_bytes, get_write_ptr(cb_a, (i+1)%2), tile_bytes);
  }

  // Compute on current tile in buffer i%2
  cb_wait_front(cb_a, 1);
  compute_tile(buf);
  cb_pop_front(cb_a, 1);
}
```

NVIDIA equivalent: software pipelining in CUDA, but L1/L2 cache hides some of this. Tenstorrent requires explicit double buffering for performance.

### multi-chip parallelism

**pipeline parallelism:**

layer N on chip 0, layer N+1 on chip 1, etc.

```
Chip 0: attention layer
    ↓ (ethernet)
Chip 1: MLP layer
    ↓ (ethernet)
Chip 2: next attention layer
    ↓
...
```

microbatching: split batch into micro-batches, pipeline through chips.

```
Time 0: Chip 0 processes micro-batch 0
Time 1: Chip 0 processes micro-batch 1, Chip 1 processes micro-batch 0
Time 2: Chip 0 processes micro-batch 2, Chip 1 processes micro-batch 1, Chip 2 processes micro-batch 0
...
```

steady state: all chips busy, throughput = slowest stage.

**data parallelism:**

replicate model on each chip, split batch.

```
Chip 0: samples 0-7
Chip 1: samples 8-15
Chip 2: samples 16-23
...

Forward pass → compute gradients → all-reduce gradients → update weights
```

all-reduce over Ethernet slower than NVLink. 400G helps (Blackhole) but still 1.5× slower than NVLink.

workaround: gradient accumulation (reduce all-reduce frequency), pipeline parallelism (avoid all-reduce).

**tensor parallelism:**

shard tensors across chips. example: split attention heads.

```
12 attention heads, 4 chips → 3 heads per chip

Each chip computes 3 heads independently.
Concatenate results via Ethernet.
```

requires careful sharding to balance compute and communication. works well for coarse-grained sharding (full attention heads), poorly for fine-grained (sharding within heads).

## performance characteristics

### raw compute

| metric         | Wormhole n300 | Blackhole p150a | NVIDIA H100 (PCIe) |
| -------------- | ------------- | --------------- | ------------------ |
| FP16 TFLOPS    | ~230          | 372             | 1,513              |
| FP8 TFLOPS     | 466           | 745             | 3,026              |
| power (W)      | 300           | ~300            | 350                |
| TFLOPS/W (FP8) | 1.55          | 2.48            | 8.65               |
| price ($)      | 1,399         | TBD (~$3k)      | 30,000             |
| $/TFLOP (FP8)  | 3.00          | ~4.03           | 9.92               |

H100 has 6.5× throughput, 3.5× efficiency, 21× cost.

Tenstorrent wins on price/performance. H100 wins on absolute performance.

### memory bandwidth

| metric    | Wormhole   | Blackhole       | H100       |
| --------- | ---------- | --------------- | ---------- |
| capacity  | 12GB GDDR6 | 32GB GDDR6      | 80GB HBM3  |
| bandwidth | 336 GB/s   | ~500 GB/s (est) | 2,000 GB/s |
| BW/TFLOP  | 0.72 GB/s  | 0.67 GB/s       | 0.66 GB/s  |

H100 has 6× memory bandwidth, 2.5× capacity. critical for memory-bound workloads (LLM inference, embedding tables).

Tenstorrent trades bandwidth for cost. inference with small batch sizes (latency-sensitive) hits bandwidth wall.

### interconnect bandwidth

| metric     | Wormhole            | Blackhole           | H100             |
| ---------- | ------------------- | ------------------- | ---------------- |
| intra-chip | NoC (~TB/s)         | NoC (~TB/s)         | ~5 TB/s internal |
| inter-chip | 16×100G = 200 GB/s  | 12×400G = 600 GB/s  | NVLink: 900 GB/s |
| latency    | ~1-10 μs (ethernet) | ~1-10 μs (ethernet) | ~1 μs (NVLink)   |

Blackhole narrows gap (600 vs 900 GB/s) but latency remains higher. impacts all-reduce heavy workloads.

### benchmark results

> [!note] benchmark data
> specific LLaMA-7B and ResNet-50 numbers are estimates based on architecture analysis. authoritative benchmarks from Tenstorrent show Falcon-7B at 76-77 tokens/s (batch 32) and Llama-3.1-8B targeting 20 tokens/s/user.

**LLaMA-7B inference (estimated, batch=1, seq=128):**

- Wormhole n300: ~15 tokens/s (estimate)
- H100: ~60 tokens/s

H100 4× faster (memory bandwidth bound).

**LLaMA-7B inference (estimated, batch=32, seq=128):**

- Wormhole n300: ~200 tokens/s (estimate)
- H100: ~600 tokens/s

H100 3× faster (compute bound, H100's higher throughput wins).

**ResNet-50 inference (batch=64):**

- Wormhole: ~2,000 images/s (estimate)
- H100: ~8,000 images/s

H100 4× faster.

**cost-normalized (images/s per $1k):**

- Wormhole n300: 1,430 images/s per $1k
- H100: 267 images/s per $1k

Wormhole 5.4× better on cost-normalized throughput.

## use cases and tradeoffs

### where tenstorrent wins

**cost-sensitive inference:**

serving models where cost/inference matters more than latency. batch inference, async APIs.

**edge deployment:**

Grayskull (75W) fits edge servers. comparable to Jetson AGX Orin (60W, 200 TOPS INT8).

**custom dataflow:**

researchers needing non-standard execution (custom quantization, sparsity patterns, model-specific optimizations).

explicit programming model allows fine-grained control impossible in CUDA.

**vendor diversification:**

avoiding NVIDIA lock-in. open-source stack (Apache 2.0), commodity interconnect.

### where NVIDIA wins

**raw performance:**

H100 has 4-8× throughput. latency-critical workloads (interactive chat, real-time inference) need absolute speed.

**memory capacity:**

H100 has 80GB, A100 has 80GB, H200 has 141GB. large models (70B+) require high memory.

Wormhole's 12GB insufficient. Blackhole's 32GB helps but still less than H100.

**software maturity:**

CUDA ecosystem: 15+ years, cuDNN, TensorRT, Triton, CUTLASS.

PyTorch/TensorFlow/JAX deeply optimized for CUDA.

Tenstorrent: <5 years public development. MLIR stack is young, optimization passes less mature.

**training at scale:**

data-parallel training with frequent all-reduce. NVLink's low latency (ns) vs Ethernet (μs) matters.

8×H100 NVLink cluster scales better than 8×Blackhole Ethernet.

**ecosystem:**

Hugging Face models, pre-trained weights, model zoos assume CUDA.

Tenstorrent requires manual porting, optimization.

## systems architecture

### wormhole n300 deep dive

dual-chip module. each chip: 80 Tensix cores, 12GB GDDR6, 8×100G Ethernet.

chips connected via Ethernet (not direct die-to-die). introduces latency but uses standard switches.

**inter-chip communication:**

chip 0 wants to send tensor to chip 1:

1. writer kernel on chip 0 writes to Ethernet port
2. Ethernet switch routes packet to chip 1
3. reader kernel on chip 1 receives from Ethernet port

latency: ~1-2 μs. bandwidth: up to 800 Gbps (8×100G).

**use case: pipeline parallelism:**

split transformer layer-wise. layer 0-11 on chip 0, layer 12-23 on chip 1.

```
Input → Chip 0 (layers 0-11) → Ethernet → Chip 1 (layers 12-23) → Output
```

forward pass: 1 transfer (activations). backward pass: 1 transfer (gradients).

total transfers: 2 per sample. at 800 Gbps, 8MB activation takes ~80 μs.

if compute time per chip >80 μs, Ethernet doesn't bottleneck.

### blackhole galaxy architecture

32 chips, 4×8 mesh, 2D torus topology.

each chip connects to 4 neighbors (N, S, E, W). wrap-around edges (torus).

**routing example:**

chip 0 sends to chip 31:

- direct path: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 15 → 23 → 31 (10 hops)
- torus wrap: 0 → 24 → 25 → 26 → 27 → 28 → 29 → 30 → 31 (8 hops)

routing algorithm selects shortest path. each hop adds ~200 ns latency.

**collective operations:**

all-reduce across 32 chips. ring algorithm: 32 steps.

per-step transfer: 1/32 of tensor. at 400G per link, 1GB tensor takes:

- per-step: 1GB/32 = 32MB. 32MB @ 50GB/s = 640 μs
- total: 32 steps × 640 μs = 20 ms

compare H100 NVSwitch all-reduce: ~1-2 ms for 1GB (10-20× faster).

workaround: pipeline parallelism (avoid frequent all-reduce), gradient accumulation (reduce frequency).

### big RISC-V cores (blackhole)

16 cores, 4 clusters of 4. 64-bit dual-issue, SiFive design.

**use cases:**

1. **control plane:** schedule kernels, manage Ethernet, coordinate multi-chip.
2. **preprocessing:** image decode, tokenization, data augmentation.
3. **host offload:** reduce PCIe traffic, run logic on-device.

**example: preprocessing pipeline:**

host sends compressed images via PCIe → big RISC-V cores decode JPEG → Tensix cores run model.

eliminates host-side decode + PCIe transfer. saves bandwidth, reduces latency.

**programming:**

big cores run standard RISC-V Linux. compile with GCC/Clang.

```c
// Run on big RISC-V core
void preprocess(uint8_t* input, float* output, int size) {
  for (int i = 0; i < size; ++i) {
    output[i] = (input[i] / 255.0f - 0.5f) * 2.0f;  // Normalize
  }
  // Send to Tensix cores via NoC
  noc_write(output, tensix_buffer_addr, size * sizeof(float));
}
```

baby cores (752 total) handle fine-grained Tensix control. big cores handle coarse-grained orchestration.

## software ecosystem gaps

### missing pieces

**automatic kernel tuning:**

CUDA has CUTLASS (template-based GEMM), Triton (Python DSL for kernels).

Tenstorrent: no equivalent. users hand-tune tile sizes, circular buffer allocation.

**model zoo:**

Hugging Face Transformers, Diffusers assume CUDA.

Tenstorrent: manual porting required. tt-metal repo has examples (BERT, ResNet, LLaMA) but limited coverage.

**third-party libraries:**

CUDA: DeepSpeed, Megatron, vLLM.

Tenstorrent: none. users implement custom parallelism, optimizations.

**debugger:**

CUDA: cuda-gdb, Nsight Compute, Nsight Systems.

Tenstorrent: basic logging, print debugging. no interactive debugger for RISC-V kernels.

**profiler:**

CUDA: nvprof, Nsight profiling tools.

Tenstorrent: Tracy integration (partial), manual instrumentation.

### timeline estimate

**today (2025):** basic functionality. compile PyTorch/JAX via MLIR. hand-optimize critical kernels.

**2026:** improved MLIR passes (fusion, tiling). model zoo coverage (transformers, diffusion).

**2027:** auto-tuning, profiler, debugger. competitive software parity with 2023 CUDA.

**2028+:** third-party ecosystem (libraries, frameworks).

Tenstorrent is 3-5 years behind NVIDIA in software maturity. acceptable if cost/openness outweigh convenience.

## technical challenges

### explicit memory management burden

programmer allocates circular buffers, sizes them correctly, coordinates producer/consumer.

misstep: buffer overflow (producer outpaces consumer), buffer underflow (consumer starves).

**example error:**

```cpp
// Allocate CB with 2 tiles
cb_reserve_back(cb_id, 2);

// Reader writes 3 tiles (OVERFLOW)
for (int i = 0; i < 3; ++i) {
  cb_push_back(cb_id, 1);  // Third push wraps around, corrupts data
}
```

debugging: no runtime checks. corruption manifests as incorrect results, not errors.

CUDA: L1/L2 cache automatic. user only manages shared memory (max 48KB-164KB). Tenstorrent: user manages all 1.5MB SRAM.

### multi-chip synchronization

Ethernet latency (μs) requires coarse-grained synchronization.

**barrier example:**

32 chips synchronize after layer. each chip sends "done" signal to coordinator, waits for "proceed" signal.

latency: 32 chips × 1 μs (send) + 1 μs (receive) = ~33 μs.

if layer compute time is 1 ms, 33 μs = 3% overhead (acceptable).

if layer compute time is 100 μs, 33 μs = 33% overhead (unacceptable).

solution: pipeline to hide synchronization. while chip 0 computes layer N, chip 1 computes layer N+1.

### thermal constraints

Blackhole Galaxy: 32 chips × 300W = 9.6kW. power supply delivers 7.5kW → chips run below max TDP.

datacenter cooling: 7.5kW in 6U chassis requires liquid cooling or high-CFM fans.

compare 8×H100 NVLink: 8 × 700W = 5.6kW (more manageable air cooling).

power efficiency matters. Tenstorrent's 2.5 TFLOPS/W vs H100's 8.6 TFLOPS/W means longer runtime for same work → higher total energy.

cost-sensitive batch inference: energy matters less (already amortized over many requests). latency-critical: energy matters more (idle power + peak power).

## comparison to other accelerators

### cerebras (wafer-scale)

CS-3: entire wafer (46,225 mm²), 900,000 cores, 44GB on-chip SRAM.

**vs Tenstorrent:**

- Cerebras: extreme on-chip memory, eliminates DRAM bottleneck.
- Tenstorrent: modular, scales via Ethernet, cheaper per TFLOP.

Cerebras targets giant models fitting on single wafer. Tenstorrent targets distributed workloads over commodity fabric.

### graphcore (IPU)

IPU: 1,472 cores, 900MB on-chip SRAM, BSP programming model.

**vs Tenstorrent:**

- Graphcore: automatic data movement (BSP model abstracts synchronization).
- Tenstorrent: explicit data movement (manual control).

Graphcore easier to program but less flexible. Tenstorrent harder but allows custom dataflow.

Graphcore struggled commercially (Softbank acquired 2024). Tenstorrent ongoing.

### SambaNova (dataflow)

SambaNova SN40L: dataflow architecture, reconfigurable tiles.

**vs Tenstorrent:**

- SambaNova: spatial dataflow (map compute graph to tiles).
- Tenstorrent: temporal execution (loop over tiles).

SambaNova optimizes graph-level parallelism. Tenstorrent optimizes per-op parallelism.

both target enterprise inference. SambaNova closed-source, Tenstorrent open-source.

### AMD (GPU)

MI300X: 192GB HBM3, 1,300 TFLOPS FP16, 5.2 TB/s bandwidth.

**vs Tenstorrent:**

- AMD: CUDA-like (ROCm, HIP), drop-in replacement.
- Tenstorrent: different programming model, not CUDA-compatible.

AMD competes directly with NVIDIA. Tenstorrent differentiates on cost, openness, programmability.

## open-source ecosystem

### tt-metal (Apache 2.0)

https://github.com/tenstorrent/tt-metal [@ttmetal2024]

low-level runtime, device kernels, host API.

**structure:**

```
tt_metal/
  hw/              # Hardware specs, NoC configs
  impl/            # Runtime implementation
  kernels/         # Common kernels (matmul, softmax, etc.)
  programming_examples/  # Tutorials
  tt_eager/        # Python bindings (tt-nn)
```

active development. ~50 commits/week (2025).

### tt-mlir (Apache 2.0)

https://github.com/tenstorrent/tt-mlir [@ttmlir2024]

MLIR dialects, compiler passes.

**structure:**

```
tt-mlir/
  include/mlir-dialects/  # TTIR, TTNN, TTMetal dialects
  lib/Conversion/         # Lowering passes
  lib/Transforms/         # Optimization passes
  runtime/                # Runtime integration
```

younger than tt-metal. fewer contributors, faster iteration.

## future roadmap

### near-term (2025-2026)

- Blackhole production availability
- MLIR optimization passes (fusion, memory planning)
- PyTorch 2.0 integration (torch.compile backend)
- model zoo expansion (Llama, Mistral, Stable Diffusion)

### mid-term (2026-2027)

- next-gen silicon (post-Blackhole): 3nm, HBM3, higher core count
- 64-128 chip clusters
- auto-tuning framework (tile size search, layout optimization)
- profiler and debugger tools

### long-term (2027+)

- 1,000+ chip clusters
- custom silicon licensing (hyperscalers integrate Tensix IP)
- competitive training performance (not just inference)
- ecosystem parity with CUDA (libraries, frameworks, tools)

## references

### documentation

- Tenstorrent website: https://tenstorrent.com/
- [@tenstorrent2024metalium] - TT-Metalium programming guide
- [@tenstorrent2024wormhole] - Wormhole specifications
- [@tenstorrent2024blackhole] - Blackhole specifications
- GitHub repositories:
  - [@ttmetal2024] - tt-metal: low-level runtime and kernels
  - [@ttmlir2024] - tt-mlir: MLIR compiler infrastructure

### whitepapers and presentations

- [@vasiljevic2024blackhole] - Hot Chips 2024: Blackhole & TT-Metalium architecture presentation
- [@semianalysis2021wormhole] - SemiAnalysis Wormhole analysis: scale-out architecture details
- [@semianalysis2022blackhole] - SemiAnalysis Blackhole & Grendel deep dive
- ServeTheHome Blackhole coverage
- The Register: RISC-V packed Blackhole chips (2024)

### academic papers

- [@thuning2024attention] - "Attention in SRAM on Tenstorrent Grayskull" demonstrates SRAM-based attention mechanisms
  - https://github.com/moritztng/grayskull-attention
- [@pizzini2025tenstorrent] - "Assessing Tenstorrent's RISC-V MatMul Acceleration Capabilities" provides detailed Tensix core analysis (ISC HPC 2025)
- [@brown2024stencils] - "Accelerating stencils on the Tenstorrent Grayskull RISC-V accelerator" shows 5× energy efficiency vs Xeon (SC24)

see also: [[thoughts/GPU programming]], [[thoughts/MLIR]], [[thoughts/XLA]], [[thoughts/Compiler]], [[thoughts/PyTorch]]
