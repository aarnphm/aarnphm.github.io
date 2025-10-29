---
date: "2022-12-23"
description: or accelerated linear algebra
id: XLA
modified: 2025-10-29 02:15:38 GMT-04:00
tags:
  - seed
  - ml
  - compilers
title: XLA
---

> a domain-specific compiler for linear algebra that optimizes JAX computations.

The idea is to _fuse multiple operations into single optimized kernels_, eliminating intermediate memory allocations and reducing kernel launch overhead.

```python
import jax.numpy as jnp


def compute(x, y, z):
  return jnp.sum(x + y * z)


# without XLA: 3 kernel launches
# 1. multiply: y * z → temp1 (memory allocation)
# 2. add: x + temp1 → temp2 (memory allocation)
# 3. reduce_sum: temp2 → result (memory allocation)
# total: 3 kernel launches, 2 intermediate buffers

# with XLA: 1 fused kernel
# single kernel computes entire expression
# reads x, y, z once → writes result once
# total: 1 kernel launch, 0 intermediate buffers
```

see also: [OpenXLA](https://github.com/openxla/xla)

see also: [[thoughts/PJRT|PJRT]], [[thoughts/MLIR|MLIR]], [[thoughts/PyTorch]], [[thoughts/Jax|JAX]], [[thoughts/Compiler|compiler]], [[thoughts/Autograd|autograd]], [[thoughts/GPU programming|GPU programming]]

## architecture

### compilation pipeline

XLA transforms high-level ML operations through multiple IR levels:

```
JAX/TensorFlow Python
    ↓ (tracing)
Jaxpr (JAX) / GraphDef (TF)
    ↓ (lowering)
StableHLO
    ↓ (conversion)
HLO (High-Level Operations)
    ↓ (optimization passes)
Optimized HLO
    ↓ (backend codegen)
LLO (Low-Level Operations)
    ↓ (target specific)
LLVM IR (CPU) / PTX (GPU) / Custom (TPU)
    ↓ (assembly)
Native machine code
```

**frontend abstraction:** frameworks emit StableHLO, a portable ML operation set with versioning guarantees

**optimization layer:** XLA's core optimization passes operate on HLO IR

**backend abstraction:** PJRT provides uniform runtime interface across hardware

### HLO: high-level operations

HLO is XLA's primary intermediate representation. functional, SSA-form IR representing tensor computations.

**core properties:**

- immutable tensors (value semantics)
- pure functions (no side effects except I/O)
- static shapes (dynamic shapes handled separately)
- explicit memory layout specification

fundamental operation categories:

**element-wise operations:**

```
HloInstruction* add = builder.AddInstruction(
    HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, lhs, rhs));

// element-wise: map, abs, exp, log, tanh, etc.
// broadcasting follows NumPy semantics
```

**reduction operations:**

```
HloInstruction* sum = builder.AddInstruction(
    HloInstruction::CreateReduce(
        reduced_shape, operand, init_value,
        {reduction_dim}, sum_computation));

// reductions: sum, product, max, min, argmax
// specify dimensions and reduction computation
```

**dot operations (matmul/convolution):**

```
// general matrix multiplication
DotDimensionNumbers dims;
dims.add_lhs_contracting_dimensions(1);  // contract on dim 1
dims.add_rhs_contracting_dimensions(0);  // contract on dim 0

HloInstruction* dot = builder.AddInstruction(
    HloInstruction::CreateDot(
        output_shape, lhs, rhs, dims, precision));

// convolution: specialized dot with windowing
HloInstruction* conv = builder.AddInstruction(
    HloInstruction::CreateConvolve(...));
```

**data movement:**

```
// reshape, transpose, slice, dynamic-slice
// concat, pad, reverse, broadcast
// these operations are often "free" via layout optimization
```

**control flow:**

```
// while loop
HloInstruction* while_op = builder.AddInstruction(
    HloInstruction::CreateWhile(
        shape, condition_computation,
        body_computation, init));

// conditional
HloInstruction* cond = builder.AddInstruction(
    HloInstruction::CreateConditional(
        shape, pred, true_comp, false_comp));

// call: function invocation
```

**HLO example for matmul:**

```
HloModule SimpleMatmul

ENTRY %matmul.v2 (a: f32[128,256], b: f32[256,512]) -> f32[128,512] {
  %a = f32[128,256]{1,0} parameter(0)
  %b = f32[256,512]{1,0} parameter(1)
  ROOT %dot = f32[128,512]{1,0} dot(f32[128,256]{1,0} %a,
                                     f32[256,512]{1,0} %b),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
```

layout notation `{1,0}` means dimension 1 is minor (row-major). XLA optimizes layouts automatically.

### StableHLO

StableHLO is the portable operation set built on MLIR, succeeding MHLO.

**versioning guarantees:**

- 5-year backward compatibility (older StableHLO → newer XLA)
- 2-year forward compatibility (newer StableHLO → older XLA within window)
- MLIR bytecode serialization with version metadata

**operation coverage:** ~100 operations spanning:

- element-wise arithmetic and comparison
- linear algebra (dot_general, convolution)
- reductions and scans
- dynamic shape operations
- quantization operations
- control flow (while, conditional, case)
- collective operations (all-reduce, all-gather)

StableHLO ops map cleanly to HLO but include serialization versioning:

```mlir
func.func @matmul(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>)
    -> tensor<128x512xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
  return %0 : tensor<128x512xf32>
}
```

XLA converts StableHLO → HLO during compilation. StableHLO enables framework interoperability: JAX, TensorFlow, PyTorch all emit StableHLO.

## fusion: the killer optimization

fusion combines multiple operations into single kernels. primary source of XLA speedup.

### fusion types

**vertical fusion (producer-consumer):**

fuse operations in data dependency chain:

```python
import jax
import jax.numpy as jnp


@jax.jit
def relu_grad(x):
  return jnp.where(x > 0, 1.0, 0.0)


# without fusion:
# kernel 1: compare x > 0 → temp (memory write/read)
# kernel 2: select based on temp → result (memory write/read)

# with fusion:
# single kernel: loads x, computes comparison and select, writes result
# eliminates intermediate buffer and kernel launch
```

**horizontal fusion (sibling operations):**

fuse independent operations reading same inputs:

```python
@jax.jit
def stats(x):
  mean = jnp.mean(x)
  var = jnp.var(x)
  return mean, var


# without fusion:
# kernel 1: reduce_sum for mean
# kernel 2: reduce_sum for variance
# x is read from memory twice

# with fusion:
# single kernel: reads x once, computes both reductions
# memory bandwidth halved
```

**multi-output fusion:**

fuse operations producing multiple outputs:

```python
@jax.jit
def layer_norm_stats(x):
  mean = jnp.mean(x, axis=-1, keepdims=True)
  centered = x - mean
  variance = jnp.mean(centered**2, axis=-1, keepdims=True)
  return mean, variance


# fusion produces both mean and variance in single pass
# intermediate 'centered' kept in registers, never written to memory
```

### fusion strategies

XLA uses pattern-based fusion with greedy algorithm:

**loop fusion:** element-wise and reduction ops with compatible iteration spaces

```python
@jax.jit
def fused_activation(x, bias, scale):
  # all element-wise, same shape → single fused kernel
  return jnp.tanh((x + bias) * scale)


# generates single GPU kernel:
# for each element i:
#   temp1 = x[i] + bias[i]
#   temp2 = temp1 * scale[i]
#   output[i] = tanh(temp2)
# all temporaries in registers
```

**input fusion:** fuse reductions with their element-wise producers

```python
@jax.jit
def softmax(x):
  exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
  return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)


# fusion plan:
# 1. fuse: max reduction with subtract and exp
# 2. fuse: sum reduction with divide
# result: 2 kernels instead of 5
```

**fusion constraints:**

fusion isn't always beneficial:

- **memory limits:** fused kernel must fit in register file / shared memory
- **reuse patterns:** if intermediate used multiple times, fusion may increase recomputation
- **operation compatibility:** can't fuse across synchronization boundaries
- **large operations:** optimized GEMM/convolution libraries (cuBLAS, cuDNN) shouldn't be fused

XLA analyzes cost model:

```
benefit = memory_traffic_saved + kernel_launch_overhead_saved
cost = register_pressure + recomputation

fuse if benefit > cost
```

## core optimization passes

### algebraic simplification

constant folding, identity elimination, algebraic rewrites:

```python
@jax.jit
def redundant(x):
  return x * 1.0 + 0.0 - 0.0  # → x


# XLA simplifies:
# x * 1.0 → x (multiply by identity)
# x + 0.0 → x (add identity)
# x - 0.0 → x (subtract identity)
```

strength reduction:

```python
@jax.jit
def power_of_two(x):
  return x * 8.0  # → x << 3 (if integer context)
  return x / 4.0  # → x * 0.25 (multiply cheaper than divide)
```

reassociation for parallelism:

```python
# ((a + b) + c) + d → (a + b) + (c + d)
# enables SIMD parallelism in reduction tree
```

### layout optimization

memory layout affects performance by orders of magnitude.

**row-major vs column-major:**

```python
# matrix stored as [M, N]
# row-major {1,0}: elements in row are contiguous
# column-major {0,1}: elements in column are contiguous

# XLA chooses layout based on access patterns:
# - row-major for row-wise operations
# - column-major for GEMM (cuBLAS prefers column-major)
```

**layout propagation:**

XLA propagates layouts through computation:

```
matmul → transpose → matmul
# instead of: compute matmul, transpose output, use as input
# optimize to: compute matmul in transposed layout directly
# eliminates transpose operation entirely
```

**padding for alignment:**

```python
# shape [127, 255] padded to [128, 256] for:
# - aligned memory access (coalescing on GPU)
# - vectorization (SIMD friendly sizes)
# - tile sizes (match hardware block dimensions)
```

### buffer assignment

liveness analysis determines when buffers can be reused:

```python
@jax.jit
def sequential_ops(x):
  a = x + 1.0  # buffer for 'a'
  b = a * 2.0  # can reuse 'a' buffer (a dead after this)
  c = b - 3.0  # can reuse 'a'/'b' buffer
  return c


# XLA allocates single buffer, reused for a, b, c
# instead of 3 separate allocations
```

**aliasing analysis:**

identify when input and output can share memory:

```python
@jax.jit
def in_place_update(x):
  return x.at[0].set(1.0)  # output can alias input (careful with purity!)


# XLA may avoid copy if safe
```

**memory planning:**

XLA schedules operations to minimize peak memory:

```
operation sequence: A → B → C → D
memory profiles:
  A then B: peak = 1000MB
  A then C then B: peak = 800MB
XLA may reorder operations for memory efficiency
```

### rematerialization

trade computation for memory by recomputing values:

```python
@jax.jit
def checkpoint_example(x):
  a = expensive_computation(x)  # large intermediate
  b = f(a)
  c = g(a)  # 'a' needed again
  return b + c


# without rematerialization: store 'a' in memory
# with rematerialization: recompute 'a' when needed for 'c'
# saves memory at cost of recomputation
```

critical for training large models where activations dominate memory.

JAX provides explicit control:

```python
from jax import checkpoint


@jax.jit
def train_step(params, x):
  @checkpoint
  def forward(params, x):
    # large forward pass
    return loss

  loss, grads = jax.value_and_grad(forward)(params, x)
  return loss, grads


# checkpointed blocks: save only inputs and outputs
# recompute activations during backward pass
```

### all-reduce optimization

for distributed training, collective operations are performance-critical.

**ring all-reduce:**

```
instead of: gather all gradients to one device → broadcast
use: each device exchanges with neighbors in ring
bandwidth optimal: each device sends/receives (N-1)/N * data_size
```

**hierarchical all-reduce:**

```
multi-node cluster:
1. all-reduce within each node (fast NVLink/shared memory)
2. all-reduce across nodes (slower network)
3. broadcast result within nodes
```

**all-reduce fusion:**

```python
# fuse multiple small all-reduces into single large one
# amortizes latency overhead
# XLA automatically batches across layers
```

## JAX integration

JAX uses XLA as compilation backend via `jax.jit`.

### tracing mechanism

JAX traces Python functions to Jaxpr (JAX expression), then lowers to StableHLO/HLO:

```python
import jax
import jax.numpy as jnp


def f(x):
  return x**2 + 2 * x + 1


# first call: trace
jitted_f = jax.jit(f)

# tracing captures computation on abstract values
x_abstract = jax.ShapedArray((10,), jnp.float32)

# produces Jaxpr:
# { lambda ; a:f32[10].
#   let b = integer_pow[y=2] a
#       c = mul 2.0 a
#       d = add b c
#       e = add d 1.0
#   in (e,) }
```

Jaxpr is functional IR with explicit data dependencies. XLA lowers Jaxpr → StableHLO → HLO → optimized code.

### staged compilation

**trace once, execute many:**

```python
@jax.jit
def matmul(A, B):
  return A @ B


# first call with shape [100, 200] @ [200, 300]
result1 = matmul(jnp.ones((100, 200)), jnp.ones((200, 300)))
# traces and compiles

# subsequent calls with same shapes: reuse compiled code
result2 = matmul(jnp.ones((100, 200)), jnp.ones((200, 300)))
# no recompilation

# different shapes trigger recompilation
result3 = matmul(jnp.ones((50, 100)), jnp.ones((100, 150)))
# traces and compiles new version
```

**recompilation overhead:**

```python
# bad: varying shapes cause recompilation
for i in range(1000):
  x = jnp.ones((i, i))  # different shape each iteration!
  result = jitted_f(x)  # recompiles every iteration

# good: fixed shapes
x = jnp.ones((1000, 1000))
for i in range(1000):
  result = jitted_f(x)  # compiled once, reused
```

use `static_argnums` for values affecting control flow:

```python
@jax.jit(static_argnums=(1,))
def conditional_norm(x, ord):
  if ord == 1:
    return jnp.sum(jnp.abs(x))
  elif ord == 2:
    return jnp.sqrt(jnp.sum(x**2))
  else:
    return jnp.max(jnp.abs(x))


# compiles separate versions for each 'ord' value
result = conditional_norm(x, ord=2)  # compiles L2 version
```

### control flow primitives

Python control flow doesn't work with tracing:

```python
@jax.jit
def broken(x):
  if x < 3:  # error: can't branch on traced value
    return x**2
  else:
    return x + 1
```

use JAX control flow primitives:

**cond for conditionals:**

```python
from jax import lax


@jax.jit
def conditional(x):
  return lax.cond(
    x < 3,
    lambda x: x**2,  # true branch
    lambda x: x + 1,  # false branch
    x,
  )


# XLA compiles both branches, selects at runtime
# more efficient than Python if
```

**while_loop for loops:**

```python
@jax.jit
def power_method(A, num_iters):
  def cond_fun(state):
    i, v = state
    return i < num_iters

  def body_fun(state):
    i, v = state
    v = A @ v
    v = v / jnp.linalg.norm(v)
    return i + 1, v

  init_val = (0, jnp.ones(A.shape[0]))
  _, v = lax.while_loop(cond_fun, body_fun, init_val)
  return v


# XLA optimizes loop body (fusion, etc.)
```

**fori_loop for counted loops:**

```python
@jax.jit
def cumsum(x):
  def body_fun(i, acc):
    return acc + x[i]

  return lax.fori_loop(0, len(x), body_fun, 0.0)
```

**scan for sequential computation:**

```python
@jax.jit
def running_mean(x):
  def scan_fn(carry, x_i):
    mean, count = carry
    new_count = count + 1
    new_mean = mean + (x_i - mean) / new_count
    return (new_mean, new_count), new_mean

  init = (0.0, 0)
  _, means = lax.scan(scan_fn, init, x)
  return means


# scan is XLA-optimized for sequential dependencies
```

### dynamic shapes

XLA primarily operates on static shapes, but dynamic shapes are supported:

```python
@jax.jit
def slice_until_zero(x):
  # find first zero, slice up to that point
  idx = jnp.argmax(x == 0)  # dynamic index
  return lax.dynamic_slice(x, (idx,), (1,))


# XLA uses special dynamic shape operations
# potentially less optimized than static shapes
```

polymorphic shapes (experimental):

```python
from jax.experimental import polymorphic_api as poly


@jax.jit
def generic_matmul(A, B):
  return A @ B


# can compile for symbolic shapes [m, k] @ [k, n]
# reuses compiled code for all concrete m, k, n
```

### autodiff through XLA

JAX composes differentiation with JIT compilation:

```python
import jax
import jax.numpy as jnp


@jax.jit
def loss(params, x, y):
  pred = params['w'] @ x + params['b']
  return jnp.mean((pred - y) ** 2)


# differentiate
grad_fn = jax.grad(loss)

# JIT compile gradient
jitted_grad = jax.jit(grad_fn)

# XLA compiles both forward and backward pass
# fusion applies to both directions
grads = jitted_grad(params, x, y)
```

forward-mode AD:

```python
from jax import jvp


@jax.jit
def f(x):
  return x**3


# Jacobian-vector product
primals, tangents = jvp(f, (x,), (v,))
# XLA fuses forward pass and JVP computation
```

reverse-mode AD:

```python
from jax import vjp


@jax.jit
def f(x):
  return jnp.sum(x**2)


# vector-Jacobian product
primals, vjp_fn = vjp(f, x)
grads = vjp_fn(1.0)  # cotangents
# XLA optimizes backward pass with fusion
```

second-order derivatives:

```python
hessian = jax.jacfwd(jax.grad(f))
# XLA compiles forward-over-reverse mode
# chooses optimal AD mode based on shapes
```

## backend code generation

### CPU backend (LLVM)

XLA lowers HLO → LLVM IR → x86/ARM assembly.

**vectorization:**

```python
@jax.jit
def vector_add(x, y):
  return x + y


# XLA generates LLVM IR with vector intrinsics:
# %1 = load <8 x float>, <8 x float>* %x_ptr
# %2 = load <8 x float>, <8 x float>* %y_ptr
# %3 = fadd <8 x float> %1, %2
# store <8 x float> %3, <8 x float>* %out_ptr

# LLVM lowers to AVX/NEON instructions
```

**parallelization:**

XLA uses parallel loops for multi-core CPUs:

```python
@jax.jit
def parallel_matmul(A, B):
  return A @ B


# XLA may generate OpenMP-style parallel loops
# or use Eigen for multi-threaded BLAS
```

### GPU backend (NVIDIA)

XLA generates NVPTX IR → PTX → SASS (GPU assembly).

**kernel fusion:**

```python
@jax.jit
def gelu(x):
  return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))


# XLA fuses entire GELU into single CUDA kernel
# typical framework: 7+ kernel launches
# XLA: 1 kernel launch
```

generated CUDA kernel structure:

```cuda
__global__ void fused_gelu(float* x, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x_val = x[idx];
        float x_cubed = x_val * x_val * x_val;
        float inner = x_val + 0.044715f * x_cubed;
        float scaled = sqrtf(2.0f / M_PI) * inner;
        float tanh_val = tanhf(scaled);
        out[idx] = 0.5f * x_val * (1.0f + tanh_val);
    }
}
```

**memory coalescing:**

XLA optimizes memory access patterns:

```python
# transpose: read rows, write columns (inefficient)
# XLA inserts shared memory tile to coalesce accesses
```

**cuDNN/cuBLAS integration:**

```python
@jax.jit
def optimized_conv(x, kernel):
  return lax.conv_general_dilated(x, kernel, ...)


# XLA recognizes convolution pattern
# calls cuDNN instead of generating custom kernel
# same for matmul → cuBLAS
```

**multi-GPU:**

```python
from jax.experimental import multihost_utils
import jax

# 8 GPUs, data parallel
devices = jax.devices()


@jax.jit
def train_step(params, batch):
  loss, grads = jax.value_and_grad(loss_fn)(params, batch)
  grads = lax.pmean(grads, axis_name='batch')  # all-reduce
  return loss, grads


# XLA generates:
# 1. per-device computation kernels
# 2. NCCL all-reduce for gradient synchronization
# 3. optimized communication schedule
```

### TPU backend

custom XLA backend for Google's Tensor Processing Units.

TPUs use **systolic array** architecture: matrix multiply units arranged in 2D grid.

XLA generates TPU-specific instructions:

```python
@jax.jit
def tpu_matmul(A, B):
  return A @ B


# XLA tiles matrices to fit TPU dimensions
# generates MXU (matrix multiply unit) instructions
# optimizes data layout for systolic array
```

TPU v4: 128x128 systolic array, bfloat16/int8 compute.

### multi-device execution

XLA supports heterogeneous execution:

```python
from jax import pmap


@pmap
def parallel_computation(x):
  # runs on all available devices
  return x**2 + x


# XLA compiles once, replicates across devices
# automatic collective operations for communication
```

SPMD (Single Program Multiple Data):

```python
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

devices = mesh_utils.create_device_mesh((4, 2))  # 4x2 device grid
sharding = PositionalSharding(devices)


@jax.jit
def sharded_matmul(A, B):
  return A @ B


# annotate sharding
A = jax.device_put(A, sharding.reshape(4, 2))
# XLA generates device-local computation
# inserts collective-permute for cross-device communication
```

## performance characteristics

### when XLA helps

**compute-bound workloads:**

many small operations benefit most from fusion:

```python
@jax.jit
def transformer_mlp(x, W1, W2):
  return jax.nn.gelu(x @ W1) @ W2


# unfused: matmul → gelu → matmul (3 kernel launches)
# fused: matmul → fused_gelu_matmul (saves gelu kernel + intermediate)
# typical speedup: 1.3-2x
```

**fusion opportunities:**

```python
# excellent for XLA
@jax.jit
def element_wise_heavy(x):
  return jnp.tanh(jnp.exp(x) / (1.0 + jnp.exp(x)))


# speedup: 5-10x (eliminates intermediate buffers)


# marginal benefit
@jax.jit
def just_matmul(A, B):
  return A @ B


# speedup: 1.05-1.2x (cuBLAS already optimized)
```

### when XLA doesn't help

**memory-bound operations:**

```python
@jax.jit
def large_transpose(x):  # shape: [10000, 10000]
  return x.T


# bottleneck: memory bandwidth
# XLA can't improve (hardware limited)
# speedup: ~1.0x
```

**already-optimized libraries:**

```python
# large GEMM: cuBLAS is near-optimal
# XLA adds compilation overhead, minimal runtime gain
@jax.jit
def huge_matmul(A, B):  # [8192, 8192] @ [8192, 8192]
  return A @ B


# speedup: 1.0-1.1x
```

**small workloads:**

```python
@jax.jit
def tiny_computation(x):  # x.shape = (10,)
  return x + 1


# compilation overhead >> execution time
# first call: 50-200ms (compilation)
# subsequent calls: <1ms (execution)
# not worth it for one-off computations
```

### compilation overhead

**cold start:**

```python
import time
import jax.numpy as jnp


def f(x):
  return jnp.sum(x**2 + 2 * x + 1)


x = jnp.ones(1000000)

# first call: tracing + compilation
start = time.time()
result = jax.jit(f)(x)
result.block_until_ready()
print(f'first call: {(time.time() - start) * 1000:.2f}ms')
# typical: 100-500ms

# second call: cached
start = time.time()
result = jax.jit(f)(x)
result.block_until_ready()
print(f'second call: {(time.time() - start) * 1000:.2f}ms')
# typical: 0.1-1ms
```

**persistent compilation cache:**

```bash
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
# subsequent runs reuse compiled code
```

**ahead-of-time compilation:**

```python
from jax import jit
from jax.experimental import aot

# lower and compile ahead of time
lowered = jit(f).lower(x)
compiled = lowered.compile()

# serialize for later use
serialized = compiled.runtime_executable().serialize()

# load and execute (no compilation)
loaded = aot.load_from_serialized(serialized)
result = loaded(x)
```

### typical speedups

real-world performance gains:

| workload                    | unfused  | XLA       | speedup  |
| --------------------------- | -------- | --------- | -------- |
| element-wise chain (10 ops) | baseline | fused     | 8-12x    |
| transformer layer           | baseline | fused     | 1.5-2.5x |
| softmax                     | baseline | fused     | 3-5x     |
| layer norm                  | baseline | fused     | 2-4x     |
| large matmul (>1024)        | baseline | optimized | 1.0-1.2x |
| small matmul (<256)         | baseline | optimized | 1.2-1.8x |
| RNN cell (per step)         | baseline | fused     | 2-3x     |

compilation overhead amortizes over ~100-1000 executions for typical models.

## comparison with other compilers

### vs TVM

**TVM:** tensor compilation framework with Relay (graph IR) and TIR (tensor IR).

| aspect      | XLA                              | TVM                             |
| ----------- | -------------------------------- | ------------------------------- |
| approach    | domain-specific (linear algebra) | general tensor compilation      |
| IR levels   | StableHLO → HLO → LLO            | Relay → TIR → target code       |
| auto-tuning | limited (mostly heuristics)      | AutoTVM, Ansor (genetic search) |
| scheduling  | automatic fusion                 | manual schedule primitives      |
| backends    | CPU, GPU, TPU                    | CPU, GPU, mobile, FPGA, custom  |
| integration | TensorFlow, JAX native           | external via TVMC               |

TVM emphasizes auto-tuning: exhaustively search schedules for optimal performance.

XLA emphasizes integration: native compilation for TF/JAX with good-enough performance.

### vs IREE

**IREE:** MLIR-based ML compiler targeting edge/mobile/datacenter.

| aspect        | XLA                        | IREE                     |
| ------------- | -------------------------- | ------------------------ |
| IR foundation | custom (HLO)               | MLIR (Linalg, Flow, HAL) |
| target focus  | datacenter (TPU, GPU, CPU) | edge, mobile, embedded   |
| compilation   | JIT and AOT                | primarily AOT            |
| latency       | optimized for throughput   | optimized for latency    |
| binary size   | large (includes runtime)   | small (minimal runtime)  |

IREE uses MLIR's progressive lowering:

```
StableHLO → Linalg → Flow → Stream → HAL → target
```

XLA and IREE share StableHLO frontend, diverge at optimization strategy.

### vs TorchScript / TorchInductor

**TorchScript:** PyTorch's tracing/scripting compilation.

**TorchInductor:** PyTorch 2.0's compiler (backend for `torch.compile`).

| aspect      | XLA                      | TorchInductor            |
| ----------- | ------------------------ | ------------------------ |
| tracing     | JAX-style functional     | Python bytecode analysis |
| fusion      | pattern-based HLO fusion | Triton template codegen  |
| backend     | LLVM, NVPTX, TPU         | Triton, C++, CUDA        |
| integration | JAX, TensorFlow          | PyTorch native           |
| maturity    | production (10+ years)   | newer (2023)             |

TorchInductor generates Triton kernels: Python DSL for GPU programming.

XLA generates PTX/SASS: compiler chooses fusion strategy.

### vs Triton

**Triton:** GPU kernel programming DSL (Python syntax).

| aspect      | XLA                          | Triton                          |
| ----------- | ---------------------------- | ------------------------------- |
| abstraction | graph-level fusion           | kernel-level programming        |
| control     | automatic (compiler decides) | manual (programmer controls)    |
| expertise   | none (use JAX)               | GPU programming knowledge       |
| flexibility | limited by fusion patterns   | arbitrary kernels               |
| ease        | write NumPy, get performance | write explicit memory hierarchy |

Triton for custom kernels, XLA for general compilation.

JAX supports custom Triton kernels via `jax.extend.triton`:

```python
import jax
from jax.extend import triton as jax_triton
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(0)
  offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  mask = offs < N
  x = tl.load(x_ptr + offs, mask=mask)
  y = tl.load(y_ptr + offs, mask=mask)
  tl.store(out_ptr + offs, x + y, mask=mask)


def triton_add(x, y):
  out = jnp.empty_like(x)
  N = x.size
  grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

  kernel = jax_triton.triton_call(x, y, out, kernel=add_kernel, grid=grid, BLOCK_SIZE=1024)
  return kernel


# combine: XLA for graph-level, Triton for custom kernels
```

## advanced features

### SPMD partitioning

Single Program Multiple Data: partition computation across devices.

```python
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P

# create device mesh
devices = mesh_utils.create_device_mesh((4, 2))  # 4x2 grid
mesh = Mesh(devices, axis_names=('data', 'model'))


# annotate partitioning
@jax.jit
def sharded_matmul(x, w):
  # x: [batch, features] sharded on 'data' axis
  # w: [features, classes] sharded on 'model' axis
  return x @ w


# specify sharding
with mesh:
  x_sharded = jax.device_put(x, P('data', None))
  w_sharded = jax.device_put(w, P(None, 'model'))
  result = sharded_matmul(x_sharded, w_sharded)
  # result sharded: P('data', 'model')

# XLA generates:
# 1. local matmuls on each device
# 2. all-gather for 'model' dimension
# 3. communication schedule minimizing latency
```

### pipeline parallelism

split model across devices, pipeline micro-batches:

```python
from jax.experimental import pipeline


def layer_1(x):
  return jax.nn.relu(x @ W1)


def layer_2(x):
  return jax.nn.relu(x @ W2)


def layer_3(x):
  return x @ W3


stages = [layer_1, layer_2, layer_3]


@pipeline.pipeline(stages, num_microbatches=4)
def pipelined_model(x):
  return x


# XLA schedules micro-batches across stages:
# time 0: device 0 processes batch 0 stage 1
# time 1: device 0 processes batch 1 stage 1
#         device 1 processes batch 0 stage 2
# ...
# overlaps computation and communication
```

### gradient checkpointing

rematerialization for memory efficiency:

```python
from jax import checkpoint


def expensive_layer(x, params):
  # large activation tensors
  h1 = jax.nn.relu(x @ params['w1'])
  h2 = jax.nn.relu(h1 @ params['w2'])
  h3 = jax.nn.relu(h2 @ params['w3'])
  return h3


@checkpoint
def checkpointed_layer(x, params):
  return expensive_layer(x, params)


@jax.jit
def forward(params, x):
  h = checkpointed_layer(x, params)
  return jnp.sum(h)


# XLA optimization:
# forward pass: save only inputs/outputs
# backward pass: recompute h1, h2, h3 as needed
# trades 2x compute for 3x memory savings
```

selective checkpointing:

```python
# checkpoint only expensive blocks
def transformer_block(x, params):
  attn_out = attention(x, params['attn'])
  mlp_out = checkpoint(mlp)(attn_out, params['mlp'])  # checkpoint MLP only
  return mlp_out


# XLA recomputes MLP activations in backward pass
# keeps attention activations (cheaper to store)
```

### collective operations

XLA optimizes distributed training primitives:

**all-reduce:**

```python
from jax import lax


@jax.jit
def data_parallel_update(local_grads):
  # average gradients across devices
  avg_grads = lax.pmean(local_grads, axis_name='devices')
  return avg_grads


# XLA generates ring all-reduce (NCCL on GPU)
# bandwidth-optimal: O((N-1)/N) efficiency
```

**all-gather:**

```python
@jax.jit
def gather_predictions(local_preds):
  # gather predictions from all devices
  all_preds = lax.all_gather(local_preds, axis_name='devices')
  return all_preds


# XLA optimizes communication pattern
```

**reduce-scatter:**

```python
@jax.jit
def reduce_scatter_grads(grads):
  # reduce and scatter in one operation
  scattered = lax.psum_scatter(grads, axis_name='devices')
  return scattered


# XLA fuses reduce and scatter
# saves communication round-trip
```

### custom call mechanism

integrate hand-written kernels:

```python
from jax import core
from jax.interpreters import xla


# register custom operation
def custom_op_abstract(x):
  return core.ShapedArray(x.shape, x.dtype)


def custom_op_lowering(ctx, x):
  # emit XLA custom call
  return xla.custom_call(
    'my_custom_kernel', result_types=[ctx.avals_out[0]], operands=[x], backend_config={'param': 42}
  )


custom_op = core.Primitive('custom_op')
custom_op.def_abstract_eval(custom_op_abstract)
xla.register_lowering(custom_op, custom_op_lowering)


# use in JAX
@jax.jit
def f(x):
  return custom_op.bind(x)


# XLA calls registered C++/CUDA kernel
# enables integration with cuDNN, cuBLAS, custom CUDA
```

## debugging and introspection

### XLA_FLAGS environment variables

```bash
# dump HLO to files
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"

# dump optimized HLO
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=/tmp/xla_dump"

# visualize fusion decisions
export XLA_FLAGS="--xla_dump_hlo_graph_path=/tmp/xla_dump"

# disable specific optimizations
export XLA_FLAGS="--xla_disable_hlo_passes=fusion"

# CPU: enable LLVM IR dump
export XLA_FLAGS="--xla_dump_to=/tmp/xla --xla_dump_hlo_as_text \
                  --xla_dump_hlo_as_html --xla_dump_hlo_as_dot"

# GPU: profile kernel execution
export XLA_FLAGS="--xla_gpu_enable_reduction_epilogue_fusion=false"
```

### inspecting compiled code

```python
import jax


@jax.jit
def f(x):
  return jnp.sum(x**2)


# get HLO
x = jnp.ones(1000)
lowered = jax.jit(f).lower(x)
hlo_text = lowered.as_text()
print(hlo_text)

# output:
# HloModule jit_f
# ENTRY %main.4 (Arg_0.1: f32[1000]) -> f32[] {
#   %Arg_0.1 = f32[1000]{0} parameter(0)
#   %multiply.2 = f32[1000]{0} multiply(f32[1000]{0} %Arg_0.1, ...)
#   ROOT %reduce.3 = f32[] reduce(...), to_apply=%add
# }
```

visualize computation:

```python
# requires graphviz
lowered = jax.jit(f).lower(x)
print(lowered.as_text(dialect='hlo'))

# generate DOT graph (with XLA_FLAGS)
# produces fusion_*.dot files
```

### performance profiling

**JAX profiler:**

```python
import jax
import jax.profiler


@jax.jit
def train_step(params, batch):
  # training code
  return loss, grads


# profile execution
with jax.profiler.trace('/tmp/jax_profile'):
  for batch in dataset:
    loss, grads = train_step(params, batch)

# analyze with TensorBoard
# tensorboard --logdir /tmp/jax_profile
```

**nsys (NVIDIA Nsight Systems):**

```bash
nsys profile -o profile python train.py

# visualize kernel timeline, memory transfers, NCCL calls
nsys-ui profile.nsys-rep
```

### common pitfalls

**recompilation on every call:**

```python
# bad: shape changes every iteration
for i in range(100):
  x = jnp.ones(i * 10)  # different shape
  result = jax.jit(f)(x)  # recompiles 100 times!

# good: use padding for fixed shape
max_size = 1000
for i in range(100):
  x = jnp.ones(i * 10)
  x_padded = jnp.pad(x, (0, max_size - len(x)))
  result = jax.jit(f)(x_padded)  # compiles once
```

**mixing Python and JAX control flow:**

```python
# bad: Python if inside jit
@jax.jit
def f(x):
  if x.shape[0] > 100:  # concrete value needed, but x is traced
    return x.sum()
  return x.mean()


# good: use lax.cond
@jax.jit
def f(x):
  return lax.cond(x.shape[0] > 100, lambda x: x.sum(), lambda x: x.mean(), x)
```

**unnecessary device transfers:**

```python
# bad: result.copy() forces device → host transfer
for i in range(1000):
  result = jax.jit(f)(x)
  print(result.copy())  # slow! synchronizes every iteration

# good: accumulate on device, transfer once
results = []
for i in range(1000):
  results.append(jax.jit(f)(x))
results = jnp.stack(results)
print(results)  # single transfer
```

## production use cases

### Google

XLA originated at Google for TensorFlow compilation.

**TensorFlow serving:** XLA compiles SavedModels for inference

**Cloud TPU:** XLA is the only compilation path for TPU

**internal infrastructure:** large-scale training uses XLA for efficiency

### DeepMind

**AlphaFold:** JAX + XLA for protein structure prediction

scientific computing workloads benefit from fusion:

- many element-wise operations (normalization, attention)
- automatic differentiation through XLA
- multi-GPU/TPU scaling via SPMD

**scientific ML:** molecular dynamics, quantum chemistry simulations

### research and open source

**Stable Diffusion (JAX):** diffusion models compiled with XLA

faster iteration than PyTorch for research prototyping

**Hugging Face Transformers:** JAX/Flax models use XLA

**Google Research:** JAX is standard for ML research

publications using JAX increase exponentially (2020: ~10, 2024: ~1000+)

### commercial deployments

**Cohere:** large language model serving with JAX + XLA + TPU

**Anthropic:** Claude training infrastructure uses JAX

**Character.AI:** conversational AI inference with XLA optimization

## practical examples

### simple fusion example

```python
import jax
import jax.numpy as jnp


# define computation
def fused_ops(x, y, z):
  return jnp.sum(x + y * z)


# compile with XLA
jitted = jax.jit(fused_ops)

# test
x = jnp.ones(1000000)
y = jnp.ones(1000000) * 2
z = jnp.ones(1000000) * 3

result = jitted(x, y, z).block_until_ready()
# first call: traces and compiles (~100ms)
# subsequent calls: ~0.1ms

# inspect HLO
print(jitted.lower(x, y, z).as_text())
```

HLO output shows fusion:

```
HloModule jit_fused_ops

%fused_computation {
  %p0 = f32[1000000] parameter(0)
  %p1 = f32[1000000] parameter(1)
  %p2 = f32[1000000] parameter(2)
  %multiply = f32[1000000] multiply(%p1, %p2)
  %add = f32[1000000] add(%p0, %multiply)
  ROOT %reduce = f32[] reduce(%add, const_0), to_apply=%add_reduce
}

ENTRY %main {
  %Arg_0 = f32[1000000] parameter(0)
  %Arg_1 = f32[1000000] parameter(1)
  %Arg_2 = f32[1000000] parameter(2)
  ROOT %fusion = f32[] fusion(%Arg_0, %Arg_1, %Arg_2),
                       kind=kLoop, calls=%fused_computation
}
```

single fusion operation replaces 3 separate kernels.

### control flow with XLA

```python
import jax
from jax import lax


@jax.jit
def conditional_compute(x, mode):
  def l1_norm(x):
    return jnp.sum(jnp.abs(x))

  def l2_norm(x):
    return jnp.sqrt(jnp.sum(x**2))

  def linf_norm(x):
    return jnp.max(jnp.abs(x))

  # switch based on mode
  return lax.switch(mode, [l1_norm, l2_norm, linf_norm], x)


x = jnp.array([1.0, -2.0, 3.0, -4.0])

print(conditional_compute(x, 0))  # L1: 10.0
print(conditional_compute(x, 1))  # L2: 5.477
print(conditional_compute(x, 2))  # Linf: 4.0

# XLA compiles all branches
# runtime selection via switch opcode
```

### efficient cumulative operations with scan

```python
import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def efficient_cumsum(x):
  def scan_fn(carry, x_i):
    new_carry = carry + x_i
    return new_carry, new_carry

  _, cumsum = lax.scan(scan_fn, 0.0, x)
  return cumsum


# vs naive implementation
@jax.jit
def naive_cumsum(x):
  result = jnp.zeros_like(x)
  for i in range(len(x)):
    result = result.at[i].set(jnp.sum(x[: i + 1]))
  return result


x = jnp.arange(1000.0)

# efficient_cumsum: XLA optimizes scan primitive
# naive_cumsum: loop unrolling, O(n^2) operations

import time

# benchmark
for _ in range(10):  # warmup
  efficient_cumsum(x).block_until_ready()

start = time.time()
for _ in range(1000):
  efficient_cumsum(x).block_until_ready()
print(f'scan: {(time.time() - start) * 1000:.2f}ms')

# scan is 10-100x faster for sequential dependencies
```

### matrix multiplication with different layouts

```python
import jax
import jax.numpy as jnp


@jax.jit
def matmul_chain(A, B, C):
  # (A @ B) @ C
  return (A @ B) @ C


# XLA optimizes layout and operation order
A = jnp.ones((128, 256))
B = jnp.ones((256, 128))
C = jnp.ones((128, 512))

result = matmul_chain(A, B, C)

# XLA may reorder: A @ (B @ C) if more efficient
# layout optimization for cache locality
# calls optimized BLAS (cuBLAS on GPU)

lowered = jax.jit(matmul_chain).lower(A, B, C)
print(lowered.as_text())
# inspect optimization decisions
```

### custom gradient with checkpointing

```python
import jax
import jax.numpy as jnp
from jax import checkpoint


def expensive_forward(x, w1, w2, w3, w4):
  h1 = jax.nn.relu(x @ w1)
  h2 = jax.nn.relu(h1 @ w2)
  h3 = jax.nn.relu(h2 @ w3)
  h4 = jax.nn.relu(h3 @ w4)
  return jnp.sum(h4)


# without checkpointing
@jax.jit
def train_no_checkpoint(params, x):
  loss = expensive_forward(x, *params)
  grads = jax.grad(lambda p: expensive_forward(x, *p))(params)
  return loss, grads


# with checkpointing
@jax.jit
def train_checkpoint(params, x):
  @checkpoint
  def forward(p):
    return expensive_forward(x, *p)

  loss = forward(params)
  grads = jax.grad(forward)(params)
  return loss, grads


# memory usage:
# no checkpoint: stores h1, h2, h3, h4
# checkpoint: stores only inputs/outputs, recomputes in backward

x = jnp.ones((128, 512))
params = [jnp.ones((512, 512)) for _ in range(4)]

loss1, grads1 = train_no_checkpoint(params, x)
loss2, grads2 = train_checkpoint(params, x)

# same result, different memory/compute tradeoff
assert jnp.allclose(loss1, loss2)
```

### distributed data parallel training

```python
import jax
import jax.numpy as jnp
from jax import pmap, lax

# simulate multi-device environment
num_devices = jax.local_device_count()


@pmap
def data_parallel_step(params, batch):
  # each device processes local batch shard
  def loss_fn(p):
    preds = p['w'] @ batch['x'] + p['b']
    return jnp.mean((preds - batch['y']) ** 2)

  loss, grads = jax.value_and_grad(loss_fn)(params)

  # synchronize gradients across devices
  grads = lax.pmean(grads, axis_name='devices')

  return loss, grads


# replicate parameters across devices
params = {'w': jnp.ones((512, 128)), 'b': jnp.zeros(128)}
params = jax.tree_map(lambda x: jnp.stack([x] * num_devices), params)

# shard batch across devices
batch = {
  'x': jnp.ones((num_devices, 64, 512)),  # batch_size=64 per device
  'y': jnp.ones((num_devices, 64, 128)),
}

loss, grads = data_parallel_step(params, batch)

# XLA generates:
# 1. per-device forward/backward kernels
# 2. all-reduce (NCCL) for gradient sync
# 3. optimized communication overlap with computation

print(f'per-device loss: {loss}')
print(f'synchronized grads shape: {grads["w"].shape}')
# grads['w']: [num_devices, 512, 128]
```

## references

XLA documentation and specifications:

- XLA architecture: https://openxla.org/xla/architecture
- HLO reference: https://openxla.org/xla/operation_semantics
- StableHLO specification: https://openxla.org/stablehlo/spec
- PJRT C++ API: https://openxla.org/xla/pjrt/cpp_api_overview

JAX and compilation:

- JAX documentation: https://docs.jax.dev/
- JAX JIT compilation: https://docs.jax.dev/en/latest/jit-compilation.html
- JAX AOT lowering: https://docs.jax.dev/en/latest/aot.html

academic papers:

- "Compiling machine learning programs via high-level tracing" (Frostig et al., 2018): JAX design and XLA integration
- "XLA: Optimizing Compiler for Machine Learning" (Google, 2017): original XLA paper
- "Operator Fusion in XLA: Analysis and Evaluation" (Snider & Liang, 2023): fusion strategies and performance analysis

related projects:

- OpenXLA initiative: https://openxla.org/
- IREE compiler: https://iree.dev/
- Torch-MLIR: https://github.com/llvm/torch-mlir
