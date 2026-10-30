---
date: '2022-12-23'
description: tensor compiler for CPU, GPU and TPU execution, with notes on fusion, JAX staging and performance measurement.
id: XLA
modified: 2026-10-30 09:05:44 GMT-04:00
seealso:
  - '[[thoughts/PJRT|PJRT]]'
  - '[[thoughts/MLIR|MLIR]]'
  - '[[thoughts/PyTorch|PyTorch]]'
  - '[[thoughts/Compiler|Compiler]]'
  - '[[thoughts/Autograd|autograd]]'
  - '[[thoughts/GPU programming|GPU programming]]'
tags:
  - ml
  - compilers
title: XLA
---

XLA (Accelerated Linear Algebra) compiles tensor computations for CPUs, GPUs, and TPUs. In [[thoughts/Jax|JAX]], `jit` stages a region of a program for compilation. This gives XLA several operations to optimize together: it can combine kernels, choose memory layouts, and reuse storage once values are dead. The executable still has to move data and perform the arithmetic. [Architecture](https://openxla.org/xla/architecture).

## architecture

### compilation pipeline

For a JAX function, the stages are:

1. **Trace.** JAX runs Python with tracers that describe input properties, usually shapes and dtypes, and records a jaxpr.
2. **Lower.** JAX converts that computation into StableHLO.
3. **Compile.** XLA optimizes the computation and generates an executable for the selected backend.
4. **Execute.** The runtime supplies device buffers and launches the executable.

These stages can be inspected separately. Lowered StableHLO shows what was passed to the compiler; optimized executable text shows later decisions. Neither is a timing measurement. [JAX AOT API](https://docs.jax.dev/en/latest/aot.html).

### HLO: high-level operations

HLO represents tensor operations such as element-wise arithmetic, reductions, matrix products, and control flow. A logical tensor value does not prescribe a fresh physical allocation. Buffer assignment can reuse storage when lifetimes permit it. [HLO operation semantics](https://openxla.org/xla/operation_semantics).

### StableHLO

StableHLO is the versioned operation set at the framework/compiler boundary. It uses [[thoughts/MLIR|MLIR]] and specifies operations, types, and execution semantics. Its portability contract concerns the represented computation; runtime support and backend capabilities still determine where an exported program can run. [Specification](https://openxla.org/stablehlo/spec).

## fusion

Consider this reduction over equal-length vectors:

$$
s = \sum_{i=1}^{n}(x_i+y_i z_i).
$$

A separate element-wise multiply and add could each write a temporary vector. A fused implementation can compute intermediate values inside a kernel, avoiding those full-vector writes and reads. On a GPU this can also reduce launch overhead. XLA's GPU backend uses fusion, generated kernels, and library calls. The backend and input shape determine which implementation it selects. [GPU architecture](https://openxla.org/xla/gpu_architecture).

```python
import jax
import jax.numpy as jnp


def compute(x, y, z):
  return jnp.sum(x + y * z)


x = jnp.arange(8, dtype=jnp.float32)
y = jnp.ones_like(x)
z = jnp.full_like(x, 2)

lowered = jax.jit(compute).lower(x, y, z)
print(lowered.as_text())
compiled = lowered.compile()
print(compiled.as_text())
print(compiled(x, y, z).block_until_ready())  # 44.0
```

The example exposes the compiler output for this machine. Counting source operators does not establish a kernel count. A reduction may need multiple stages, and fusion can increase register pressure or duplicate computation. Inspect the optimized program and a device trace before claiming that fusion helped.

## core optimization passes

The useful questions concern where bytes and values go:

- **Algebraic simplification:** which operations can disappear? Floating-point rewrites can change rounding, so compare numerical results at the precision the application needs. [JAX numerical FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs).
- **Layout assignment:** how are logical dimensions stored? A transpose can sometimes become a layout choice; incompatible consumers can require a physical copy.
- **Buffer assignment:** which dead values leave storage that another result can reuse?
- **Rematerialization:** would recomputing a value reduce peak live memory enough to justify the extra work?

These are coupled decisions: eliminating an intermediate changes its lifetime, and changing a layout affects the kernels that can consume it. [Compiler architecture](https://openxla.org/xla/architecture), [GPU layout and scheduling](https://openxla.org/xla/gpu_architecture).

## JAX integration

### tracing mechanism

Python side effects occur while tracing. Ordinary `print` and mutation of a Python list do not become repeated device operations. Keep transformed functions pure and return state explicitly. JAX provides separate callback and debugging APIs when an effect is required. [JIT tracing](https://docs.jax.dev/en/latest/jit-compilation.html).

### staged compilation

A compiled specialization depends on input properties and compilation settings. Changing shapes, dtypes, or static arguments can require another compilation. Define the jitted function outside a hot loop so its cache can be reused. A cache hit still executes the computation; cached compilation does not mean cached outputs. [JIT caching](https://docs.jax.dev/en/latest/jit-compilation.html#jit-and-caching).

### control flow primitives

A Python `if` needs a concrete boolean during tracing. An input-dependent scalar predicate therefore needs a staged control-flow operation such as `lax.cond`. Both branches are traced and must return compatible structures, shapes, and dtypes. [Control flow](https://docs.jax.dev/en/latest/control-flow.html).

```python
@jax.jit
def piecewise(x):
  return jax.lax.cond(x < 3, lambda t: 3.0 * t**2, lambda t: -4.0 * t, x)


print(piecewise(jnp.float32(2)))  # 12.0
print(piecewise(jnp.float32(4)))  # -16.0
```

A Python loop with a fixed trip count can be unrolled during tracing, enlarging the compiled graph. `lax.scan` represents a fixed-length loop with a carry whose shape and dtype stay fixed. That can reduce compilation size. Its runtime advantage depends on the workload; an unrolled linear scan still performs linear work.

### dynamic shapes

JAX export supports symbolic dimensions so one exported program can cover a family of shapes. This saves repeated tracing and lowering. The exported program can still compile separately for each concrete shape. Shape-polymorphic export therefore gives no general promise of one machine executable for every size. [Shape polymorphism](https://docs.jax.dev/en/latest/export/shape_poly.html).

### autodiff through XLA

JAX applies automatic-differentiation transformations before the resulting computation is compiled. For example, `jax.jit(jax.grad(loss))` stages the gradient computation. Choosing forward or reverse mode belongs to the JAX transformation being requested. XLA then optimizes the resulting program.

`jax.checkpoint` controls which autodiff residuals are saved and which are recomputed during the backward pass. The memory and arithmetic tradeoff depends on the checkpoint boundary and the function. [Gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html).

## backend code generation

The backend chooses target-specific implementations, including generated code and optimized library calls. CPU, GPU, and TPU executables have different schedules and memory constraints. A source expression alone cannot tell us which kernel, library, or communication algorithm ran.

[[thoughts/PJRT|PJRT]] exposes device discovery, buffers, compilation, and executable invocation through a common API. It is a runtime interface; supporting PJRT does not mean that a plugin must use XLA internally. [PJRT overview](https://openxla.org/xla/pjrt/cpp_api_overview).

## performance characteristics

### compilation overhead

Suppose compilation takes time $C$, a warmed compiled call takes $t_c$, and a reference call takes $t_r$. Ignoring other startup costs and assuming these times stay fixed, compilation pays for itself after $N$ calls when

$$
C + Nt_c < Nt_r.
$$

If $t_r>t_c$, this requires $N>C/(t_r-t_c)$. If the compiled call is slower, repeated calls cannot recover the compilation cost under this model. Measure these quantities for the same workload, dtype, and device.

### performance profiling

JAX dispatch can return before device work finishes. Timing a Python call without synchronization can measure dispatch alone. Place inputs on the target device before timing when transfers are outside the intended measurement. Separate first-call latency from warmed calls. [Benchmarking JAX](https://docs.jax.dev/en/latest/benchmarking.html), [asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html).

This follows the `compute` example above. It measures completed calls on already-created inputs:

```python
from statistics import median
from time import perf_counter

jax.block_until_ready((x, y, z))
run = jax.jit(compute)

start = perf_counter()
run(x, y, z).block_until_ready()
first_call_seconds = perf_counter() - start

samples = []
for _ in range(20):
  start = perf_counter()
  run(x, y, z).block_until_ready()
  samples.append(perf_counter() - start)

print({
  'first_call_seconds': first_call_seconds,
  'warm_median_seconds': median(samples),
  'backend': jax.default_backend(),
  'jax_version': jax.__version__,
})
```

The first call may benefit from caches populated earlier in the process. Use a fresh process and state the cache policy for a cold measurement. A vector with eight elements checks the example's behavior; realistic performance work needs representative sizes and a reference implementation.

## advanced features

Sharding places parts of arrays on different devices. A partitioned computation may need collectives to supply missing data or combine partial results. The sharding and operation determine the communication; selecting a device mesh alone does not define an efficient pipeline schedule. [GPU partitioning](https://openxla.org/xla/gpu_architecture).

For an external C or CUDA implementation, JAX's FFI requires a registered native target, result metadata, and a compatible calling convention. Differentiation and batching need appropriate rules. A Python placeholder for a custom operation supplies none of the native implementation. [FFI guide](https://docs.jax.dev/en/latest/ffi.html).

## debugging and introspection

Start with `jax.make_jaxpr(compute)(x, y, z)`, then inspect the lowering and executable text from the example. Use a runtime profile to locate kernels, copies, synchronization, and collectives. Compiler cost analysis gives estimates; a runtime trace records what executed.

## references

- [XLA source](https://github.com/openxla/xla) and [OpenXLA](https://openxla.org/).
- [JAX documentation](https://docs.jax.dev/) and [AOT inspection](https://docs.jax.dev/en/latest/aot.html).
- Other compiler projects: [IREE](https://iree.dev/) and [Torch-MLIR](https://github.com/llvm/torch-mlir).
