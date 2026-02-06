---
date: '2025-10-05'
description: building a JIT compiler in Python, from simple AST lowering to IR-based optimization
id: python-bytecode-jit
modified: 2025-11-09 01:38:09 GMT-05:00
tags:
  - compilers
  - technical
title: simple JIT compiler
---

see also: [@bolz2009tracing; @salib2004starkiller] for JIT in Python

three compilation strategies of increasing sophistication:

1. **TinyCJIT** (`minimal_jit.py`) — single-pass AST → C translation. 50-150ms compile, 20-80× runtime speedup on bandwidth-bound kernels
2. **IRCompiler** (`ir_compiler.py`) — multi-pass compilation through SSA IR with type inference and optimization passes. 200-500ms compile, enables constant folding and DCE
3. **Compiler** (`compiler.py`) — adaptive dispatch: complexity heuristic chooses TinyCJIT for simple functions, IRCompiler for complex ones

execution flow mirrors V8 (Ignition → TurboFan) and PyPy (tracing JIT): fast baseline tier, optimizing tier for hot code.

## tier 1: TinyCJIT — direct AST lowering

bypasses the interpreter by lowering restricted Python to C. single-pass translation with no intermediate representation.

### pipeline

```
Python AST → C code generation → clang/gcc → shared object → ctypes binding
```

compilation steps:

1. `inspect.getsource` + `ast.parse` to extract function AST
2. walk AST, emit C (restricted subset: for/range loops, array subscripts, binary ops)
3. hash source, cache `.so` in `/tmp/minimal_cjit/`
4. invoke `clang -O3 -fPIC -shared -lm`
5. `ctypes.CDLL` to load, wrap with numpy → pointer marshaling

decorator usage:

```python
from ctypes import POINTER, c_float, c_int

jit = TinyCJIT(verbose=True)


@jit(
  restype=None,
  argtypes=[POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int],
)
def vector_add(a, b, out, n):
  for i in range(n):
    out[i] = a[i] + b[i]
```

### code generation

two AST visitors emit C:

- `_ExprEmitter`: expressions → C syntax (`a[i] + b[i]` → `(a[i] + b[i])`)
- `_CBodyGenerator`: statements → C blocks, tracks locals

example:

```python
def saxpy(a, x, y, out, n):
  for i in range(n):
    out[i] = a * x[i] + y[i]
```

generates:

```c
void saxpy(float* a, float* x, float* y, float* out, int n) {
  for (int i = 0; i < n; i += 1) {
    out[i] = (a * x[i]) + y[i];
  }
}
```

type inference: scalar type inferred from first pointer argument (`float*` → `float`). locals discovered during traversal.

### performance

vector add (100k floats):

```
python loop:   12.4 ms
TinyCJIT:       0.18 ms
speedup:        68×
```

speedup sources: no interpreter dispatch, no reference counting, compiler vectorization (SSE/AVX), register allocation.

compile time: 50-150ms (invoke clang, hash-based caching).

vs Numba: comparable runtime (both produce native code), TinyCJIT 3-5× faster compile (C compiler vs LLVM). Numba has richer type inference and broader Python support.

### BLAS benchmarks

[[thoughts/JIT/blas.py]] compares TinyCJIT vs Numba on linear algebra kernels. bandwidth-bound ops (saxpy, dot): competitive. compute-bound (gemm): Numba wins via better vectorization.

## tier 2: IRCompiler — SSA-based optimization

TinyCJIT is fast but rigid: single-pass translation with no optimization. IRCompiler introduces an intermediate representation (IR) in SSA form, enabling type inference and optimization passes.

### pipeline

```
Python AST → IR lowering → type inference → optimization passes → IR → C → clang/gcc → .so
```

compilation steps:

1. `IRBuilder` lowers AST to SSA IR (basic blocks, phi nodes for loops)
2. `TypeInference` propagates types through SSA graph
3. `ConstantFolding` evaluates compile-time expressions
4. `DeadCodeElimination` removes unreachable instructions
5. `PhiElimination` converts out of SSA for C codegen
6. `CCodegen` emits C from IR
7. compile and bind via ctypes (same as TinyCJIT)

decorator usage identical to TinyCJIT:

```python
compiler = IRCompiler(verbose=True, optimize=True)


@compiler(restype=c_float, argtypes=[c_float, c_float])
def add_mul(a, b):
  tmp = a + b
  return tmp * 2.0
```

### IR design

SSA form with typed values:

```
entry:
  %const1 = const 2.0 : float
  %binop1 = add %a, %b : float
  %binop2 = mul %binop1, %const1 : float
  ret %binop2
```

phi nodes for loops:

```
loop_header:
  %iv1 = phi [%const2, entry], [%iv_next3, loop_body2] : int
  %cond1 = lt %iv1, %n : bool
  br %cond1, loop_body2, loop_exit3
```

### optimization passes

**constant folding**: `tmp = 2.0 * 3.0` → `tmp = 6.0` at compile time

**dead code elimination**: removes unreachable basic blocks and unused values

**type inference**: propagates `IRType.FLOAT`/`IRType.INT`/`IRType.PTR_FLOAT` through SSA graph

### performance

compile time: 200-500ms (IR passes add overhead vs TinyCJIT's 50-150ms).

runtime: comparable to TinyCJIT for simple kernels. optimizations help on constant-heavy code.

limitations: phi elimination for nested loops incomplete. reductions (dot product) may be incorrect. use TinyCJIT for complex control flow.

## tier 3: Compiler — adaptive dispatch

unified interface: chooses compilation strategy based on function complexity.

### complexity heuristic

AST visitor counts nodes:

- loops: +5
- calls: +2
- binary ops: +1

threshold (default 10): complexity < 10 → TinyCJIT, ≥ 10 → IRCompiler.

### usage

```python
jit = Compiler(mode='auto', verbose=True, complexity_threshold=10)


@jit(restype=c_float, argtypes=[c_float, c_float])
def simple_add(a, b):  # complexity ~2 → TinyCJIT
  return a + b


@jit(restype=c_float, argtypes=[c_float, c_float])
def complex_compute(a, b):  # complexity ~15 → IRCompiler
  tmp1 = a + b
  tmp2 = tmp1 * 2.0
  tmp3 = tmp2 - a
  tmp4 = tmp3 / b
  return tmp4 + a
```

modes:

- `mode='auto'`: heuristic-based (default)
- `mode='fast'`: always TinyCJIT
- `mode='optimized'`: always IRCompiler

compilation stats tracked in `jit.stats`:

```python
{'fast': 1, 'optimized': 1}
```

### design rationale

mirrors production JITs (V8, PyPy): fast baseline tier for quick startup, optimizing tier for hot code. adaptive dispatch amortizes compilation cost: simple functions get fast compile, complex functions get optimizations.

real-world strategy: profile at runtime, tier up on hot paths. this demo uses static complexity heuristic for simplicity.

## implementation

- [[thoughts/JIT/minimal_jit.py]] — TinyCJIT implementation
- [[thoughts/JIT/ir_compiler.py]] — IRCompiler with SSA IR
- [[thoughts/JIT/compiler.py]] — unified Compiler with adaptive dispatch
- [[thoughts/JIT/ir.py]] — IR data structures and optimization passes

## references

CPython internals: `Python/ceval.c` (eval loop), `Python/compile.c` (bytecode compiler), `Python/specialize.c` (PEP 659).

libraries: [bytecode](https://github.com/MatthieuDartiailh/bytecode), [codetransformer](https://github.com/llllllllll/codetransformer).

posts: [CPython Peephole Optimizer](https://akaptur.com/blog/2014/08/02/the-cpython-peephole-optimizer-and-you/), [Python 3.13 JIT](https://tonybaloney.github.io/posts/python-gets-a-jit.html), [Inline Caching](https://bernsteinbear.com/blog/inline-caching/).
