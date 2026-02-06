---
date: '2025-10-05'
description: educational JIT compiler implementations in Python
id: bytecode-jit-readme
layout: L->ET|A
modified: 2026-01-02 02:21:33 GMT-05:00
tags:
  - seed
  - compilers
title: python JIT compiler
---

three-tier JIT compiler demonstrating compilation strategies from simple to sophisticated. see [[thoughts/JIT/python bytecode jit|python bytecode jit]] for detailed writeup.

## tier 1: TinyCJIT (`minimal_jit.py`)

single-pass AST → C translation. 50-150ms compile, 20-80× runtime speedup.

```python
jit = TinyCJIT()


@jit(
  restype=None,
  argtypes=[POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int],
)
def vector_add(a, b, out, n):
  for i in range(n):
    out[i] = a[i] + b[i]
```

## tier 2: IRCompiler (`ir_compiler.py` + `ir.py`)

multi-pass compilation through SSA IR. 200-500ms compile, enables optimizations (constant folding, DCE, type inference).

```python
compiler = IRCompiler(verbose=True, optimize=True)


@compiler(restype=c_float, argtypes=[c_float, c_float])
def add_mul(a, b):
  return (a + b) * 2.0
```

## tier 3: Compiler (`compiler.py`)

adaptive dispatch: complexity heuristic chooses TinyCJIT (simple) or IRCompiler (complex). mirrors V8/PyPy tiered compilation.

```python
jit = Compiler(mode='auto', complexity_threshold=10)


@jit(restype=c_float, argtypes=[c_float, c_float])
def compute(a, b):
  return a + b  # auto-selects TinyCJIT
```

## benchmarks (`blas.py`)

linear algebra kernels: saxpy, dot, gemv, gemm. compares TinyCJIT/IRCompiler/Compiler vs Numba.

findings: Compiler competitive on bandwidth-bound (saxpy), Numba wins on reductions (dot/gemv via LLVM), Compiler wins on blocked gemm (cache tiling).

## demos

run from `content/thoughts/jit/`:

```bash
python minimal_jit.py    # TinyCJIT vector add demo
python ir_compiler.py    # IRCompiler demo
python compiler.py       # adaptive Compiler demo
python blas.py          # BLAS benchmarks
```

## utilities

**tracing_jit.py**: trace-based JIT with loop detection, guard insertion, deoptimization.

**bytecodes.py**: bytecode manipulation examples (constant folding, DCE, inlining).

**numba_jit.py**: Numba comparison baseline.

```bash
python tracing_jit.py    # tracing JIT demo
python bytecodes.py      # bytecode transformations
```
