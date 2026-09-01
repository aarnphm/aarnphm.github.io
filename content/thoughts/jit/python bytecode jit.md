---
date: '2025-10-05'
description: small Python AST to C compilers, an experimental SSA path, and the limits of their benchmark harness
id: python-bytecode-jit
modified: 2026-08-27 09:11:56 GMT-04:00
tags:
  - compilers
  - technical
title: small Python compiler
---

The name of this directory is historical. Its main compilers do not operate on Python bytecode, watch a running function, or promote hot code. They recover a function's source during decoration and eagerly compile a restricted Python AST to C.

There are two backends and one static dispatcher:

| file                         | actual role                                                          | trustworthy boundary                                                          |
| ---------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `minimal_jit.py`             | direct AST to C lowering                                             | simple scalar arithmetic and positive-step loops with an explicit C signature |
| `ir.py` and `ir_compiler.py` | experimental SSA-like IR, small optimization passes, then C lowering | straight-line arithmetic; loop-carried scalar values are currently wrong      |
| `compiler.py`                | one-time syntax-based choice between the two backends                | records which backend it chose; it does not profile or recompile              |

This makes the code useful for learning the boundary between source recovery, lowering, an intermediate representation, native compilation, and an ABI. Its scope ends at a compiler experiment.

## direct AST lowering

`TinyCJIT` lowers a small Python subset directly to C. Compilation happens when the decorator runs. Calling the decorated function later enters the compiled shared library through `ctypes`.

The pipeline is:

```text
Python source -> AST -> C source -> clang or gcc -> shared library -> ctypes wrapper
```

`inspect.getsource` and `ast.parse` recover the function. Two visitors emit C expressions and statements. The generated C is hashed for the cache, compiled with optimization enabled, loaded with `ctypes.CDLL`, and wrapped so NumPy arrays can be passed as pointers.

The C ABI is explicit. `TinyCJIT` does not infer parameter types. The caller supplies `restype` and `argtypes`, while named local temporaries are emitted as `float`. A correct SAXPY signature therefore distinguishes its scalar coefficient from its array pointers:

```python
from ctypes import POINTER, c_float, c_int

jit = TinyCJIT()


@jit(
  restype=None,
  argtypes=[
    c_float,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_float),
    c_int,
  ],
)
def saxpy(a, x, y, out, n):
  for i in range(n):
    out[i] = a * x[i] + y[i]
```

The generated function has the corresponding signature:

```c
void saxpy(float a, float* x, float* y, float* out, int n) {
  for (int i = 0; i < n; i += 1) {
    out[i] = (a * x[i]) + y[i];
  }
}
```

The accepted subset is narrower than Python. In particular, loop code always emits an increasing `index < stop` condition, so zero and negative `range` steps do not preserve Python semantics. The NumPy wrapper also trusts the declared element type. A mismatched pointer type crosses the ABI boundary without Python's usual safety checks and can crash or corrupt the process, as the [`ctypes` documentation](https://docs.python.org/3/library/ctypes.html) warns.

## the IR experiment

`IRCompiler` inserts an intermediate representation between the AST and C. The current route is:

```text
Python source -> AST -> basic blocks and SSA names -> small passes -> C -> shared library
```

The implementation demonstrates why an IR is useful. It gives constant folding and unused-result removal a representation that is easier to inspect than Python syntax. It also makes data-flow bugs visible.

For straight-line arithmetic, the path lowers expressions, propagates a limited set of integer, float, and pointer types, folds constants, removes some unused results, eliminates phi nodes, and emits C. The pass names sound broader than their current implementations. Type propagation handles only a few instruction classes, and dead-code elimination does not compute block reachability.

The loop path has a harder defect. The induction variable is carried through a phi node, while scalar reductions are not. A dot product can therefore return the initial accumulator or another wrong value because the accumulator is never carried through the loop edge. The source says this directly in `ir_compiler.py`, where loop-carried phis for reductions are disabled. Dot products, matrix-vector products, and matrix multiplication through this backend are currently incorrect.

That failure is the useful lesson. SSA construction has to represent every value that flows around a loop. Naming only the induction variable gives valid-looking IR and wrong programs.

## static strategy selection

`Compiler` parses the recovered source once and computes a syntax score:

$$
C(f) = 5L + 2K + B,
$$

where $L$ counts loop nodes, $K$ counts call nodes, and $B$ counts binary operations. The default threshold sends smaller scores to `TinyCJIT` and larger scores to `IRCompiler`.

This happens during decoration. There are no hotness counters, guards, runtime traces, on-stack replacement, or tier promotion. Decorator expressions also belong to the recovered syntax and can inflate the score. The score also ignores backend support, so it can route a reduction into a backend that miscompiles it.

## the nearby production reference

CPython's experimental JIT starts from its specializing adaptive interpreter. When a hot `JUMP_BACKWARD` or `RESUME` site enters tracing, CPython translates specialized bytecode into micro-ops, optimizes the trace, and installs an executor. That executor can run in the Tier 2 micro-op interpreter or, in a JIT-enabled build, as copy-and-patch machine code. The details live in CPython's current [JIT internals](https://github.com/python/cpython/blob/main/InternalDocs/jit.md) and [build documentation](https://docs.python.org/3/using/configure.html#cmdoption-enable-experimental-jit).

The contrast matters. Production tiering responds to observed execution and preserves a path back to the interpreter when assumptions fail. `Compiler` makes a static source-level choice before the function runs.

V8's current tiers provide another reference point: Ignition interprets bytecode, Sparkplug supplies a baseline compiler, and Maglev and TurboFan provide optimizing tiers. PyPy uses meta-tracing around its RPython interpreter [@bolz2009tracing]. Those systems are useful architecture references. They do not describe what this directory implements.

## evidence before speed

The existing `blas.py` harness prints timing winners without checking outputs. It also times already compiled wrappers for the eager C backends while Numba may still compile on its first call. Its results cannot support claims about compilation cost or runtime speed.

A valid harness needs to establish this order:

1. Compare every generated result with a trusted implementation under a stated tolerance.
2. Give mutating kernels fresh input and output arrays for every implementation.
3. Measure cold compilation in a fresh cache separately from warm calls.
4. Record CPU, compiler, flags, Python version, package versions, input sizes, and run counts.
5. Inspect generated assembly or compiler reports before attributing a result to vectorization.

Until that protocol passes, the benchmark only documents the correctness bug. A fast wrong reduction has a speedup ratio with no semantic content.

## directory map

- [[thoughts/jit/minimal_jit.py]] contains the eager AST to C decorator.
- [[thoughts/jit/ir.py]] and [[thoughts/jit/ir_compiler.py]] contain the experimental IR path.
- [[thoughts/jit/compiler.py]] contains the static syntax-based dispatcher.
- [[thoughts/jit/blas.py]] is a benchmark harness that needs correctness gates before its timings mean anything.
- [[thoughts/jit/tracing_jit.py]] extracts a loop from the AST and emits C. Despite its name, it has no runtime trace recorder, guards, or deoptimization.
- [[thoughts/jit/bytecodes.py]] contains one working bytecode-folding example alongside several conceptual or version-stale transformations.
- [[thoughts/jit/numba_jit.py]] is a comparison harness whose compilation timers also need repair.

The directory is best read beside [PEP 659](https://peps.python.org/pep-0659/) and CPython's JIT internals. Starkiller is also nearby history, though it was a static type inferencer and compiler rather than a runtime JIT [@salib2004starkiller].
