---
date: '2025-10-05'
description: map of the Python AST, IR, bytecode, and benchmark experiments in this directory
id: bytecode-jit-readme
layout: L->ET|A
modified: 2026-08-27 09:11:56 GMT-04:00
tags:
  - seed
  - compilers
title: Python compiler experiments
---

This directory contains compiler experiments with three different objects: Python source syntax, a small custom IR, and CPython bytecode. The main path is eager Python AST to C compilation. See [[thoughts/jit/python bytecode jit|small Python compiler]] for the design and its correctness boundary.

## working map

| file             | purpose                                                                         | current status                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `minimal_jit.py` | recover source, lower a restricted AST to C, compile, and bind through `ctypes` | works for the demonstrated straight-line expressions and positive-step loops when the caller supplies the right C types |
| `ir.py`          | IR values, instructions, blocks, small analysis passes, and C printing          | educational; some pass names describe more than the implementation currently does                                       |
| `ir_compiler.py` | lower an AST through the custom IR and compile the resulting C                  | straight-line arithmetic works; loop-carried scalar reductions are wrong                                                |
| `compiler.py`    | choose a backend from a static AST complexity score                             | selection happens once during decoration; there is no runtime tiering                                                   |
| `tracing_jit.py` | extract a loop from the AST and compile it to C                                 | trace-shaped source transformation with no guards or deoptimization                                                     |
| `bytecodes.py`   | inspect and rewrite CPython bytecode                                            | constant folding works for the constructed example; several other demos are conceptual or stale on current opcodes      |
| `blas.py`        | compare generated numerical kernels with Numba                                  | timings are invalid until the harness rejects wrong outputs and separates cold compilation from warm calls              |
| `numba_jit.py`   | Numba comparison experiments                                                    | timing labels need the same cold and warm separation                                                                    |

## what the main path compiles

The direct compiler runs during decoration:

```text
Python source -> AST -> C -> system compiler -> shared library -> ctypes wrapper
```

The IR path inserts basic blocks, SSA-style names, and small optimization passes before C generation. Its current loop lowering carries the induction variable across iterations and fails to carry scalar accumulators. Any reduction routed through it can return a plausible wrong value.

`Compiler` uses the following static score:

$$
C(f) = 5L + 2K + B.
$$

Here $L$ is the number of loop nodes, $K$ the number of calls, and $B$ the number of binary operations in the recovered source. The score does not measure hotness or backend support.

## running the demonstrations

Use the repository's locked Python environment from its root:

```bash
uv run --frozen python content/thoughts/jit/minimal_jit.py
uv run --frozen python content/thoughts/jit/ir_compiler.py
uv run --frozen python content/thoughts/jit/compiler.py
uv run --frozen python content/thoughts/jit/bytecodes.py
```

`blas.py` remains useful as a bug reproducer. Its printed winners are not performance evidence because the harness does not compare outputs before timing them.
