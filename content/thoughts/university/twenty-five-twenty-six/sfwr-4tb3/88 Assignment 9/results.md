---
date: '2026-03-17'
description: circa algebraic optimization
id: results
modified: 2026-03-24 20:07:10 GMT-04:00
tags:
  - sfwr4tb3
  - assignment
title: code optimization
---

## A1

3 procedures across 2 files:

**P0.ipynb** (parser):

- `term()`: the `else` branch (where operands are not both constants) calls `CG.genBinaryOp(op, x, y)` unconditionally. modified to check:
  - if `op == TIMES` and one operand is `Const` with value `1`, use the other operand directly (skip the multiply)
  - if `op == DIV` and `y` is `Const` with value `1`, use `x` directly (skip the divide)
  - if `op == TIMES` and one operand is `Const` whose value is a power of 2, compute the shift amount (`log2`) and call a shift-left code generation instead of `genBinaryOp(TIMES, ...)`

- `simpleExpression()`: the `else` branch similarly calls `CG.genBinaryOp(op, x, y)` unconditionally. modified to check:
  - if `op == PLUS` and one operand is `Const` with value `0`, use the other operand directly (skip the add)
  - if `op == MINUS` and `y` is `Const` with value `0`, use `x` directly (skip the subtract)

**CGwat.ipynb** (code generator):

- `genBinaryOp()`: does not originally support shift left (`i32.shl`). modified to handle a new shift-left case, emitting `i32.shl` when invoked with a shift-left operation. alternatively, a new procedure (e.g., `genShiftLeft(x, n)`) can be introduced that loads `x`, pushes the shift amount as `i32.const n`, and emits `i32.shl`.

the optimizations are algebraic identities:

| expression       | identity                | optimization                            |
| ---------------- | ----------------------- | --------------------------------------- |
| `x + 0`, `0 + x` | additive identity       | omit `i32.add`                          |
| `x - 0`          | subtractive identity    | omit `i32.sub`                          |
| `x × 1`, `1 × x` | multiplicative identity | omit `i32.mul`                          |
| `x div 1`        | division identity       | omit `i32.div_s`                        |
| `x × 2^n`        | strength reduction      | replace `i32.mul` with `i32.shl` by `n` |

optimizations 1 and 2 (identities for 0 and 1) are handled purely in P0, because the parser already has access to `Const` nodes and can skip the `CG.genBinaryOp` call when the identity element is detected. optimization 3 (shift left) requires CGwat to emit `i32.shl`, which it did not previously support in `genBinaryOp`.

for detecting powers of 2: a value `v` is a power of 2 iff `v > 0 and (v & (v - 1)) == 0`. the shift amount is `v.bit_length() - 1`.

## A2

**Part 1 - Why does P0 generate different code for the three assignments?**

The P0 parser evaluates expressions left-to-right following the grammar's associativity. In `simpleExpression()`, the `while` loop processes additive operators left-to-right. Constant folding in P0 only triggers when BOTH operands of a binary operation are `Const` (the check `type(x) == Const == type(y)`).

- `y := x + 3 + 4`: parsed as `(x + 3) + 4`. The first operand `x` is `Var`, `3` is `Const`. Since they're not both `Const`, `CG.genBinaryOp(PLUS, x, 3)` emits `i32.add`. The result is a stack `Var`, so `(result) + 4` also emits `i32.add`. Two additions generated.

- `y := 3 + 4 + x`: parsed as `(3 + 4) + x`. Both `3` and `4` are `Const`, so constant folding fires: `x.val = 3 + 4 = 7`. Then `7 + x` where `7` is `Const` and `x` is `Var`, not both `Const`, so one `genBinaryOp` call. One addition generated with `i32.const 7`.

- `y := 3 + x + 4`: parsed as `(3 + x) + 4`. `3` is `Const`, `x` is `Var`, not both `Const`, so `genBinaryOp` emits `i32.add`. Then `(result) + 4` also emits `i32.add`. Two additions generated.

The key: constant folding is a peephole optimization that only applies when both operands are compile-time constants. Left-to-right evaluation means only `3 + 4 + x` gets the two constants adjacent in the parse tree.

**Part 2 - Can compilers optimize the first and third assignments to add 7?**

**Strict arithmetic** (overflow is an error): yes. Since both constants 3 and 4 are positive, the overflow behavior is preserved. If `x + 3` overflows (positive), then `x + 7` also overflows. If `x + 3` does not overflow but `(x + 3) + 4` does, then `x + 7` also overflows (since the mathematical sum is the same). The optimization would NOT be valid in general if the constants had different signs (e.g., `x + MAX_INT + (-1)` could overflow at the first step while `x + (MAX_INT - 1)` might not).

**Modulo arithmetic** (wraparound, no overflow error): yes. Two's complement addition is associative: `(x + 3) + 4 ≡ x + (3 + 4) ≡ x + 7 (mod 2^n)` for any x. The optimization is always valid with modulo arithmetic regardless of the constants' signs.

## A3

This is a code question (WAT hand-translation). The answer is in the notebook cell.

## A4

Given: `a = 1`, `b = 3`, `procedure q(x: integer): x := a + 1; b := a`, call `q(a)`.

**Call by value:** x is a local copy of a's value. x = 1 (copy). Execute: `x := a + 1 = 1 + 1 = 2` (only x changes, a stays 1). `b := a = 1`. Final: **a = 1, b = 1**.

**Call by result:** x is an uninitialized local variable. On return, x is copied to the actual parameter a. Execute: `x := a + 1 = 1 + 1 = 2` (a still 1). `b := a = 1`. On exit: `a := x = 2`. Final: **a = 2, b = 1**.

**Call by reference:** x is an alias for a (same memory location). Execute: `x := a + 1 = 1 + 1 = 2`, but since x IS a, this sets a = 2. `b := a = 2`. Final: **a = 2, b = 2**.

|              | a   | b   |
| ------------ | --- | --- |
| by value     | 1   | 1   |
| by result    | 2   | 1   |
| by reference | 2   | 2   |

The difference between result and reference: with call by result, changes to x during execution don't affect a until return. With call by reference, changes to x immediately affect a, which is why `b := a` sees the updated value (2) under call by reference but the original value (1) under call by result.

## A5

Given: `procedure sum(i, l, h, x: integer): s := 0; i := l; while i < h do s := s + x; i := i + 1`

Call: `sum(a, 1, n, a × a)` with all parameters by name.

By the definition of call by name, each formal parameter is textually replaced by the actual parameter expression, evaluated in the caller's scope each time it is referenced. Substituting i → a, l → 1, h → n, x → a × a:

```
s := 0
a := 1
while a < n do
  s := s + a × a
  a := a + 1
```

The key: `x` is replaced by `a × a`, and since `a` changes each iteration (via `i := i + 1` becoming `a := a + 1`), the value of `x` changes dynamically.

Trace:

- a = 1: s = 0 + 1×1 = 1, then a = 2
- a = 2: s = 1 + 2×2 = 5, then a = 3
- a = 3: s = 5 + 3×3 = 14, then a = 4
- ...
- a = n-1: s = ... + (n-1)², then a = n
- Loop exits when a = n.

Final value of s = 1² + 2² + 3² + ... + (n-1)² = (n-1)·n·(2n-1) / 6.
