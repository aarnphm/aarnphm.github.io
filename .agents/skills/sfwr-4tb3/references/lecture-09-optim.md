---
name: lecture-09-optim
description: code optimization — three-address code, basic blocks, CSE, LICM, strength reduction, dead code, algebraic identities, constant folding
---

# lecture 09 — code optimization

"optimization" is a misnomer: it's just improvement in size, time, memory, or energy. no compiler is optimal for every use.

## intermediate rep: three-address code

every statement has at most three operands: $x := y\ \text{op}\ z$, $x := \text{op}\ y$, $x := y$, $\text{if}\ x\ \text{rel}\ y\ \text{goto}\ L$, $\text{goto}\ L$, $x := \text{call}\ p(y_1, \ldots)$, $\text{return}\ x$.

three-address code decouples optimization from the source and the target. machine-independent optimizations run on it; machine-dependent ones run later on target code.

## basic blocks and flow graphs

a **basic block** is a maximal straight-line sequence of three-address code: exactly one entry (the first instruction) and one exit (the last). leaders begin new basic blocks:

- the first instruction
- any jump target
- any instruction immediately after a jump

edges connect blocks that can flow into each other, forming the **control-flow graph** (CFG).

## common subexpression elimination (CSE)

represent the code as a directed acyclic graph (DAG) per basic block. while parsing left-to-right, look up each subexpression in a table. if present, reuse its number; else allocate a new one. two identical subtrees become one node.

example: $a \times b + a \times b \times c$ becomes $t_0 \times c + t_0$ where $t_0 = a \times b$.

a programmer-level equivalent is to introduce a temporary $t := a \times b$ and reuse it, but without CSE, the code will still load $a$, $b$ into registers each time.

CSE often fires in **generated** code, for example array indexing: $a[i][j]$ with $a : [0..9] \to [0..9] \to \text{integer}$ expands to $a + i \times 40 + j \times 4$, used three times in $a[i][j] := a[i][j] + 1$ style expressions.

## loop invariant code motion (LICM)

An invariant expression may be hoisted when doing so preserves effects and traps, including when the loop executes zero times. Check aliasing and writes as well as the variables named in the expression. For a pure, non-trapping expression, transform:

```
while i < n do
  x := a + b × c  // a, b, c don't change
  i := i + 1
```

into:

```
t := a + b × c
while i < n do
  x := t
  i := i + 1
```

**preheader** is the block before the loop where hoisted code lives.

## strength reduction

replace expensive ops with cheaper equivalents. classic examples:

| original                                   | replacement                                  |
| :----------------------------------------- | :------------------------------------------- |
| $x \times 2$                               | $x + x$ or $x \ll 1$                         |
| $x \times 2^n$                             | $x \ll n$                                    |
| $x \div 2^n$, unsigned or nonnegative $x$  | logical right shift by $n$                   |
| $x \bmod 2^n$, unsigned or nonnegative $x$ | $x\ \&\ (2^n - 1)$                           |
| $i \times 4$ in loop                       | maintain $t$ with $t := t + 4$ per iteration |

For negative signed operands, truncation toward zero differs from arithmetic right shift: `-3 / 2` truncates to `-1`, while shifting gives `-2`. A mask also does not preserve a signed remainder such as `-3 rem 2 = -1`. Check the language and target's rounding, overflow, and valid shift counts before applying these identities.

The induction-variable transformation replaces a multiply per iteration with an addition; preserve initialization, update order, and overflow behavior.

## dead code elimination

identities:

- $\text{if false then}\ S \equiv \text{skip}$
- $\text{if true then}\ S\ \text{else}\ T \equiv S$
- $\text{while false do}\ S \equiv \text{skip}$
- a basic block with no incoming edges is dead

dead code often arises from debug/logging code, conditional compilation, or machine-specific variants.

## algebraic identities and constant folding

**constant folding**: evaluate constant expressions at compile time. $3 + 4 \to 7$. the P0 compiler does this with arbitrary-precision integers during delayed code generation; at run-time, it uses machine integers. the subtlety: compile-time and run-time may disagree on overflow.

**algebraic identities**:

- $x + 0 = x$, $x \times 1 = x$, $x \times 0 = 0$
- $x \land \text{true} = x$, $x \lor \text{false} = x$, $x \lor \text{true} = \text{true}$
- $x \le x = \text{true}$, $x < x = \text{false}$
- $\lnot \lnot x = x$

Apply these identities only when the operand domain and evaluation effects permit them. Removing an expression may remove a trap, memory access, or procedure call; floating-point NaNs also invalidate identities such as $x \le x = \text{true}$.

## typical exam question shapes

1. "given this P0 program, show the three-address code" → convert statement-by-statement
2. "given this three-address code, identify basic blocks and draw the CFG" → mark leaders, group
3. "identify common subexpressions in this assignment" → look for syntactically identical subtrees
4. "apply LICM to this loop" → find loop invariants, hoist to preheader
5. "given this assignment, apply strength reduction" → spot $\times 2^n$, $\times \text{const}$ in loop, etc.
6. "constant-fold this expression" → evaluate compile-time constants

## interaction with code generation

CSE, LICM, and strength reduction all assume that subsequent code generation is **not smart enough** to notice these. a naive generator loads $a$, $b$, $c$ fresh each time. the optimizations rewrite the intermediate form so the generator sees the reduced expression.

## annotating risc-v for optimization questions

questions like "out of registers" / "common subexpressions in matrix operations" want you to:

1. show the naive codegen
2. count register pressure or redundant memory ops
3. propose the optimization (CSE, LICM, spill, rematerialization)
4. show the improved codegen
