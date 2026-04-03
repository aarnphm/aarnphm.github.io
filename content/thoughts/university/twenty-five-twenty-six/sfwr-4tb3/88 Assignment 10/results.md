---
date: '2026-03-31'
description: code optimization, common subexpression elimination, basic blocks
id: results
modified: 2026-03-31 16:58:55 GMT-04:00
tags:
  - sfwr4tb3
  - assignment
title: 'Assignment 10: Code Optimization'
---

## Q1: Basic Blocks of p2 [4 points]

The generated RISC-V code for `p2`:

```asm
	.data
x_:	.space 4
y_:	.space 4
	.text
	.globl main
main:
	jal ra, p2
	addi a0, zero, 0
	addi a7, zero, 93
	scall
	.globl p2
p2:
	addi sp, sp, -16
	sw ra, 12(sp)
	sw s0, 8(sp)
	addi s0, sp, 16
	addi s7, zero, 1
	la s10, x_
	sw s7, 0(s10)
L1:
	la s10, y_
	lw s7, 0(s10)
	addi s3, zero, 2
	bge s7, s3, L2
L3:
	la s10, x_
	lw s7, 0(s10)
	addi s3, s7, 3
	la s2, x_
	sw s3, 0(s2)
	j L1
L2:
	lw ra, 12(sp)
	lw s0, 8(sp)
	addi sp, sp, 16
	ret
```

There are five basic blocks, delineated by the labels that are targets of branch/jump instructions:

**Block 1** (`main:`): program entry, calls `p2` via `jal ra, p2`. The procedure call ends this block.

**Block 2** (after `jal`): system exit sequence (`addi a0, zero, 0; addi a7, zero, 93; scall`), executed after `p2` returns.

**Block 3** (`p2:`): procedure prologue (save `ra`, `s0`, set up frame pointer) and the statement `x := 1` (load constant 1, store to `x_`). This block ends where `L1` begins, since `L1` is a jump target.

**Block 4** (`L1:`): the while loop. This block contains both the condition check `y < 2` (load `y`, load 2, `bge` to `L2` if `y >= 2`) and the loop body `x := x + 3` (load `x`, add 3, store to `x_`), ending with `j L1`. The label `L3` appears between the condition and the body, but `L3` is _not_ the target of any branch instruction, so it does not start a new basic block. The conditional branch `bge` is permitted inside a basic block per the definition.

**Block 5** (`L2:`): procedure epilogue (restore `ra`, `s0`, deallocate stack frame) and `ret`.

In terms of source code: block 3 corresponds to `x := 1`, block 4 corresponds to the entire `while y < 2 do x := x + 3` (condition and body together), and block 5 is the implicit return at the end of the program.

---

## Q2: Annotating RISC-V Code [8 points]

The annotated for the generated RISC-V for:

```
var x, y, z: integer
program p1
  z := x + 3
  x := (x + y) × (x + 3) + (x + y)
```

```asm
p1:
	addi sp, sp, -16       ; prologue: allocate stack frame
	sw ra, 12(sp)          ; save return address
	sw s0, 8(sp)           ; save frame pointer
	addi s0, sp, 16        ; set frame pointer
	la s2, x_              ; (temp) load address of x
	lw s4, 0(s2)           ; s4 := x
	addi s11, s4, 3        ; s11 := x + 3
	la s6, z_              ; (temp) load address of z
	sw s11, 0(s6)          ; z := x + 3       (store s11 to z_)
	la s7, y_              ; (temp) load address of y
	lw s8, 0(s7)           ; s8 := y
	add s10, s4, s8        ; s10 := x + y
	mul s5, s10, s11       ; s5 := (x + y) × (x + 3)    [s11 reused — CSE!]
	add s3, s5, s10        ; s3 := s5 + (x + y)          [s10 reused — CSE!]
	la s9, x_              ; (temp) load address of x
	sw s3, 0(s9)           ; x := s3           (store result to x_)
	lw ra, 12(sp)          ; epilogue: restore return address
	lw s0, 8(sp)           ; restore frame pointer
	addi sp, sp, 16        ; deallocate stack frame
	ret                    ; return
```

**Common subexpression elimination** saves two computations:

1. `x + 3` is computed once (`addi s11, s4, 3`) and reused: first for storing to `z`, then as the right operand of the multiplication `(x + y) × (x + 3)`. Without CSE, a second `addi` instruction and a load of `x` would be needed.

2. `x + y` is computed once (`add s10, s4, s8`) and reused: first as the left operand of `(x + y) × (x + 3)`, then as the right operand of the final addition `... + (x + y)`. Without CSE, a second `add` and two loads (`x` and `y`) would be needed.

The entire program body is one basic block (no labels between `p1:` and the epilogue), so all subexpressions are candidates for sharing. The `reguse` dictionary tracks `('add', s4, 3) → s11` and `('add', s4, s8) → s10`, allowing the code generator to reuse these registers instead of recomputing.

---

## Q3: Out of Registers [4 points]

```python
compileString(
  """
var a, b, c, d, e, f: integer
program p
  a := a + b + c + d + e + f
""",
  target='riscv',
)
```

The compiler has 10 available registers (`s2`–`s11`). The expression `a + b + c + d + e + f` is parsed left-to-right as `((((a + b) + c) + d) + e) + f`:

| step | operation               | registers allocated        |
| ---- | ----------------------- | -------------------------- |
| 1    | load `a`                | 1                          |
| 2    | load `b`                | 2                          |
| 3    | `a + b`                 | 3                          |
| 4    | load `c`                | 4                          |
| 5    | `(a+b) + c`             | 5                          |
| 6    | load `d`                | 6                          |
| 7    | `((a+b)+c) + d`         | 7                          |
| 8    | load `e`                | 8                          |
| 9    | `(((a+b)+c)+d) + e`     | 9                          |
| 10   | load `f`                | 10                         |
| 11   | `((((a+b)+c)+d)+e) + f` | **11 — out of registers!** |

Because the P0 compiler keeps all subexpressions in registers for potential reuse (common subexpression elimination) and never frees registers within a basic block, each load and each arithmetic result permanently consumes a register. With 6 distinct variables and 5 addition results, 11 registers are needed, exceeding the available 10.

---

## Q4: Common Subexpressions in Matrix Operations [8 points]

### Explicit address calculation

For matrices `a`, `b`, `c` of type `[0..N-1] → [0..N-1] → integer`, `size(integer) = 4`, and row stride `N × 4`:

$$c[i][j] + a[i][k] \times b[k][j]$$

becomes:

$$*(adr(c) + i \times (N \times 4) + j \times 4) + *(adr(a) + i \times (N \times 4) + k \times 4) \times *(adr(b) + k \times (N \times 4) + j \times 4)$$

### DAG table with common subexpressions

| expression  | number |
| :---------- | :----- |
| `adr(c)`    | `$1`   |
| `i`         | `$2`   |
| `N × 4`     | `$3`   |
| `$2 × $3`   | `$4`   |
| `$1 + $4`   | `$5`   |
| `j`         | `$6`   |
| `4`         | `$7`   |
| `$6 × $7`   | `$8`   |
| `$5 + $8`   | `$9`   |
| `*$9`       | `$10`  |
| `adr(a)`    | `$11`  |
| `$11 + $4`  | `$12`  |
| `k`         | `$13`  |
| `$13 × $7`  | `$14`  |
| `$12 + $14` | `$15`  |
| `*$15`      | `$16`  |
| `adr(b)`    | `$17`  |
| `$13 × $3`  | `$18`  |
| `$17 + $18` | `$19`  |
| `$19 + $8`  | `$20`  |
| `*$20`      | `$21`  |
| `$16 × $21` | `$22`  |
| `$10 + $22` | `$23`  |

The corresponding three-address code:

```
$1 := adr(c)
$2 := i
$3 := N × 4
$4 := $2 × $3
$5 := $1 + $4
$6 := j
$7 := 4
$8 := $6 × $7
$9 := $5 + $8
$10 := *$9
$11 := adr(a)
$12 := $11 + $4
$13 := k
$14 := $13 × $7
$15 := $12 + $14
$16 := *$15
$17 := adr(b)
$18 := $13 × $3
$19 := $17 + $18
$20 := $19 + $8
$21 := *$20
$22 := $16 × $21
$23 := $10 + $22
```

The shared subexpressions are:

- **`$4 = i × (N × 4)`**: the row offset for index `i`, shared between `c[i][j]` and `a[i][k]` (both matrices are indexed by `i` in the first dimension)
- **`$8 = j × 4`**: the column offset for index `j`, shared between `c[i][j]` and `b[k][j]` (both indexed by `j` in the second dimension)
- **`$3 = N × 4`** and **`$7 = 4`**: the row stride and element size constants, shared across all three matrix accesses
- **`$2 = i`**, **`$6 = j`**, **`$13 = k`**: index variables loaded once and reused

Without common subexpression elimination, the expression would require computing `i × (N × 4)` twice, `j × 4` twice, and loading each index variable and constant multiple times. In the context of matrix multiplication (which evaluates this expression in a triply-nested loop), these savings are significant.
