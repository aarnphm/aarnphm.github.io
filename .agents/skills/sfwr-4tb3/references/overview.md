---
name: overview
description: course architecture, P0 language, and compiler module layout
---

# overview

## course shape

sfwr 4tb3 builds a compiler for P0, a pascal-flavoured pedagogical language. the architecture is modular: each phase is its own jupyter notebook, imported via `%run`.

```
P0              the parser (entry point)
 ↳ SC           the scanner (getSym, KEYWORDS, IDENT, symbol classes)
 ↳ ST           the symbol table (newDecl, find, openScope, closeScope)
 ↳ CG           a code generator (one of CGast / CGwat / CGriscv / CGmips)
```

the parser type-checks via lookups in `ST` and calls `CG` procedures to emit code. swap `CG` for a different target and the same `P0` parser produces different output.

## P0 language (syntax, abridged)

declarations: $\texttt{const}\ c = e$, $\texttt{type}\ t = T$, $\texttt{var}\ x: T$, $\texttt{procedure}\ p(v: T) \to (r: U)$.

statements: assignment $x := e$, multi-assignment $x, y := e, f$, procedure-with-results $x \leftarrow p(a, b)$, `if`/`else`, `while`, sequencing `;`.

types: $\texttt{integer}$, $\texttt{boolean}$, arrays $[l..u] \to T$, records $(f_1: T_1, f_2: T_2)$, refs $\texttt{Ref}\ T$ (for reference parameters).

predefined identifiers: `integer`, `boolean`, `true`, `false`, `read`, `write`, `writeln`.

syntax is indentation-sensitive (like python). blocks are formed by indentation or `()`.

## compiler modules (files on disk)

each lecture/lab carries a recent snapshot. canonical source is under lecture 05, but labs and assignments include updated versions as features accrue:

| file            | role                       | key entries                                                                                                                                                                                                                        |
| :-------------- | :------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SC.ipynb`      | scanner                    | `getSym()`, `mark()`, `KEYWORDS`, `IDENT`, symbol constants                                                                                                                                                                        |
| `ST.ipynb`      | symbol table               | classes `Var`, `Ref`, `Const`, `Type`, `Proc`, `StdProc`, `Record`, `Array`; `newDecl`, `find`, `openScope`/`closeScope`                                                                                                           |
| `P0.ipynb`      | parser & type-checker      | `factor`, `term`, `simpleExpression`, `expression`, `statement*`, `typ`, `procedureDecl`, `program`                                                                                                                                |
| `CGast.ipynb`   | AST pretty-printer codegen | classes `UnaryOp`, `BinaryOp`, `Assignment`, `Call`, `Seq`, `IfThen`, `IfElse`, `While`, `ArrayIndexing`, `FieldSelection`; `genVar`, `genConst`, `genBinaryOp`, `genAssign`, `genCall`, `genProcStart/Entry/Exit`, `genLocalVars` |
| `CGwat.ipynb`   | wasm codegen               | same surface as `CGast`, emits WAT text                                                                                                                                                                                            |
| `CGriscv.ipynb` | risc-v codegen             | emits risc-v asm; `regs` is the available-register set in the lecture 08 snapshot                                                                                                                                                  |
| `CGmips.ipynb`  | mips codegen               | same surface, emits mips asm                                                                                                                                                                                                       |

## AST classes

(from `CGast`) — binary/unary op, assignment, call, sequence, if-then, if-else, while, array-indexing, field-selection. expressions return records with `.tp` (type) and a target-specific location. variables carry `.lev` (scope level) and an address/register in their symbol table entry.

## naming conventions in code

- $x$ often means "an expression or entry being processed"
- `reg`, $reg_1$, $reg_2$ are freshly-allocated registers (risc-v/mips)
- In the lecture 08 RISC-V generator, `SP = 'sp'` names the runtime stack pointer; `regs` contains available registers, removed by `obtainReg` and returned by `releaseReg`.
- `$name` prefixes indicate wasm locals/globals/functions
- `memsize` tracks statically-allocated memory during codegen
- `$_fp` is the local frame pointer; `$_memsize` is the global memory top

For grading weights or assessment rules, read the relevant syllabus or question. The local notebooks and their tests provide the implementation context; a lecture snapshot does not establish the requirements of a later assignment.
