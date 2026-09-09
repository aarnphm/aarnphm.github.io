---
name: lecture-06-stack-wasm
description: stack-based target (webassembly), translation schemes for expressions, statements, arrays, records
---

# lecture 06 — a stack architecture as target

## wasm in one breath

- stack architecture, no registers, operands pushed/popped
- byte code, block-structured (`if … end`, `block … end`, `loop … end`)
- statically typed: `i32`, `i64`, `f32`, `f64`. P0 uses `i32` everywhere
- four stores: code (immutable), memory (byte-addressable), global (typed values), stack (operands + call frames + labels)

## stores abstractly

$$
\begin{aligned}
c  &: [0..FN) \to \text{seq}(\text{byte}) &&\text{code} \\
m  &: [0..\text{MemSize}) \to \text{byte} &&\text{memory (pages of } 2^{16} \text{ bytes)} \\
g  &: [0..\text{GlobalVars}) \to \text{Value} &&\text{globals} \\
s  &: [0..\text{StackSize}) \to \text{Value} &&\text{stack} \\
sp &: [0..\text{StackSize}] &&\text{stack pointer}
\end{aligned}
$$

## instruction cheat-sheet (semantics as assignments)

| instruction                               | effect                                        | trap condition                                                               |
| :---------------------------------------- | :-------------------------------------------- | :--------------------------------------------------------------------------- |
| `i32.const i`                             | $s[sp], sp := i,\ sp+1$                       |                                                                              |
| `i32.add`                                 | $s[sp-2], sp := s[sp-2] + s[sp-1],\ sp-1$     |                                                                              |
| `i32.sub` / `i32.mul`                     | pop two, push result                          |                                                                              |
| `i32.div_s` / `i32.rem_s`                 | signed division / remainder                   | zero divisor; division also traps for minimum signed integer divided by $-1$ |
| `i32.eq`/`ne`/`lt_s`/`gt_s`/`le_s`/`ge_s` | comparisons push $1$ or $0$                   |                                                                              |
| `i32.eqz`                                 | $1$ if top is $0$, else $0$                   |                                                                              |
| `i32.load offset=n`                       | $s[sp-1] := m[s[sp-1]+n]$                     | bounds                                                                       |
| `i32.store offset=n`                      | $m[s[sp-2]+n] := s[sp-1];\ sp -= 2$           | bounds                                                                       |
| `local.get x` / `local.set x`             | frame-local r/w                               |                                                                              |
| `global.get x` / `global.set x`           | global r/w                                    |                                                                              |
| `local.tee x`                             | `local.set` but leaves the value on the stack |                                                                              |

These are abbreviated stack effects. For numeric traps and rounding, use the [WebAssembly numeric semantics](https://webassembly.github.io/spec/core/exec/numerics.html); remainder of the minimum signed integer by $-1$ is zero.

control:

- `block L … end` → `br L` jumps to the instruction after `end`
- `loop L … end` → `br L` jumps to the instruction after `loop`
- `if … else … end` → pops condition; $\ne 0$ picks `if`, else `else`
- `br_if L` → pops; jumps iff non-zero
- `call $f` → pushes args before, pops results after

## translation scheme: expressions

| $E$                              | $\text{code}(E)$                                                                |
| :------------------------------- | :------------------------------------------------------------------------------ |
| $x$ (local)                      | `local.get $x`                                                                  |
| $x$ (global)                     | `global.get $x`                                                                 |
| $n$                              | `i32.const n`                                                                   |
| $E_1\ \text{op}\ E_2$ arithmetic | $\text{code}(E_1);\ \text{code}(E_2);\ \texttt{i32.op}$                         |
| $-E$                             | `i32.const 0;` $\text{code}(E);$ `i32.sub`                                      |
| comparisons                      | $\text{code}(E_1);\ \text{code}(E_2);\ \texttt{i32.cmp}$                        |
| $\text{not}\ E$                  | $\text{code}(E);$ `i32.eqz` (fuse negation with comparison if possible)         |
| $E_1\ \text{and}\ E_2$           | $\text{code}(E_1);$ `if (result i32)` $\text{code}(E_2)$ `else i32.const 0 end` |
| $E_1\ \text{or}\ E_2$            | $\text{code}(E_1);$ `if (result i32) i32.const 1 else` $\text{code}(E_2)$ `end` |

## translation scheme: statements

| $S$                                                | $\text{code}(S)$                                                          |
| :------------------------------------------------- | :------------------------------------------------------------------------ |
| $x := E$                                           | $\text{code}(E);$ `set $x` (local or global)                              |
| $x_1, \ldots, x_n := E_1, \ldots, E_n$             | push all, then `set` them in reverse                                      |
| $x_1, \ldots, x_m \leftarrow p(E_1, \ldots, E_n)$  | push args, `call $p`, `set $x_m; … ; set $x_1`                            |
| $S_1; \ldots; S_n$                                 | $\text{code}(S_1); \ldots; \text{code}(S_n)$                              |
| $\text{if}\ E\ \text{then}\ S$                     | $\text{code}(E);$ `if` $\text{code}(S)$ `end`                             |
| $\text{if}\ E\ \text{then}\ S_1\ \text{else}\ S_2$ | $\text{code}(E);$ `if` $\text{code}(S_1)$ `else` $\text{code}(S_2)$ `end` |
| $\text{while}\ E\ \text{do}\ S$                    | `loop $L` $\text{code}(E);$ `if` $\text{code}(S);$ `br $L end end`        |

## declarations

| $D$                                            | $\text{code}(D)$                                                                                         |
| :--------------------------------------------- | :------------------------------------------------------------------------------------------------------- |
| $\texttt{var}\ x: \texttt{integer}$ (local)    | `(local $x i32)`                                                                                         |
| $\texttt{var}\ x: \texttt{integer}$ (global)   | `(global $x (mut i32) i32.const 0)`                                                                      |
| $\texttt{procedure}\ p(v: T) \to (r: U)\ D\ S$ | `(func $p (param $v i32) (result i32) (local $r i32)` $\text{code}(D);\ \text{code}(S);$ `local.get $r)` |

## program skeleton

```wat
(module
  (import "P0lib" "write" (func $write (param i32)))
  (import "P0lib" "writeln" (func $writeln))
  (import "P0lib" "read" (func $read (result i32)))
  code(D_1)
  (func $program
    code(D_2)
    code(S)
  )
  (memory 1)
  (start $program)
)
```

## arrays

global arrays are consecutive in memory. `x.adr` stored in the symbol table. with $A = [l..u] \to T$:

- `var x: A` allocates $\text{size}(A) = (u-l+1) \times \text{size}(T)$ bytes, stores `x.adr := memsize; memsize += size(A)`
- array indexing $x[E]$:
  ```wat
  code(E)
  i32.const l          ;; x.lower
  i32.sub
  i32.const size(T)
  i32.mul
  code(x)              ;; base address
  i32.add
  i32.load
  ```
- array assignment $x[E] := F$ is the same pattern with $\text{code}(F);$ `i32.store` at the end.

local arrays use a runtime frame pointer `$_fp` and a global `$_memsize`:

```wat
(local $x i32)
(local $_fp i32)
global.get $_memsize
local.set $_fp
global.get $_memsize
local.tee $x
i32.const size(A)
i32.add
global.set $_memsize
;; body
local.get $_fp
global.set $_memsize   ;; epilogue: restore memsize
```

`local.tee` is the trick to set `$x` and leave the value on the stack for the subsequent `i32.add`.

## records

consecutive in memory like arrays, but fields have a static $\text{offset}(f)$ relative to record base. $x.f$ emits:

```wat
code(x)              ;; address of the record
i32.const offset(f)
i32.add
```

then `i32.load` or `i32.store` as needed.

## array parameters and aliasing

only a pointer is passed. arrays-passed-by-value are local constants; must be copied if the callee wants to mutate. the current P0 compiler permits aliasing; calls like `q(x, x)` can silently share state. `memory.copy` copies $n$ bytes from address $s$ to $d$ (takes $d, s, n$ on the stack).

## running wasm from a notebook

three hosts shown:

- browser via `runwasm` (JS `WebAssembly.compile` + `.instantiate`)
- `pywasm` for pure-python interpretation
- `wasmtime` for native compilation via linker

all three import `P0lib.write`, `P0lib.writeln`, `P0lib.read` and expect `(start $program)` in the module.

## quick wat tool reference

- `wat2wasm file.wat` → `file.wasm` (binary)
- `wasm2wat file.wasm` → textual, but locals and functions become numeric indices; labels become `@1`, `@2` nesting depth
