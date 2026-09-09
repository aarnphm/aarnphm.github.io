---
name: lecture-07-datatypes
description: further data types (floats, sets, records) and control structures (exceptions, indirect calls, jump tables) on wasm
---

# lecture 07 — further data types and control structures

## floating-point numbers

wasm has `f32`/`f64` with IEEE-754 semantics. instructions mirror integer arithmetic: `f64.add`, `f64.sub`, `f64.mul`, `f64.div`, comparisons `f64.eq/ne/lt/gt/le/ge`, conversions `f64.convert_i32_s`, `i32.trunc_f64_s`. Floating-point comparisons do not have signed/unsigned variants. With NaN, equality and ordered comparisons return false; `ne` returns true. See the [WebAssembly numeric semantics](https://webassembly.github.io/spec/core/exec/numerics.html) for instruction-specific behavior.

grammar extensions for P0 float literals: $[\text{int}]\ \texttt{'.'}\ \text{num}\ [\texttt{'e'}\ \text{int}]$ etc. the scanner emits a dedicated float token.

## sets

bitset representation: $\text{set}\,[l..u]$ with $u - l + 1 \le 32$ packs into a 32-bit word. membership $e \in x$ → test bit $(e - l)$ of $x$. union/intersection/difference are `i32.or`, `i32.and`, `i32.and` with negation. singleton $\{e\}$ → `1 << (e - l)`. $x + \{e\}$ / $x - \{e\}$ lower to or/and with mask.

for larger sets, span multiple words and lift each op bitwise.

## bit operations and `tee`

wasm has `i32.and`, `i32.or`, `i32.xor`, `i32.shl`, `i32.shr_s`, `i32.shr_u`. `local.tee $x` pops the top, writes to `$x`, and leaves the value on the stack — used to set and reuse a value in one shot, especially for allocating on the dynamic memory stack.

## records, revisited

same memory layout as arrays: consecutive fields, field offsets computed statically. $x.f$ adds $\text{offset}(f)$ to the base address. records nested inside arrays: $x[i].f$ multiplies index by $\text{size}(\text{record})$, then adds $\text{offset}(f)$.

## exceptions (see `exception.wat`)

wasm's exception proposal adds `try … catch … end` and `throw $tag`. in the course, exceptions are implemented either via the proposal or by explicit `block`/`br_if` unrolling:

```wat
(func $f
  try
    call $mayThrow
  catch $exn
    ;; handler
  end)
```

the P0 translation for $\texttt{try}\ S\ \texttt{except}\ \text{handler}$ pushes a handler label, executes $S$, and on `throw` branches to the matched `catch`.

## indirect calls (see `indirect.wat`)

`call_indirect (type $sig)` lets you call a function by table index. useful for virtual dispatch, function pointers, and polymorphic callbacks. requires `(table funcref …)` and `(elem (i32.const idx) $f1 $f2 …)` to populate. the function value on the stack is an `i32` index into the table.

## jump tables (see `jumptable.wat`)

`br_table L_0 L_1 … L_n L_d` pops an index and branches to the label at that index, falling back to the default $L_d$. used to compile `case` statements efficiently:

```wat
(block $default
  (block $c2 (block $c1 (block $c0
    local.get $x
    br_table $c0 $c1 $c2 $default
  ) code_for_case_0 br $default)
    code_for_case_1 br $default)
      code_for_case_2 br $default)
  code_for_default)
```

## infinite loops

`infiniteloop.txt` / `verylongloop.txt` exist to show that wasm validates but doesn't prove termination — the host environment needs a timeout or fuel mechanism (like `wasmtime`'s epoch interruption or the browser's long-task kill).
