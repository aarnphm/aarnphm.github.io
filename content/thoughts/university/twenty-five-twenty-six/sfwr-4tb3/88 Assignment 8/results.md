---
date: '2026-03-17'
description: RISC-V
id: results
modified: 2026-03-17 23:00:42 GMT-04:00
tags:
  - sfwr4tb3
  - assignment
title: fermat and seed parser
---

## A1

### seed() implementation

the `seed` function returns the millisecond portion of the current time. this keeps the value small enough that `a × r` (where `a = 16807`) won't overflow a 32-bit signed integer on the first iteration: $16807 \times 999 = 16{,}790{,}193 < 2^{31}$.

```python
def seed(_: Machine, args: list[int]) -> list[int]:
  from datetime import datetime

  return [datetime.now().microsecond // 1000]
```

### primality.wat

```wasm
;;  var r: integer
;;  (global variable for random number generator state)
(module
(import "P0lib" "write" (func $write (param i32)))
(import "P0lib" "writeln" (func $writeln))
(import "P0lib" "read" (func $read (result i32)))
(import "P0lib" "seed" (func $seed (result i32)))
(global $r (mut i32) i32.const 0)

;;  procedure randint(lower, upper: integer) → (rand: integer)
;;    const a = 16807
;;    const c = 11
;;    const m = 65535
;;      r := (a × r + c) mod m
;;      rand := r mod (upper - lower) + lower
(func $randint (param $lower i32) (param $upper i32) (result i32)
(local $rand i32)
(local $0 i32)
;;  r := (16807 × r + 11) mod 65535
i32.const 16807
global.get $r
i32.mul
i32.const 11
i32.add
i32.const 65535
i32.rem_s
global.set $r
;;  rand := r mod (upper - lower) + lower
global.get $r
local.get $upper
local.get $lower
i32.sub
i32.rem_s
local.get $lower
i32.add
local.set $rand
local.get $rand
)

;;  procedure power(a: integer, n: integer, p: integer) → (res: integer)
;;    res, a := 1, a mod p
;;    while n > 0 do
;;      if n mod 2 = 1 then
;;        res, n := (res × a) mod p, n - 1
;;      a, n := (a × a) mod p, n div 2
(func $power (param $a i32) (param $n i32) (param $p i32) (result i32)
(local $res i32)
(local $0 i32)
;;  res, a := 1, a mod p
i32.const 1
local.get $a
local.get $p
i32.rem_s
local.set $a
local.set $res
;;  while n > 0 do
loop
local.get $n
i32.const 0
i32.gt_s
if
;;    if n mod 2 = 1 then
local.get $n
i32.const 2
i32.rem_s
i32.const 1
i32.eq
if
;;      res, n := (res × a) mod p, n - 1
local.get $res
local.get $a
i32.mul
local.get $p
i32.rem_s
local.get $n
i32.const 1
i32.sub
local.set $n
local.set $res
end
;;    a, n := (a × a) mod p, n div 2
local.get $a
local.get $a
i32.mul
local.get $p
i32.rem_s
local.get $n
i32.const 2
i32.div_s
local.set $n
local.set $a
br 1
end
end
local.get $res
)

;;  procedure likelyPrime(n: integer, k: integer)
;;    var i, p, a: integer
;;      i, p := k, 1
;;      while (i > 0) and (p = 1) do
;;        a ← randint(1, n - 1)
;;        p ← power(a, n - 1, n)
;;        i := i - 1
;;      if p = 1 then write(1) else write(0)
(func $likelyPrime (param $n i32) (param $k i32)
(local $i i32)
(local $p i32)
(local $a i32)
(local $0 i32)
;;  i, p := k, 1
local.get $k
i32.const 1
local.set $p
local.set $i
;;  while (i > 0) and (p = 1) do
loop
local.get $i
i32.const 0
i32.gt_s
if (result i32)
local.get $p
i32.const 1
i32.eq
else
i32.const 0
end
if
;;    a ← randint(1, n - 1)
i32.const 1
local.get $n
i32.const 1
i32.sub
call $randint
local.set $a
;;    p ← power(a, n - 1, n)
local.get $a
local.get $n
i32.const 1
i32.sub
local.get $n
call $power
local.set $p
;;    i := i - 1
local.get $i
i32.const 1
i32.sub
local.set $i
br 1
end
end
;;  if p = 1 then write(1) else write(0)
local.get $p
i32.const 1
i32.eq
if
i32.const 1
call $write
else
i32.const 0
call $write
end
)

;;  program primalityTest
;;    var n, k: integer
;;      r ← seed()
;;      n ← read(); k ← read()
;;      likelyPrime(n, k)
(global $_memsize (mut i32) i32.const 0)
(func $program
(local $n i32)
(local $k i32)
(local $0 i32)
;;  r ← seed()
call $seed
global.set $r
;;  n ← read()
call $read
local.set $n
;;  k ← read()
call $read
local.set $k
;;  likelyPrime(n, k)
local.get $n
local.get $k
call $likelyPrime
)
(memory 1)
(start $program)
)
```

## A2

### 1. generated code length

the function body in the code section is **87 bytes** (3 bytes for local declarations + 84 bytes of instructions).

the full code section payload is 89 bytes. the complete binary file is 179 bytes.

### 2. annotated hex dump

<pre style="font-family:monospace;color:royalblue">
0x000000: 00 61 73 6D    .asm    magic
0x000004: 01 00 00 00    ....    version
0x000008: 01 0C 03 60    ...`    typesecid, size typesec (12), #types (3), functype (0)
0x00000c: 01 7F 00 60    ...`    #params (1), i32, #results (0), functype (1)
0x000010: 00 00 60 00    ..`.    #params (0), #results (0), functype (2), #params (0)
0x000014: 01 7F 02 2C    ...,    #results (1), i32, importsecid, size importsec (44)
0x000018: 03 05 50 30    ..P0    #imports (3), len('P0lib') (5), 'P', '0'
0x00001c: 6C 69 62 05    lib.    'l', 'i', 'b', len('write') (5)
0x000020: 77 72 69 74    writ    'w', 'r', 'i', 't'
0x000024: 65 00 00 05    e...    'e', func import, typeidx 0, len('P0lib') (5)
0x000028: 50 30 6C 69    P0li    'P', '0', 'l', 'i'
0x00002c: 62 07 77 72    b.wr    'b', len('writeln') (7), 'w', 'r'
0x000030: 69 74 65 6C    itel    'i', 't', 'e', 'l'
0x000034: 6E 00 01 05    n...    'n', func import, typeidx 1, len('P0lib') (5)
0x000038: 50 30 6C 69    P0li    'P', '0', 'l', 'i'
0x00003c: 62 04 72 65    b.re    'b', len('read') (4), 'r', 'e'
0x000040: 61 64 00 02    ad..    'a', 'd', func import, typeidx 2
0x000044: 03 02 01 01    ....    funcsecid, size funcsec (2), #typeidx (1), typeidx 1
0x000048: 05 03 01 00    ....    memsecid, size memsec (3), #mem (1), 0x00 (only min)
0x00004c: 01 06 06 01    ....    min 1 (page), globalsecid, size globalsec (6), #globals (1)
0x000050: 7F 01 41 14    ..A.    i32, mutable (var), i32.const, 20 (_memsize)
0x000054: 0B 08 01 03    ....    end (expr), startsecid, size startsec (1), funcidx 3
0x000058: 0A 59 01 57    .Y.W    codesecid, size codesec (89), #functions (1), size code (87)
0x00005c: 01 02 7F 41    ...A    #local decls (1), 2 vars, i32, i32.const
0x000060: 00 41 03 36    .A.6    0 (addr a[1]), i32.const, 3, i32.store
0x000064: 02 00 41 04    ..A.    align 2, offset 0, i32.const, 4 (addr a[2])
0x000068: 41 07 36 02    A.6.    i32.const, 7, i32.store, align 2
0x00006c: 00 41 08 41    .A.A    offset 0, i32.const, 8 (addr a[3]), i32.const
0x000070: 05 36 02 00    .6..    5, i32.store, align 2, offset 0
0x000074: 41 0C 41 09    A.A.    i32.const, 12 (addr a[4]), i32.const, 9
0x000078: 36 02 00 41    6..A    i32.store, align 2, offset 0, i32.const
0x00007c: 10 41 07 36    .A.6    16 (addr a[5]), i32.const, 7, i32.store
0x000080: 02 00 41 01    ..A.    align 2, offset 0, i32.const, 1
0x000084: 21 00 03 40    !..@    local.set, 0 ($i), loop, void
0x000088: 20 00 41 05     .A.    local.get, 0 ($i), i32.const, 5 (N)
0x00008c: 4C 04 40 20    L.@     i32.le_s, if, void, local.get
0x000090: 00 41 01 6B    .A.k    0 ($i), i32.const, 1 (lower), i32.sub
0x000094: 41 04 6C 41    A.lA    i32.const, 4 (elem size), i32.mul, i32.const
0x000098: 00 6A 28 02    .j(.    0 (base addr), i32.add, i32.load, align 2
0x00009c: 00 41 07 46    .A.F    offset 0, i32.const, 7, i32.eq
0x0000a0: 04 40 20 00    .@ .    if, void, local.get, 0 ($i)
0x0000a4: 10 00 0B 20    ...     call, 0 ($write), end (if), local.get
0x0000a8: 00 41 01 6A    .A.j    0 ($i), i32.const, 1, i32.add
0x0000ac: 21 00 0C 01    !...    local.set, 0 ($i), br, 1 (loop)
0x0000b0: 0B 0B 0B       ...     end (if), end (loop), end (func)
</pre>

## A3

### 1. extending the scanner

in `SC.ipynb`, two new symbol constants are added:

```python
NOTELEMENT = 52
DIFFERENCE = 53
```

in `getSym()`, two new branches recognize the unicode characters `∉` (U+2209) and `∖` (U+2216):

```python
elif ch == '∉': getChar(); sym = NOTELEMENT
...
elif ch == '∖': getChar(); sym = DIFFERENCE
```

the production for `symbol` is extended to include `∉` and `∖` alongside the existing set operators.

### 2. extending the parser

the grammar is extended at two places:

```
term ::= factor {("×" | "div" | "mod" | "∩" | "∖" | "and") factor}
expression ::= simpleExpression
    {("=" | "≠" | "<" | "≤" | ">" | "≥" | "∈" | "∉" | "⊆" | "⊇") simpleExpression}
```

`∖` binds at the same level as `∩` (in `term`), and `∉` binds at the same level as `∈` (in `expression`). so `a ∉ b ∖ c ∪ d` parses as `a ∉ ((b ∖ c) ∪ d)`.

in `P0.ipynb`, the imports are extended:

```python
from SC import ..., NOTELEMENT, DIFFERENCE, ...
```

`term()` is extended to include `DIFFERENCE` in the while condition:

```python
while SC.sym in {TIMES, DIV, MOD, INTERSECTION, DIFFERENCE, AND}:
    ...
    elif op in {INTERSECTION, DIFFERENCE} and type(x.tp) == Set == type(y.tp):
        x = CG.genBinaryOp(op, x, y)
    ...
```

`expression()` is extended to include `NOTELEMENT`:

```python
while SC.sym in {EQ, NE, LT, LE, GT, GE, ELEMENT, NOTELEMENT, SUBSET, SUPERSET}:
    ...
    elif (op in (ELEMENT, NOTELEMENT) and x.tp == Int) or \
        (op in (SUBSET, SUPERSET) and type(x.tp) == Set):
        x = CG.genUnaryOp(op, x); y = simpleExpression()
        if type(y.tp) == Set: x = CG.genRelation(op, x, y)
        else: mark('set expected')
    else: mark('bad type')
```

type checking: for `a ∉ b`, if `a` is not `Int`, error `bad type`; if `b` is not a set, error `set expected`. for `a ∖ b`, if both are not sets, error `bad type`.

### 3. extending the AST (CGast.ipynb)

imports extended with `NOTELEMENT, DIFFERENCE`. `BinaryOp.__str__()` extended:

```python
'∉' if self.op == NOTELEMENT else \
'∖' if self.op == DIFFERENCE else \
```

### 4. extending the code generator (CGwat.ipynb)

imports extended with `NOTELEMENT, DIFFERENCE`.

**`genUnaryOp`**: `NOTELEMENT` uses the same code as `ELEMENT` (convert integer to bitmask via `1 << i`):

```python
elif op in {ELEMENT, NOTELEMENT}:
    asm.append('local.set $0')
    asm.append('i32.const 1')
    asm.append('local.get $0')
    asm.append('i32.shl')
    x = Var(Int); x.lev = Stack
```

**`genBinaryOp`**: `DIFFERENCE` implements $s \setminus t = s \cap \complement t$. the right operand (already on stack) is complemented via XOR with the universe, then ANDed with the left operand:

```python
elif op == DIFFERENCE:
    loadItem(y)
    u = (1 << x.tp.length) - 1
    u = u << x.tp.lower
    asm.append('i32.const ' + hex(u))
    asm.append('i32.xor')
    loadItem(x)
    asm.append('i32.and')
    x = Var(x.tp); x.lev = Stack
```

**`genRelation`**: `NOTELEMENT` applies `i32.and` then `i32.eqz` (the negation of the element test):

```python
['i32.and', 'i32.eqz'] if op == NOTELEMENT else \
```

for `i ∉ s`, the generated code is: compute bitmask `1 << i`, AND with `s`, then `eqz`. the result is 1 (true) iff bit `i` is not set in `s`.

for `s ∖ t`, the generated code is: load `t`, XOR with universe (complement), load `s`, AND. this computes $s \cap \complement t$.

### 5. evaluating the implementation

the Sieve of Eratosthenes is run repeatedly to compare four implementations.

P0 with set operations compiles to bitwise operations on 32-bit integers, where each set element maps to a single bit. this is the representation-level advantage: the entire sieve state fits in one machine word, and operations like union, intersection, complement, set difference are single WASM instructions (`i32.or`, `i32.and`, `i32.xor`).

_expected relative performance_ (fastest to slowest for the sieve with $N=32$):

1. _P0/WASM via wasmtime or browser JS engine_: WASM compiled to native machine code by the JIT. the sieve operates on a single `i32`, so the inner loop is a handful of register operations. expected to be within a small constant factor of C.
2. _Java (Oracle JVM)_: `HashSet<Integer>` requires boxing each integer, hashing, and heap allocation per element. orders of magnitude more work per set operation than a bitmask. the JIT eventually optimizes the hot path, but the data structure overhead dominates. for $10^8$ repetitions, wall clock is in the seconds range.
3. _P0/WASM via pywasm_: pywasm is a pure-Python WASM interpreter. every WASM instruction is dispatched through Python's interpreter loop. roughly $10^3\text{--}10^4\times$ slower than native WASM execution.
4. _Python (CPython)_: `set` objects use hash tables. each `add`/`remove`/`in` test involves hashing, comparisons, and dynamic memory management. for $10^6$ repetitions, wall clock is in the seconds range.

observation:

- P0's bitwise set representation on a 32-element universe turns $O(n)$ set operations into $O(1)$ machine instructions.
- the Java and Python versions use general-purpose set data structures (hash sets) that are asymptotically equivalent but carry constant factors of $100\text{--}1000\times$ per operation due to hashing, boxing, and cache misses.
- the pywasm interpreter is slow bc it's interpreting bytecode in Python, adding another layer of indirection on top of the efficient representation.
