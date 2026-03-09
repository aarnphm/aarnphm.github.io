---
date: "2026-03-09"
description: stack controls
id: results
modified: 2026-03-09 08:20:44 GMT-04:00
tags:
  - sfwr4tb3
  - assignment
title: wasm and compile string
---

## A1

Half-open intervals `[a .. b)` meaning `[a .. b - 1]`.

Only the parser (`P0.ipynb`) needs modification, specifically `typ()`. The scanner already recognizes `)` as `RPAREN`.

Grammar:

```
type ::= ident
       | "[" expression ".." expression ("]" | ")") "→" type
       | "(" typedIds ")"
       | "set" "[" expression ".." expression "]"
```

After parsing the upper bound expression, check `RBRAK` (closed) vs `RPAREN` (half-open):

```python
if SC.sym == RBRAK:
    getSym(); halfopen = False
elif SC.sym == RPAREN:
    getSym(); halfopen = True
else: mark("']' or ')' expected")
...
length = y.val - x.val if halfopen else y.val - x.val + 1
x = Type(CG.genArray(Array(z, x.val, length)))
```

## A2

Binary constants with `0b` prefix.

Only the scanner (`SC.ipynb`) needs modification, specifically `number()`.

Grammar:

```
number ::= '0' 'b' binarydigit {binarydigit} | digit {digit}
digit ::= '0' | ... | '9'
binarydigit ::= '0' | '1'
```

When the first digit is `'0'` and the next character is `'b'`, parse binary digits and accumulate in base 2:

```python
def number():
    global sym, val
    sym, val = NUMBER, 0
    if ch == '0':
        getChar()
        if ch == 'b':
            getChar()
            if ch not in ('0', '1'): mark('binary digit expected')
            while ch == '0' or ch == '1':
                val = 2 * val + int(ch)
                getChar()
            if val >= 2**31: mark('number too large')
            return
    while '0' <= ch <= '9':
        val = 10 * val + int(ch)
        getChar()
    if val >= 2**31: mark('number too large')
```

## A3

Hand-translated `randgcd` to WASM following the translation scheme from the course notes.

`randint`: linear congruential generator with constants $a = 16807$, $c = 11$, $m = 65535$. Updates global `r`, returns `r \mod \text{bound}`.

`gcd`: Euclid's subtraction algorithm. While loop uses `loop`/`if`/`br 1`/`end`/`end` pattern. Inner if-else for the comparison.

`program`: chains `read` → `randint` → `write` for both `x` and `y`, then `gcd(x, y)` → `write`.

With `read()` returning 41: $r_1 = (16807 \times 41 + 11) \mod 65535 = 33748$, $x = 33748 \mod 100 = 48$. $r_2 = (16807 \times 33748 + 11) \mod 65535 = 62757$, $y = 62757 \mod 100 = 57$. $\gcd(48, 57) = 3$.

Output: `48 57 3`.

## A4

Hand-optimized `fibonacci` in WASM using only parameters and the stack (no local variables).

The `if (result i32)` construct allows both branches to push exactly one `i32`:

```wasm
(func $fib (param $n i32) (result i32)
  local.get $n
  i32.const 1
  i32.le_s
  if (result i32)
    local.get $n
  else
    local.get $n
    i32.const 1
    i32.sub
    call $fib
    local.get $n
    i32.const 2
    i32.sub
    call $fib
    i32.add
  end
)
```

The `program` function chains `read` → `fib` → `write` directly on the stack with zero locals:

```wasm
(func $program
  call $read
  call $fib
  call $write
)
```

With `read()` returning 7: $\text{fib}(7) = 13$.
