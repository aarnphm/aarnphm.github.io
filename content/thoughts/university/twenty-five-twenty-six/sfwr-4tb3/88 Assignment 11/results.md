---
date: "2026-04-07"
description: earley parsing, PEG, packrat parsing
id: results
modified: 2026-04-07 21:37:56 GMT-04:00
tags:
  - sfwr4tb3
  - assignment
title: "Assignment 11: Earley and PEG Parsing"
---

## Q1: Steps with Earley's Parser [6 points]

### Part 1: Earley parsing table for `a+a+a` with grammar G1

G1 = (`S→E`, `E→a`, `E→E+E`)

| step |   | set  | item             |
|:-----|:--|:-----|:-----------------|
| 0    |   | s[0] | S→ •E, 0        |
| 1    | P | s[0] | E→•a, 0         |
| 2    | P | s[0] | E→•E+E, 0       |
| 3    | M | s[1] | E→a•, 0         |
| 4    | C | s[1] | S→E•, 0         |
| 5    | C | s[1] | E→E•+E, 0       |
| 6    | M | s[2] | E→E+•E, 0       |
| 7    | P | s[2] | E→•a, 2         |
| 8    | P | s[2] | E→•E+E, 2       |
| 9    | M | s[3] | E→a•, 2         |
| 10   | C | s[3] | E→E•+E, 2       |
| 11   | C | s[3] | E→E+E•, 0       |
| 12   | M | s[4] | E→E+•E, 2       |
| 13   | C | s[3] | S→E•, 0         |
| 14   | C | s[3] | E→E•+E, 0       |
| 15   | M | s[4] | E→E+•E, 0       |
| 16   | P | s[4] | E→•a, 4         |
| 17   | P | s[4] | E→•E+E, 4       |
| 18   | M | s[5] | E→a•, 4         |
| 19   | C | s[5] | E→E+E•, 2       |
| 20   | C | s[5] | E→E•+E, 4       |
| 21   | C | s[5] | E→E+E•, 0       |
| 22   | C | s[5] | E→E•+E, 2       |
| 23   | C | s[5] | E→E+E•, 0       |
| 24   | C | s[5] | S→E•, 0         |
| 25   | C | s[5] | E→E•+E, 0       |

The item `S→E•, 0` appears in s[5], confirming that `a+a+a` is accepted.

### Part 2: Two derivation sequences

The ambiguous grammar G1 = (`S→E`, `E→a`, `E→E+E`) allows two distinct leftmost derivations of `a+a+a`:

**Derivation 1** (left-associative, `(a+a)+a`):

$$S \Rightarrow E \Rightarrow E+E \Rightarrow E+E+E \Rightarrow a+E+E \Rightarrow a+a+E \Rightarrow a+a+a$$

Here the left `E` in `E+E` is expanded first as `E+E`, yielding `(E+E)+E`.

**Derivation 2** (right-associative, `a+(a+a)`):

$$S \Rightarrow E \Rightarrow E+E \Rightarrow a+E \Rightarrow a+E+E \Rightarrow a+a+E \Rightarrow a+a+a$$

Here the left `E` in `E+E` is expanded as `a`, and the right `E` is then expanded as `E+E`, yielding `E+(E+E)`.

---

## Q2: All Trees with Earley's Parser [6 points]

Two lines are modified from the original `parse` function:

```python
def parse(g, x, log = False):
    global s
    n = len(x); x = '^' + x + '$'; S, π = g[0][0], g[0][2:]
    s = [{(S, '', π, 0)}] + [set() for _ in range(n)]
    if log: print('   s[0]: ', S, '→ •', π, ', 0', sep='')
    for i in range(n + 1):
        v = set()
        while v != s[i]:
            e = (s[i] - v).pop(); v.add(e)
            A, σ, τ, j = e
            if len(τ) > 0 and τ[0] == x[i + 1]:
                f = (A, σ + τ[0], τ[1:], j)
                s[i + 1].add(f)
                if log: print('M  s[', i + 1, ']: ', f[0], '→', f[1], '•', f[2], ', ', f[3], sep='')
            elif len(τ) > 0:
                for f in ((r[0], '', r[2:], i) for r in g if r[0] == τ[0]):
                    s[i].add(f)
                    if log: print('P  s[', i, ']: ', f[0], '→', f[1], '•', f[2], ', ', f[3], sep='')
            else:
                # MODIFIED: wrap completed nonterminal with tree notation A(σ)
                for f in ((B, μ + ν[0] + '(' + σ + ')', ν[1:], k) for (B, μ, ν, k) in s[j] if len(ν) > 0 and ν[0] == A):
                    s[i].add(f)
                    if log: print('C  s[', i, ']: ', f[0], '→', f[1], '•', f[2], ', ', f[3], sep='')
    # MODIFIED: return set of all parse trees
    return {σ for (A, σ, τ, j) in s[n] if A == S and τ == '' and j == 0}
```

**Line 1** (complete step): Changed `μ + ν[0]` to `μ + ν[0] + '(' + σ + ')'`. When nonterminal `A` completes with recognized string `σ`, the parent item records `A(σ)` instead of just `A`.

**Line 2** (return): Changed `(S, π, '', 0) in s[n]` to `{σ for (A, σ, τ, j) in s[n] if A == S and τ == '' and j == 0}`. Collects all distinct tree strings from completed start-symbol items.

---

## Q3: Arithmetic Expressions with PEG [6 points]

PEG grammar:

```
S ← E
E ← T ('+' T)*
T ← F ('×' F)*
F ← P'²' / P
P ← 'a' / '(' E ')'
```

```python
class Backtrack:
    src: str

    def literal(self, k: int, a: str):
        if self.src.startswith(a, k): return k + len(a)

    def S(self, k):
        return self.E(k)

    def E(self, k):
        k = self.T(k)
        if k is None: return None
        while True:
            r = self.literal(k, '+')
            if r is None: return k
            r = self.T(r)
            if r is None: return k
            k = r

    def T(self, k):
        k = self.F(k)
        if k is None: return None
        while True:
            r = self.literal(k, '×')
            if r is None: return k
            r = self.F(r)
            if r is None: return k
            k = r

    def F(self, k):
        r = self.P(k)
        if r is not None:
            s = self.literal(r, '²')
            if s is not None: return s
            return r
        return None

    def P(self, k):
        r = self.literal(k, 'a')
        if r is not None: return r
        r = self.literal(k, '(')
        if r is not None:
            r = self.E(r)
            if r is not None:
                r = self.literal(r, ')')
                return r
        return None

    def parse(self, s: str):
        self.src = s; return self.S(0) == len(s)
```

---

## Q4: Packrat Parsing for Statements [6 points]

PEG grammar:

```
statement ← ident selector ':=' ident / ident (',' ident)* ':=' ident (',' ident)* / ident (',' ident)* '←' ident '(' ident ')'
selector  ← ('[' ident ']' / '.' ident)*
ident     ← 'a' / ... / 'z'
```

### Part 1: Backtracking parser

```python
class StatementBacktrack:
    src: str

    def literal(self, k: int, a: str):
        if self.src.startswith(a, k): return k + len(a)

    def ident(self, k):
        if k < len(self.src) and 'a' <= self.src[k] <= 'z':
            return k + 1
        return None

    def selector(self, k):
        while True:
            r = self.literal(k, '[')
            if r is not None:
                r = self.ident(r)
                if r is not None:
                    r = self.literal(r, ']')
                    if r is not None:
                        k = r; continue
            r = self.literal(k, '.')
            if r is not None:
                r = self.ident(r)
                if r is not None:
                    k = r; continue
            return k

    def statement(self, k):
        r = self.ident(k)
        if r is not None:
            r2 = self.selector(r)
            r3 = self.literal(r2, ':=')
            if r3 is not None:
                r4 = self.ident(r3)
                if r4 is not None:
                    return r4
        r = self.ident(k)
        if r is not None:
            while True:
                r2 = self.literal(r, ',')
                if r2 is None: break
                r3 = self.ident(r2)
                if r3 is None: break
                r = r3
            r2 = self.literal(r, ':=')
            if r2 is not None:
                r3 = self.ident(r2)
                if r3 is not None:
                    while True:
                        r4 = self.literal(r3, ',')
                        if r4 is None: break
                        r5 = self.ident(r4)
                        if r5 is None: break
                        r3 = r5
                    return r3
        r = self.ident(k)
        if r is not None:
            while True:
                r2 = self.literal(r, ',')
                if r2 is None: break
                r3 = self.ident(r2)
                if r3 is None: break
                r = r3
            r2 = self.literal(r, '←')
            if r2 is not None:
                r3 = self.ident(r2)
                if r3 is not None:
                    r4 = self.literal(r3, '(')
                    if r4 is not None:
                        r5 = self.ident(r4)
                        if r5 is not None:
                            r6 = self.literal(r5, ')')
                            if r6 is not None:
                                return r6
        return None

    def parse(self, s: str):
        self.src = s; return self.statement(0) == len(s)
```

### Part 2: Memoizing (packrat) parser

```python
class StatementMemoizing(StatementBacktrack):
    def parse(self, s: str):
        self.src = s
        self.memo = {}
        return self.statement(0) == len(s)

    def ident(self, k):
        if ('ident', k) not in self.memo:
            self.memo[('ident', k)] = super().ident(k)
        return self.memo[('ident', k)]

    def selector(self, k):
        if ('selector', k) not in self.memo:
            self.memo[('selector', k)] = super().selector(k)
        return self.memo[('selector', k)]

    def statement(self, k):
        if ('statement', k) not in self.memo:
            self.memo[('statement', k)] = super().statement(k)
        return self.memo[('statement', k)]
```

Each nonterminal parsing function checks the memo table `(name, position)` before computing. If cached, returns immediately. Otherwise computes via `super()`, stores the result, and returns it. The `super()` calls resolve to the backtracking implementations, which in turn call `self.ident`, `self.selector` etc. on the memoizing instance, so memoization cascades through all recursive calls.
