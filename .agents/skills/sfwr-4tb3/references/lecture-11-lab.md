---
name: lecture-11-lab
description: local Lab 11 Earley, backtracking, and memoizing parser exercises
---

# Lab 11

Read the named question in `99 Lab 11/` or its assignment counterpart in `88 Assignment 11/`. The local lab uses Python parsers. Haskell and OCaml examples elsewhere in the generalized-parsing lecture do not establish the lab's API.

## Earley sets

`01 Steps with Earley's Parser.ipynb` asks for sets of dotted productions with origin indices. Use the question's start production and indexing convention.

1. Initialize the first set, then close it under prediction and completion, including nullable productions.
2. Scan matching terminals into the next set.
3. Close that set under prediction and completion before scanning again.
4. Accept when the completed start item with origin zero is in the final set.

For assignment questions requesting all trees, preserve all derivations or back-pointers contributing to an item. A set of recognition items alone discards that history. Trace the supplied implementation's representation before adding tree construction.

## Backtracking and memoization

`02 Packrat Parsing.ipynb` uses:

```text
S  ← &(AB c) a* BC
AB ← (a AB b)?
BC ← (b BC c)?
```

Its `Backtrack` class returns the next input position on success and `None` on failure. Position zero is a valid successful result. `parse` checks that the result equals the input length. `Memoizing` caches by `(nonterminal, position)` and resets the cache for each input, including cached failures.

Read those classes and test cells before extending them. Preserve the positive lookahead's original position and optional productions' zero-length success. A memo table avoids recomputing the same rule at the same position; check the work performed inside each memoized rule before claiming linear runtime.

## Statement parsing

`03 Packrat Parsing for Statements.ipynb` supplies this grammar:

```text
statement  ← assignment / call
assignment ← designator ':=' designator
call       ← designator '(' designator ')'
designator ← ident ('.' ident)*
ident      ← 'a' / ... / 'z'
```

It uses `StatementBacktrack` and `StatementMemoizing`. Extend those definitions and preserve the fallback from an unsuccessful assignment to a call at the original position. Follow the actual exercise if an assignment variant asks for additional constructs.

## Arithmetic PEG questions

Read `88 Assignment 11/03 Arithmetic Expressions with PEG.ipynb` for its grammar, operators, and required semantic values. Ordered choice and greedy repetition affect recognition; a repetition-based expression grammar also needs an explicit accumulation rule to produce left-associative values or trees.

The [generalized-parsing reference](lecture-10-general.md) covers the algorithms. Use the local question for the answer shape and verification cases.
