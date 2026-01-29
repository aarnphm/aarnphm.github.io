# bases compiler + runtime (quartz implementation)

status: active
last updated: 2026-01-28

this directory contains the obsidian bases compiler, interpreter, and runtime helpers used by quartz to render `.base` files. it is designed to match obsidian bases syntax and semantics with deterministic evaluation and consistent diagnostics.

You can test it out with any of the base file in my vault here:

```bash
pnpm exec tsx quartz/scripts/inspect-base.ts content/antilibrary.base > /tmp/antilibrary-ast-ir.json

jq '.expressions[] | {context, kind, source, ast}' /tmp/antilibrary-ast-ir.json
jq '.expressions[] | {context, kind, ir}' /tmp/antilibrary-ast-ir.json
```

## scope

- parse base expressions (filters, formulas, summaries, property expressions)
- compile expressions to bytecode ir
- interpret bytecode with a deterministic stack vm
- resolve file, note, formula, and property values
- render views (table, list, cards/gallery, board, calendar, map)
- surface parse and runtime diagnostics in base output

## architecture (pipeline)

1. parse `.base` yaml (plugin: `quartz/plugins/transformers/bases.ts`)
2. parse expressions into ast (`compiler/parser.ts`)
3. compile ast to ir (`compiler/ir.ts`)
4. evaluate ir per row with caches (`compiler/interpreter.ts`)
5. render views and diagnostics (`render.ts`)

## modules

- `compiler/lexer.ts`: tokenizer with span tracking and regex support
- `compiler/parser.ts`: pratt parser for expression grammar and error recovery
- `compiler/ir.ts`: bytecode instruction set + compiler
- `compiler/interpreter.ts`: stack vm, value model, coercions, methods, functions
- `compiler/diagnostics.ts`: diagnostics types and helpers
- `compiler/schema.ts`: summary config schema and builtins
- `compiler/properties.ts`: property expression builder for columns and config keys
- `render.ts`: view rendering and diagnostics output
- `query.ts`: summaries and view summary helpers
- `types.ts`: base config types and yaml parsing helpers

## value model (runtime)

runtime values are tagged unions with explicit kinds:

- null, boolean, number, string
- date, duration
- list, object
- file, link
- regex, html, icon, image

coercions are permissive to match obsidian behavior. comparisons prefer type-aware equality (links resolve to files when possible, dates compare by time, etc), with fallbacks when resolution fails.

## expression features (spec parity)

- operators: `==`, `!=`, `>`, `<`, `>=`, `<=`, `&&`, `||`, `!`, `+`, `-`, `*`, `/`, `%`
- member and index access
- function calls and method calls
- list literals and regex literals
- `this` binding with embed-aware scoping
- list helpers (`filter`, `map`, `reduce`) using implicit locals `value`, `index`, `acc`
- summary context helpers: `values` (column values) and `rows` (row files)

## diagnostics

- parser diagnostics are collected with spans at compile time
- runtime diagnostics are collected during evaluation and deduped per context
- base views render diagnostics above the view output

## this scoping

- main base file: `this` resolves to the base file
- embedded base: `this` resolves to the embedding file
- row evaluation: `file` resolves to the row file

## performance decisions

- bytecode ir keeps evaluation linear and stable
- per-build backlink index avoids n^2 scans
- property cache memoizes property expressions per file
- formula cache memoizes formula evaluation per file

## view rendering

- table, list, cards/gallery, board, calendar, map
- map rendering expects coordinates `[lat, lon]` and map config fields
- view filters combine with base filters via logical and
