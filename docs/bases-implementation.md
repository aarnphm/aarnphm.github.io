# bases implementation plan

status: draft
last updated: 2026-01-28

## scope

implement a production-grade bases compiler and runtime that matches obsidian bases syntax, views, formulas, and summaries. focus is correctness and performance at scale. this document enumerates concrete steps, owners are implicit.

## deliverables

- new expression parser with ebnf grammar and ast mapping
- shared evaluator for filters, formulas, summaries
- property resolver that matches obsidian file._ and note._ specs
- indices for tags, links, backlinks
- test suite with spec fixtures and regression coverage
- migration path with dual-eval diffing

## milestones

1. spec and parser foundation
2. runtime core and indices
3. view rendering integration
4. migration and cleanup

## work items

### 1) spec and parser foundation

- formalize grammar and lexer tokens in docs/bases-compiler.md
- define ast types with spans and source mapping
- implement lexer with bracket access and string escape support
- implement pratt parser with precedence and error recovery
- add parser tests for obsidian examples and edge cases

### 2) runtime core and indices

- implement value model (null, boolean, number, string, date, duration, list, object, file, link)
- implement builtin functions and methods per obsidian spec
- define coercion rules and error modes (strict vs permissive)
- implement bytecode ir and interpreter
- build per-build indices for tags, links, backlinks
- add property cache and memoized formula evaluation

### 3) view rendering integration

- resolve formula.\* in sort, groupBy, and order paths
- replace existing summaries with full builtin set + formula summaries
- ensure file.\* coverage (basename, size, embeds, properties)
- implement this keyword scoping for base, embed, and sidebar
- update base emitter to use compiled evaluators

### 4) migration and cleanup

- add feature flag to select old vs new evaluator per base
- implement dual evaluation diff tool for test vaults
- add golden tests for content/antilibrary.base
- document migration notes and remove old parser after stabilization

### migration checklist (draft)

- add `basesCompiler` feature flag in config, default false
- wire flag into base transformer and emitter to choose evaluator
- add dual-eval mode that runs both engines and diffs outputs
- define diff format (view type, row count, first mismatch, summary mismatch)
- store diffs in a reproducible report file under docs or tmp
- add a cli path to run dual-eval on a subset of bases
- add a fallback mode that switches to legacy evaluator on error
- declare cutover criteria: zero diffs for N bases, perf within target, no runtime diagnostics

## tests

- parser unit tests per grammar production
- eval tests for each builtin function and method
- property resolver tests for file._ and note._
- view-level snapshot tests for table, list, cards, map
- performance tests at 10k and 100k synthetic notes

## risks

- semantic mismatch with obsidian edge cases, need fixtures
- type coercion ambiguity across mixed-type comparisons
- performance regression if caching is incomplete

## dependencies

- no new deps by default
- if using a parser dependency or oxc, confirm before adding
