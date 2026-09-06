---
name: lecture-05-parser
description: construction of a parser — scanner/parser separation, symbol table, P0 parser structure, error handling, data representation
---

# lecture 05 — the construction of a parser

(notebook title: "6. The Construction of a Parser". directory is `05`.)

this is the lecture where the P0 compiler's actual code appears. `P0.ipynb`, `SC.ipynb`, `ST.ipynb`, `CGast.ipynb`, `CGwat.ipynb`, `P0ParsingTest.ipynb`, `P0TypeCheckingTest.ipynb` all live here.

## scanner / parser separation

the scanner (`SC.ipynb`) reads characters, produces symbols:

- `getSym()` — reads the next symbol, stores in `sym` (symbol kind) and `val` (value, if any)
- `mark(msg)` — reports an error at the current position; parsing continues
- `KEYWORDS` — dict from keyword strings to their symbol constants
- symbol constants: `IDENT`, `NUMBER`, `PLUS`, `MINUS`, `TIMES`, `DIV`, `MOD`, `EQ`, `NE`, `LT`, `LE`, `GT`, `GE`, `BECOMES`, `LPAREN`, `RPAREN`, `IF`, `THEN`, `ELSE`, `WHILE`, `DO`, `VAR`, `TYPE`, `PROCEDURE`, `RECORD`, `ARRAY`, `PROGRAM`, `INTEGER`, `BOOLEAN`, ...

the parser (`P0.ipynb`) consumes symbols, produces AST/code. procedures follow the grammar:

- `factor()` — $\text{factor} \to \text{ident} \mid \text{number} \mid \texttt{'('}\,\text{expression}\,\texttt{')'}$
- `term()` — $\text{term} \to \text{factor}\,\{\,(\texttt{'*'} \mid \texttt{'/'} \mid \texttt{mod} \mid \texttt{div})\,\text{factor}\,\}$
- `simpleExpression()` — $\text{simpleExpression} \to [\texttt{'+'} \mid \texttt{'-'}]\,\text{term}\,\{\,(\texttt{'+'} \mid \texttt{'-'})\,\text{term}\,\}$
- `expression()` — $\text{expression} \to \text{simpleExpression}\,[\,\text{rel}\,\text{simpleExpression}\,]$
- `statement()` — $\text{statement} \to \text{assignment} \mid \text{call} \mid \text{if} \mid \text{while} \mid \text{block}$
- `statementSuite()` — $\text{statementSuite} \to \text{stmt} \mid (\text{indent}\,\text{stmt}^*\,\text{dedent})$
- `typ()` — $\text{typ} \to \texttt{integer} \mid \texttt{boolean} \mid \text{ident} \mid \texttt{'['}\,\text{expr}\,\texttt{'..'}\,\text{expr}\,\texttt{']'}\,\texttt{'} \to \texttt{'}\,\text{typ} \mid \texttt{'('}\,\text{fields}\,\texttt{')'}$
- `procedureDecl()` — procedure declaration
- `program()` — top-level

## the symbol table (ST.ipynb)

symbol table entries are plain classes:

```python
class Var:   def __init__(self, tp, lev, adr): ...
class Ref:   def __init__(self, tp, lev, adr): ...   # reference param
class Const: def __init__(self, tp, val): ...
class Type:  def __init__(self, val): ...             # type alias; val is the underlying
class Proc:  def __init__(self, par, res, lev, adr): ...
class StdProc: def __init__(self, par, res): ...      # predefined procedures
class Record:  def __init__(self, fields): ...
class Array:   def __init__(self, base, lower, length): ...
```

operations:

- `Init()` — start a fresh table with predefined types and procs
- `openScope()` — push a new scope
- `closeScope()` — pop
- `newDecl(name, entry)` — add to current scope; `mark` if duplicate
- `find(name)` — walk from innermost scope outward; returns the entry or marks undefined

the `lev` field on `Var`/`Ref`/`Proc` tracks scope depth. level $0$ is globals, `curlev` increments on procedure entry.

## type compatibility

```python
def compatible(xt, yt):
  # structural compatibility
  # integer ~ integer, boolean ~ boolean
  # array types compatible if same base and same length (lower/upper don't matter)
  # record types compatible if same fields (names + types in order)
  # refs unwrap to their target type
```

## parser-codegen coordination

the parser calls codegen procedures (`genBinaryOp`, `genVar`, `genConst`, `genAssign`, `genCall`, `genProcEntry`, `genProcExit`, etc.). the code generator is swappable — `CGast` pretty-prints, `CGwat` emits WAT, `CGriscv` emits risc-v asm. they have the same surface.

## error handling

- scanner/parser errors: `mark(msg)` records at position, returns, parsing continues
- type errors: same pattern, `mark('incompatible operands of + at ...')`
- the parser never raises; the test harness checks that `mark` was called (or not) for each test input

## data representation at runtime

- **integer**: 32-bit two's complement (i32 in wasm, word in risc-v)
- **boolean**: i32 ($0$ or $1$), not bit-packed
- **array**: contiguous memory, $[l..u] \to T$ takes $(u - l + 1) \times \text{size}(T)$ bytes
- **record**: contiguous memory, fields at static offsets
- **reference**: address (i32)

global variables have static addresses assigned by the symbol table (field `adr`). local variables live on the wasm stack or in risc-v registers/stack.

## coping with errors in the grammar

techniques used in P0:

1. **sync sets**: after an error, skip tokens until a sync symbol (`;`, `end`, `)`, keyword)
2. **error productions**: extra rules for common mistakes
3. **continued parsing**: `mark` without raising, so the rest of the program can still be checked

## testing

`P0ParsingTest.ipynb` runs `compileString(src)` against expected output (emitted code or error messages). `P0TypeCheckingTest.ipynb` focuses on type errors. `P0test.ipynb` runs full end-to-end compilation + execution for WASM.
