---
name: lecture-05-parser
description: P0 scanner, symbol table, parser, error handling, and local notebook entry points
---

# Lecture 05: parser construction

Use `05 Construction of a Parser/SC.ipynb`, `ST.ipynb`, and `P0.ipynb` for this lecture. For an assignment, use its own copies. The following facts were checked against the lecture 05 definitions; later snapshots may differ.

## Scanner and errors

`getSym()` advances the scanner's `sym` and `val`. `identKW()` classifies identifiers through `KEYWORDS`; inspect that mapping before extending tokens. Built-in type names such as `integer` and `boolean` are declared by `P0.program()` in the symbol table.

`SC.mark(msg)` raises an `Exception` containing the line, position, and message. Parsing stops through that exception. `P0.compileString` catches and re-raises the message. Preserve this behavior when adding diagnostics; check the requested snapshot before assuming recovery or continued parsing.

## Symbol table

| Entry or operation                          | Lecture 05 contract                                          |
| ------------------------------------------- | ------------------------------------------------------------ |
| `Var(tp)`, `Ref(tp)`, `Res(tp)`             | Variable, reference parameter, result parameter              |
| `Const(tp, val)`                            | Typed constant                                               |
| `Type(tp)`                                  | Type entry with underlying type in `.val`                    |
| `Proc(par, res)`, `StdProc(par, res)`       | Procedure parameter/result entries                           |
| `init()`                                    | Initializes the scope stack                                  |
| `newDecl(name, entry)`                      | Adds `.name` and `.lev`, checks duplicate declarations       |
| `find(name)`                                | Searches scopes and reports an undefined name through `mark` |
| `openScope()`, `closeScope()`, `topScope()` | Manage or inspect the innermost scope                        |

Read array, record, and set classes in the same notebook when extending those types. Code generators assign target-specific locations; those fields are not extra constructor arguments to `Var` or `Proc`.

## Parser and type checking

Follow the function for the requested grammar production: `factor`, `term`, `simpleExpression`, `expression`, `statement`, `typ`, `procedureDecl`, or `program`. Preserve scanner lookahead and the semantic value returned to the caller.

`compatible(xt, yt)` handles equality, sets, arrays, and records. Read its actual predicate when changing compatibility: the lecture snapshot compares record fields through `zip`, and a generic claim about structural equivalence is insufficient to describe that implementation. Diagnose differences from the question's requirements explicitly.

The parser calls the selected generator through `CG`, including expression, declaration, statement, and procedure hooks. Read both the call site and the matching generator implementation when extending that interface.

## Verification

Use `P0ParsingTest.ipynb` for parsing and `P0TypeCheckingTest.ipynb` for type errors. Code generation needs its target's tests; run an execution harness when the request depends on runtime behavior. Execute only the relevant setup and test cells, accounting for notebook imports and output files. A request for an explanation can use a hand-derived trace, identified as such.
