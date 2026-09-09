---
name: content-index
description: lookup guide for lecture, lab, and assignment notebooks under content/thoughts/university/twenty-five-twenty-six/sfwr-4tb3/
---

# content index

All paths are relative to `content/thoughts/university/twenty-five-twenty-six/sfwr-4tb3/`. Use this index when the request lacks an exact path. Read the matching `.ipynb` directly; if an indexed path has moved, search the named lab or assignment with `rg --files`, excluding `.ipynb_checkpoints`. Directory descriptions are a snapshot, not evidence that a question is still absent.

## lectures

| dir                                            | title                                                                      | key ref                    |
| :--------------------------------------------- | :------------------------------------------------------------------------- | :------------------------- |
| `00 Notebooks on Compiler Construction`        | course intro, tooling setup                                                | —                          |
| `01 Language and Syntax`                       | grammars, derivations, chomsky hierarchy, BNF/EBNF, CST vs AST             | `lecture-01-syntax.md`     |
| `02 Regular Languages`                         | regex AST, NFA, DFA, thompson's construction, equivalence, minimization    | `lecture-02-regex.md`      |
| `03 Analysis of Context-Free Languages`        | pushdown automata, LL(k)/LR(k), FIRST/FOLLOW, recursive descent            | `lecture-03-cfl.md`        |
| `04 Syntax-Directed Translation`               | attribute grammars, synthesized/inherited, AST construction, type checking | `lecture-04-sdt.md`        |
| `05 Construction of a Parser`                  | P0 parser structure, SC/ST, error handling, runtime data rep               | `lecture-05-parser.md`     |
| `06 A Stack Architecture as Target`            | WebAssembly codegen, translation schemes for expr/stmt/array/record        | `lecture-06-stack-wasm.md` |
| `07 Further Data Types and Control Structures` | floats, sets, bit ops, exceptions, indirect calls, jump tables             | `lecture-07-datatypes.md`  |
| `08 A RISC Architecture as Target`             | risc-v and mips codegen, register conventions, ABI                         | `lecture-08-risc.md`       |
| `09 Code Optimization`                         | three-address code, basic blocks, CSE, LICM, strength reduction            | `lecture-09-optim.md`      |
| `10 Generalized Parsing`                       | combinator / earley / PEG / packrat / probabilistic                        | `lecture-10-general.md`    |

## labs (weekly, 1–11)

| dir         | contents                                                                                                                                                                       | covered in          |
| :---------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------ |
| `99 Lab 1`  | different derivations, equivalent languages, copy-language derivation, english ambiguity, coroutines, random number generator with generators, integer scanner with generators | lecture 01          |
| `99 Lab 2`  | NLTK, precedence and associativity, grammar for $a^i b^n c^i d^n$                                                                                                              | lecture 01          |
| `99 Lab 3`  | unix grep and sed, explaining regex, writing regex, three formalizations (regex/NFA/DFA)                                                                                       | lecture 02          |
| `99 Lab 4`  | RE for odd a's even b's, minimizing an FSA, proof of shunting law                                                                                                              | lecture 02          |
| `99 Lab 5`  | determinizing and minimizing FSA, Flex [tutorial], LL and LR grammars, attribute grammar for binary numbers, analyzing RE grammar                                              | lectures 02, 03, 04 |
| `99 Lab 6`  | backreferences [tutorial], evaluating arithmetic expressions, infix to postfix, static vs dynamic binding                                                                      | lecture 04, 05      |
| `99 Lab 7`  | ASCII operators, explaining WASM, translating to WASM, valid WASM, sum of 10 numbers (WASM)                                                                                    | lecture 06          |
| `99 Lab 8`  | translating linear search to WASM, modifying linear search, WASM binary format [tutorial], reverse-engineering textual WASM                                                    | lectures 06, 07     |
| `99 Lab 9`  | shift left and shift right (WASM), parameter passing [tutorial], parameter passing                                                                                             | lecture 07          |
| `99 Lab 10` | annotating risc-v code                                                                                                                                                         | lecture 08, 09      |
| `99 Lab 11` | steps with earley's parser, packrat parsing, packrat parsing for statements                                                                                                    | lecture 10          |

## assignments (graded, 1–11)

| dir                | question titles                                                                                                                                                      | covered in      |
| :----------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- |
| `88 Assignment 1`  | NLTK parse trees (a1.py, a3.py, a5.py)                                                                                                                               | lecture 01      |
| `88 Assignment 2`  | railroad-diagrams pretty-printer (rr-2.2-SNAPSHOT-java11), expression grammar EBNF                                                                                   | lecture 01      |
| `88 Assignment 3`  | data processing task                                                                                                                                                 | lecture 02      |
| `88 Assignment 4`  | (see results.md)                                                                                                                                                     | lecture 02      |
| `88 Assignment 5`  | sanitizing pathnames (Flex+C), removing unnecessary double quotes in CSV (sed), JSON string→native values (sed), RE with counted repetition (`{n}`, `{n,m}`, `{n,}`) | lectures 02, 04 |
| `88 Assignment 6`  | (empty dir at time of indexing)                                                                                                                                      | lecture 05      |
| `88 Assignment 7`  | half-open intervals, binary constants, translating procedures to WASM, translating recursion to WASM (fibonacci.wat, randgcd.wat)                                    | lecture 06      |
| `88 Assignment 8`  | extending the P0 library (WASM), reverse-engineering binary WASM (primes.wasm/.wat, primality.wat), extending P0 with bitwise set operations                         | lectures 06, 07 |
| `88 Assignment 9`  | algebraic optimizations in P0 (WASM), arithmetic expressions with constants, procedure parameters, call by value vs result vs reference, call by name                | lectures 07, 09 |
| `88 Assignment 10` | basic blocks and common subexpressions, annotating risc-v code, out of registers, common subexpressions in matrix operations                                         | lectures 08, 09 |
| `88 Assignment 11` | steps with earley's parser, all trees with earley's parser, arithmetic expressions with PEG, packrat parsing for statements                                          | lectures 10, 11 |

## canonical compiler modules

these travel with every lab/assignment from week 7 onwards (updated snapshots):

- `P0.ipynb` — parser & type checker
- `SC.ipynb` — scanner
- `ST.ipynb` — symbol table
- `CGast.ipynb` — AST pretty-printer codegen
- `CGwat.ipynb` — WAT codegen
- `CGriscv.ipynb` — risc-v codegen
- `CGmips.ipynb` — mips codegen

accompanying test harnesses:

- `P0ParsingTest.ipynb`, `P0TypeCheckingTest.ipynb` — parser / type check tests
- `CGastTest.ipynb`, `CGwatTest.ipynb` — codegen tests
- `SCTest.ipynb`, `STTest.ipynb` — module tests
- `P0test.ipynb` — end-to-end WASM execution

when extending the compiler for an assignment, the idiom is to modify the copy local to that assignment folder, not the lecture 05 canonical snapshot.

## auxiliary files worth noting

- `07 Further Data Types and Control Structures/exception.wat`, `indirect.wat`, `jumptable.wat`, `infiniteloop.txt`, `verylongloop.txt` — reference wasm for control-structure lectures
- `08 A RISC Architecture as Target/MIPS Green Card.pdf`, `RISC-V Reference Card.pdf`, `RISC-V Assembly Programmer's Manual.md` — ISA references
- `10 Generalized Parsing/f0.hs`, `f1.hs`, `f2.hs`, `f2maybe.hs` — haskell combinator parsers
- `10 Generalized Parsing/parsing.ml` — ocaml parser

## quick question→path lookup

- "how do i parse $a^i b^n c^i d^n$" → `99 Lab 2/03 Grammar for aⁱbⁿcⁱdⁿ.ipynb` (lecture 01 material)
- "explain regex {n,m}" → `88 Assignment 5/04 RE with Counted Repetition.ipynb` (see lecture-02-regex.md)
- "reverse engineer this wasm" → `99 Lab 8/04 Reverse-Engineering Textual WASM.ipynb` or `88 Assignment 8/02 Reverse-Engineering Binary WASM.ipynb`
- "annotate this risc-v" → `99 Lab 10/01 Annotating RISCV Code.ipynb` or `88 Assignment 10/02 Annotating RISC-V Code.ipynb`
- "out of registers" → `88 Assignment 10/03 Out of Registers.ipynb`
- "earley by hand" → `99 Lab 11/01 Steps with Earley's Parser.ipynb` or `88 Assignment 11/01, 02`
- "PEG for arithmetic" → `88 Assignment 11/03 Arithmetic Expressions with PEG.ipynb`
- "packrat for statements" → `99 Lab 11/03 Packrat Parsing for Statements.ipynb` or `88 Assignment 11/04`
- "call by name / value / reference" → `88 Assignment 9/04, 05`
- "bitwise sets in P0" → `88 Assignment 8/03 Extending P0 with Bitwise Set Operations (WASM).ipynb`
- "infix to postfix" → `99 Lab 6/02 Infix to Postfix.ipynb`
- "static vs dynamic binding" → `99 Lab 6/03 Static vs Dynamic Binding.ipynb`
