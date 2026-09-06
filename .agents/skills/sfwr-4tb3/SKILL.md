---
name: sfwr-4tb3
description: Help with SFWR 4TB3 coursework, the P0 compiler, and related course notebooks.
argument-hint: '<question or notebook path>'
---

# sfwr-4tb3

compiler construction. everything under `content/thoughts/university/twenty-five-twenty-six/sfwr-4tb3/` is covered: 11 lectures, 11 weekly labs, 11 graded assignments, plus the P0 compiler source.

## entry rules

- start with `references/content-index.md` to locate the exact notebook for any lab/assignment mention. it has a question→path lookup table.
- read `references/overview.md` when the question needs course architecture or module layout (SC, ST, CGast, CGwat, CGriscv, CGmips, P0)
- match the question to the lecture reference:
  - grammars, derivations, chomsky hierarchy, BNF/EBNF, parse trees, coroutines/generators: `references/lecture-01-syntax.md`
  - regex / NFA / DFA / language equality / counted repetition: `references/lecture-02-regex.md`
  - pushdown automata, LL(k)/LR(k), FIRST/FOLLOW, recursive descent: `references/lecture-03-cfl.md`
  - attribute grammars, synthesized/inherited, AST construction, type checking, infix→postfix: `references/lecture-04-sdt.md`
  - P0 parser structure, scanner, symbol table, error handling, runtime data: `references/lecture-05-parser.md`
  - wasm codegen, stack translation, arrays, records: `references/lecture-06-stack-wasm.md`
  - floats, sets, bit ops, exceptions, jumptables: `references/lecture-07-datatypes.md`
  - risc-v / mips, register allocation, register annotation: `references/lecture-08-risc.md`
  - three-address code, CSE, LICM, strength reduction, constant folding: `references/lecture-09-optim.md`
  - combinator parsing, earley, PEG, packrat, probabilistic: `references/lecture-10-general.md`
  - earley step tables, packrat for statements, lab-shape problems: `references/lecture-11-lab.md`
  - "how do i attack a practice-final / lab / assignment question": `references/exam-playbook.md`
- work from the matching references and notebook. If an authorized `p0-compiler` specialist is available and the task benefits from delegation, give it the bounded question and relevant paths; otherwise complete the work locally.

## working on a jhub4tb3 notebook

the questions on `jhub4tb3.cas.mcmaster.ca/user/phama10/` are practice/exam notebooks aarnphm ran in class. the local mirror lives under `content/thoughts/university/twenty-five-twenty-six/sfwr-4tb3/`. attack a question by:

1. grab the question-name (e.g. "Regular Expressions with Exponentiation") and locate the closest lecture: regex questions → lecture 02, parsing-with-PEG → lecture 10, annotating risc-v → lecture 08/09, etc.
2. read the relevant `.ipynb` cells for the class definitions, signatures, and worked examples already in the notebook (`P0.ipynb`, `SC.ipynb`, `ST.ipynb`, `CGast.ipynb`, `CGwat.ipynb`, `CGriscv.ipynb`, `Regular Languages.ipynb`)
3. extend the existing code pattern rather than inventing new APIs. the course's idiom is pattern-matching with `match ... case ClassName(field=bind):`.
4. if the problem adds a new operator/construct, the usual three moves are:
   - add an AST class with `__init__` storing children and `__repr__` for printing
   - add a case to the recursive translator (`RegExToFSA` / `compile` / code generator)
   - if equality or nullability is asked, also extend `equalRegEx` / `matches_empty`

## voice

- match the notebook voice: Sekerinski writes precise, minimal prose with explicit EBNF grammars and tables of translation schemes. mirror that when showing code or schemes.
- when showing WAT, indent by two spaces and keep instructions on their own line.
- when showing regex AST, use the class names `ε`, `Sym`, `Choice`, `Conc`, `Star` verbatim.

## do not

- invent helper functions that don't fit the existing module boundary
- write filesystem-touching code in the parser/scanner
- introduce a Python idiom absent from the current notebook. Reuse its `match`/`case`, `frozenset`/`fset`, and class definitions; use dataclasses only when the notebook actually imports and uses them.
