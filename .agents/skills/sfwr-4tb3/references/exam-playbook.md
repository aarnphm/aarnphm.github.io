---
name: exam-playbook
description: how to attack typical 4TB3 practice final / exam questions — question shapes, attack patterns, anti-patterns
---

# exam playbook

the test is on jupyterhub. programming questions live in cells; you type python, run it, verify. the "Practice Final Exam" folder on jhub4tb3 has one `.ipynb` per question, titled like `02 Regular Expressions with Exponentiation [8 points].ipynb`.

## question-shape taxonomy

### 1. "extend regex with operator X" (lecture 02)

drill into `references/lecture-02-regex.md`. the three moves:

1. add AST class with `__init__` and `__repr__`
2. add a case to `RegExToFSA.ToFSA` (use `merge` to combine transition functions, fresh `QC` for new states)
3. extend any semantic helpers mentioned (equality, nullability, language iterator)

**tell-tale signs**: the notebook already has `class RegEx: pass`, `class ε(RegEx)`, `class Sym(RegEx)`, etc. the question gives a new operator like $E^n$, $E?$, $E^+$, $E\{m,n\}$, or a character class $[abc]$.

### 2. "translate this P0 program to WAT" (lecture 06)

- look up the matching row in the translation-scheme table
- for locals: `local.get`/`local.set`; for globals: `global.get`/`global.set`
- arrays: `i32.const lower; i32.sub; i32.const size; i32.mul; base; i32.add`, then `load`/`store`
- records: `i32.const offset(f); i32.add`, then `load`/`store` (where $\text{offset}(f)$ is the field offset)
- while/if: use block/loop/br_if if the question wants forward branches; else use `if … else … end`

### 3. "annotate this risc-v code" (lecture 08)

- match each instruction to the P0 source line it implements
- explicitly track what's in each register (e.g. $t_0 = x$, $t_1 = y$)
- flag register reuse and stack-saves

### 4. "optimize this code" (lecture 09)

- constant-fold literal expressions first
- look for repeated subexpressions (CSE)
- find loop invariants (LICM)
- look for $\times 2^n$, $\times k$ in a loop (strength reduction)
- apply algebraic identities: $x + 0 = x$, $x \times 1 = x$, $x \times 0 = 0$, $x \land \text{false} = \text{false}$

### 5. "earley parse this input by hand" (lecture 10/11)

- write each $s_i$ as a set of items $[A \to \alpha \bullet \beta, k]$
- apply predict → scan → complete to fixpoint
- check membership of $[S' \to S \bullet, 0] \in s_n$ at the end
- if asked for the parse tree: trace back-pointers

### 6. "write a PEG for this language" (lecture 10/11)

- kill left-recursion: $A \leftarrow A\,x \mid y$ becomes $A \leftarrow y\,x^*$
- use prioritized choice $/$ for ambiguity (like dangling-else)
- test against sample inputs in the notebook

### 7. "fix this broken codegen" (any codegen lecture)

- run the cell; read the error
- compare the generated code against the translation scheme
- common bugs: wrong sign (`sub` vs `add`), swapped operands, missing `local.get` before op, missing `i32.const` for constants, wrong offset for record fields

## what the grader wants

- **code that runs**: P0test.ipynb or CGwatTest.ipynb usually has the test harness. run it and watch for green
- **matches the translation scheme**: the grader has the canonical translation; deviating is risk
- **no dead code or comments**: the course style is minimal

## anti-patterns to avoid

- inventing new helper classes when you could extend an existing one
- writing WAT with 4-space indent (the notebook uses 2)
- using f-strings with embedded WAT operators instead of the `emit(...)` pattern
- forgetting to `releaseReg` after consuming a register in risc-v codegen
- writing `x++` style; P0 and python use $x := x + 1$
- adding try/except for flow control; the course uses `mark('error msg')` for parser errors

## how to use this skill during an exam

1. read the question, identify which lecture's playbook applies
2. open the matching reference file in this skill
3. open the relevant notebook — read the existing class/function signatures
4. extend, don't replace
5. run the test cell
6. if green, move on. if red, read the traceback and fix the specific line
