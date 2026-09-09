---
name: sfwr-4tb3
description: Solve SFWR 4TB3 notebook exercises and explain or extend the course's P0 compiler.
argument-hint: '<question or notebook path>'
---

# SFWR 4TB3

Course material lives under `content/thoughts/university/twenty-five-twenty-six/sfwr-4tb3/`. Start with the supplied question and notebook. If only a lab, assignment, or title is given, use the [content index](references/content-index.md) to locate it. Read the relevant cells and local module copies; assignment snapshots can differ from lecture copies.

The question, executable definitions, and test cells determine the required answer. References below are summaries and navigation aids. Verify exact APIs, error handling, translation schemes, and register conventions in the target notebook before relying on a summary. For an unavailable JupyterHub question, locate its local mirror; request the exact cells if the mirror lacks the question.

Load only the reference relevant to the task:

| Topic                                       | Reference                                                   |
| ------------------------------------------- | ----------------------------------------------------------- |
| Course and module layout                    | [Overview](references/overview.md)                          |
| Grammars, derivations, BNF/EBNF, generators | [Syntax](references/lecture-01-syntax.md)                   |
| Regex, automata, counted repetition         | [Regular languages](references/lecture-02-regex.md)         |
| PDA, FIRST/FOLLOW, LL/LR                    | [Context-free languages](references/lecture-03-cfl.md)      |
| Attribute grammars, ASTs, type checking     | [Syntax-directed translation](references/lecture-04-sdt.md) |
| SC, ST, P0, error handling                  | [Parser](references/lecture-05-parser.md)                   |
| WAT translation and memory layout           | [Stack target](references/lecture-06-stack-wasm.md)         |
| Sets, floats, exceptions, indirect calls    | [Data types](references/lecture-07-datatypes.md)            |
| RISC-V/MIPS and register allocation         | [RISC target](references/lecture-08-risc.md)                |
| CSE, LICM, strength reduction               | [Optimization](references/lecture-09-optim.md)              |
| Earley, PEG, packrat, probabilistic parsing | [Generalized parsing](references/lecture-10-general.md)     |
| Lab/assignment 11 exercises                 | [Lab 11](references/lecture-11-lab.md)                      |
| Choosing an approach to an exercise         | [Exam playbook](references/exam-playbook.md)                |

Preserve the notebook's signatures, AST names, notation, and module boundaries. A syntax extension may lower to existing primitives; add a new AST node only when the question or required semantics calls for one. Assignment 5's counted repetition uses `repeat`/`repeatRange`, while an exercise supplying `Exp` requires its own case.

For explanations, show the requested derivation, table, or code with the course's notation. For implementation requests, edit the assignment-local copy and run the relevant supplied test cells, fixing failures introduced by the change. Distinguish executed results from hand-derived output. Report missing runtimes or notebook cells precisely, and complete the reasoning that remains possible.
