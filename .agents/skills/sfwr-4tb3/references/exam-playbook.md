---
name: exam-playbook
description: choose an approach to a 4TB3 exercise from its question, local code, and test cells
---

# Exercise approaches

Use the exact question and its local definitions. A title identifies the topic; the supplied API, grammar, and tests determine the implementation. Read one matching reference when it adds needed detail.

| Question                  | Useful next step                                                                                                                                                           |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Extend regex syntax       | Check whether the parser should lower to existing primitives or the exercise requires a new constructor. See [regular languages](lecture-02-regex.md).                     |
| Translate P0 to WAT       | Use the target notebook's translation scheme and code generator. Track operand order, storage class, offsets, and branch structure.                                        |
| Annotate RISC-V           | Associate each instruction with its source expression and track register contents. Check the local free-register set and calling convention.                               |
| Optimize code             | Identify the required transformation, then preserve evaluation order, side effects, traps, overflow, and division semantics. See [optimization](lecture-09-optim.md).      |
| Construct Earley sets     | Close each set under prediction and completion, including the initial set, before scanning into the next. Retain origin indices and check the accepting item.              |
| Write PEG or packrat code | Match the supplied grammar and parser language. Preserve ordered choice, zero-length success, failure caching, and full-input acceptance. See [lab 11](lecture-11-lab.md). |
| Repair code generation    | Compare the failing output with the local translation scheme, fix the owning hook, then rerun the failing case.                                                            |

For implementation work, retain the notebook's function signatures and run the supplied tests for the affected question. Add a boundary case when the extension introduces one, such as zero repetitions or an empty parse. Resolve failures caused by the change; report missing runtime dependencies if execution is unavailable. Never present a predicted result as a test run.

For a hand derivation or explanation, provide the requested reasoning and final result. A code change, full test suite, or new helper abstraction is useful only when the request needs it. Formatting and naming should follow the supplied notebook; do not infer grading rules from style preferences.
