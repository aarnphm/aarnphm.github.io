---
date: '2024-10-08'
description: Mechanical rules for solving a class of problems, with propositional formation rules as an example.
id: effective procedure
modified: 2026-10-06 09:10:27 GMT-04:00
tags:
  - math
title: effective procedure
---

In [[thoughts/logic]], an effective procedure gives a finite set of exact instructions that can be followed mechanically. Each step specifies what to do next, without requiring a new insight from the person carrying it out. To solve every problem in a given class, the procedure must also halt with a correct answer for every input in that class. A procedure that can run forever on some inputs needs a narrower claim about what it solves. [Stanford CS103, effective computation and decidability](https://web.stanford.edu/class/archive/cs/cs103/cs103.1222/lectures/21/Small.pdf).

## formation rules for propositional calculus

A well-formed formula (wff) is an expression built according to the grammar of a logical language. For this language, use propositional variables, negation, and four binary connectives: conjunction $\cdot$, disjunction $\vee$, implication $\supset$, and biconditional $\equiv$. The symbols $\alpha$ and $\beta$ below stand for formulas.

1. **Base rule:** every propositional variable is a wff.
2. **Negation rule:** if $\alpha$ is a wff, then $\neg\alpha$ is a wff.
3. **Binary connective rules:** if $\alpha$ and $\beta$ are wffs, then these expressions are wffs:

   $$
   (\alpha \cdot \beta),\qquad
   (\alpha \vee \beta),\qquad
   (\alpha \supset \beta),\qquad
   (\alpha \equiv \beta).
   $$

4. **Exhaustiveness:** an expression is a wff only if finitely many applications of these rules build it from propositional variables. This is an [inductive definition](https://avigad.github.io/lamr/propositional_logic.html#syntax).

For example, start with $p$ and $q$, form $\neg q$, then form $(p \vee \neg q)$. The expression $(p\;q)$ fails because the grammar has no rule that joins two formulas with a space.

These rules give an effective way to check syntax: recognize a variable, or identify an allowed outer connective and check its smaller parts. Each recursive check has a shorter input, so the process terminates. Whether a well-formed formula is true requires a separate interpretation of its variables and connectives.
