---
name: lecture-01-syntax
description: language and syntax — grammars, derivations, parse trees, BNF/EBNF, chomsky hierarchy, concrete vs abstract syntax
---

# lecture 01 — language and syntax

## grammar formally

a grammar $G = (T, N, P, S)$:

- $T$ — terminals
- $N$ — nonterminals (disjoint from $T$)
- $P$ — productions: pairs $A \to \alpha$ with $A \in (T \cup N)^*$ containing at least one nonterminal, $\alpha \in (T \cup N)^*$
- $S \in N$ — start symbol

a **derivation** from $S$ substitutes the rhs for the lhs of a production until only terminals remain. the **language** $L(G) = \{ \alpha \in T^* : S \Rightarrow^* \alpha \}$.

two derivations differ if they apply productions in different orders; they may or may not yield different parse trees.

## leftmost vs rightmost derivation

- **leftmost**: always expand the leftmost nonterminal
- **rightmost**: always expand the rightmost nonterminal

for an unambiguous grammar, leftmost and rightmost derivations are unique given a sentence. **ambiguous** grammars have multiple parse trees for some sentence.

## parse trees

the tree structure of a derivation: root is $S$, each internal node is a nonterminal expanded by some production, leaves are terminals (or $\epsilon$). ambiguous grammar $\iff$ some sentence has $\ge 2$ distinct parse trees.

classic ambiguity examples:

- dangling else: `if E then if F then S else T`
- english pp-attachment: "i saw the man with the telescope"

## chomsky hierarchy

grammar type determined by the form of productions:

| type | name              | form                                                          | recognizer         |
| :--- | :---------------- | :------------------------------------------------------------ | :----------------- |
| 0    | unrestricted      | $\alpha \to \beta$, $\alpha$ has a nonterminal                | turing machine     |
| 1    | context-sensitive | $\alpha A \beta \to \alpha \gamma \beta$ ($\gamma$ non-empty) | linear-bounded TM  |
| 2    | context-free      | $A \to \gamma$                                                | pushdown automaton |
| 3    | regular           | $A \to aB$ or $A \to a$ or $A \to \epsilon$                   | finite automaton   |

programming languages are typically context-free (syntax) with some context-sensitive checks handled post-parse (type checking, scope resolution).

## BNF and EBNF

**BNF**: $A ::= \alpha \mid \beta \mid \gamma$ — each line lists alternatives.

**EBNF** extends BNF:

- $\{ \alpha \}$ — zero or more
- $[ \alpha ]$ — optional (zero or one)
- $( \alpha )$ — grouping

every EBNF grammar reduces to BNF by introducing fresh nonterminals for repetition and optionality.

## syntax diagrams

visual EBNF: boxes for nonterminals, ovals/rounded-boxes for terminals, arrows for sequencing, branching for choice, loop-back arrows for repetition. tools like `rr.war` (railroad-diagrams) generate these from EBNF.

## concrete vs abstract syntax trees

- **concrete syntax tree** (CST): reflects every grammar rule, including parens and punctuation
- **abstract syntax tree** (AST): only load-bearing structure — operators and operands, discarding parens, semicolons, keywords that don't carry semantic weight

example: $(2 + 3) \times 4$ has a CST with paren nodes but an AST like `Mul(Add(2, 3), 4)` with no paren information (the tree shape already encodes grouping).

## coroutines and generators (lab 1)

the notebook introduces python generators for scanners: a scanner yields tokens one at a time. pattern:

```python
def scanner(s: str):
  pos = 0
  while pos < len(s):
    if s[pos].isdigit():
      n, pos = 0, pos
      while pos < len(s) and s[pos].isdigit():
        n, pos = 10 * n + int(s[pos]), pos + 1
      yield ('NUM', n)
    elif s[pos].isalpha():
      ...
    else:
      pos += 1
```

the consumer uses `next()` or iteration. this is the pattern lab 1's "Integer Scanner with Generators" question wants.
