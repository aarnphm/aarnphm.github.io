---
name: lecture-10-general
description: generalized parsing — combinator parsers, earley's algorithm, PEG, packrat, probabilistic grammars
---

# lecture 10 — generalized parsing

four parsing strategies covered, with tradeoffs.

## combinator parsing ($LL(k)$)

each EBNF production becomes a parsing function. for production $B \to E$, define $B(s) = \text{parse } E \text{ from } s$. rules lean on exceptions for backtracking when $LL(k)$ fails.

- works for $LL(1)$ grammars out of the box
- $LL(2)$ needs two-symbol lookahead; general $LL(k)$ needs $k$-symbol
- no left-recursion (leads to infinite recursion)
- can have side effects (scanner state, error messages, attributes)

## earley's parser

works for **arbitrary CFGs** without backtracking. $O(n^3)$ worst case, $O(n^2)$ for unambiguous, $O(n)$ for "most practical" grammars.

normalize: assume start symbol $S$ appears only on the lhs of one production $S \to \pi$. if not, add $S' \to S$.

### earley items

an item is a production with a dot and an origin index: $[A \to \alpha \bullet \beta, k]$. it means: we're trying to parse $A$, have matched $\alpha$ starting at position $k$, and still need to match $\beta$.

### the state sets

$s_0, s_1, \ldots, s_n$ where $s_i$ holds all items active after consuming $i$ input symbols.

three operations:

- **predict**: $[A \to \alpha \bullet B\,\beta, k] \in s_i$ and $B \to \gamma$ is a production → add $[B \to \bullet \gamma, i]$ to $s_i$
- **scan**: $[A \to \alpha \bullet a\,\beta, k] \in s_i$ and input at position $i$ is $a$ → add $[A \to \alpha\,a \bullet \beta, k]$ to $s_{i+1}$
- **complete**: $[B \to \gamma \bullet, k] \in s_i$ (completed item) → for every $[A \to \alpha \bullet B\,\beta, j] \in s_k$, add $[A \to \alpha\,B \bullet \beta, j]$ to $s_i$

input is accepted iff $[S \to \pi \bullet, 0] \in s_n$.

### complexity

in $s_i$, up to $O(i)$ items; over $n$ positions, $O(n^2)$ items. prediction and scanning need $O(i)$ work each; completion may need $O(i^2)$. summing: $O(n^3)$.

### building the parse tree

tag each item with back-pointers during `scan` and `complete`. then trace from the accepting item back through the chain. for ambiguous grammars, multiple completing items produce multiple trees.

## parsing expression grammars (PEG)

key differences from CFG:

- **prioritized choice** $e_1 / e_2$: try $e_1$; if it fails, try $e_2$ at the same position
- **greedy repetition** $e^*$, $e^+$, $e?$: match as many as possible, no backtracking into the repetition
- no left-recursion
- ordered, deterministic
- predicates: $\&e$ (positive lookahead, succeeds if $e$ matches but consumes nothing), $!e$ (negative lookahead)

PEG example (dangling-else resolved): $\text{IfStmt} \leftarrow \texttt{'if'}\ E\ \texttt{'then'}\ S\ (\texttt{'else'}\ S)?$ — the optional clause is greedy, binding the `else` to the nearest `if`.

PEG longest-match: $\texttt{'<<='} / \texttt{'<<'} / \texttt{'<='}$ selects the longest match that appears first.

**gotcha**: $a^*\,a$ in PEG matches nothing. $a^*$ greedily consumes all `a`s; the trailing `a` can't match. write $a^+$ or $a\,a^*$.

## packrat parsing

Packrat parsing memoizes a rule's result at each input position, including failures. There are $O(|N| \times n)$ rule-position entries. Linear time for a fixed grammar also depends on bounded work per entry or memoizing the relevant subexpressions; a rule that repeatedly scans a long suffix can still cost more.

space-time tradeoff: memo table is large. worth it when the grammar has heavy shared prefixes; otherwise packrat can be slower than recursive descent.

## side effects and attribute grammars

| strategy                       | side effects during parsing?                                                     |
| :----------------------------- | :------------------------------------------------------------------------------- |
| $LL(k)$ / $LR(k)$              | yes, in rule actions                                                             |
| combinator (with backtracking) | risky — may undo                                                                 |
| packrat                        | return semantic values with memoized results; a cache does not undo side effects |
| earley                         | attach results to items; not all items are used                                  |

## probabilistic grammars

productions weighted by probability summing to $1$ per nonterminal. notation: $A \to \alpha\ @\ p$. uses: generate sentences with a probability distribution, disambiguate parses (pick tree with highest product of rule probabilities).

## comparison table

| aspect               | $LL(k)$                          | packrat                       | earley                           |
| :------------------- | :------------------------------- | :---------------------------- | :------------------------------- |
| grammar restrictions | $LL(k)$ conds, no left-recursion | no left-recursion             | arbitrary                        |
| nondeterminism       | nondet choice/rep, greedy impl   | greedy only                   | nondet choice/rep                |
| time complexity      | $O(n)$                           | $O(n)$                        | $O(n^3)$ worst, $O(n)$ practical |
| space                | $O(n)$ stack                     | $O(\lvert N \rvert \times n)$ | $O(n^2)$                         |
| side effects         | ok                               | no                            | attach to items                  |
| scannerless          | rarely                           | yes (domain-specific)         | possible                         |

## attribute grammars for parsing expression grammars

from the notebook: a parsing attribute grammar threads attributes through rule applications. example counting pairs of `a`s:

$$
\begin{aligned}
S &\leftarrow A\ !. &&\{ S.\text{count} = A.\text{count} \} \\
A &\leftarrow \texttt{'a'}\,\texttt{'a'}\,!\texttt{'a'}\,A &&\{ A.\text{count} = A_2.\text{count} + 1 \} \\
A &\leftarrow (\texttt{'a'} / \texttt{'b'})\,A &&\{ A.\text{count} = A_2.\text{count} \} \\
A &\leftarrow \epsilon &&\{ A.\text{count} = 0 \}
\end{aligned}
$$

the $!\texttt{'a'}$ ensures `aa` isn't followed by a third `a`; the prioritized choice prefers `aa` over any single character.
