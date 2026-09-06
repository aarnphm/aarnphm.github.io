---
name: lecture-11-lab
description: lab 11 — step-by-step earley's parser, packrat parsing, packrat for statements; plus assignment 11 extensions
---

# lecture 11 / lab 11

the three lab 11 notebooks walk through:

1. **steps with earley's parser** — given a grammar and input, produce the sequence of earley state sets $s_0, s_1, \ldots, s_n$ by hand
2. **packrat parsing** — define a packrat parser for a small grammar in haskell or ocaml with memoization
3. **packrat parsing for statements** — extend packrat to handle statement blocks, sequencing, `if`, `while`

the assignment 11 counterparts push further:

- **all trees with earley's parser** — for ambiguous grammars, enumerate all parse trees
- **arithmetic expressions with PEG** — write the PEG for `expr`, `term`, `factor` with prioritized choice
- **packrat parsing for statements** — same as lab, but graded

## earley by hand (the method)

given grammar $G$ and input $w = a_1\,a_2\,\ldots\,a_n$:

1. start: $s_0 = \{ [S' \to \bullet S, 0] \}$ plus all predictions from it
2. for each $i$ from $0$ to $n-1$:
   - scan: for each item $[A \to \alpha \bullet a_{i+1}\,\beta, k]$ in $s_i$, add $[A \to \alpha\,a_{i+1} \bullet \beta, k]$ to $s_{i+1}$
   - in $s_{i+1}$: repeatedly apply predict (on items with nonterminal after dot) and complete (on completed items) until fixpoint
3. accept iff $[S' \to S \bullet, 0] \in s_n$

### worked example (tiny)

grammar:

$$
S \to a\,S\,b \mid \epsilon
$$

input: `a a b b`

- $s_0$: $[S' \to \bullet S, 0]$, $[S \to \bullet a\,S\,b, 0]$, $[S \to \bullet, 0]$, $[S' \to S \bullet, 0]$ (via complete)
- scan `a` ($i=0 \to 1$): $[S \to a \bullet S\,b, 0]$ into $s_1$; then predict $[S \to \bullet a\,S\,b, 1]$, $[S \to \bullet, 1]$; complete $[S \to a\,S \bullet b, 0]$
- scan `a` ($i=1 \to 2$): $[S \to a \bullet S\,b, 1]$; predict $[S \to \bullet a\,S\,b, 2]$, $[S \to \bullet, 2]$; complete $[S \to a\,S \bullet b, 1]$
- scan `b` ($i=2 \to 3$): $[S \to a\,S\,b \bullet, 1]$; complete cascades $[S \to a\,S \bullet b, 0]$
- scan `b` ($i=3 \to 4$): $[S \to a\,S\,b \bullet, 0]$; complete $[S' \to S \bullet, 0]$ → accept

## PEG for P0 statements

packrat-for-statements shape:

$$
\begin{aligned}
\text{Stmt}      &\leftarrow \text{Assign} / \text{IfStmt} / \text{WhileStmt} / \text{Block} / \text{Call} \\
\text{Assign}    &\leftarrow \text{Var}\ \texttt{':='}\ \text{Expr} \\
\text{IfStmt}    &\leftarrow \texttt{'if'}\ \text{Expr}\ \texttt{'then'}\ \text{Suite}\ (\texttt{'else'}\ \text{Suite})? \\
\text{WhileStmt} &\leftarrow \texttt{'while'}\ \text{Expr}\ \texttt{'do'}\ \text{Suite} \\
\text{Suite}     &\leftarrow \text{INDENT}\ \text{StmtList}\ \text{DEDENT} / \text{Stmt} \\
\text{StmtList}  &\leftarrow \text{Stmt}\ (\text{NEWLINE}\ \text{Stmt})^*
\end{aligned}
$$

the $/$ is prioritized choice; $*$ / $?$ are greedy. watch out for left-recursion: $\text{Expr} \leftarrow \text{Expr}\ \texttt{'+'}\ \text{Term} / \text{Term}$ is illegal; rewrite as $\text{Expr} \leftarrow \text{Term}\,(\texttt{'+'}\ \text{Term})^*$.

## packrat implementation pattern (haskell)

```haskell
parseA :: String -> Int -> Maybe (AST, Int)
parseA input pos = case Map.lookup (A, pos) memo of
  Just r  -> r
  Nothing -> let r = ...parsing logic...
             in memoInsert (A, pos) r ; r
```

in ocaml, use a mutable `Hashtbl` keyed by `(nonterminal, pos)`.

the trick: every parser function is $\text{input} \to \text{pos} \to \text{option}\,(\text{ast} \times \text{pos})$ and memoizes on $(\text{nt}, \text{pos})$. that gives $O(|N| \times n)$ entries, $O(1)$ lookup, so total time is linear.

## all-trees with earley

for ambiguous grammars, the `complete` step can produce multiple items that share completed nonterminals but different histories. to enumerate all trees:

1. for each item, record **all** back-pointer pairs that led to it, not just one
2. from $[S' \to S \bullet, 0] \in s_n$, enumerate combinations of back-pointers recursively
3. tree count grows exponentially for genuinely ambiguous sentences (this is expected)

## arithmetic with PEG

canonical answer:

$$
\begin{aligned}
\text{Expr}   &\leftarrow \text{Term}\,((\texttt{'+'} / \texttt{'-'})\,\text{Term})^* \\
\text{Term}   &\leftarrow \text{Factor}\,((\texttt{'*'} / \texttt{'/'})\,\text{Factor})^* \\
\text{Factor} &\leftarrow \texttt{'('}\,\text{Expr}\,\texttt{')'} / \text{Number} / \text{Ident} \\
\text{Number} &\leftarrow [0\text{-}9]^+ \\
\text{Ident}  &\leftarrow [a\text{-}zA\text{-}Z][a\text{-}zA\text{-}Z0\text{-}9]^*
\end{aligned}
$$

note the $+$ / $-$ inside a single choice with repetition, encoding left-associativity via iteration.

## common exam traps

- **left recursion in PEG**: illegal, must be rewritten with $*$ or $+$
- **ambiguous else in EBNF**: the straight CFG is ambiguous; PEG's greedy $?$ resolves it without grammar gymnastics
- **misreading earley items**: the origin index $k$ is where the production _started_, not the current position
- **completing a not-yet-completed item**: complete only fires when the dot is at the end
- **scanning past the input**: $s_{n+1}$ is never constructed; accept is checked in $s_n$
