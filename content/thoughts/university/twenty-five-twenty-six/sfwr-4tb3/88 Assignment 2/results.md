---
date: '2026-01-15'
description: AST et al.
id: results
modified: 2026-01-26 22:35:23 GMT-05:00
tags:
  - sfwr4tb3
  - assignment
title: regular constructions and language parser
---

## A1

Consider expression made up of identifiers $a, b, c, d$ and binary operators $+, -$ like shown

$$
\begin{align*}
a &+ b &+ c \\
a &- b &- c \\
a &- b &+ c - d \\
a &+ b &- c + d
\end{align*}
$$

Write grammars as below with NLTK and draw the parse trees with NLTK

> [!question] 1.
>
> Write a grammar such that $+$ binds tighter than $-$ and both $+$ and $-$ associate to the left. That is, $a+b+c$ is parsed as $\left( a+b \right)+c$ and $a-b+c-d$ as $\left( a - \left( b+c \right) \right) - d$. Draw the parse tree for $a+b+c$ and $a-b+c-d$!

> [!question] 2.
> Write a grammar such that $-$ binds tighter than $+$ and both $-$ and $+$ associate to the left. That is $a+b+c$ is parsed as as $\left( a+b \right) + c$ and $a-b+c-d$ as $\left( a-b \right) + \left( c-d \right)$. Draw the parse tree for $a+b+c$ and $a-b+c-d$!

> [!question] 3.
> Write a grammar such that $+$ and $-$ bind equally strongly and associate to the left. That is, $a+b+c$ is parsed as $\left( a+b \right)+c$ is parsed as $\left( a+b \right) + c$ and $a-b+c-d$ as $\left( \left( a-b \right) +c \right) - d$. Draw the parse tree for $a+b+c$ and $a-b+c-d$!

> [!question] 4.
> Write a grammar such that $+$ and $-$ bind equally strongly and associate to the right. That is $a+b+c$ is parsed as $a+\left( b+c \right)$ and $a-b+c-d$ as $a-\left( b+\left( c-d \right) \right)$. Draw the parses trees for $a+b+c$ and $a-b+c-d$!

> [!question] 5.
>
> Now consider expression made up of identifiers $a, b, c, d$ binary operator $+$ and unary operator $-$ such that
>
> $$
> \begin{aligned}
> &-a+b+c
> &a+-b+c
> \end{aligned}
> $$
>
> Write a grammar such that $-$ binds tighter than $+$ and $+$ associates to the left. That is, $-a+b+c$ is parsed as $\left( \left( -a \right)+b \right)+c$ and $a+-b+c$ as $a+\left( \left( -b \right) + c \right)$. Draw the parse tree for $-a+b+c$ and $a+-b+c$!

> [!question] 6.
> Write a grammar such that $+$ binds tighter than $-$ and $+$ associates to the left. That is, $-a+b+c$ is parsed as $-\left( \left( a+b \right) +c \right)$ and $a+-b+c$ as $a+\left( -\left( b+c \right) \right)$. Draw the parse trees for $-a+b+c$ and $a+-b+c$!

![[thoughts/university/twenty-five-twenty-six/sfwr-4tb3/88 Assignment 2/a1.py]]

## A2

Consider following grammar for arithmetic expressions:

```
expression  →  [ '+' | '–' ] term { ( '+' | '–' ) term }
term  →  factor { ( '×' | '/' ) factor }
factor  →  number | identifier | '(' expression ')'
```

### part 1.

Use the LaTeX `mdwtools` package to

1. pretty-print above grammar, as in:
   ![[thoughts/university/twenty-five-twenty-six/sfwr-4tb3/88 Assignment 2/img/expressiongrammar.webp]]
2. to draw the syntax diagrams of the three nonterminals, as in:
   ![[thoughts/university/twenty-five-twenty-six/sfwr-4tb3/88 Assignment 2/img/termdiagram.webp]]

For this, create a file `prettyprintgrammar.tex` from the terminal and use this as template:

```latex
\documentclass{article}
\usepackage[rounded]{syntax}

\title{Syntax Diagrams for Expressions}
\author{Aaron Pham}

\setlength{\linewidth}{160mm}
\setlength
{\textwidth}{160mm}

\begin{document}
\maketitle

\begin{grammar}
...
\end{grammar}


\begin{syntdiag} % for expression
...
\end{syntdiag}

\begin{syntdiag} % for term
...
\end{syntdiag}

\begin{syntdiag} % for factor
...
\end{syntdiag}


\end{document}

```

You can run `pdflatex` from the terminal with `pdflatex prettyprintgrammar.tex`

### part 3.

Use the railroad diagram generator RR to draw the syntax diagram! You can either use the website http://bottlecaps.de/rr/ui or run RR locally. To run RR locally:

1. Run `java -jar ./rr-2.2-SNAPSHOT-java11/rr.war -gui` inside the unzipped folder. That should print the message `Now listening on http://localhost:8080/`.
2. Open `http://localhost:8080/` in your web browser.

Note that RR uses a W3C standard for EBNF. Insert the grammar and the generated SVG or PNG diagrams in the cell below.

![[thoughts/university/twenty-five-twenty-six/sfwr-4tb3/88 Assignment 2/diagram]]

## A3

Procedure `derivable` from the course notes can be used for unrestricted grammars, not just context-sensitive grammars. The procedure will terminate with `true` or `false` for context-sensitive grammars (_decision procedure_) but may or may not terminate for unrestricted grammars (_semi-decision procedure_).

```python
from collections.abc import Iterator


class Grammar:
  def __init__(
    self, T: set[str], N: set[str], P: set[tuple[str, str]], S: str
  ):
    self.T, self.N, self.P, self.S = T, N, P, S

  def L(self, log=False, stats=False) -> Iterator[str]:
    dd, d = set(), {self.S}
    if log:
      print('    ', self.S)
    while d != set():
      if stats:
        print('# added derivations:', len(d))
      if log:
        print()
      dd.update(d)
      d = set()
      for π in sorted(dd, key=len):
        for σ, τ in self.P:  # production σ → τ
          i = π.find(σ, 0)
          while i != -1:  # π == π[0:i] + σ + π[i + len(σ):]
            χ = π[0:i] + τ + π[i + len(σ) :]
            χ = χ.replace('  ', ' ')
            if (χ not in dd) and (χ not in d):
              if all(a in self.T for a in χ.split()):
                yield χ.strip()
              if log:
                print('    ', π, '⇒', χ)
              d.add(χ)
            i = π.find(σ, i + 1)


def derivable(
  G: Grammar, ω: str, log=False, stats=False
) -> bool:  # G must be context-sensitive
  dd, d, ω = set(), {G.S}, ω.strip()
  if log:
    print('    ', G.S)
  while d != set():
    if stats:
      print('# added derivations:', len(d))
    if log:
      print()
    dd.update(d)
    d = set()
    for π in sorted(dd, key=len):
      for σ, τ in G.P:  # production σ → τ
        i = π.find(σ, 0)
        while i != -1:  # π == π[0:i] + σ + π[i + len(σ):]
          χ = π[0:i] + τ + π[i + len(σ) :]
          χ = χ.replace('  ', ' ')
          if (χ not in dd) and (χ not in d):
            if χ.strip() == ω:
              return True
            elif len(χ.strip()) <= len(ω):
              if log:
                print('    ', π, '⇒', χ)
              d.add(χ)
          i = π.find(σ, i + 1)
  return False


setattr(Grammar, 'derivable', derivable)
```

Consider the language $\{ a^{n}b^{2n}c^{n} \mid n \ge 1 \}$. Write a grammar, $G$, for this language, and use procedure `derivable` to check that `a b b c, a a b b b b c c, a a a b b b b b b c c c` are derivable, but `a b c, a b b b c, a b b c c, a a b b c c` are not.! The grammar must be monotonic, meaning context-sensitive.

![[thoughts/university/twenty-five-twenty-six/sfwr-4tb3/88 Assignment 2/a3.py]]

## A4

Explain in simple words the languages described by the following regular expressions. Avoid paraphrasing the regular expressions!

> [!question] 1
> $(a^{*}b^{*})^{*}$

All strings over the alphabet $\{a, b\}$, including the empty string.

> [!question] 2
> $(a^{*}[b])^{*}$

All strings over the alphabet $\{a, b\}$, including the empty string (same language as question 1).

> [!question] 3
> $(a^{*}ba^{*}b)^{*} a^{*}$

Strings over $\{a, b\}$ with an even number of $b$'s, including zero.

> [!question] 4
> $(a^{*}[ba^{*}c])^{*}$

Strings over $\{a, b, c\}$ where deleting all $a$'s leaves a sequence of $bc$ pairs, possibly empty.

> [!question] 5
> $(a\mid ba)^{*}[b]$

Strings over $\{a, b\}$ with no consecutive $b$'s.

> [!question] 6
> $a^{*}(ba+)^{*}$

Strings over $\{a, b\}$ where every $b$ is immediately followed by at least one $a$.
