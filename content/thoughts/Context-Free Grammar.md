---
date: '2025-04-22'
description: formal grammar defined as 4-tuple with non-terminals, terminals, production rules, and start symbol, recognized by pushdown automata.
id: Context-Free Grammar
modified: 2026-01-15 08:27:04 GMT-05:00
tags:
  - math/discrete
title: Context-Free Grammar
---

> [!abstract]
>
> 4-tuple $(N, \Sigma, P, S)$
>
> - $N$: A finite set of non-terminal
> - $\Sigma$: A finite set of terminal $N \cap \Sigma = \emptyset$
> - $P$: is a finite subset of $N \times (N \cup \Sigma)^{*}$
> - $S$: $S \in N$, the start symbols

## Pushdown Automata

> [!abstract]
>
> $M = (Q, \Sigma, \Gamma, \delta, s, \perp, F)$
