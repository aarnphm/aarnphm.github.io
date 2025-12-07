---
date: "2025-04-22"
description: formal grammar defined as 4-tuple with non-terminals, terminals, production rules, and start symbol, recognized by pushdown automata.
id: Context-Free Grammar
modified: 2025-10-29 02:15:19 GMT-04:00
tags:
  - math
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
