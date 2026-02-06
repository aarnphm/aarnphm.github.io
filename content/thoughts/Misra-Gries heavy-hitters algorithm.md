---
date: '2024-10-11'
description: extends Boyer-Moore finding algorithm
id: Misra-Gries heavy-hitters algorithm
modified: 2025-10-29 02:15:28 GMT-04:00
tags:
  - algorithm
title: Misra-Gries heavy-hitters algorithm
---

one of the earliest [[thoughts/data]] streaming algorithm.

## problem.

> Given the bag $b$ of $n$ elements and an integer $k \geq 2$. Find the values that occur more than $n/k$ times in $b$

idea: two passes over the values in $b$, while storing at most $k$ values from $b$ and their number of occurrences.

Assume the bag is available in array $b[0:n-1]$ of $n$ elements, then a _==heavy-hitter==_ of bag $b$ is a value
that occurs more than $n/k$ times in $b$ for some integer $k \geq 2$

## pseudocode.

```pseudo
\begin{algorithm}
\caption{Misra--Gries}
\begin{algorithmic}
\State $t \gets \{\}$
\State $d \gets 0$
\For{$i \gets 0$ to $n-1$}
    \If{$b[i] \notin t$}
        \State $t \gets t \cup \{b[i]\}$
        \State $d \gets d + 1$
    \Else
        \State $t \gets t \cup \{b[i]\}$
    \EndIf
    \If{$d = k$}
        \State Delete $k$ distinct values from $t$
        \State Update $d$
    \EndIf
\EndFor
\end{algorithmic}
\end{algorithm}
```
