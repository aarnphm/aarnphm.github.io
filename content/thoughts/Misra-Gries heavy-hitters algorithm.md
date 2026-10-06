---
date: '2024-10-11'
description: Find frequent values with bounded counters and verify them in a second pass.
id: Misra-Gries heavy-hitters algorithm
modified: 2026-10-06 09:10:15 GMT-04:00
tags:
  - algorithm
title: Misra-Gries heavy-hitters algorithm
---

Misra–Gries keeps a small set of candidate values while reading data in order. It generalises Boyer–Moore majority voting: setting $k=2$ leaves one candidate counter. The [original paper](https://www.cs.utexas.edu/~misra/scannedPdf.dir/FindRepeatedElements.pdf) gives the two-pass algorithm.

## problem.

> Given a sequence $b$ of $n$ elements and an integer $k \geq 2$, find every value whose frequency $f(x)$ satisfies $f(x)>n/k$.

There can be at most $k-1$ such _==heavy hitters==_: $k$ values each occurring more than $n/k$ times would require more than $n$ elements.

Keep a map $c$ with at most $k-1$ keys. An arriving value increments its counter, or takes a free slot with count $1$. When an untracked value arrives at a full map, decrement every counter and remove zero entries. That arrival is consumed by the cancellation step.

These counters record surviving occurrences. A second pass counts the candidates in the original sequence and applies the threshold. This requires replayable input; a one-pass stream leaves candidate values and lower bounds on their frequencies.

## pseudocode.

```pseudo
\begin{algorithm}
\caption{Misra--Gries}
\begin{algorithmic}
\Require Sequence $b$, length $n$, integer $k \geq 2$
\State $c \gets$ empty map
\For{each $x$ in $b$}
    \If{$x \in \operatorname{keys}(c)$}
        \State $c[x] \gets c[x]+1$
    \ElIf{$|c|<k-1$}
        \State $c[x] \gets 1$
    \Else
        \For{each $y$ in a copy of $\operatorname{keys}(c)$}
            \State $c[y] \gets c[y]-1$
            \If{$c[y]=0$}
                \State Remove key $y$ from $c$
            \EndIf
        \EndFor
    \EndIf
\EndFor
\For{each $y$ in $\operatorname{keys}(c)$}
    \State $c[y] \gets 0$
\EndFor
\For{each $x$ in $b$}
    \If{$x \in \operatorname{keys}(c)$}
        \State $c[x] \gets c[x]+1$
    \EndIf
\EndFor
\Return $\{x \in \operatorname{keys}(c) : k \cdot c[x]>n\}$
\end{algorithmic}
\end{algorithm}
```

## why cancellation works.

Each decrement step removes one occurrence of each of $k$ distinct values: the $k-1$ tracked values and the arriving value. This is the cancellation argument in [Misra's proof](https://www.cs.utexas.edu/~misra/Notes.dir/HeavyHitters.pdf).

Let $D$ be the number of these steps and $\widehat f(x)$ the counter at the end of the first pass, taking absent keys as zero. The removed groups contain $kD$ occurrences, so

$$
D \leq \left\lfloor\frac{n}{k}\right\rfloor.
$$

A group removes at most one occurrence of any particular value. Therefore

$$
f(x)-D \leq \widehat f(x) \leq f(x).
$$

If $f(x)>n/k$, then $\widehat f(x)>0$, so every heavy hitter survives. A survivor can still fall below the threshold in the original input, which is why the exact recount matters.

## example.

Take $k=3$ and $b=(a,b,a,c,a,b,d,e)$. The map holds at most two keys.

| After reading | Counters      |
| ------------- | ------------- |
| $a,b,a$       | $\{a:2,b:1\}$ |
| $c$           | $\{a:1\}$     |
| $a,b$         | $\{a:2,b:1\}$ |
| $d$           | $\{a:1\}$     |
| $e$           | $\{a:1,e:1\}$ |

The arrivals $c$ and $d$ each cancel three distinct occurrences. The recount gives $f(a)=3>8/3$ and $f(e)=1\leq 8/3$, so the result is $\{a\}$. Both first-pass counters were $1$; their equal residual counts concealed different frequencies.

## cost.

With expected constant-time hash-map operations, both passes take expected $O(n)$ total time and $O(k)$ words of extra space. One decrement step touches $k-1$ counters. There are at most $\lfloor n/k\rfloor$ such steps, so their total work is linear. This assumes keys and counters fit in machine words.

With a balanced search tree and constant-time key comparisons, the bound becomes $O(n\log k)$ time. The [original paper](https://www.cs.utexas.edu/~misra/scannedPdf.dir/FindRepeatedElements.pdf) studies the comparison-based setting. The memory saving is useful when $k$ is small relative to the number of distinct input values.
