---
date: '2024-01-30'
description: Fisher-Yates shuffle algorithm
id: Fisher-Yates
modified: 2026-06-05 15:08:05 GMT-04:00
tags:
  - seed
title: Fisher-Yates
---

Produces an _unbiased_ permutation of $n$ items: each of the $n!$ orderings comes out with probability $1/n!$, given a uniform source of random integers.

Why it works: step $i$ picks $j$ uniformly from $i+1$ choices, so one run makes $n \cdot (n-1) \cdots 2 = n!$ equally likely choice sequences. Each sequence yields a different permutation, so the map from choice sequences to permutations is a bijection.

The common bug draws $j$ from all of $\{0, \dots, n-1\}$ at every step. That gives $n^n$ equally likely sequences mapped onto $n!$ permutations. For $n=3$ that is $27$ sequences over $6$ permutations, and $6 \nmid 27$, so some permutations must appear more often than others.

The version below is Durstenfeld's in-place form (1964), which Knuth lists as Algorithm P in TAOCP vol. 2. It costs $O(n)$ time and $O(1)$ extra space; the original 1938 pen-and-paper method struck out items and cost $O(n^2)$.

Pseudocode:

```pseudo
\begin{algorithm}
\caption{Fisher-Yates shuffle}
\begin{algorithmic}
\REQUIRE An array $A$ of length $n$
\FOR{$i = n-1$ \TO $1$}
    \STATE $j \gets$ random integer such that $0 \leq j \leq i$
    \STATE swap $A[i]$ and $A[j]$
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

Underscore.js `_.sample` runs the same swap forward and stops after $n$ steps. The first $n$ slots then hold a uniform random sample without replacement, so a partial shuffle costs $O(n)$ swaps (plus the array copy) instead of a full pass:

```js title="FisherYates.js"
function sample(obj, n, guard) {
  if (n == null || guard) {
    if (!isArrayLike(obj)) obj = values(obj)
    return obj[random(obj.length - 1)]
  }
  var sample = toArray(obj)
  var length = getLength(sample)
  n = Math.max(Math.min(n, length), 0)
  var last = length - 1
  for (var index = 0; index < n; index++) {
    var rand = random(index, last)
    var temp = sample[index]
    sample[index] = sample[rand]
    sample[rand] = temp
  }
  return sample.slice(0, n)
}
```
