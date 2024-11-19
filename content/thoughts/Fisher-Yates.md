---
id: Fisher-Yates
tags:
  - seed
date: "2024-01-30"
description: Fisher-Yates shuffle algorithm
title: Fisher-Yates
---

Produced an _unbiased_ permutation: every permutation is equally likely.

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

Implementation of modern Fisher-Yates algorithm

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
