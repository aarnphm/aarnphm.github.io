---
id: Sorting
tags:
  - sfwr2c03
date: "2024-01-24"
title: Sorting
---

### correctness of `BestTwoSum`

Let $\text{TS(start, end)} = {(L[i], L[j]) \mid (L[i] + L[j] = w) \land (\text{start} \leq i \leq j \leq end)}$

```prolog
result := empty bag
i, j := 0, N-1
while i < j do
  if L[i] + L[j] = w then
    add (L[i], L[j]) to result
    i,j := i+1,j-1
  else if L[i] + L[j] < w then
    i := i+1
  else
    j := j-1
return result /* result = TS(L, 0, N-1) */
```

### selection sort.
```prolog
Input: L[0...N) of N values
For pos := 0 to N-2 do
  min := pos
  For i := pos+1 to N-1 do
    if L[i] < L[min] then
      min := i
  swap L[pos] and L[min]
```
Comparison: $\sum_{\text{pos}=0}^{N-2}(N-1-pos) = \Theta(N^2)$, changes $2(N-1) = \Theta(N)$

### insertion sort.
```prolog
Input: L[0...N) of N values
For pos := 1 to N-1 do
  v := L[pos]
  p := pos
  while p > 0 and v< L[p-1] do
    L[p] := L[p-1]
    p := p-1
  L[p] := v
```

Comparison: $\leq \text{pos} = \sum_{\text{pos}=1}^{N-1} pos = \frac{N(N-1)}{2}$, changes $\leq \text{pos} = \sum_{\text{pos}=1}^{N-1}(1+pos) = \frac{N(N-1)}{2} + N - 1$

![[thoughts/university/sfwr-2c03/images/sumary-sorting.png]]

### merge sort.
- divide-and-conquer

### A lower bound for general-purpose sorting
_assume we have a list of $L \lbrack 0 \dots N)$ of $N$ distinct values_

$S$: All possible lists $L$ that are treated the same by A such that $C: L[i] < L[j]$

> [!question]
> Can we improve mergesort O(N) memory?

### quick sort.

Complexity of quicksort

$$
T(N) = \begin{cases}
1 & \text{if } N \leq 1; \\\
T(N-1) + N & \text{if } N > 1
\end{cases}
$$

recursion tree: