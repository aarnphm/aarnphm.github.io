---
id: Sorting
tags:
  - sfwr2c03
date: "2024-01-24"
title: Sorting
---

### correctness of `BestTwoSum`

Let $\text{TS(start, end)} = {(L[i], L[j]) \space | \space (L[i] + L[j] = w) \land (\text{start} \leq i \leq j \leq end)}$

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
```
