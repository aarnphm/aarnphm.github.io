---
date: '2024-01-24'
description: sorting algorithms including selection sort, insertion sort, merge sort, quicksort with complexity analysis and lower bounds for comparison-based sorting.
id: Sorting
modified: 2026-06-05 15:08:41 GMT-04:00
tags:
  - sfwr2c03
title: Sorting
---

### correctness of `BestTwoSum`

Sorting puts values in order while preserving every input occurrence. That order can also rule out candidates in a search.

For this version of `BestTwoSum`, assume $L[0\dots N)$ is sorted in strictly increasing order and $w$ is the target sum. Each pair uses two distinct positions. Define

$$
\operatorname{TS}(a,b)
= \{(L[p],L[q]) \mid a \leq p < q \leq b,\ L[p]+L[q]=w\}.
$$

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
return result /* result = TS(0, N-1) */
```

At the start of each iteration, the pairs already found together with $\operatorname{TS}(i,j)$ are exactly $\operatorname{TS}(0,N-1)$:

- If $L[i]+L[j]<w$, every remaining partner for $L[i]$ is at most $L[j]$. None can reach the target, so advancing $i$ discards no answer.
- If $L[i]+L[j]>w$, every remaining partner for $L[j]$ is at least $L[i]$. Decreasing $j$ is safe for the same reason.
- On equality, record the pair. Distinct values mean neither endpoint has another partner for this target, so both pointers can move.

The interval shrinks each time. When $i\geq j$, it contains no pair of distinct positions, so the result is complete. The scan takes $\mathcal{O}(N)$ time and $\Theta(1)$ auxiliary space, excluding the returned pairs.

The distinct-value assumption matters. With $L=[1,1,2,2]$ and $w=3$, the code returns two copies of $(1,2)$, while there are four matching index pairs. Enumerating all index pairs with duplicates requires handling their multiplicities and can produce $\Theta(N^2)$ output. With a single value $L=[2]$ and $w=4$, there is no answer because the same position cannot be used twice.

### selection sort.

Before iteration `pos`, the prefix contains the smallest `pos` values in sorted order. The inner loop finds a minimum of the remaining suffix. Swapping it into `pos` extends that prefix by one value.

```prolog
Input: L[0...N) of N values
For pos := 0 to N-2 do
  min := pos
  For i := pos+1 to N-1 do
    if L[i] < L[min] then
      min := i
  swap L[pos] and L[min]
```

For $N\geq 1$, the number of key comparisons is independent of input order:

$$
\sum_{\mathrm{pos}=0}^{N-2}(N-1-\mathrm{pos})
= \frac{N(N-1)}{2} = \Theta(N^2).
$$

The code makes $N-1$ swaps, including swaps of a position with itself. Counting two array writes per swap gives $2(N-1)$ writes; assignments to a temporary variable are excluded. The algorithm uses $\Theta(1)$ auxiliary space. It is unstable: swapping the minimum of $[2_a,2_b,1]$ into the first position leaves $[1,2_b,2_a]$, reversing the two equal keys.

### insertion sort.

Here the prefix is sorted, but it need not contain the smallest values in the whole list. Save the next value, shift each larger predecessor right, then put the saved value in the gap.

```prolog
Input: L[0...N) of N values
For pos := 1 to N-1 do
  v := L[pos]
  p := pos
  while p > 0 and v < L[p-1] do
    L[p] := L[p-1]
    p := p-1
  L[p] := v
```

At outer iteration `pos`, there are at most $\mathrm{pos}$ key comparisons and $\mathrm{pos}$ shifts, followed by one array write for `v`. For $N\geq 1$, summing over the outer iterations gives

$$
C \leq \sum_{p=1}^{N-1}p=\frac{N(N-1)}{2},
\qquad
W \leq \sum_{p=1}^{N-1}(p+1)=\frac{N(N-1)}{2}+N-1.
$$

These count comparisons of values and writes to the array, excluding index checks and local assignments. Reverse-sorted distinct input reaches both bounds. Already-sorted input needs one key comparison and one array write per insertion, giving $\Theta(N)$ time.

An inversion is a pair of positions whose values are in the wrong order. Each shift removes one inversion involving the saved value. If the input has $I$ inversions, the running time is $\Theta(N+I)$. The strict comparison `v < L[p-1]` leaves equal predecessors in place, so this version is stable. Stability means equal keys retain their original relative order. [Cornell's sorting notes](https://www.cs.cornell.edu/courses/cs2110/2025fa/lectures/lec07/) give the sorted-prefix argument and stability definition.

| Algorithm     | Worst-case key comparisons | Worst-case array writes | Auxiliary space | Stable? |
| ------------- | -------------------------- | ----------------------- | --------------- | ------- |
| SelectionSort | $\Theta(N^2)$              | $\Theta(N)$             | $\Theta(1)$     | No      |
| InsertionSort | $\Theta(N^2)$              | $\Theta(N^2)$           | $\Theta(1)$     | Yes     |

### merge sort.

Split the array in half, sort each half recursively, then merge the two sorted ranges. During the merge, the next output is the smaller of the two first unconsumed values. Once one range is exhausted, copy the rest of the other range.

For example, merging $[1,4,7]$ with $[2,3,8]$ emits $1,2,3,4,7,8$. Each output removes one value from the unconsumed input, so merging ranges with a combined length of $N$ takes $\Theta(N)$ time.

For $N>1$, the recurrence is

$$
T(N)=T(\lfloor N/2\rfloor)+T(\lceil N/2\rceil)+\Theta(N),
\qquad T(0),T(1)=\Theta(1).
$$

Each level processes $\Theta(N)$ values in total, and there are $\Theta(\log N)$ levels. This gives $\Theta(N\log N)$ time for the standard algorithm. A reusable merge buffer needs $\Theta(N)$ auxiliary space; the recursion stack needs $\Theta(\log N)$. Taking the value from the left range first on ties preserves stability. See [MIT 6.006, lecture 3](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/mit6_006f11_lec03/).

### A lower bound for general-purpose sorting

Assume the algorithm learns about keys only through comparisons. For $N$ distinct keys, there are $N!$ possible input orders. A correct algorithm must distinguish them because each order requires its own permutation of the input positions.

Represent the comparisons as a binary decision tree, with one branch for each outcome of a comparison between distinct keys. A tree of height $h$ has at most $2^h$ leaves, so

$$
2^h\geq N!
\quad\Longrightarrow\quad
h\geq\lceil\log_2(N!)\rceil=\Omega(N\log N).
$$

For even $N$, the largest $N/2$ factors in $N!$ are each at least $N/2$, so $\log_2(N!)\geq(N/2)\log_2(N/2)$. This gives the lower bound directly. It applies to worst-case comparison sorting. Counting sort and radix sort use information about key representations, so they fall outside this model. [MIT 6.006, lecture 7](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/mit6_006f11_lec07/) develops this distinction.

> [!question]
> Can we improve mergesort's $\Theta(N)$ auxiliary memory?

The buffer belongs to this array-merge implementation. It is not a space lower bound for sorting. Heapsort gives worst-case $\Theta(N\log N)$ time with $\Theta(1)$ auxiliary space, at the cost of stability. In-place merging techniques also exist, with more involved implementations. A linked-list merge can relink nodes without an array buffer, though a recursive version still uses stack space.

### quick sort.

Choose a pivot and partition the remaining values into values smaller and larger than it. Sorting those parts completes the sort because partitioning has already placed them on the correct side of the pivot. Equal keys require an explicit rule; three-way partitioning groups them together and recurses only on the smaller and larger parts.

For distinct keys and linear-time partitioning, if the pivot has rank $k$,

$$
T(N)=T(k)+T(N-k-1)+\Theta(N).
$$

Consistently balanced splits give $\Theta(N\log N)$ time. Choosing an extreme key each time gives the worst case:

$$
T(N) = \begin{cases}
\Theta(1) & \text{if } N \leq 1, \\
T(N-1) + \Theta(N) & \text{if } N > 1.
\end{cases}
$$

The recursion tree is then a chain: partition work is proportional to $N+(N-1)+\cdots+2=\Theta(N^2)$. For example, always choosing the first key as pivot gives these splits on already-sorted distinct input.

A uniformly random pivot gives expected $\Theta(N\log N)$ time for any fixed input of distinct keys. The expectation is over pivot choices, and the quadratic worst case remains possible. Ordinary recursive quicksort can use $\Theta(N)$ stack space in that worst case. Recursing on the smaller partition and processing the larger one in a loop bounds stack space by $\mathcal{O}(\log N)$, since each nested call handles at most half as many values. [Cornell CS 2112](https://www.cs.cornell.edu/courses/cs2112/2021fa/lectures/lecture.html?id=sorting) covers partitioning and randomized expected time.
