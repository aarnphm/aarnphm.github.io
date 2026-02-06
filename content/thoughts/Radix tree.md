---
date: '2024-11-18'
description: space-optimized prefix trie where single-child nodes merge with parents, enabling o(k) lookup, insertion, and deletion operations.
id: Radix tree
modified: 2025-10-29 02:15:33 GMT-04:00
tags:
  - technical
title: Radix tree
---

A prefix [[thoughts/university/twenty-three-twenty-four/sfwr-2c03/Hash tables|trie]] in which each node that is the only child is merged with its parent.

![[thoughts/images/Patricia_trie.svg]]

_By Claudio Rocchini - Own work, CC BY 2.5, [wikimedia](https://commons.wikimedia.org/w/index.php?curid=2118795)_

result: number of all internal nodes is at most the radix $r$ of the tree, where $r=2^{x} \forall x \in \mathbb{R}^d \cap x \ge 1$

Edge can be labelled with sequences of elements as well as single elements.

key at each node is compared chunk-of-bits, where quantity of bits in any given chunk is the radix $r$ of the radix tree:

- $r=2$ then radix trie is binary, which minimise sparsity at the expense of maximising trie-depth
- $r \ge 4$ is a power of two, then it is a r-ary trie, which lessen the depth at the expense of some sparseness

**Lookup pseudocode**:

```pseudo
\begin{algorithm}
\caption{Lookup}
\begin{algorithmic}
\State $\text{traverseNode} \gets \text{root}$
\State $\text{elementsFound} \gets 0$
\While{traverseNode $\neq \text{null} \land \neg \text{traverseNode}.\text{isLeaf}() \land \text{elementsFound} < \text{length}(x)$}
    \State nextEdge $\gets$ select edge from traverseNode.edges where edge.label is a prefix of $x.\text{suffix}(\text{elementsFound})$
    \If{nextEdge $\neq \text{null}$}
        \State traverseNode $\gets$ nextEdge.targetNode
        \State elementsFound $\gets$ elementsFound + length(nextEdge.label)
    \Else
        \State traverseNode $\gets$ null
    \EndIf
\EndWhile
\State \Return traverseNode $\neq \text{null} \land \text{traverseNode}.\text{isLeaf}() \land \text{elementsFound} = \text{length}(x)$
\end{algorithmic}
\end{algorithm}
```

## complexity

Permits lookup, deletion, insertion in $O(k)$ rather than $O(\log n)$

Normally $k \ge \log n$, but in a balanced tree every comparison is a string comparison requires $O(k)$ worse-case time. Whereas in a trie all comparison require constant times, but takes $m$ comparisons to look up a string length $m$
