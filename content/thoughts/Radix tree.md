---
date: '2024-11-18'
description: compressed prefix trie with multi-symbol edge labels and terminal markers for stored keys.
id: Radix tree
modified: 2026-10-01 09:05:21 GMT-04:00
tags:
  - technical
title: Radix tree
---

A compressed prefix [[thoughts/university/twenty-three-twenty-four/sfwr-2c03/Hash tables|trie]]. Each edge can store several symbols, so a chain of nonterminal nodes with one child becomes a single edge. A node is _terminal_ when its root-to-node path spells a stored key. Terminal nodes may still have children.[^trie]

![[thoughts/images/Patricia_trie.svg]]

_By Claudio Rocchini - Own work, CC BY 2.5, [wikimedia](https://commons.wikimedia.org/w/index.php?curid=2118795)_

The radix counts the possible values of one branching digit. For digits of $b$ bits,

$$
r = 2^b, \qquad b \in \mathbb{N},\quad b \ge 1.
$$

Thus $r=2$ uses one bit per digit; $r=256$ uses one byte. Larger digits reduce the number of uncompressed levels and can leave more unused child slots. A compressed edge label can span several digits.[^art]

The radix bounds a node's fan-out. The total number of internal nodes depends on the keys. For a tree with $I$ internal nodes and $L$ leaves, **if every internal node has at least two children**, counting edges gives

$$
2I \le I + L - 1
\quad\Longrightarrow\quad
I \le L - 1.
$$

A unary root or a terminal node with one child falls outside that assumption.

Take `car`, `cart`, and `cat`. The root has an edge labelled `ca`, followed by edges `r` and `t`. The `r` node stores `car` and has a further `t` edge for `cart`. Keeping its terminal marker is necessary: checking only whether the node is a leaf would reject `car`.

Looking up `cart` consumes `ca`, `r`, then `t`. Looking up `ca` reaches a nonterminal node, so it fails. Looking up `c` stops inside the first edge and also fails. These are exact-membership queries; a prefix query has a different acceptance rule.

**Lookup pseudocode**:

The root always exists, including for an empty dictionary. Outgoing labels are nonempty and have distinct first symbols. Marking the root terminal stores the empty key.

```pseudo
\begin{algorithm}
\caption{Lookup}
\begin{algorithmic}
\State $v \gets \text{root}$
\State $i \gets 0$
\While{$i < \text{length}(x)$}
    \State $e \gets$ edge from $v$ whose label starts with $x[i]$, or null
    \If{$e = \text{null}$}
        \State \Return false
    \EndIf
    \If{$e.\text{label}$ is not a prefix of $x.\text{suffix}(i)$}
        \State \Return false
    \EndIf
    \State $i \gets i + \text{length}(e.\text{label})$
    \State $v \gets e.\text{targetNode}$
\EndWhile
\State \Return $v.\text{terminal}$
\end{algorithmic}
\end{algorithm}
```

## complexity

For a query of $k$ symbols, lookup takes $O(k)$ time when selecting a child takes $O(1)$. Successful edge matches consume disjoint pieces of the query, and a mismatch ends the search. Compression reduces pointer traversals while retaining the symbol comparisons.

Insertion splits an edge where keys diverge. Deletion clears a terminal marker, removes an empty branch, and merges any newly nonterminal unary path. Both can take $O(k)$ with constant-time child access and labels represented by slices into stored keys. Copying long labels or scanning large child lists adds work.[^art]

A balanced comparison tree needs $O(\log n)$ _key comparisons_ for $n$ stored keys. Comparing strings from their beginnings can inspect $O(k)$ symbols each time, giving an $O(k\log n)$ worst-case lookup bound. Keep the units explicit: key comparisons and symbol comparisons count different operations.

[^trie]: James Aspnes, [Radix search](https://www.cs.yale.edu/homes/aspnes/pinewiki/RadixSearch.html), on prefix-free keys, terminal markers, and Patricia compression.

[^art]: Leis, Kemper, and Neumann, [The Adaptive Radix Tree](https://www.db.in.tum.de/~leis/papers/ART.pdf), sections III.A, III.E, and III.F, on digit width, path compression, and update operations.
