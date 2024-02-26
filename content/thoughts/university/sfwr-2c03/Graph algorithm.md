---
id: Graph algorithm
tags:
  - sfwr2c03
date: "2024-02-26"
title: Graph algorithm
---

See also [[thoughts/university/sfwr-2c03/graph-algo.pdf|slides]]

*Node* as [[thoughts/Information Theory|information]] and *edges* as [[thoughts/relationships|relationship]] between [[thoughts/data|data]]

### Directed acyclic graph (DAG)

Application: [[thoughts/Merkle DAG|Merkle DAG]]

### undirected.

> [!note] $(\mathcal{N}, \mathcal{E})$
> - $\mathcal{N}$: set of vertices
> - $\mathcal{E}$: set of _undirected_ edges: $\mathcal{E} \subseteq \mathcal{N} \times \mathcal{N}$

_path_ is a sequence of nodes and edges connect two nodes.

> A graph is **connected** if there is a path between every pair of vertices.

In a weight undirected graph each edge has a weight: $w: \mathcal{E} \to \mathbb{R}$

See also [[thoughts/Group theory#Graph isomorphism|Graph isomorphism]]

### directed.

> [!note] $(\mathcal{N}, \mathcal{E})$
> - $\mathcal{N}$: set of vertices
> - $\mathcal{E}$: set of edges containing node pairs: $\mathcal{E} \subseteq \mathcal{N} \times \mathcal{N}$

> [!important] _path_ difference
> sequence of nodes and edges are directional, edges are ordered pairs.

> [!note] _cycle_
> path with at-least one edge from a node

> **Strongly component**: maximal sub-graph in which all node pairs are strongly connected.

### matrix [[thoughts/representations|representation]]

Let $\mathcal{G} = (\mathcal{N}, \mathcal{E})$ be a directed graph with $n \in \mathcal{N} \land id(n) \text{ with } 0 \leq id(n) \leq |\mathcal{N}|$

Let $M = | \mathcal{N} | \times | \mathcal{N} |$ matrix
