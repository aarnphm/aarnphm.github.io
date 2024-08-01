---
id: Graphs
tags:
  - sfwr2c03
date: "2024-02-26"
title: Graphs
---

See also [[thoughts/university/sfwr-2c03/graph-algo.pdf|slides]]

*Node* as [[thoughts/Information Theory|information]] and *edges* as relationship between [[thoughts/data|data]]

## Directed acyclic graph (DAG)

Application: [[thoughts/Merkle DAG|Merkle DAG]]

## undirected.

> [!note] $(\mathcal{N}, \mathcal{E})$
> - $\mathcal{N}$: set of vertices
> - $\mathcal{E}$: set of _undirected_ edges: $\mathcal{E} \subseteq \mathcal{N} \times \mathcal{N}$

_path_ is a sequence of nodes and edges connect two nodes.

> A graph is **connected** if there is a path between every pair of vertices.

In a weight undirected graph each edge has a weight: $w: \mathcal{E} \to \mathbb{R}$

See also [[thoughts/Group theory#Graph isomorphism|Graph isomorphism]]

## directed.

> [!note] $(\mathcal{N}, \mathcal{E})$
> - $\mathcal{N}$: set of vertices
> - $\mathcal{E}$: set of edges containing node pairs: $\mathcal{E} \subseteq \mathcal{N} \times \mathcal{N}$

> [!important] _path_ difference
> sequence of nodes and edges are directional, edges are ordered pairs.

> [!note] _cycle_
> path with at-least one edge from a node

> **Strongly component**: maximal sub-graph in which all node pairs are strongly connected.

## matrix [[thoughts/representations|representation]]

Let $\mathcal{G} = (\mathcal{N}, \mathcal{E})$ be a directed graph with $n \in \mathcal{N} \land id(n) \text{ with } 0 \leq id(n) \leq |\mathcal{N}|$

Let $M = | \mathcal{N} | \times | \mathcal{N} |$ matrix

> For every pairs of nodes $(m, n)$ set $M[id(m), id(n)] \coloneqq (m, n) \in \mathcal{E}$

## The adjacency list representation

> [!important]
> Let $A \lbrack 0 \dots |\mathcal{N}|$ be an array of _bags_
> For every edge $(m, n \in \mathcal{E})$ add $n$ to $(m,n)$ to bag $A \lbrack id(m) \rbrack$

| ops   | complexity    |
|--------------- | --------------- |
| add/remove nodes | $\Theta(\|\mathcal{N}\|)$ (copy array)   |
| add/remove edges $(n, m)$  | $\Theta(\|out(n)\|)$  (adding to bag)  |
| check an edge $(n, m)$ exists  | $\Theta(\|out(n)\|)$  (searching bags)  |
| iterate over all _incoming_ edges of $n$ | $\Theta(\|\mathcal{E}\|)$  (scan all bags)  |
| iterate over all _outgoing_ edges of $n$ | $\Theta(\|out(n)\|)$  (scan a bag)  |
| Check or change the weight of $(n, m)$ | $\Theta(1)$  |

## comparison.

> **Dense** graph: $|\mathcal{E}| \approx \Theta(|\mathcal{N}|^2)$
> **Sparse** graph: $|\mathcal{E}| \approx \Theta(|\mathcal{N}|)$

## Traversing undirected graph.

### Depth-first search (DFS)

![[thoughts/university/twenty-three-twenty-four/sfwr-2c03/images/example-graph-dfs.png]]
```pseudo
\begin{algorithm}
\caption{DFS-R(G, marked, n)}
\begin{algorithmic}
\REQUIRE $G = (\mathcal{N}, \mathcal{E}), \text{marked}, n \in \mathcal{N}$
\FORALL{$ (n, m) \in \mathcal{E} $}
  \IF{$\neq \text{marked}[m]$}
    \STATE $\text{marked}[m] \coloneqq \text{true}$
    \STATE $\text{DFS-R}{(G, \text{marked}, m)}$
  \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

$marked \coloneqq \lbrace n \longmapsto (n \neq s) \mid n \in \mathcal{N} \rbrace$


> [!important] Conclusion
>
> - all nodes to which $n_3$ is connected
> - $\mathcal{G}$ is **not** a connected graph
> - The order of recursive call determines all $n_3$ is connected to.

#### Complexity

- $|\mathcal{N}|$ memory for _marked_ and at-most $|\mathcal{N}|$ recursive calls
- inspect each node at-most once and traverse each edge once: $\Theta(|\mathcal{N}| + |\mathcal{E}|)$

#### Connected-components
```pseudo
\begin{algorithm}
\caption{DFS-CC-R(G, cc, n)}
\begin{algorithmic}
\REQUIRE $G = (\mathcal{N}, \mathcal{E}), cc, n \in \mathcal{N}$
\FORALL{$(n, m) \in \mathcal{E}$}
  \IF{$cc[m] = \text{unmarked}$}
    \STATE $cc[m] \coloneqq cc[n]$
    \STATE $\text{DFS-CC-R}(G, cc, m)$
  \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}
```
```pseudo
\begin{algorithm}
\caption{COMPONENTS(G, s)}
\begin{algorithmic}
\REQUIRE $G = (\mathcal{N}, \mathcal{E}), s \in \mathcal{N}$
\STATE $cc \coloneqq \{ n \mapsto \text{unmarked} \}$
\FORALL{$n \in \mathcal{N}$}
  \IF{$cc[n] = \text{unmarked}$}
    \STATE $cc[n] \coloneqq n$
    \STATE $\text{DFS-CC-R}(G, cc, n)$
  \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

#### Two-colourability

> [!important] Bipartite graph
>
> A graph is _bipartite_ if we can partition the nodes in two sets such that no two nodes in the same set share an edge.

```pseudo
\begin{algorithm}
\caption{DFS-TC-R(G, colors, n)}
\begin{algorithmic}
\REQUIRE $G = (\mathcal{N}, \mathcal{E}), \text{colors}, n \in \mathcal{N}$
\FORALL{$(n, m) \in \mathcal{E}$}
  \IF{$\text{colors}[m] = 0$}
    \STATE $\text{colors}[m] \coloneqq -\text{colors}[n]$
    \STATE $\text{DFS-TC-R}(G, \text{colors}, m)$
  \ELSIF{$\text{colors}[m] = \text{colors}[n]$}
    \STATE \textbf{print} "This graph is not bipartite."
  \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

```pseudo
\begin{algorithm}
\caption{TwoColors(G)}
\begin{algorithmic}
\REQUIRE $G = (\mathcal{N}, \mathcal{E})$
\STATE $\text{colors} \coloneqq \{ n \mapsto 0 \mid n \in \mathcal{N} \}$
\FORALL{$n \in \mathcal{N}$}
  \IF{$\text{colors}[n] = 0$}
    \STATE $\text{colors}[n] \coloneqq 1$
    \STATE $\text{DFS-TC-R}(G, \text{colors}, n)$
  \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

### Breadth-first search (BFS)

```pseudo
\begin{algorithm}
\caption{BFS(G, s)}
\begin{algorithmic}
\REQUIRE $G = (\mathcal{N}, \mathcal{E}), s \in \mathcal{N}$
\STATE $\text{marked} \coloneqq \{ n \mapsto (n \neq s) \mid n \in \mathcal{N} \}$
\STATE $Q \coloneqq \text{a queue holding only } s$
\WHILE{$\neg\text{Empty}(Q)$}
  \STATE $n \coloneqq \text{Dequeue}(Q)$
  \FORALL{$(n, m) \in \mathcal{E}$}
    \IF{$\neg\text{marked}[m]$}
      \STATE $\text{marked}[m] \coloneqq \text{true}$
      \STATE $\text{Enqueue}(Q, m)$
    \ENDIF
  \ENDFOR
\ENDWHILE
\end{algorithmic}
\end{algorithm}
```

#### Single-source shortest path

> Given an undirected graph *without weight* and node $s \in \mathcal{N}$, find a shortest path from node $s$ to all nodes $s$ can reach.

```pseudo
\begin{algorithm}
\caption{BFS-SSSP(G, s)}
\begin{algorithmic}
\REQUIRE $G = (\mathcal{N}, \mathcal{E}), s \in \mathcal{N}$
\STATE $\text{distance} \coloneqq \{ n \mapsto \infty \mid n \in \mathcal{N} \}$
\STATE $\text{distance}[s] \coloneqq 0$
\STATE $Q \coloneqq \text{a queue holding only } s$
\WHILE{$\neg\text{Empty}(Q)$}
  \STATE $n \coloneqq \text{Dequeue}(Q)$
  \FORALL{$(n, m) \in \mathcal{E}$}
    \IF{$\text{distance}[m] = \infty$}
      \STATE $\text{distance}[m] \coloneqq \text{distance}[n] + 1$
      \STATE $\text{Enqueue}(Q, m)$
    \ENDIF
  \ENDFOR
\ENDWHILE
\end{algorithmic}
\end{algorithm}
```
