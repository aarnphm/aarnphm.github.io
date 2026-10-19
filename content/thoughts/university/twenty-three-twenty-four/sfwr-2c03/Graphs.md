---
date: '2024-02-26'
description: Graph representations and traversal, with the invariants behind DFS, BFS, connected components, two-colouring and unweighted shortest paths.
id: Graphs
modified: 2026-10-12 09:12:00 GMT-04:00
tags:
  - sfwr2c03
title: Graphs
---

See also [[thoughts/university/twenty-three-twenty-four/sfwr-2c03/elementary-graph-algorithm.pdf|slides]].

A graph records which objects are related. Vertices can carry data, and edges encode the relation we chose to study. That choice determines what [[thoughts/Information Theory|information]] the graph retains. A road graph might use intersections as vertices and road segments as edges.

Write $G = (\mathcal{N}, \mathcal{E})$, with $N = |\mathcal{N}|$ vertices and $M = |\mathcal{E}|$ edges. The definitions below use finite graphs without parallel edges; the undirected examples also exclude self-loops.

## Directed acyclic graph (DAG)

A DAG is a directed graph with no directed cycle. It admits a _topological ordering_: every edge points from an earlier vertex to a later one. This is useful for prerequisites, since a cycle would require a task to precede itself. A DAG can still have several paths between two vertices and can be disconnected.[^topological]

Application: [[thoughts/Merkle DAG|Merkle DAG]].

## undirected.

> [!note] $G = (\mathcal{N}, \mathcal{E})$
>
> An undirected edge is an unordered pair $\{u,v\}$ of distinct vertices:
>
> $$\mathcal{E} \subseteq \bigl\{\{u,v\}: u,v \in \mathcal{N},\ u \neq v\bigr\}.$$
>
> The edge can be followed in either direction.

A _path_ here is a sequence of distinct vertices with an edge between each consecutive pair. Its length is its number of edges. A _walk_ may repeat vertices and edges. The path containing only one vertex has length zero.

A graph is **connected** if every pair of vertices has a path between them. A connected component is a maximal set of mutually reachable vertices; an isolated vertex forms its own component.

A weighted graph also assigns a value to each edge:

$$w: \mathcal{E} \to \mathbb{R}.$$

The meaning and allowed signs of those weights depend on the problem. Distance, travel time and cost impose different assumptions.

See also [[thoughts/Group theory#Graph isomorphism|Graph isomorphism]].

## directed.

> [!note] $G = (\mathcal{N}, \mathcal{E})$
>
> A directed edge is an ordered pair:
>
> $$\mathcal{E} \subseteq \mathcal{N} \times \mathcal{N}.$$
>
> The edge $(u,v)$ goes from $u$ to $v$. It says nothing about whether $(v,u)$ exists.

> [!important] _path_ difference
>
> A directed path follows every edge in its specified direction. Reachability can therefore be one-way.

> [!note] _cycle_
>
> A directed cycle follows one or more edges back to its starting vertex, with no other repeated vertex. A self-loop is a cycle of length one.

Two vertices are **strongly connected** if each is reachable from the other. A **strongly connected component** is a maximal set with this property. Ignoring edge directions instead gives the _weakly connected_ components. Ordinary DFS from one vertex computes its reachable set; that set need not be a strongly connected component.[^scc]

## matrix [[thoughts/representations|representation]]

Assign each vertex a unique index with a bijection

$$\operatorname{id}: \mathcal{N} \to \{0,\ldots,N-1\}.$$

The adjacency matrix $A \in \{0,1\}^{N\times N}$ of a directed graph has entries

$$
A[\operatorname{id}(u),\operatorname{id}(v)] =
\begin{cases}
1 & \text{if } (u,v) \in \mathcal{E},\\
0 & \text{otherwise.}
\end{cases}
$$

For an undirected graph, use $\{u,v\} \in \mathcal{E}$ in the condition, so $A$ is symmetric. A weighted matrix stores weights together with an absence marker; zero cannot mean "no edge" if zero-weight edges are allowed.

A conventional matrix stores $\Theta(N^2)$ entries. Edge lookup and update take $\Theta(1)$ time. Listing a vertex's neighbors requires scanning a row, taking $\Theta(N)$ time even when the vertex has few neighbors.[^matrix]

## The adjacency list representation

Store an array $L[0,\ldots,N-1]$ of neighbor lists. For a directed edge $(u,v)$, put $v$ in $L[\operatorname{id}(u)]$. For an undirected edge $\{u,v\}$, put each endpoint in the other's list. Write $\operatorname{Adj}(u)$ for that list in the algorithms below.

The lists store $M$ directed entries or $2M$ undirected entries, plus one list header per vertex. Total storage is $\Theta(N+M)$.[^graph]

Assume outgoing lists are unsorted linked lists, insertion prepends a node, and vertex indices are already known. Let $d^+(u)$ be the outdegree. These are worst-case costs, including the constant cost of inspecting an empty list.[^lists]

| operation                                                  | time               |
| ---------------------------------------------------------- | ------------------ |
| Insert $(u,v)$ when absence is already known               | $\Theta(1)$        |
| Check for $(u,v)$, or remove it by endpoints               | $\Theta(1+d^+(u))$ |
| List all outgoing edges of $u$                             | $\Theta(1+d^+(u))$ |
| List incoming edges by scanning every outgoing list        | $\Theta(N+M)$      |
| Find the weight of $(u,v)$, or update it by endpoints      | $\Theta(1+d^+(u))$ |
| Read or update a weight through an existing edge reference | $\Theta(1)$        |

If insertion must reject duplicates, it first pays for lookup. An incoming-list index makes incoming traversal proportional to indegree, at the cost of maintaining both directions. Vertex insertion depends on how the outer array grows. Removing a vertex must also remove its incident edges; a full scan of outgoing lists takes $\Theta(N+M)$.

## comparison.

For a family of simple graphs, _dense_ usually means $M=\Theta(N^2)$, while _sparse_ often means $M=O(N)$. Usage varies, so state the bound when it matters. Lists save space on sparse graphs. A matrix provides constant-time edge membership tests at the cost of allocating an entry for every vertex pair.

## Traversing undirected graph.

DFS and BFS discover vertices reachable from a source $s$. Both must record a discovery before exploring onward, so a cycle cannot keep rediscovering the same vertex.

For a small example, take

$$
\mathcal{N}=\{a,b,c,d,e\},\qquad
\mathcal{E}=\bigl\{\{a,b\},\{a,c\},\{b,d\},\{c,d\}\bigr\}.
$$

Starting from $a$, either search reaches $a,b,c,d$. Vertex $e$ stays unreachable. We will inspect neighbors alphabetically when describing a particular visit order.

### Depth-first search (DFS)

DFS follows an unvisited neighbor recursively, then returns to inspect the remaining neighbors. Marking on entry prevents the undirected edge back to the parent from starting another recursive call.[^dfs]

```jsx imports={Zoomable,GraphTraversal}
<Zoomable label="DFS and BFS on the course example graph">
  <GraphTraversal
    mode="dfs"
    start={0}
    caption="The course example graph on $n_0,\ldots,n_9$, with each $\operatorname{Adj}(n)$ in ascending index order. Pick DFS or BFS and a source $s$, then step. The salmon square is the vertex at work (the top of the call stack, or the vertex just dequeued) and the salmon band is the edge it inspects. A dark outline marks a vertex on the stack or in the queue; sage marks a vertex whose call has returned (DFS) or whose list is scanned (BFS). Heavy edges form the search tree. DFS times $d$ and $f$ tick one clock at each entry and return. From $n_1$ or $n_8$ only two vertices are marked: the graph is disconnected."
  />
</Zoomable>
```

```pseudo
\begin{algorithm}
\caption{DFS-R(G, marked, n)}
\begin{algorithmic}
\REQUIRE $G = (\mathcal{N}, \mathcal{E}), \text{marked}, n \in \mathcal{N}$
\STATE $\text{marked}[n] \coloneqq \text{true}$
\FORALL{$m \in \operatorname{Adj}(n)$}
  \IF{$\neg \text{marked}[m]$}
    \STATE $\text{DFS-R}(G, \text{marked}, m)$
  \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

Initialize every vertex as unmarked, then call the procedure on the source:

$$
\text{marked} \coloneqq \{n \mapsto \text{false} \mid n \in \mathcal{N}\},
\qquad \text{DFS-R}(G,\text{marked},s).
$$

> [!important] Conclusion
>
> On return, a vertex is marked exactly when it is reachable from $s$. In the small example, the discovery order is $a,b,d,c$. Changing neighbor order can change the DFS tree and visit order; the reachable set stays the same. The graph is disconnected because $e$ is unmarked.

#### Complexity

The marked array takes $\Theta(N)$ space. The recursion stack has depth at most $N$, so auxiliary space is $O(N)$. A long path can exhaust a language runtime's call stack; an explicit stack avoids that recursion limit.

Let $R$ be the reachable vertices and $M_R$ the edges in their connected component. Initialization and traversal take $\Theta(N+M_R)$ time: every reached vertex is entered once, and each undirected edge is inspected twice, once from each endpoint. The worst case is $\Theta(N+M)$. This bound assumes adjacency lists; scanning matrix rows gives a worst-case bound of $\Theta(N^2)$.

#### Connected-components

Start a new DFS at each still-unlabelled vertex. Give it a new component label and pass that label to every vertex it reaches. The outer loop includes isolated vertices.[^cc]

```pseudo
\begin{algorithm}
\caption{DFS-CC-R(G, cc, n)}
\begin{algorithmic}
\REQUIRE $G = (\mathcal{N}, \mathcal{E}), cc[n] \neq \text{unmarked}$
\FORALL{$m \in \operatorname{Adj}(n)$}
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
\caption{COMPONENTS(G)}
\begin{algorithmic}
\REQUIRE $G = (\mathcal{N}, \mathcal{E})$
\STATE $cc \coloneqq \{ n \mapsto \text{unmarked} \mid n \in \mathcal{N} \}$
\FORALL{$n \in \mathcal{N}$}
  \IF{$cc[n] = \text{unmarked}$}
    \STATE $cc[n] \coloneqq n$
    \STATE $\text{DFS-CC-R}(G, cc, n)$
  \ENDIF
\ENDFOR
\RETURN $cc$
\end{algorithmic}
\end{algorithm}
```

Each vertex is labelled once across all calls, giving $\Theta(N+M)$ time and $O(N)$ auxiliary space. The example has components $\{a,b,c,d\}$ and $\{e\}$. Two vertices are connected exactly when their labels agree. This procedure is for undirected components; following directed edges this way does not compute strongly connected components.

#### Two-colourability

> [!important] Bipartite graph
>
> A graph is _bipartite_ if its vertices can be partitioned into two sets with every edge joining different sets. Either set may be empty.

Use colours $1$ and $-1$, reserving $0$ for uncoloured vertices. Each newly reached vertex gets the opposite colour from its parent. An edge joining equal colours proves that the current component cannot be two-coloured; return that failure through every recursive caller.[^bipartite]

```pseudo
\begin{algorithm}
\caption{DFS-TC-R(G, colors, n)}
\begin{algorithmic}
\REQUIRE $G = (\mathcal{N}, \mathcal{E}), \text{colors}[n] \neq 0$
\FORALL{$m \in \operatorname{Adj}(n)$}
  \IF{$\text{colors}[m] = 0$}
    \STATE $\text{colors}[m] \coloneqq -\text{colors}[n]$
    \IF{$\neg \text{DFS-TC-R}(G, \text{colors}, m)$}
      \RETURN $\text{false}$
    \ENDIF
  \ELSIF{$\text{colors}[m] = \text{colors}[n]$}
    \RETURN $\text{false}$
  \ENDIF
\ENDFOR
\RETURN $\text{true}$
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
    \IF{$\neg \text{DFS-TC-R}(G, \text{colors}, n)$}
      \RETURN $\text{false}$
    \ENDIF
  \ENDIF
\ENDFOR
\RETURN $\text{true}$
\end{algorithmic}
\end{algorithm}
```

Colours alternate along a path. An odd cycle would require the starting vertex to receive the opposite of its own colour when the cycle closes. An undirected graph is bipartite exactly when it has no odd cycle. The example admits the partition $\{a,d,e\}$ and $\{b,c\}$; adding $\{b,c\}$ creates the triangle $a,b,c,a$ and makes it fail. The check takes $O(N+M)$ time and $O(N)$ auxiliary space.

### Breadth-first search (BFS)

BFS uses a first-in, first-out queue. Mark the source before enqueueing it, and mark each neighbor when it is enqueued. This gives each reached vertex one queue entry.[^bfs]

```pseudo
\begin{algorithm}
\caption{BFS(G, s)}
\begin{algorithmic}
\REQUIRE $G = (\mathcal{N}, \mathcal{E}), s \in \mathcal{N}$
\STATE $\text{marked} \coloneqq \{ n \mapsto (n = s) \mid n \in \mathcal{N} \}$
\STATE $Q \coloneqq \text{a queue holding only } s$
\WHILE{$\neg\text{Empty}(Q)$}
  \STATE $n \coloneqq \text{Dequeue}(Q)$
  \FORALL{$m \in \operatorname{Adj}(n)$}
    \IF{$\neg\text{marked}[m]$}
      \STATE $\text{marked}[m] \coloneqq \text{true}$
      \STATE $\text{Enqueue}(Q, m)$
    \ENDIF
  \ENDFOR
\ENDWHILE
\RETURN $\text{marked}$
\end{algorithmic}
\end{algorithm}
```

The example is processed in order $a,b,c,d$. All vertices one edge from $a$ leave the queue before any vertex two edges away. BFS has the same worst-case time and auxiliary-space bounds as DFS with adjacency lists.

#### Single-source shortest path

For an unweighted graph, shortest means fewest edges. BFS reaches vertices in distance layers, so the first discovery of a vertex fixes its shortest distance. A shorter route would have discovered it from an earlier layer.[^bfs]

```pseudo
\begin{algorithm}
\caption{BFS-SSSP(G, s)}
\begin{algorithmic}
\REQUIRE $G = (\mathcal{N}, \mathcal{E}), s \in \mathcal{N}$
\STATE $\text{distance} \coloneqq \{ n \mapsto \infty \mid n \in \mathcal{N} \}$
\STATE $\text{parent} \coloneqq \{ n \mapsto \text{none} \mid n \in \mathcal{N} \}$
\STATE $\text{distance}[s] \coloneqq 0$
\STATE $Q \coloneqq \text{a queue holding only } s$
\WHILE{$\neg\text{Empty}(Q)$}
  \STATE $n \coloneqq \text{Dequeue}(Q)$
  \FORALL{$m \in \operatorname{Adj}(n)$}
    \IF{$\text{distance}[m] = \infty$}
      \STATE $\text{distance}[m] \coloneqq \text{distance}[n] + 1$
      \STATE $\text{parent}[m] \coloneqq n$
      \STATE $\text{Enqueue}(Q, m)$
    \ENDIF
  \ENDFOR
\ENDWHILE
\RETURN $(\text{distance},\text{parent})$
\end{algorithmic}
\end{algorithm}
```

The example gives distances $0,1,1,2,\infty$ for $a,b,c,d,e$. To recover a path to a reachable target, follow its parent pointers back to $s$, then reverse the sequence. Unreachable vertices retain infinite distance and no parent. Different neighbor orders can select different shortest paths of the same length.

This also works on directed graphs when adjacency lists contain outgoing neighbors. Unequal weights need a different algorithm: if a direct edge costs $10$ and a two-edge route costs $1+1=2$, BFS selects the direct edge because it minimizes edge count.

[^topological]: Sedgewick and Wayne, [Topological](https://github.com/kevin-wayne/algs4/blob/master/src/main/java/edu/princeton/cs/algs4/Topological.java).

[^scc]: Sedgewick and Wayne, [KosarajuSharirSCC](https://github.com/kevin-wayne/algs4/blob/master/src/main/java/edu/princeton/cs/algs4/KosarajuSharirSCC.java), for the definition and a separate SCC algorithm.

[^matrix]: Pat Morin, _Open Data Structures_, [adjacency matrices](https://opendatastructures.org/ods-python/12_1_AdjacencyMatrix_Repres.html).

[^graph]: Sedgewick and Wayne, [Graph](https://github.com/kevin-wayne/algs4/blob/master/src/main/java/edu/princeton/cs/algs4/Graph.java) and [Digraph](https://github.com/kevin-wayne/algs4/blob/master/src/main/java/edu/princeton/cs/algs4/Digraph.java).

[^lists]: Pat Morin, _Open Data Structures_, [adjacency lists](https://opendatastructures.org/ods-python/12_2_AdjacencyLists_Graph_a.html). Operation costs depend on the list implementation and on which indexes are maintained.

[^dfs]: Sedgewick and Wayne, [DepthFirstSearch](https://github.com/kevin-wayne/algs4/blob/master/src/main/java/edu/princeton/cs/algs4/DepthFirstSearch.java).

[^cc]: Sedgewick and Wayne, [CC](https://github.com/kevin-wayne/algs4/blob/master/src/main/java/edu/princeton/cs/algs4/CC.java).

[^bipartite]: Sedgewick and Wayne, [Bipartite](https://github.com/kevin-wayne/algs4/blob/master/src/main/java/edu/princeton/cs/algs4/Bipartite.java).

[^bfs]: Sedgewick and Wayne, [BreadthFirstPaths](https://github.com/kevin-wayne/algs4/blob/master/src/main/java/edu/princeton/cs/algs4/BreadthFirstPaths.java).
