---
date: '2026-05-26'
description: or just new thing
id: Sets
modified: 2026-06-04 14:14:20 GMT-04:00
seealso:
  - '[[thoughts/topology]]'
  - '[[thoughts/pdfs/munkres-topology.pdf|Topology, by Munkres]]'
tags:
  - math
title: Sets
---

A set is a collection of distinct objects, called its _elements_ or _members_. Set theory studies which axiom systems make this idea behave; the working axiom system is [ZFC](https://en.wikipedia.org/wiki/Zermelo–Fraenkel_set_theory).

The field began with Cantor's work on infinite cardinalities. [[thoughts/Wittgenstein#Russell's paradox and the vicious circle principle|Russell's paradox]] is the canonical obstruction you hit when first studying naive set theory: the set $R = \{x \mid x \notin x\}$ both contains and excludes itself. ZFC blocks this at the level of comprehension: the axiom of separation only lets you carve a subset $\{x \in A \mid \dots\}$ out of an existing set $A$, so the unrestricted $R$ is never formed. The axiom of foundation additionally forbids $x \in x$ chains.

Sets are the substrate for [[thoughts/algebraic geometry|algebraic structures]] and the mathematical spaces studied in #math/topology.

## notation

| symbol           | meaning                                                                        |
| ---------------- | ------------------------------------------------------------------------------ |
| $x \in A$        | $x$ is an element of $A$                                                       |
| $x \not\in A$    | $x$ is not an element/belongs to $A$                                           |
| $A \subseteq B$  | every element of $A$ is in $B$                                                 |
| $A \subsetneq B$ | A is a _proper subset_ of B given that $A \subset B$ and A is different from B |
| $A \cup B$       | union: $\{x \mid x \in A \text{ or } x \in B\}$                                |
| $A \cap B$       | intersection: $\{x \mid x \in A \text{ and } x \in B\}$                        |
| $A \setminus B$  | difference: $\{x \in A \mid x \notin B\}$                                      |
| $\emptyset$      | the empty set                                                                  |
| $\mathcal{P}(A)$ | power set: all subsets of $A$                                                  |
| $A \times B$     | cartesian product                                                              |

> [!NOTE]
>
> $\subseteq$ and $\subsetneq$ is considered ::inclusion:: and ::proper inclusion{h4}:: respectively
>
> We can also express the notion of "A and B has no {{sidenotes[common items]: We can also say that A and B are disjoint}} via the empty set", or $A \cap B = \emptyset$

### empty

_the set with no elements_, also see [[thoughts/Wittgenstein#Russell's paradox and the vicious circle principle|Russell's paradox]]

for union and intersection we can define

$$
\begin{aligned}
  A \cup \emptyset &= A \\
  A \cap \emptyset &= \emptyset
\end{aligned}
$$

### contrapositive and converse

_"if...then" would often concern relation between \_statement_, _contrapositive_ or _converse_

| logic           | notation                                   |
| --------------- | ------------------------------------------ |
| If $P$ then $Q$ | $P \implies Q$                             |
| contrapositive  | $(\text{not } Q) \implies (\text{not } P)$ |

## open

A _topology_ on a set $X$ is a collection $\tau \subseteq \mathcal{P}(X)$ whose members are called _open sets_, satisfying (munkres §12):

- $\emptyset, X \in \tau$
- arbitrary unions of open sets are open
- finite intersections of open sets are open

The pair $(X, \tau)$ is a _topological space_. The same $X$ can carry many topologies: the discrete topology ($\tau = \mathcal{P}(X)$), the indiscrete topology ($\tau = \{\emptyset, X\}$), and any topology in between.

For $X = \mathbb{R}$ with the standard topology, $U \subseteq \mathbb{R}$ is open iff every $x \in U$ has some $\varepsilon > 0$ with $(x - \varepsilon, x + \varepsilon) \subseteq U$.

## closed

A set $C \subseteq X$ is _closed_ if its complement $X \setminus C$ is open. Equivalently (munkres §17):

- $\emptyset, X$ are closed
- arbitrary intersections of closed sets are closed
- finite unions of closed sets are closed

Closed and open are not exclusive. In the discrete topology every set is both. In $\mathbb{R}$ with the standard topology, $[a, b]$ is closed, $(a, b)$ is open, and $[a, b)$ is neither. The half-open structure is what makes the lower limit topology distinct from the standard one.

The _closure_ $\overline{A}$ is the smallest closed set containing $A$; the _interior_ $\mathrm{int}(A)$ is the largest open set inside $A$. Their difference $\overline{A} \setminus \mathrm{int}(A)$ is the boundary $\partial A$.
