---
date: '2026-05-26'
description: or just new thing
id: Sets
modified: 2026-06-04 17:08:14 GMT-04:00
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
| $A \cup B$       | [[#($A cup B$)\|union]]: $\{x \mid x \in A \text{ or } x \in B\}$              |
| $A \cap B$       | [[#($A cap B$)\|intersection]]: $\{x \mid x \in A \text{ and } x \in B\}$      |
| $A \setminus B$  | [[#difference\|difference]]: $\{x \in A \mid x \notin B\}$                     |
| $\emptyset$      | the [[#empty\|emptyset]]                                                       |
| $\mathcal{P}(A)$ | power set: all subsets of $A$                                                  |
| $A \times B$     | cartesian product                                                              |
| $\text{not } P$  | negation                                                                       |

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

"if...then" would often concern relation between _statement_, _contrapositive_ or _converse_

| logic           | notation                                   |
| --------------- | ------------------------------------------ |
| If $P$ then $Q$ | $P \implies Q$                             |
| contrapositive  | $(\text{not } Q) \implies (\text{not } P)$ |
| converse        | $Q \implies P$                             |

> note that statement and contrapositive are _logically equivalent_, but converse does not affect about its truthy/falsity of the original state
>
> Note that only _iff_ holds if _converse also holds_, i.e $P \iff Q$

## set operations

We can visualize the basic operations and rules of set theory via Venn diagrams

### ($A \cup B$)

The union contains all elements that are in $A$, or in $B$, or in both.

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[scale=1.5]
  \begin{scope}
    \clip (0,0) circle (1) (1.2,0) circle (1);
    \foreach \x in {-2.2,-1.95,...,2.7} {
      \draw[line width=0.35pt] (\x,-1.3) -- ++(2.6,2.6);
    }
  \end{scope}
  \draw[thick] (0,0) circle (1) node[left=2] {$A$};
  \draw[thick] (1.2,0) circle (1) node[right=2] {$B$};
\end{tikzpicture}
\end{document}
```

### ($A \cap B$)

The intersection contains all elements that are in both $A$ and $B$.

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[scale=1.5]
  \begin{scope}
    \clip (0,0) circle (1);
    \clip (1.2,0) circle (1);
    \foreach \x in {-0.6,-0.35,...,1.5} {
      \draw[line width=0.35pt] (\x,-1.3) -- ++(2.6,2.6);
    }
  \end{scope}
  \draw[thick] (0,0) circle (1) node[left=2] {$A$};
  \draw[thick] (1.2,0) circle (1) node[right=2] {$B$};
\end{tikzpicture}
\end{document}
```

### ($A \setminus B$)

The difference (or relative complement) contains all elements that are in $A$ but not in $B$.

It is also known as _complement_ of B relative to A, or "complement of B in A"

```tikz
\usepackage{tikz}
\definecolor{flexokired}{HTML}{fdb2a2}
\begin{document}
\begin{tikzpicture}[scale=1.5]
  \begin{scope}
    \clip (0,0) circle (1);
    \begin{scope}[even odd rule]
      \clip (-1.3,-1.3) rectangle (2.5,1.3) (1.2,0) circle (1);
      \foreach \x in {-2.2,-1.95,...,1.5} {
        \draw[line width=0.35pt] (\x,-1.3) -- ++(2.6,2.6);
      }
    \end{scope}
  \end{scope}
  \draw[thick] (0,0) circle (1) node[left=2] {$A$};
  \draw[thick] (1.2,0) circle (1) node[right=2] {$B$};
  \node[font=\normalsize] at (0.34,-1.28) {$A$};
  \draw[line width=0.45pt] (0.56,-1.18) -- (0.66,-1.38);
  \node[font=\normalsize] at (0.9,-1.28) {$B$};
\end{tikzpicture}
\end{document}
```

### distributive

For any three sets $A, B,$ and $C$:
$$A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$$

```tikz
\usepackage{tikz}
\definecolor{flexokigreen}{HTML}{cdd597}
\begin{document}
\begin{tikzpicture}[scale=1.2]
  \begin{scope}
    \clip (90:0.8) circle (1);
    \begin{scope}
      \clip (210:0.8) circle (1);
      \fill[flexokigreen, opacity=0.8] (-3,-3) rectangle (3,3);
    \end{scope}
    \begin{scope}[even odd rule]
      \clip (210:0.8) circle (1) (-3,-3) rectangle (3,3);
      \begin{scope}
        \clip (330:0.8) circle (1);
        \fill[flexokigreen, opacity=0.8] (-3,-3) rectangle (3,3);
      \end{scope}
    \end{scope}
  \end{scope}
  \draw[thick] (90:0.8) circle (1) node[above=2] {$A$};
  \draw[thick] (210:0.8) circle (1) node[below left=2] {$B$};
  \draw[thick] (330:0.8) circle (1) node[below right=2] {$C$};
\end{tikzpicture}
\end{document}
```

### De Morgan's Laws

For any three sets $A, B,$ and $C$:
$$A \setminus (B \cup C) = (A \setminus B) \cap (A \setminus C)$$

```tikz
\usepackage{tikz}
\definecolor{flexokired}{HTML}{fdb2a2}
\begin{document}
\begin{tikzpicture}[scale=1.2]
  \begin{scope}[even odd rule]
    \clip (210:0.8) circle (1) (-3,-3) rectangle (3,3);
    \begin{scope}
      \clip (330:0.8) circle (1) (-3,-3) rectangle (3,3);
      \fill[flexokired, opacity=0.5] (90:0.8) circle (1);
    \end{scope}
  \end{scope}
  \draw[thick] (90:0.8) circle (1) node[above=2] {$A$};
  \draw[thick] (210:0.8) circle (1) node[below left=2] {$B$};
  \draw[thick] (330:0.8) circle (1) node[below right=2] {$C$};
\end{tikzpicture}
\end{document}
```

> [!IMPORTANT] DeMorgan's laws verbatim
>
> _The complement of the union equals the intersection of the complements_
>
> _The complement of the intersection equals the union of the complements_

### Power set

> [!IMPORTANT] correct notation
>
> a distinction between object $a$, which is an _element of the set_ $A$, and one-element set $\{a\}$, which is a _subset of_ $A$
>
> If $A$ is the set $\{a, b, c\}$ then
>
> $a \in A,\;\;\;\;\{a\} \subset A,\;\;\;\; \{a\} \in \mathcal{P}(A)$

### Arbitrary Unions and Intersection

_union of the elements of_ $\mathcal{A}$ is defined by

$$
\bigcup_{A \in \mathcal{A}}\; A = \{x \mid x \in A \text{ for at least one } A \in \mathcal{A}\}
$$

_intersection of the elements of_ $\mathcal{A}$ is defined by

$$
\bigcap_{A \in \mathcal{A}}\; A = \{x \mid x \in A \text{ for every } A \in \mathcal{A}\}
$$

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
