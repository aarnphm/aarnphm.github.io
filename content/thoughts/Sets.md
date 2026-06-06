---
date: '2026-05-26'
description: basic set theory, operations, cardinality, topology, and ZFC.
id: Sets
modified: 2026-06-05 23:55:12 GMT-04:00
seealso:
  - '[[thoughts/topology|topology]]'
  - '[[thoughts/pdfs/munkres-topology.pdf|Topology, by Munkres]]'
  - '[[thoughts/pdfs/Basic Topology, Amstrong.pdf|Basic Topology, Amstrong]]'
tags:
  - math/sets
  - math/topology
title: Sets
---

A set is a collection of distinct objects, called its _elements_ or _members_.

Set theory studies which axiom systems make this idea behave; the working axiom system is [ZFC](https://en.wikipedia.org/wiki/Zermelo–Fraenkel_set_theory).

The field began with Cantor's work on infinite cardinalities.

[[thoughts/Wittgenstein#Russell's paradox and the vicious circle principle|Russell's paradox]] is the canonical obstruction you hit when first studying naive set theory: the set $R = \{x \mid x \notin x\}$ both contains and excludes itself.

[[#Zermelo-Fraenkel set theory|ZFC]] addresses this via comprehension, where the axiom of separation only lets you carve a subset $\{x \in A \mid \dots\}$ out of an existing set $A$, so the unrestricted $R$ is never {{sidenotes[formed.]: The axiom of foundation additionally forbids $x \in x$ chains.}}

Sets are the substrate for [[thoughts/algebraic geometry|algebraic structures]] and the mathematical spaces studied in #math/topology.

## notation

| symbol           | meaning                                                                   |
| ---------------- | ------------------------------------------------------------------------- |
| $x \in A$        | $x$ is an element of $A$                                                  |
| $x \not\in A$    | $x$ is not an element of $A$                                              |
| $A \subseteq B$  | every element of $A$ is in $B$                                            |
| $A \subsetneq B$ | $A$ is a _proper subset_ of $B$: $A \subseteq B$ and $A \neq B$           |
| $A \cup B$       | [[#($A cup B$)\|union]]: $\{x \mid x \in A \text{ or } x \in B\}$         |
| $A \cap B$       | [[#($A cap B$)\|intersection]]: $\{x \mid x \in A \text{ and } x \in B\}$ |
| $A \setminus B$  | [[#difference\|difference]]: $\{x \in A \mid x \notin B\}$                |
| $\emptyset$      | [[#empty\|emptyset]]                                                      |
| $\mathcal{P}(A)$ | [[#power set\|power set]]: all subsets of $A$                             |
| $A \times B$     | [[#Cartesian products\|Cartesian products]]                               |
| $\neg P$         | negation                                                                  |

> [!NOTE]
>
> $\subseteq$ and $\subsetneq$ are ::inclusion:: and ::proper inclusion{h4}:: respectively
>
> We can also express the notion that "$A$ and $B$ have no {{sidenotes[common items]: We can also say that $A$ and $B$ are disjoint.}}" via the empty set, or $A \cap B = \emptyset$

## set-builder notation

The [set-builder notation](https://en.wikipedia.org/wiki/Set-builder_notation) {{sidenotes[expression]: this is domain-bound format, which is safe.}} $\{x \in A \mid P(x)\}$ means _start with an existing set $A$, then keep exactly the elements satisfying $P$_.

$$
\{x \in A \mid P(x)\} \subseteq A
$$

> [!caution] naive form
>
> $\{x \mid P(x)\}$
>
> _this actually assumes that every predicate determines a set, where Russell's paradox would then choose_ $P(x)$ _to be_ $x \notin x$

ZFC solves this via separations:

$$
\forall\;A\;\exists\;B\;\forall x\;(x \in B \iff x \in A \land P(x))
$$

> [!math] extensionality
>
> Sets are determined by their elements:
>
> $$
> \forall\;A\;\forall\;B\;(\forall x\;(x \in A \iff x \in B) \implies A = B)
> $$
>
> Order and repetition ::does not matter:: when we consider members of {{sidenotes[a set.]: We will consider [surjective](https://en.wikipedia.org/wiki/Bijection,_injection_and_surjection#Surjection), [bijective](https://en.wikipedia.org/wiki/Bijection,_injection_and_surjection#Bijection), and [injective](https://en.wikipedia.org/wiki/Bijection,_injection_and_surjection#Injection) sets afterwards.}}
>
> $$
> \{1,2,3\} = \{3,2,1,1\}\;\; \text{ (axiom of extensionality)}
> $$

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

> A statement and its contrapositive are _logically equivalent_. The converse can have a different truth value.
>
> If the converse also holds, then $P \iff Q$.

The universal and existential quantifiers are the grammar behind most set statements:

| logic                         | notation           |
| ----------------------------- | ------------------ |
| for all elements              | $\forall x \in A$  |
| there exists an element       | $\exists x \in A$  |
| there exists a unique element | $\exists!x \in A$  |
| no element exists             | $\nexists x \in A$ |

Negation flips quantifiers:

$$
\neg(\forall x \in A,\;P(x)) \iff \exists x \in A,\;\neg P(x)
$$

$$
\neg(\exists x \in A,\;P(x)) \iff \forall x \in A,\;\neg P(x)
$$

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
    \foreach \x in {-0.6,-0.35,...,2.7} {
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

It is also known as the _complement_ of $B$ relative to $A$, or the complement of $B$ in $A$.

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

For sets $A,B,C$:

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

> [!NOTE] order of operations
>
> neither commutative nor associative
>
> $A \cup (B \cap C)$ and $(A \cup B) \cap C$:
>
> ```tikz
> \usepackage{tikz}
> \definecolor{flexokigreen}{HTML}{cdd597}
> \begin{document}
> \begin{tikzpicture}[scale=0.9]
>   % Define circles
>   \def\circleA{(0,0) circle (1.3)}
>   \def\circleB{(1.3,0.9) circle (1.3)}
>   \def\circleC{(1.3,-0.9) circle (1.3)}
>
>   % Left diagram: A \cup (B \cap C)
>   \begin{scope}[shift={(0,0)}]
>     % Shade A
>     \fill[flexokigreen, opacity=0.8] \circleA;
>     % Shade B \cap C
>     \begin{scope}
>       \clip \circleB;
>       \fill[flexokigreen, opacity=0.8] \circleC;
>     \end{scope}
>
>     % Draw outlines
>     \draw[thick] \circleA;
>     \draw[thick] \circleB;
>     \draw[thick] \circleC;
>
>     % Labels
>     \node at (-1.6, 0) {\textbf{\textsf{A}}};
>     \node at (2.9, 0.9) {\textbf{\textsf{B}}};
>     \node at (2.9, -0.9) {\textbf{\textsf{C}}};
>
>     \node[below] at (0.65, -2.5) {\textbf{\textsf{\textit{A} $\cup$ (\textit{B} $\cap$ \textit{C})}}};
>   \end{scope}
>
>   % Right diagram: (A \cup B) \cap C
>   \begin{scope}[shift={(7,0)}]
>     % Shade (A \cup B) \cap C
>     \begin{scope}
>       \clip \circleC;
>       \fill[flexokigreen, opacity=0.8] \circleA;
>       \fill[flexokigreen, opacity=0.8] \circleB;
>     \end{scope}
>
>     % Draw outlines
>     \draw[thick] \circleA;
>     \draw[thick] \circleB;
>     \draw[thick] \circleC;
>
>     % Labels
>     \node at (-1.6, 0) {\textbf{\textsf{A}}};
>     \node at (2.9, 0.9) {\textbf{\textsf{B}}};
>     \node at (2.9, -0.9) {\textbf{\textsf{C}}};
>
>     \node[below] at (0.65, -2.5) {\textbf{\textsf{(\textit{A} $\cup$ \textit{B}) $\cap$ \textit{C}}}};
>   \end{scope}
>
> \end{tikzpicture}
> \end{document}
> ```

under the same [[#de Morgan's laws|de Morgan's laws]] we include the _second_ distributive law

$$
\begin{aligned}
  A \setminus (B\cup C) &= (A \setminus B) \cap (A \setminus C) \\
  A \setminus (B\cap C) &= (A \setminus B) \cup (A \setminus C)
\end{aligned}
$$

### de Morgan's laws

For sets $A,B,C$:

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

> [!IMPORTANT] de Morgan's laws verbatim
>
> _The complement of the union equals the intersection of the complements_
>
> _The complement of the intersection equals the union of the complements_

### sets algebra

For a fixed universe $X$, we write $A^c = X \setminus A$.

| law             | union form                              | intersection form                       |
| --------------- | --------------------------------------- | --------------------------------------- |
| identity        | $A \cup \emptyset = A$                  | $A \cap X = A$                          |
| domination      | $A \cup X = X$                          | $A \cap \emptyset = \emptyset$          |
| idempotent      | $A \cup A = A$                          | $A \cap A = A$                          |
| commutative     | $A \cup B = B \cup A$                   | $A \cap B = B \cap A$                   |
| associative     | $(A \cup B) \cup C = A \cup (B \cup C)$ | $(A \cap B) \cap C = A \cap (B \cap C)$ |
| absorption      | $A \cup (A \cap B) = A$                 | $A \cap (A \cup B) = A$                 |
| complement      | $A \cup A^c = X$                        | $A \cap A^c = \emptyset$                |
| double negative | $(A^c)^c = A$                           | $(X \setminus A)^c = A$                 |

Set difference is intersection with a complement:

$$
A \setminus B = A \cap B^c
$$

The symmetric difference keeps elements that appear in exactly one set:

$$
A\;\triangle\;B = (A \setminus B) \cup (B \setminus A)
$$

Equivalently, membership in $A \triangle B$ is exclusive-or:

$$
x \in A\;\triangle\;B \iff (x \in A \land x \notin B) \lor (x \notin A \land x \in B)
$$

### power set

> [!IMPORTANT] correct notation
>
> a distinction between object $a$, which is an _element of the set_ $A$, and one-element set $\{a\}$, which is a _subset of_ $A$
>
> If $A$ is the set $\{a, b, c\}$ then
>
> $a \in A,\;\;\;\;\{a\} \subset A,\;\;\;\; \{a\} \in \mathcal{P}(A)$

The clean model is a subset as its characteristic function:

$$
S \subseteq A \quad\leftrightarrow\quad \chi_S: A \to \{0,1\}
$$

where $\chi_S(x)=1$ exactly when $x \in S$. Thus $\mathcal{P}(A)$ bijects with $\{0,1\}^A$, the set of all functions from $A$ to $\{0,1\}$.

For $A = \{a,b,c\}$, fix the coordinate order $(a,b,c)$. A subset is then a length-$3$ bit-vector:

$$
\{a,c\} \leftrightarrow 101
$$

Thus $\mathcal{P}(A)$ bijects with $\{0,1\}^3$:

$$
\mathcal{P}(\{a,b,c\}) \cong \{0,1\}^3
$$

If $A$ has $n$ elements, then each subset of $A$ is one binary string in $\{0,1\}^n$. Each coordinate answers one membership question, so $n$ yes/no questions give $2^n$ subsets.

> [!math] Cantor's theorem
>
> For every set $A$, there is no surjection $f: A \to \mathcal{P}(A)$.
>
> Proof:
>
> assume a surjection $f$ exists, and form the diagonal set
>
> $$
> D = \{a \in A \mid a \notin f(a)\}
> $$
>
> Since $f$ is surjective, $D = f(d)$ for some $d \in A$. Then $d \in D \iff d \notin f(d) \iff d \notin D$, contradiction.
>
> Therefore $\mathcal{P}(A)$ has strictly larger cardinality than $A$.

### arbitrary unions and intersection

_union of the elements of_ $\mathcal{A}$ is defined by

$$
\bigcup_{A \in \mathcal{A}}\; A = \{x \mid x \in A \text{ for at least one } A \in \mathcal{A}\}
$$

_intersection of the elements of_ $\mathcal{A}$ is defined by

$$
\bigcap_{A \in \mathcal{A}}\; A = \{x \mid x \in A \text{ for every } A \in \mathcal{A}\}
$$

> [!IMPORTANT] universality of emptyset
> If $\emptyset \in \mathcal{A}$, the union is not forced to be empty. The empty set contributes no elements, then the other members still contribute theirs.
>
> If $\mathcal{A} = \emptyset$, then $\bigcup_{A \in \mathcal{A}} A = \emptyset$.
>
> If $\mathcal{A} = \emptyset$ and we are working inside a universe $X$, then every $x \in X$ vacuously satisfies the defining property for intersection, so
>
> $$
> \bigcap_{A \in \emptyset} A = X
> $$

An _indexed family_ of sets is a function $I \to \mathcal{P}(X)$, usually written $\{A_i\}_{i \in I}$ instead of $i \mapsto A_i$.

$$
\bigcup_{i \in I} A_i = \{x \in X \mid \exists i \in I,\;x \in A_i\}
$$

$$
\bigcap_{i \in I} A_i = \{x \in X \mid \forall i \in I,\;x \in A_i\}
$$

The index set $I$ is bookkeeping; the sets $A_i$ are the mathematical objects. Different indices may name the same subset.

> [!note] repetition value within a family
>
> A family can have repeated values because it is a function out of $I$. A set cannot have repeated elements because extensionality deletes repetitions.

### Cartesian products

_notion of ordered pair_ over to general sets. We define a _Cartesian product_ $A \times B$ to be the set of all ordered pairs $(a,b)$ for which $a$ is an element of $A$ and $b$ is an element of $B$.

$$
A \times B = \{(a,b) \mid\;a \in A \text{ and } b \in B\}
$$

> This assumes that the concept of "ordered pair" is given. as in $(a,b) = \{\{a\}, \{a,b\}\}$ defines the ::ordered pair:: $(a,b)$ as a {{sidenotes[collection of sets]: if $a \neq\; b$ then this definition says that $(a,b)$ is a collection containing two sets, one of which is a one-element set and the other a two-element set.<br/><br/>if $a = b$ then $(a,b)$ is a collection containing only one set $\{a\}$ since $\{a,b\} =\{a,a\}=\{a\}$ in this case.}}.
>
> The _first coordinate_ of the ordered pair is defined to be the ==element belonging to both sets==
>
> The _second coordinate_ is the element belonging to only **one of the sets**

## relations

A binary relation from $A$ to $B$ is a subset $R \subseteq A \times B$. If $(a,b) \in R$, write $aRb$.

On a set $A$, a relation $R \subseteq A \times A$ can have extra structure:

| property      | meaning                            |
| ------------- | ---------------------------------- |
| reflexive     | $\forall a \in A,\;aRa$            |
| symmetric     | $aRb \implies bRa$                 |
| antisymmetric | $(aRb \land bRa) \implies a = b$   |
| transitive    | $(aRb \land bRc) \implies aRc$     |
| total         | $\forall a,b \in A,\;aRb \lor bRa$ |

An _equivalence relation_ is reflexive, symmetric, and transitive. It partitions $A$ into equivalence classes:

$$
[a] = \{x \in A \mid x \sim a\}
$$

The quotient set $A/{\sim}$ is the set of all equivalence classes.

A _partial order_ is reflexive, antisymmetric, and transitive.

A _total order_ is a partial order where any two elements are comparable.

## functions

A function $f: A \to B$ is a relation $f \subseteq A \times B$ such that every $a \in A$ appears exactly once as a first coordinate.

$$
\forall a \in A\;\exists! b \in B\;(a,b) \in f
$$

The domain is $A$, the codomain is $B$, and the image is

$$
f(A) = \{b \in B \mid \exists a \in A,\;f(a)=b\}
$$

For $S,U \subseteq A$ and $T,V \subseteq B$:

$$
f(S) = \{f(s) \mid s \in S\}
$$

$$
f^{-1}(T) = \{a \in A \mid f(a) \in T\}
$$

[[thoughts/preimages|Preimages]] preserve the Boolean operations exactly:

$$
f^{-1}(T \cup V) = f^{-1}(T) \cup f^{-1}(V)
$$

$$
f^{-1}(T \cap V) = f^{-1}(T) \cap f^{-1}(V)
$$

$$
f^{-1}(B \setminus T) = A \setminus f^{-1}(T)
$$

Images preserve unions:

$$
f(S \cup U) = f(S) \cup f(U)
$$

Images only preserve intersections one way in general:

$$
f(S \cap U) \subseteq f(S) \cap f(U)
$$

Equality holds when $f$ is injective.

| type       | condition                                   |
| ---------- | ------------------------------------------- |
| injective  | $f(a)=f(a') \implies a=a'$                  |
| surjective | $\forall b \in B\;\exists a \in A,\;f(a)=b$ |
| bijective  | injective and surjective                    |

```tikz
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\definecolor{flexokired}{HTML}{fdb2a2}
\definecolor{flexokigreen}{HTML}{cdd597}
\begin{document}
\begin{tikzpicture}[
    scale=0.9,
    dot/.style={circle, fill=black, inner sep=1.5pt},
    arrow/.style={-{Stealth[scale=1.2]}, thick, shorten >=2pt, shorten <=2pt}
  ]

  % --- Injective ---
  \begin{scope}[shift={(0,0)}]
    \node[above] at (1, 2) {\textbf{\textsf{injective}}};
    \draw[thick, flexokigreen] (0,0) ellipse (0.8 and 1.5);
    \draw[thick, flexokired] (2,0) ellipse (0.8 and 1.5);
    \node[above] at (0, 1.6) {$A$};
    \node[above] at (2, 1.6) {$B$};

    \node[dot] (a1) at (0, 0.8) {};
    \node[dot] (a2) at (0, 0) {};
    \node[dot] (a3) at (0, -0.8) {};

    \node[dot] (b1) at (2, 1) {};
    \node[dot] (b2) at (2, 0.3) {};
    \node[dot] (b3) at (2, -0.4) {};
    \node[dot] (b4) at (2, -1.1) {};

    \draw[arrow] (a1) -- (b2);
    \draw[arrow] (a2) -- (b4);
    \draw[arrow] (a3) -- (b1);
  \end{scope}

  % --- Surjective ---
  \begin{scope}[shift={(4.5,0)}]
    \node[above] at (1, 2) {\textbf{\textsf{surjective}}};
    \draw[thick, flexokigreen] (0,0) ellipse (0.8 and 1.5);
    \draw[thick, flexokired] (2,0) ellipse (0.8 and 1.5);
    \node[above] at (0, 1.6) {$A$};
    \node[above] at (2, 1.6) {$B$};

    \node[dot] (a1) at (0, 1) {};
    \node[dot] (a2) at (0, 0.3) {};
    \node[dot] (a3) at (0, -0.4) {};
    \node[dot] (a4) at (0, -1.1) {};

    \node[dot] (b1) at (2, 0.8) {};
    \node[dot] (b2) at (2, 0) {};
    \node[dot] (b3) at (2, -0.8) {};

    \draw[arrow] (a1) -- (b1);
    \draw[arrow] (a2) -- (b2);
    \draw[arrow] (a3) -- (b3);
    \draw[arrow] (a4) -- (b2);
  \end{scope}

  % --- Bijective ---
  \begin{scope}[shift={(9,0)}]
    \node[above] at (1, 2) {\textbf{\textsf{bijective}}};
    \draw[thick, flexokigreen] (0,0) ellipse (0.8 and 1.5);
    \draw[thick, flexokired] (2,0) ellipse (0.8 and 1.5);
    \node[above] at (0, 1.6) {$A$};
    \node[above] at (2, 1.6) {$B$};

    \node[dot] (a1) at (0, 0.8) {};
    \node[dot] (a2) at (0, 0) {};
    \node[dot] (a3) at (0, -0.8) {};

    \node[dot] (b1) at (2, 0.8) {};
    \node[dot] (b2) at (2, 0) {};
    \node[dot] (b3) at (2, -0.8) {};

    \draw[arrow] (a1) -- (b2);
    \draw[arrow] (a2) -- (b1);
    \draw[arrow] (a3) -- (b3);
  \end{scope}

\end{tikzpicture}
\end{document}
```

## cardinality

Two sets have the same cardinality if there exists a bijection between them:

$$
|A| = |B| \iff \exists f: A \to B \text{ bijective}
$$

For finite sets:

$$
|A \cup B| = |A| + |B| - |A \cap B|
$$

For finite products:

$$
|A \times B| = |A||B|
$$

A set is _countably infinite_ if it bijects with $\mathbb{N}$. The integers and rationals are countable; the reals are uncountable.

> [!math] diagonal argument
>
> The set $\{0,1\}^{\mathbb{N}}$ of infinite binary sequences is uncountable. If a list claimed to contain every such sequence, construct a new sequence $b$ by setting $b_n = 1 - a_{n,n}$, where $a_{n,n}$ is the $n$th bit of the $n$th listed sequence. Then $b$ differs from row $n$ at bit $n$, so it is missing from the list.

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

## Zermelo-Fraenkel set theory

ZFC is an [axiomatic system](https://en.wikipedia.org/wiki/Axiomatic_system) that was proposed to formulate a paradox-free theory of sets to address [[thoughts/Wittgenstein#Russell's paradox and the vicious circle principle|Russell's paradox]]. Formally, it is intended to formalize a single primitive notion, that of a [hereditary](https://en.wikipedia.org/wiki/Hereditary_set) [well-founded](https://en.wikipedia.org/wiki/Well-founded_relation) set, so that all _entities_ in the universe of discourse are sets.

> These axioms of ZFC therefore refer only to [pure sets](https://en.wikipedia.org/wiki/Hereditary_set) and prevent its models from containing {{sidenotes[urelements]: elements that are not themselves sets.}}

Formally, ZFC is a one-sorted theory in [first-order logic](https://en.wikipedia.org/wiki/First-order_logic).

The only nonlogical primitive is membership $\in$. Equality is governed by extensionality, and every other construction gets encoded through sets.

| axiom              | job                                                                    |
| ------------------ | ---------------------------------------------------------------------- |
| extensionality     | same elements means same set                                           |
| empty set          | there exists a set with no elements                                    |
| pairing            | from $a,b$, form $\{a,b\}$                                             |
| union              | from a set of sets, form the set of their members                      |
| power set          | from $A$, form $\mathcal{P}(A)$                                        |
| infinity           | there exists an inductive set, giving enough material for $\mathbb{N}$ |
| separation schema  | carve subsets from existing sets by first-order predicates             |
| replacement schema | images of sets under definable functions are sets                      |
| foundation         | rules out infinite descending membership chains                        |
| choice             | for a set of nonempty sets, choose one element from each               |

Replacement is stronger than separation. Separation says "filter this set.", whereas Replacement says "send each element through a definable rule, then collect the outputs."

$$
\forall x \in A\;\exists!y\;\varphi(x,y) \implies \exists B\;\forall y\;(y \in B \iff \exists x \in A\;\varphi(x,y))
$$

The axiom of choice has many equivalent forms:

- every surjection has a right inverse
- every vector space has a basis
- every product of nonempty sets is nonempty
- every set can be well-ordered

## common fallacy

- $a \in A$ and $\{a\} \subseteq A$ say different things.
- $\emptyset \subseteq A$ for every set $A$, including $A = \emptyset$.
- $\emptyset \in A$ is a separate claim. It holds only when the empty set is one of $A$'s elements.
- $A \subseteq B$ and $B \subseteq A$ prove $A = B$ by extensionality.
- A family $\{A_i\}_{i \in I}$ remembers the index set $I$; the set $\{A_i \mid i \in I\}$ forgets repeated values.
- Complements need a universe. $A^c$ means $X \setminus A$ only after $X$ has been fixed.
- Open and closed are properties of subsets relative to a topology on $X$, not absolute properties of the raw set.
