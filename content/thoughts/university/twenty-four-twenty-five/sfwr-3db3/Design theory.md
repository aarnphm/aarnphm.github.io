---
date: '2024-12-13'
description: Functional dependencies, closure, lossless decomposition, and normal forms.
id: Design theory
modified: 2026-09-28 09:11:47 GMT-04:00
tags:
  - sfwr3db3
title: Design theory
---

> [!abstract] Keys
>
> $K$ is a candidate key of schema $R$ if $K$ determines every attribute of $R$ and no proper subset of $K$ does.
>
> A _superkey_ determines every attribute of $R$. It may contain attributes that a candidate key does not need.

_see also: [[thoughts/university/twenty-four-twenty-five/sfwr-3db3/Keys and Foreign Keys|keys]]_

## functional dependency

A functional dependency (FD) $X \to Y$ says that any two tuples agreeing on $X$ must agree on $Y$. As a schema constraint, it must hold in every legal instance. A small table can satisfy an FD by accident.

Convention: write $ABC$ for the attribute set $\{A,B,C\}$.

> [!note] properties
>
> - splitting/combining
> - trivial FDs
> - Armstrong's Axioms

> [!abstract] FDs generalise keys
>
> $X$ is a superkey exactly when $X \to R$ follows from the schema's dependencies.

### trivial

$$
\begin{aligned}
A &\to A \\
AB &\to A \\
ABC &\to AC
\end{aligned}
$$

These always hold because the right side is a subset of the left. If $D \notin ABC$, then $ABC \to AD$ is non-trivial. It is equivalent to $ABC \to D$: agreeing on $ABC$ already guarantees agreement on $A$, while agreement on $D$ adds a constraint.

### splitting/combining right side of FDs

$$
X \to A_{1} A_{2} \ldots  A_{n} \text{ holds for R }
$$

when each of $X \to A_{1}$, $X \to A_{2}$, ..., $X \to A_{n}$ _holds for_ $R$

ex: $A \to BC$ is equiv to $A \to B$ and $A \to C$

ex: $A \to F$ and $A \to G$ can be written as $A \to FG$

### Armstrong's Axioms

Let $X,Y,Z$ be sets of attributes. Reflexivity, augmentation, and transitivity are Armstrong's axioms; union and decomposition follow from them.

#### rules

| Rule          | Description                                 |
| ------------- | ------------------------------------------- |
| Reflexivity   | If $Y \subseteq X$, then $X \to Y$          |
| Augmentation  | If $X \to Y$, then $XZ \to YZ$ for any $Z$  |
| Transitivity  | If $X \to Y$ and $Y \to Z$, then $X \to Z$  |
| Union         | If $X \to Y$ and $X \to Z$, then $X \to YZ$ |
| Decomposition | If $X \to YZ$, then $X \to Y$ and $X \to Z$ |

#### dependency inference

$A \to C$ is _implied_ by $\{A \to B, B \to C\}$

#### transitivity

example: Key

List all the keys of $R(A,B,C,D)$ with the following FDs:

- $B \to C$
- $B \to D$

sol:

$$
\begin{aligned}
B \to C &\text{ and } B \to D &(\text{given})\\
B &\to CD &(\text{Union})\\
AB &\to ACD &(\text{Augmentation})\\
AB &\to ABCD &(\text{Reflexivity and Union})\\
\end{aligned}
$$

So $AB$ is a superkey. Every key must contain $A$ and $B$, since neither can be obtained from another attribute using the given FDs. Removing either prevents the set from determining all of $R$, so $AB$ is the only candidate key.

#### closure test

For an attribute set $Y$ and dependency set $F$, the **attribute closure** is

$$
Y_F^{+} = \{A \in R : F \models Y \to A\}.
$$

It contains attributes. By contrast, $F^{+}$ contains all dependencies implied by $F$. To test $Y \to Z$, check whether $Z \subseteq Y_F^{+}$. [Sibel Adali's normalization notes](https://cs.rpi.edu/~sibel/csci4380/fall2026/lecture_notes/lecture6.html) use this distinction to test whether two FD sets are equivalent.

1. Start with $C = Y$.
2. For each $U \to V$ in $F$, if $U \subseteq C$, replace $C$ with $C \cup V$.
3. Repeat until a full pass adds no attributes. Then $C = Y_F^{+}$.

For the preceding example, $B_F^{+}=BCD$ and $(AB)_F^{+}=ABCD$. The first closure omits $A$, so $B$ alone cannot be a key.

#### minimal basis

A minimal basis, also called a minimal cover in the single-attribute convention, is a simplified set $G$ with $G^{+}=F^{+}$.

> [!important] for minimal cover for FDs
>
> - Right sides are **single** attributes
> - Removing any FD changes the implied dependencies.
> - Removing any attribute from a **left side** changes the implied dependencies.

Use closure tests to make one change at a time:

1. Split each right side into single attributes.
2. For $X \to A$ and $B \in X$, compute $(X-\{B\})_G^{+}$. If it contains $A$, replace the FD with $(X-\{B\}) \to A$.
3. For $f=(X \to A)$, compute $X_{G-\{f\}}^{+}$. If it contains $A$, remove $f$.
4. Repeat the last two steps until neither changes $G$.

The left-side test asks whether the smaller determinant still determines the **right side**. It does not require recovering the removed attribute. For example, with $F=\{AB \to C, A \to C, C \to D, A \to D\}$, $B$ is removable from $AB \to C$ even though $B \notin A_F^{+}$. Remove the duplicate $A \to C$, then remove $A \to D$ by transitivity. One minimal basis is $\{A \to C,C \to D\}$.

Some texts combine equal left sides when defining a canonical cover. That convention preserves the same dependencies. See the [canonical-cover construction](https://www.db-book.com/slides-dir/PDF-dir/ch7.pdf#page=42) in _Database System Concepts_.

## Schema decomposition

Decomposition separates facts that would otherwise be repeated across rows. We want to reconstruct the original relation and enforce its dependencies after the split.

> [!note] good properties to have
>
> - **Lossless join:** joining the projections recovers exactly the original relation for every instance satisfying $F$.
> - **Dependency preservation:** $(F_{1} \cup F_{2} \cup \ldots \cup F_n)^{+} = F^{+}$, where $F_i$ is the projection of $F$ onto $R_i$.
> - Fewer update, insertion, and deletion anomalies caused by repeated facts.

> [!note]- information loss with decomposition
>
> Losslessness and dependency preservation are separate tests. A dependency can survive through several tables: with $A \to B$ in $AB$ and $B \to C$ in $BC$, local checks imply $A \to C$ even though no table contains both $A$ and $C$.

For a lossy example, project $r=\{(1,10,100),(2,10,200)\}$ onto $AB$ and $BC$. Joining on $B$ produces four tuples, including $(1,10,200)$ and $(2,10,100)$. The projections lost which $A$ belonged with which $C$.

> [!question] how can we test for losslessness?
>
> For a binary decomposition with $R_1 \cup R_2=R$, the join is lossless with respect to the FD constraints $F$ iff **at least one** of these holds:
>
> - $(R_{1} \cap R_{2}) \to R_{1}$ is in $F^{+}$.
> - $(R_{1} \cap R_{2}) \to R_{2}$ is in $F^{+}$.

Thus the shared attributes must be a superkey of either component. With $R=ABC$, $F=\{A \to B\}$, and components $AB$ and $AC$, the intersection is $A$. It determines $AB$, which suffices; $A \to AC$ need not hold. The [binary lossless-join criterion](https://www.db-book.com/slides-dir/PDF-dir/ch7.pdf#page=16) is necessary and sufficient when the constraints are FDs.

### Projection

Projecting $F$ onto $R_i$ retains all implied dependencies whose attributes lie in $R_i$. A dependency can be derived through attributes outside $R_i$.

1. Start with $F_i = \emptyset$.
2. For every subset $X \subseteq R_i$, compute $X_F^{+}$ using the original dependencies.
3. For each $A \in (X_F^{+} \cap R_i)-X$, add $X \to A$ to $F_i$.
4. Compute a minimal basis of $F_i$.

For $F=\{A \to B,B \to C\}$, the projection onto $AC$ includes $A \to C$. Merely filtering the original FD list would miss it.

## Normal forms

$$
\text{BCNF} \subseteq 3\text{NF} \subseteq 2\text{NF} \subseteq 1\text{NF}
$$

An attribute is **prime** if it belongs to at least one candidate key. The tests concern all candidate keys, including ones that were not chosen as the primary key.

| Normal form | Test                                                                                |
| ----------- | ----------------------------------------------------------------------------------- |
| 1NF         | Attribute values are atomic in the relational model being used.                     |
| 2NF         | In 1NF, and no non-prime attribute depends on a proper subset of any candidate key. |
| 3NF         | For every non-trivial $X \to A$ in $F^{+}$, $X$ is a superkey or $A$ is prime.      |
| BCNF        | For every non-trivial $X \to A$ in $F^{+}$, $X$ is a superkey.                      |

### 1NF

In the course's flat relational model, `Course(name, instructor, [student, email]*)` violates 1NF because each course contains a repeating group of student-email pairs. Store those pairs in rows of a separate enrollment relation.

### 2NF

Suppose an enrollment relation has candidate key $\{\text{StudentID},\text{CourseID}\}$ and stores `StudentName`. If $\text{StudentID} \to \text{StudentName}$, the name depends on a proper subset of the key and is repeated for every enrollment. Move the student-name association to a student relation. This is the partial-dependency problem excluded by [2NF](https://www.db-book.com/Practice-Exercises/PDF-practice-exer-dir/7.pdf).

### 3NF

For each non-trivial $X \to A$, either $X$ determines the whole relation or $A$ belongs to a candidate key. The prime-attribute exception matters when candidate keys overlap.

In the book example below, assume `BookID` is the only candidate key and $\text{AuthorID} \to \text{AuthorName}$. The dependency $\text{BookID} \to \text{AuthorID} \to \text{AuthorName}$ is transitive. The illustrated split removes a 3NF violation.

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/second-normal-form.webp|3NF decomposition of the author-name dependency]]

A violation: $\text{studio} \to \text{studioAddr}$ in a movie relation where `studio` is not a superkey and `studioAddr` is non-prime. Every movie from the same studio repeats its address.

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/three-normal-form.webp|Three normal form counter example]]
![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/three-normal-form-decomposition.webp]]

> [!theorem]
>
> Every schema with FD constraints has a lossless, dependency-preserving decomposition into 3NF.

The [3NF synthesis algorithm](https://www.cs.rpi.edu/~sibel/csci4380/fall2026/lecture_notes/lecture7.html#nf-decomposition) constructs a relation for each dependency in a minimal cover and adds a candidate-key relation if needed. Its guarantees concern that decomposition. Arbitrarily splitting a schema into 3NF components can still lose information or dependencies.

### Boyce-Codd normal form (BCNF)

> [!theorem]
>
> $R$ is in BCNF with respect to $F$ iff **every** non-trivial FD $X \to A$ in $F^{+}$ has a superkey as its left side.[^nontrivial]

[^nontrivial]: For a single-attribute right side, non-trivial means $A \notin X$.

The determinant need not be a minimal key. In $R=ABC$ with $F=\{A \to BC\}$, $AB \to C$ is allowed: $AB$ is a superkey, though $A$ alone is the candidate key.

BCNF removes redundancy caused by a non-key determinant. Other constraints, such as multivalued dependencies, can still cause redundancy. A BCNF decomposition can always be made lossless; dependency preservation is not guaranteed. [Adali's decomposition notes](https://www.cs.rpi.edu/~sibel/csci4380/fall2026/lecture_notes/lecture7.html#bcnf-decomposition) give examples of both outcomes.

#### decomposition into BCNF

For the current component $S$, use its projected dependencies $F_S$.

1. Find a non-trivial FD with determinant $X$ such that $X$ is not a superkey of $S$.
2. Compute $C=X_{F_S}^{+}$. The violation gives $X \subsetneq C \subsetneq S$.
3. Replace $S$ with $S_1=C$ and $S_2=S-(C-X)$.
4. Project the dependencies onto both components and repeat until neither has a violation.

The intersection is $X$, and $X \to C$, so each split is lossless. Recompute keys for the component being split: being a superkey is relative to that component's attributes.
