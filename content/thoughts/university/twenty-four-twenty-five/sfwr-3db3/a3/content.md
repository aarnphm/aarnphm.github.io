---
date: '2024-11-24'
description: assignment on functional dependencies, minimal cover, armstrong's axioms, normalization, and transaction concurrency control.
id: content
modified: 2025-11-09 01:40:53 GMT-05:00
tags:
  - sfwr3db3
title: DB design, concurrency and transaction
---

## Database Design

### Finding Keys

> [!question] a
>
> Consider a relation schema $R(A, B, C, D, E)$ and the set of functional dependencies
>
> $$
> \mathbf{F} = \{
> A \rightarrow BC, \,
> CD \rightarrow E, \,
> B \rightarrow D, \,
> E \rightarrow A
> \}
> $$
>
> Find all candidate keys (minimal keys) of relation $R$. Show all the steps you took to derive each key, and clearly state which of Armstrong's axioms are used in each step.

**Closure of $A$**

$A \rightarrow BC$ means $A^{+} = \{A, B, C\} (\text{Decomposition})$

$B \rightarrow D$ means $A^+ =\{A,B,C,D\} \text{(Transitivity)}$

$CD \rightarrow E$ means $A^+ =\{A,B,C,D,E\} \text{(Transitivity)}$ ($CD \in A$)

> $A$ is a candidate key

**Closure of $B$**

$B^+ = \{B\}$

$B \rightarrow D$ means $B^+ = \{B, D\}$

No other closure can be applied thus $B$ is _not_ a key

**Closure of $C$**

We can't derive all attributes, thus $C$ is _not_ a key

**Closure of $D$**

No applicable dependencies, thus $D$ is _not_ a key

**Closure of $E$**

$E \rightarrow A$ means $E^+ = \{E, A\} \text{(FD)}$

$A \rightarrow BC$ means $E^+ = \{E, A, B, C\} \text{Decomposition and Transitivity}$

$B \rightarrow D$ means $E^+ = \{E, A, B, C, D\} \text{Transitivity}$

> $E$ is a candidate key

**Closure of $CD$**

Initial $CD$ gives $(CD)^+ = \{C, D\}$

$CD \rightarrow E$ gives $(CD)^+ = \{C, D, E\} \text{FD}$

$E \rightarrow A$ gives $(CD)^+ = \{C, D, E, A\} \text{Transitivity}$

$A \rightarrow BC$ gives $(CD)^+ = \{C, D, E, A, B, C\} \text{Transitivity and Decomposition}$

> $CD$ is a candidate key

**Closure of $BC$**

Initial $BC$ gives $(BC)^+ = \{B, C\}$

$B \rightarrow D$ gives $(BC)^+ = \{B, C, D\} \text{FD}$

$CD \rightarrow E$ gives $(BC)^+ = \{B, C, D, E\} \text{Transitivity}$

$E \rightarrow A$ gives $(BC)^+ = \{B, C, D, E, A\} \text{Transitivity}$

> $BC$ is a candidate key

> [!quote] conclusion
>
> Final candidate keys are $A, E, CD, BC$

> [!question] b
>
> Is the FD $AB \rightarrow C$ entailed by **F**? Show your work that supports your answer

We will find closure of $AB$ under $F$

$(AB)^+ = \{A, B\}$

$A \rightarrow BC$ entails $A \rightarrow B$ and $A \rightarrow C$ (Decomposition)

with augmentation on $A \rightarrow C$ we have $AB \rightarrow CB$

Decomposition to $AB \rightarrow CB$ gets $AB \rightarrow C$

Therefore $AB \rightarrow C$ is entailed by $F$

### Minimal Cover

> [!question]
>
> Given the relational schema $T(A, B, C, D)$ and the FDs: $F \{ABC \rightarrow D, CD \rightarrow A, CA \rightarrow B, AD \rightarrow C, CD \rightarrow B \}$, compute the minimal cover $F^{'}$ of $F$. Show all your work (derivation) to compute $F^{'}$

We have the following FD

$$
\begin{aligned}
ABC &\rightarrow D \\
CD &\rightarrow A \\
CA &\rightarrow B \\
AD &\rightarrow C \\
CD &\rightarrow B
\end{aligned}
$$

1. RHS of FDs into single attributes

- Already in this form

2. minimize LHS by removing extraneous

FD1: $ABC \rightarrow D$

- $B$ is extraneous given that $AC \rightarrow D$ holds:
  - $(AC)^+ = \{A, C\}$
  - $CA \rightarrow B$ then add $B$ to closure
  - $ABC \rightarrow D$ then add $D$ to closure

> Update FD1: $AC \rightarrow D$

FD2: $CD \rightarrow A$

- no FD is applied is either C or D is assume extraneous, therefore remained unchanged

FD3: $CA \rightarrow B$

- can't reduce $CA \rightarrow B$ given that neither $C \rightarrow B$ and $A \rightarrow B$ holds

FD4: $AD \rightarrow C$

- can't reduce $AD \rightarrow C$ given that neither $A \rightarrow C$ and $D \rightarrow C$ holds

FD5: $CD \rightarrow B$

- can't reduce $CD \rightarrow B$ given that neither $C \rightarrow B$ and $D \rightarrow B$ holds

3. Remove redudant FDs

FD5: $CD \rightarrow B$

Can be calculated from $CD \rightarrow A$ and $CA \rightarrow B$:

Closure of $CD$ is $(CD)^+ = \{C,D\}$, $CD \rightarrow A$ gives $\{C,D,A\}$ and $CD \rightarrow B$ gives $\{C,D,A,B\}$

thus this is redudant

> [!quote] Final minimal cover is
>
> $F^{'} = \{AC \rightarrow D, CD \rightarrow A, CA \rightarrow B, AD \rightarrow C\}$

### Armstrong's Axioms

> [!question] a
>
> Given the relational schema $R(A,B, C, D, E, F)$ and FDs $F_1: \{AB \rightarrow C, A \rightarrow D, CD \rightarrow EF\}$. Show that $AB \rightarrow F$

$$
\begin{align}
A &\rightarrow D  &\text{(Given)} \\
AB &\rightarrow DB  &\text{(Augmentation w/B)} \\
AB &\rightarrow D \cup AB \rightarrow B &\text{Decomposition)} \\
AB &\rightarrow CD &\text{(Union with 2 and } AB \rightarrow C) \\
AB  &\rightarrow EF &\text{(Transitivity with 3 and }  CD \rightarrow EF) \\
AB &\rightarrow F &\text{(Decomposition)} \\
&\because
\end{align}
$$

> [!question] b
>
> Given the relational schema $R(A,B, C, D, E, F)$ and FDs $F_1: \{C \rightarrow D, BE \rightarrow A, BEF \rightarrow C \}$. Show that $BEF$ is a key

Proof: $BEF^{+} = \{A,B,C,D,E,F\}$ and $BEF$ is minimal

1. Proving closure of $BEF$

We have $BEF^{+} = \{B,E,F\}$

$BEF \rightarrow C$ and $BEF \in BEF^{+}$ by reflexivity, we add $C$ to the closure $BEF^{+} = \{B,E,F, C\}$

$C \rightarrow D$ and with Transitivity of $BEF \rightarrow C$ gives $BEF \rightarrow D$. Add $D$ to closure $BEF^{+} = \{B,E,F, C, D\}$

$BE \rightarrow A$ thus add $A$ to the closure $BEF^{+} = \{B,E,F, C, D, A\}$

Therefore by union we have prove closure of $BEF$

2. minimal of $BEF$

Case 1: Remove $B$ from $BEF$:

- Compute $EF^{+}$, and there is no FD to prove this transition, therefore $EF$ does not determine all attributes

Case 2: Remove $E$ from $BEF$:

- Compute $BF^{+}$, and there is no FD to prove this transition, therefore $EF$ does not determine all attributes

Case 3. remove $F$ from $BEF$:

- Closure of $BE$ is $BE^{+} = \{B, E\}$
- $BE \rightarrow A$ means $BE^{+} = \{B, E, A\}$
- No further can be added

Therefore BEF is minimal

> BEF is a key

### 3NF, BCNF

> [!question] 1
>
> List all functional dependencies and keys that can be inferred from this information

1. **functional dependencies**

For Company table:

FD1: $\text{companyID} \rightarrow \text{companyName, cityName, countr, assets}$

FD2: $\text{companyName, cityName} \rightarrow \text{companyID, country, assets}$

Candidate key: $\text{companyID}$ (minimal key based on FD1) and $\text{companyName, cityName}$ (based on FD2)

For Department table:

FD3: $\text{deptID} \rightarrow \text{deptName, companyID, cityName, country, deptMgrID}$

FD4: $\text{companyID, depthName} \rightarrow \text{deptID, cityName, country, deptMgrID}$

FD5: $\text{deptMgrID} \rightarrow \text{deptID}$

Candidate key: $\text{deptID}$ and $\text{companyID, depthName}$

For City table:

FD6: $\text{cityID} \rightarrow \text{cityName, country}$

FD7: $\text{cityName, country} \rightarrow \text{cityID}$

Candidate key: $\text{cityID}$ and $\text{cityName, country}$

> [!question] 2
>
> schemas satisfies either BCNF or 3NF

Note that for both Company and City tables, it satisifies BCNF. however, for Department table:

For FD3, $\text{deptID}$ is a candidate key, thus satisfies BCNF

For FD4, $\text{companyID, depthName}$ is a candidate key, thus satisfies BCNF

But for FD5 given that $\text{deptMgrID}$ is not a candidate key, thus violate BCNF

**Improvement**

- Create a new table DeptManager $\text{deptMgrID, deptID}$ with decomposition

- remove $\text{deptMgrID}$ from the original table (now $\text{deptName, companyID, cityName, country}$)

Thus should satisfy BCNF

## Transactions and Concurrency

### Schedules

Consider schedules $S_{1}, S_{2}, S_{3}$ State which of the following properties holds (or not) for each schedule: _strict, avoid cascading aborts, recoverability_. Provide brief justification for each answer

> [!question] a
>
> $S_{1}: \text{r1(X); r2(Z); r1(Z); r3(X); r3(Y); w1(X); c1; w3(Y); c3; r2(Y); w2(Z); w2(Y); c2}$

- strict: _no_ because $\text{r3}$ reads X before $T_{1}$ commits, and $\text{r2}$ reads Y before $T_{3}$ commits
- avoid cascading aborts: no, because $\text{r3}$ reads X before $T_{1}$ commits
- recoverability: yes, since $T_{2}$ reads data written by $T_{3}$ has committed

> [!question] b
>
> $S_{2}: \text{r1(X); r2(Z); r1(Z); r3(X); r3(Y); w1(X); w3(Y); r2(Y); w2(Z); w2(Y); c1; c2; c3}$

- strict: no because $T_{2}$ reads uncommitted data from $T_{3}$ before committed
- avoid cascading aborts: no because $\text{r2}(Y)$ reads an uncommitted value from $T_{3}$
- recoverability: no because $T_{2}$ reads Y written by $T_{3}$ but commits before $T_{3}$ commits

> [!question] c
>
> $S_{3}: \text{r1(X); r2(Z); r3(X); r1(Z); r2(Y); r3(Y); w1(X); w2(Z); w3(Y); w2(Y); c3; c1; c2}$

- strict: no, because $T_{2}$ writes to $Y$ after it has been modified by uncommitted $T_{3}$
- avoid cascading aborts: yes, because all reads are from initial state, not from uncommitted transaction
- recoverability: yes, because $T_{2}$ is committed after $T_3$

### Serialisability

Which of the following schedules is (conflict) serializable? For each serializable schedule, find the equivalent serial schedules.

> [!question] a
>
> $\text{r1(X); r3(X); w1(X); r2(X); w3(X)}$

$$
\begin{align*}
r_1(X) \rightarrow w_3(X) &: T_1 \text{ reads } X \text{ before } T_3 \text{ writes it} \implies T_1 \rightarrow T_3 \\
r_3(X) \rightarrow w_1(X) &: T_3 \text{ reads } X \text{ before } T_1 \text{ writes it} \implies T_3 \rightarrow T_1 \\
w_1(X) \rightarrow r_2(X) &: T_1 \text{ writes } X \text{ before } T_2 \text{ reads it} \implies T_1 \rightarrow T_2 \\
w_1(X) \rightarrow w_3(X) &: T_1 \text{ writes } X \text{ before } T_3 \text{ writes it} \implies T_1 \rightarrow T_3 \\
r_2(X) \rightarrow w_3(X) &: T_2 \text{ reads } X \text{ before } T_3 \text{ writes it} \implies T_2 \rightarrow T_3
\end{align*}
$$

The precedence graph contains a cycle between $T_{1}$ and $T_{3}$, thus this is **not conflict serializable**

> [!question] b
>
> $\text{r3(X); r2(X); w3(X); r1(X); w1(X)}$

Note that there are no conflict between $T_{2}$ and other nodes given that it only read

This is **conflict serializable** with the following equivalent serial schedules:

$$
\begin{aligned}
& T_{3} \to T_{1} \to T_{2} \\
& T_{3} \to T_{2} \to T_{1}
\end{aligned}
$$

### Locking

> [!question]
>
> Consider the following locking protocol:
>
> - Before a transaction T writes a data object A, T has to obtain an exclusive lock on A.
> - For a transaction T, we hold these exclusive locks until the end of the transaction.
> - If a transaction T reads a data object A, no lock on A is obtained.
>
> State which of the following properties are ensured by this locking protocol: serializability, conflict-serializability, recoverability, avoids cascading aborts, avoids deadlock.
> Explain and justify your answer for each property.

1. serializability

- Not ensured given that reads aren't controlled by locks, therefore two transaction can read the same data item and write to in different order (example: $\text{r1(X); r2(X); w2(X); w1(X)}$)

2. conflict-serializability

- Not ensured, same reason as above

3. recoverability

- not ensured given that if a transaction $T_j$ reads data written by $T_i$, then $T_j$ should commit only after $T_i$ commit. however, in this protocol, transaction can read uncommitted data, given that read is not locked (dirty read.)

4. avoid cascading aborts

- not ensured given that dirty read can happen (example: $\text{w1(X); r2(X)}$. In this case if $T_{1}$ aborts, $T_{2}$ will need to aboart, causing cascade aborts)

5. avoid deadlock

- ensured, given that each transaction have to obtain exclusive lock on A, as transactions can't wait for each other in a scycle since reads don't require locks.
