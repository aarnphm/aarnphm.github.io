---
date: "2024-12-11"
description: and concurrency control.
id: Transaction
modified: 2025-10-29 02:16:03 GMT-04:00
tags:
  - sfwr3db3
title: Transaction
---

see also [[thoughts/university/twenty-three-twenty-four/sfwr-3bb4/index|concurrency]]

> A sequence of read/write

## concurrency

inter-leaved processing: concurrent exec is _interleaved_ within a single CPU.

parallel processing: process are concurrently executed on _multiple_ CPUs.

```tikz style="padding-top: 3rem;gap: 5rem;"
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning}

\begin{document}
\begin{tikzpicture}[font=\small, node distance=1.5cm, >=latex]

%------------------------------
% Interleaved Processing
%------------------------------

% Place the title higher up
\node[font=\bfseries, align=center] (interleavedTitle) at (5, 1) {Interleaved (Time-Sliced) Processing};

% Draw the timeline axis lower down
\draw[->] (-0.5,2) -- (10,2) node[below]{Time};

% Processes above the time line
% P1 intervals
\draw[fill=blue!30] (0,2.4) rectangle (2,3) node[midway]{P1};
\draw[fill=blue!30] (4,2.4) rectangle (6,3) node[midway]{P1};
\draw[fill=blue!30] (8,2.4) rectangle (9.5,3) node[midway]{P1};

% P2 intervals
\draw[fill=red!30] (2,2.4) rectangle (4,3) node[midway]{P2};
\draw[fill=red!30] (6,2.4) rectangle (8,3) node[midway]{P2};

%------------------------------
% Parallel Processing
%------------------------------
\begin{scope}[yshift=-1cm]

% Title for parallel processing
\node[font=\bfseries, align=center] (parallelTitle) at (5,-3.2) {Parallel (Concurrent) Processing};

% Timelines for parallel processing
\draw[->] (-0.5,-1) -- (10,-1) node[below]{Time};
\draw[->] (-0.5,-2.5) -- (10,-2.5) node[below]{Time};

% Process 1 on core 1 (above the -1 line)
\draw[fill=blue!30] (0,-0.6) rectangle (9.5,-0.0) node[midway]{P1 running on Core 1};

% Process 2 on core 2 (above the -2.5 line)
\draw[fill=red!30] (0,-2.1) rectangle (9.5,-1.5) node[midway]{P2 running on Core 2};
\end{scope}

\end{tikzpicture}
\end{document}
```

## ACID

- atomic: either performed in its entirety (DBMS's responsibility)
- consistency: must take database from consistent state $X$ to $Y$
- isolation: appear as if they are being executed in ==isolation==
- durability: changes applied must persist, even in the events of failure

## Schedule

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/venn-schedule.webp|Venn diagram for schedule]]

> [!abstract] definition
>
> a schedule $S$ of $n$ transaction $T_{1}, T_{2}, \ldots, T_{n}$ is an _ordering_ of operations of the transactions subject to the constrain that
>
> For all transaction $T_i$ that participates in $S$, the operations of $T_i$ in $S$ **must appear in the same order** in which they occur in $T_i$

For example:

$$
S_a: R_{1}(A),R_{2}(A),W_{1}(A),W_{2}(A),\text{Abort1},\text{Commit2};
$$

- serial schedule: _does not interleave the actions of different transactions_
- equivalent schedule: effect of executing any schedule are the same

> [!quote] serialisable schedule
>
> a schedule that is equvalent to some _serial execution_ of the set of _committed_ transactions

### serial

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/serial-transaction.webp]]

> [!note] serialisable schedule
>
> ![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/serialisable-transaction.webp|Note that this is not a serial schedule, given there are interleaved operations.]]
>
> $S:R_1(A),W_1(A), R_2(A), W_2(A), R_1(B), W_1(B), R_2(B), W_2(B)$

### conflict

> [!important] operations in schedule
>
> said to be in _conflict_ if they satisfy all of the following:
>
> 1. belong to ==different== operations
> 2. access the same item $A$
> 3. at least one of the operations is a `write(A)`

| Concurrency Issue | Description                                                                           | Annotation |
| ----------------- | ------------------------------------------------------------------------------------- | ---------- |
| Dirty Read        | Reading uncommitted data                                                              | WR         |
| Unrepeatable Read | T2 changes item $A$ that was previously read by T1, while T1 is still in progress     | RW         |
| Lost Update       | T2 overwrites item $A$ while T1 is still in progress, causing T1's changes to be lost | WW         |

#### conflict serialisable schedules

Two schedules are conflict equivalent if:

- involves the same actions of the same transaction
- every pair of conflicting actions is ordered the same way

> Schedule $S$ is _conflict serialisable_ if $S$ is conflict equivalent to some serial schedule

If two schedule $S_{1}$ and $S_{2}$ are conflict equivalent then they have the same effect $S_{1} \leftrightarrow S_{2}$ by _swapping non-conflicting ops_

Every conflict serialisable schedule is serialisable

> [!NOTE] on conflict serialisable
>
> only consider **committed** transaction

### schedule with abort

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/unrecoverable-transaction.webp|Note that this schedule is unrecoverable if T2 committed]]

_However, if T2 did not commit, we abort T1 and cascade to T2_

need to avoid _cascading abort_

- if $T_i$ writes an object, then $T_j$ can read this _only after_ $T_i$ commits

### recoverable and avoid cascading aborts

**Recoverable**: a $X_\text{act}$ commits _only after_ all $X_\text{act}$ it depends on commits.

**ACA**: idea of aborting a $X_\text{act}$ can be done without cascading the abort to other $X\text{act}$

> ACA implies recoverable, ==not vice versa==

### precedent graph test

_is a schedule conflict-serialisable?_

- build a graph of all transactions $T_i$
- Edge from $T_i$ to $T_j$ if $T_i$ comes first, and makes an action that conflicts with one of $T_j$

> if graphs has no cycle then it is **conflict-serialisable**

### strict

> if a value written by $T_i$ is not read or overwritten by another $T_j$ until $T_i$ abort/commit

Are recoverable and ACA

## Lock-based concurrency control

think of mutex or a lock mechanism to control access to a data object

> transaction _must_ release the lock

> [!math] notation
>
> `Li(A)` means $T_i$ acquires lock for A, where as `Ui(A)` releases lock for A

| Lock Type | None | S        | X        |
| --------- | ---- | -------- | -------- |
| None      | OK   | OK       | OK       |
| S         | OK   | OK       | Conflict |
| X         | OK   | Conflict | Conflict |

_lock compatibility matrix_

overhead due to delays from blocking; minimize throughput

- use smallest sized object
- reduce time hold locks
- reduce hotspot

### shared locks

$S_T(A)$ for reading

### exclusive lock

$X_T(A)$ for write/read

### strict two phase locking (Strict 2PL)

- Each $X_\text{act}$ must obtain a ==S lock== on object before reading, and an ==X lock== on object before writing
- All lock held by transaction will be released when transaction is completed

> only schedule those precedence graph is acyclic

> recoverable and ACA

Example:

| T1             | T2                  |
| -------------- | ------------------- |
| ==L(A);==      |                     |
| R(A), W(A)     |                     |
|                | ==L(A); DENIED...== |
| ==L(B);==      |                     |
| R(B), W(B)     |                     |
| ==U(A), U(B)== |                     |
| Commit;        |                     |
|                | ==...GRANTED==      |
|                | R(A), W(A)          |
|                | ==L(B);==           |
|                | R(B), W(B)          |
|                | ==U(A), U(B)==      |
|                | Commit;             |

> [!note] implication
>
> - only allow safe interleavings of transactions
> - $T_{1}$ and $T_{2}$ access different objects, then no conflict and each may proceed
> - serial action

### two phase locking (2PL)

lax version of strict 2PL, where it allow $X_\text{act}$ to release locks before the end

- ==a transaction cannot request additional lock once it releases any lock==

> [!note] implication
>
> - all lock requests _must_ precede all unlock request
> - ensure _conflict serialisability_
> - two phase transaction growing phase, (or obtains lock) and shrinking phase (or release locks)

### isolation

| Isolation Level    | Description                                 |
| ------------------ | ------------------------------------------- |
| `READ UNCOMMITTED` | No read-lock                                |
| `READ COMMITTED`   | Short duration read locks                   |
| `REPEATABLE READ`  | Long duration read/lock on individual items |
| `SERIALIZABLE`     | All locks long durations                    |

### Deadlock

cycle of transactions waiting for locks to be released by each other

usually create a wait-for graph to detect cyclic actions

- wait-die: lower transactions never wait for higher priority transactions

- wound-wait: $T_i$ is higher priority than $T_j$ then $T_j$ is aborted and restarts later with same timestamp, otherwise $T_i$ waits
