---
date: '2024-12-11'
description: Schedules, serialisability, recovery, and lock-based concurrency control.
id: Transaction
modified: 2026-06-05 15:08:42 GMT-04:00
tags:
  - sfwr3db3
title: Transaction
---

see also [[thoughts/university/twenty-three-twenty-four/sfwr-3bb4/index|concurrency]]

A transaction groups database operations into one unit that either commits or aborts. In the schedules below, we track its reads, writes, and final outcome.

## concurrency

Concurrent transactions overlap in time. On one CPU core, their instructions can be interleaved; on multiple cores, instructions can also run in parallel.

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

- **Atomicity**: commit the transaction's changes together, or roll them back on abort.
- **Consistency**: a correct transaction preserves the database's constraints. The DBMS enforces declared constraints; application code remains responsible for rules it has not declared.
- **Isolation**: serialisable execution gives committed transactions the effect of some serial order. Weaker isolation levels allow specific anomalies.
- **Durability**: once a commit succeeds, its effects survive failures covered by the database's recovery guarantees.

See the [Berkeley transaction notes](https://cs186berkeley.net/notes/note11/) for the ACID model.

## Schedule

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/venn-schedule.webp|Venn diagram for schedule]]

> [!abstract] definition
>
> A schedule orders the operations of transactions $T_1,\ldots,T_n$ while preserving each transaction's own operation order.

Write $R_i(A)$ and $W_i(A)$ for reads and writes, $C_i$ for commit, and $A_i$ for abort. These examples use a single-version database: a read sees the latest preceding write unless that write has been rolled back.

A serial schedule runs each transaction to completion before starting the next. A serialisable schedule may interleave them, provided their observations and effects agree with a serial execution. Matching the final value in one example is insufficient to establish this equivalence.

### serial

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/serial-transaction.webp]]

> [!note] serialisable schedule
>
> ![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/serialisable-transaction.webp|An interleaved schedule with the same read/write order on each item as T1 followed by T2.]]
>
> The image uses assignments as shorthand for updates. Making the writes explicit:
>
> $S:R_1(A),W_1(A),R_2(A),W_2(A),R_1(B),W_1(B),R_2(B),W_2(B)$

All conflicts place $T_1$ before $T_2$, so this read/write schedule is conflict serialisable. Commit and abort handling still need checking.

### conflict

> [!important] operations in schedule
>
> Two operations conflict when they belong to different transactions, access the same item, and at least one writes it. Two reads can exchange places without changing what either reads.

| Concurrency issue | What happens                                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------- |
| Dirty read        | A transaction reads another's uncommitted write.                                                        |
| Unrepeatable read | A transaction reads an item twice and sees a change committed by another transaction between its reads. |
| Dirty write       | A transaction overwrites another's write before that writer commits or aborts.                          |
| Lost update       | A transaction writes a result computed from a stale read, overwriting another transaction's update.     |

For a lost update, start with $A=100$. Both transactions read $100$; $T_1$ adds $10$ and $T_2$ adds $20$:

$$
R_1(A),R_2(A),W_1(A\gets110),C_1,W_2(A\gets120),C_2.
$$

The final value is $120$; either serial order gives $130$. There is no dirty write here because $T_1$ commits before $T_2$ writes. See [Berenson et al., sections 3 and 4.1](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-95-51.pdf) for this distinction.

#### conflict serialisable schedules

Two schedules are **conflict equivalent** when they contain the same operations and preserve every conflicting pair's order. Equivalently, one can be obtained from the other by swapping adjacent non-conflicting operations. A schedule is **conflict serialisable** when it is conflict equivalent to a serial schedule.

Conflict serialisability implies serialisability. Its graph test can reject some view-serialisable schedules, particularly ones with blind writes. See the [Berkeley treatment](https://cs186berkeley.net/notes/note11/#conflict-serializability).

### schedule with abort

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/unrecoverable-transaction.webp|T2 commits after reading T1's uncommitted write; T1 then aborts.]]

The failure is the dependency $W_1(A),R_2(A),C_2,A_1$. Once $T_2$ commits, the database cannot retract its result as an ordinary abort. If $T_2$ is still active when $T_1$ aborts, it must also abort. Other transactions that read its writes may have to follow.

Removing aborted transactions from the serialisability analysis does not remove these read dependencies. Recovery must account for them.

### recoverable and avoid cascading aborts

A **recoverable** schedule lets a reader commit only after every transaction whose writes it read has committed. A **cascadeless** schedule, also called **ACA** (avoids cascading aborts), delays the read itself until the writer commits.

For example, $W_1(A),R_2(A),C_1,C_2$ is recoverable and has a dirty read. Moving $C_1$ before $R_2(A)$ makes it cascadeless. Thus:

$$
\text{ACA}\implies\text{recoverable}.
$$

Serialisability and recovery impose separate conditions: $W_1(A),R_2(A),C_2,C_1$ has an acyclic conflict graph and is unrecoverable. See [Database System Concepts, slides 15.22–15.24](https://www.db-book.com/Previous-editions/db4/slide-dir/ch15-2.pdf).

### precedence graph test

Build one node per transaction. Add $T_i\to T_j$ whenever an operation of $T_i$ precedes and conflicts with an operation of $T_j$.

The schedule is conflict serialisable **if and only if** this graph is acyclic. A topological ordering gives an equivalent serial order. For the lost-update example, $R_2(A)$ before $W_1(A)$ gives $T_2\to T_1$, and $R_1(A)$ before $W_2(A)$ gives $T_1\to T_2$.

### strict

A schedule is **strict** if other transactions can neither read nor overwrite an item written by $T_i$ until $T_i$ commits or finishes aborting.

$$
\text{strict}\implies\text{ACA}\implies\text{recoverable}.
$$

Strictness alone does not guarantee serialisability. The lost-update schedule above is strict and still has a cycle. [CMU's locking notes](https://15445.courses.cs.cmu.edu/spring2023/notes/16-twophaselocking.pdf) explain why strict schedules simplify rollback.

## Lock-based concurrency control

A lock manager grants access according to the locks already held by other transactions. Locks must eventually be released; the locking protocol determines when that is safe.

> [!math] notation
>
> $S_i(A)$ and $X_i(A)$ acquire shared and exclusive locks; $U_i(A)$ releases a lock.

| Requested lock | None held | S held  | X held |
| -------------- | --------- | ------- | ------ |
| S              | Granted   | Granted | Wait   |
| X              | Granted   | Wait    | Wait   |

This matrix compares locks held by different transactions.

Blocking can reduce throughput. Finer locks let transactions use different rows concurrently, at the cost of more lock-manager work and memory. Coarser locks reduce that overhead and block more unrelated work. Keep transactions short and avoid unnecessary hotspot access; choose granularity for the workload. [CMU, section 4](https://15445.courses.cs.cmu.edu/spring2023/notes/16-twophaselocking.pdf).

### shared locks

$S_i(A)$ permits $T_i$ to read $A$ while other shared-lock holders read it too.

### exclusive lock

$X_i(A)$ permits $T_i$ to read and write $A$. Other transactions must wait for either lock mode on that item.

### strict two phase locking (Strict 2PL)

Before reading, hold an S or X lock; before writing, hold an X lock. Follow the two-phase rule below and retain every X lock through commit or completed rollback.

Holding **all** locks until completion is usually called **rigorous 2PL**, or **strong strict 2PL**. Some course notes call this stronger variant strict 2PL too. The [Database System Concepts locking slides](https://web.cs.ucla.edu/classes/fall09/cs143/notes/2pl-handout.pdf) distinguish the two names.

Both variants produce conflict-serialisable, strict schedules. The example holds all locks to completion:

| $T_1$           | $T_2$                  |
| --------------- | ---------------------- |
| $X_1(A)$        |                        |
| $R_1(A),W_1(A)$ |                        |
|                 | Request $X_2(A)$; wait |
| $X_1(B)$        |                        |
| $R_1(B),W_1(B)$ |                        |
| $C_1$           |                        |
| $U_1(A),U_1(B)$ |                        |
|                 | $X_2(A)$ granted       |
|                 | $R_2(A),W_2(A)$        |
|                 | $X_2(B)$               |
|                 | $R_2(B),W_2(B)$        |
|                 | $C_2$                  |
|                 | $U_2(A),U_2(B)$        |

Commit precedes unlock. On abort, undo the writes before releasing their locks. Transactions accessing disjoint objects can still interleave.

### two phase locking (2PL)

Basic 2PL has two phases:

1. **Growing**: acquire locks or upgrade S to X; release none.
2. **Shrinking**: release locks or downgrade X to S; acquire no new locks and perform no upgrades.

The first release or downgrade ends the growing phase. Ordering transactions by their final lock acquisition gives a serialisation order. Basic 2PL permits early unlocks, so dirty reads and cascading aborts remain possible. All these 2PL variants can deadlock. See [Berkeley's locking notes](https://cs186berkeley.net/notes/note12/).

### isolation

Isolation levels specify observable behaviour. Lock duration is an implementation choice; MVCC can serve reads from older versions.

The usual SQL guarantees are:

| Isolation level    | Dirty read | Unrepeatable read | Phantom read | Serialisation anomaly |
| ------------------ | ---------- | ----------------- | ------------ | --------------------- |
| `READ UNCOMMITTED` | Allowed    | Allowed           | Allowed      | Allowed               |
| `READ COMMITTED`   | Prevented  | Allowed           | Allowed      | Allowed               |
| `REPEATABLE READ`  | Prevented  | Prevented         | Allowed      | Allowed               |
| `SERIALIZABLE`     | Prevented  | Prevented         | Prevented    | Prevented             |

A phantom changes the set of rows matching a repeated query. A serialisation anomaly makes committed results inconsistent with every serial order. Locking existing rows alone cannot prevent a new matching row from appearing; a lock-based serialisable implementation also needs range or predicate protection.

Implementations may provide stronger guarantees. PostgreSQL treats `READ UNCOMMITTED` as `READ COMMITTED`. Its `REPEATABLE READ` uses a stable snapshot and prevents phantoms, while still allowing serialisation anomalies. Its `SERIALIZABLE` mode detects dangerous dependencies and may abort a transaction, requiring a retry. See the [PostgreSQL isolation documentation](https://www.postgresql.org/docs/18/transaction-iso.html).

### Deadlock

A deadlock is a cycle of transactions waiting for locks held by each other. A **waits-for graph** has an edge $T_i\to T_j$ when $T_i$ is blocked by $T_j$. A detected cycle can be broken by aborting a participant. This graph tracks waiting; the precedence graph tracks conflicting operations already executed.

For timestamp-based prevention, older transactions have higher priority. Suppose $T_i$ requests a lock held by $T_j$:

| Rule       | $T_i$ older than $T_j$ | $T_i$ younger than $T_j$ |
| ---------- | ---------------------- | ------------------------ |
| Wait-die   | $T_i$ waits            | Abort $T_i$              |
| Wound-wait | Abort $T_j$            | $T_i$ waits              |

Wait-die permits waiting only from older to younger; wound-wait permits the reverse. Either direction rules out a cycle. Retain the original timestamp when restarting, so repeated aborts do not continually reset a transaction's priority. [CMU, section 3](https://15445.courses.cs.cmu.edu/spring2023/notes/16-twophaselocking.pdf).
