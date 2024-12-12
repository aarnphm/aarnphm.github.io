---
id: Transaction
tags:
  - sfwr3db3
date: "2024-12-11"
description: and concurrency control.
modified: 2024-12-11 07:51:44 GMT-05:00
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
- serialisable schedule: equivalent to some _serial execution_ of the transaction

### serial

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/serial-transaction.png]]

> [!note] serialisable schedule
>
> ![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/serialisable-transaction.png|Note that this is not a serial schedule, given there are interleaved operations.]]
>
> $S:R_1(A),W_1(A), R_2(A), W_2(A), R_1(B), W_1(B), R_2(B), W_2(B)$

