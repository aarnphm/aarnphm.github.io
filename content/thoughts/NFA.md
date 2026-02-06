---
date: '2024-01-30'
description: via subset construction algorithm of DFA
id: NFA
modified: 2025-10-29 02:15:29 GMT-04:00
tags:
  - math
title: non-deterministic finite automaton
---

see also [[thoughts/DFA|deterministic finite automaton]]

## definition

$$
\Sigma^{*}: \text{set of all strings based off }\Sigma
$$

$$
\begin{align*}
\text{NFA}\quad M &= (Q, \Sigma, \Delta, S, F)  \\\
Q &: \text{finite set of states} \\\
\Sigma &: \text{finite alphabet} \\\
\Delta &: Q \times \Sigma \rightarrow P(Q) \\\
S &: \text{Start states},\quad S \subseteq P(Q) \\\
F &: \text{Final states},\quad F \subseteq Q \\\
\end{align*}
$$

## examples

1. $\mathcal{L}(M) = \{ abxba \mid x \in \Sigma^{*}\}$

<details>

```mermaid
stateDiagram-v2
    direction LR
    classDef accepting fill:#4CAF50,stroke:#333,stroke-width:2px
    classDef start fill:#FFD700,stroke:#333,stroke-width:2px

    s0: q0
    s1: q1
    s2: q2
    s3: q3
    s4: q4
    s5: q5

    [*] --> s0
    s0 --> s1: a
    s1 --> s2: b
    s2 --> s2: Σ
    s2 --> s3: b
    s3 --> s4: a
    s4 --> [*]

    class s4 accepting
    class s0 start
```

</details>

2. $\mathcal{L}(M) = \{ yx \mid x = 00 \lor x =11 \land  y \in \Sigma^{*}\}$

```mermaid
stateDiagram-v2
    direction LR
    classDef accepting fill:#4CAF50,stroke:#333,stroke-width:2px
    classDef start fill:#FFD700,stroke:#333,stroke-width:2px

    s0: q0
    s1: q1
    s2: q2
    s3: q3
    s4: q4

    [*] --> s0
    s0 --> s0: 0,1
    s0 --> s1: 0
    s0 --> s3: 1
    s1 --> s2: 0
    s3 --> s4: 1
    s2 --> [*]
    s4 --> [*]

    class s2,s4 accepting
    class s0 start
```

## epsilon transition

```mermaid
stateDiagram-v2
  direction LR
  [*] --> s1
  s1 --> s2: 1
  s2 --> s3: 1
  s3 --> s4: ε
  s1 --> s4: ε
  s1 --> s1: 0
  s3 --> s3: 1
```

Given the following $M$

```mermaid
stateDiagram-v2
  direction LR
  [*] --> s1
  s1 --> s2: 1
  s2 --> s3: 1
  s3 --> s4: ε
  s1 --> s4: ε
  s1 --> s1: 0
  s3 --> s3: 1
```

$\mathcal{L}(M) = \{0^n1^m \mid n \geq 0, m \neq 1 \space, x \in \Sigma^{*}\}$
