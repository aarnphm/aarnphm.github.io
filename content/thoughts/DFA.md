---
id: DFA
tags:
  - sfwr2fa3
  - math
date: "2024-01-12"
modified: "2024-11-19"
title: Deterministic Finite Automata
---

## definition

$$
\Sigma^{*}: \text{set of all strings based off }\Sigma
$$

$$
\begin{align*}
\text{DFA}\quad M &= (Q, \Sigma, \delta, s, F)  \\\
Q &: \text{finite set of states} \\\
\Sigma &: \text{finite alphabet} \\\
\delta &: Q \times \Sigma \rightarrow Q \rightarrow \delta: Q \times \Sigma \rightarrow Q \\\
s &: \text{start state},\quad s\in{Q} \\\
F &: \text{set of final states},\quad F\subseteq{Q} \\\
\end{align*}
$$

### examples

Ex: $\Sigma = \{a, b\}$. Creates a DFA $M$ that accepts all strings that contains at least three a's.

$$
\begin{align*}
Q &= \{s_1, s_2, s_3, s_4\} \\\
s &= 1 \\\
F &= \{s_4\} \\\
\end{align*}
$$

Transition function:

$$
\begin{align*}
\delta(1, a) = s_2 \\\
\delta(1, b) = s_1 \\\
\delta(2, a) = s_3 \\\
\delta(2, b) = s_2 \\\
\delta(3, a) = s_4 \\\
\delta(3, b) = s_3 \\\
\delta(4, a) = \delta(4, b) = s_4 \\\
\end{align*}
$$

[[thoughts/representations|representation]]:

```mermaid
stateDiagram-v2
    direction LR
    classDef accepting fill:#4CAF50,stroke:#333,stroke-width:2px
    classDef start fill:#FFD700,stroke:#333,stroke-width:2px

    s1: s1
    s2: s2
    s3: s3
    s4: s4

    [*] --> s1
    s1 --> s1: b
    s1 --> s2: a

    s2 --> s2: b
    s2 --> s3: a

    s3 --> s3: b
    s3 --> s4: a

    s4 --> s4: a,b

    class s4 accepting
    class s1 start
```

> if in final string then accept, otherwise reject the string

## language.

[[thoughts/Language|Language]] of machine $\mathcal{L}(M)$ is the set of strings M accepts, such that $\mathcal{L}(M) \in \Sigma^{*}$

$$
\mathcal{L}(M) = \{w \in \Sigma^{*} | \delta(s, w) \in F\}
$$

> Assumption: $\Sigma = \{a, b\}$

> [!math] Questions
>
> Find DFA $M$ such that $\mathcal{L}(M)=$ the following
>
> 1. $\{ xab \mid x \in \Sigma^{*} \}$
> 2. $\{ x \mid |x| \% 2 = 0 \}$
> 3. $\{ x \mid x = 2^n\space ,\space n \in \mathbb{N} \}$, $\Sigma = \{0, 1\}$
> 4. $\{ x \mid "abc" \in x \}$, $\Sigma = \{a, b, c\}$
> 5. $\{ x \mid \text{a is the second last char of x} \}$
> 6. $\{ a^n \cdot b^n \mid n \ge 0 \}$
> 7. $\{ x \mid \text{a is the fifth last char of x} \}$

1.

```mermaid
stateDiagram-v2
    direction LR
    classDef accepting fill:#4CAF50,stroke:#333,stroke-width:2px
    classDef start fill:#FFD700,stroke:#333,stroke-width:2px

    s0: q0
    s1: q1
    s2: q2

    [*] --> s0
    s0 --> s0: b
    s0 --> s1: a
    s1 --> s0: a
    s1 --> s2: b
    s2 --> s0: a
    s2 --> s1: b

    class s2 accepting
    class s0 start
```

2.

```mermaid
stateDiagram-v2
    direction LR
    classDef accepting fill:#4CAF50,stroke:#333,stroke-width:2px
    classDef start fill:#FFD700,stroke:#333,stroke-width:2px

    s0: q0
    s1: q1

    [*] --> s0
    s0 --> s1: a,b
    s1 --> s0: a,b

    class s0 accepting
    class s0 start
```

3.

```mermaid
stateDiagram-v2
    direction LR
    classDef accepting fill:#4CAF50,stroke:#333,stroke-width:2px
    classDef start fill:#FFD700,stroke:#333,stroke-width:2px
    classDef dead fill:#ff6b6b,stroke:#333,stroke-width:2px

    s0: q0
    s1: q1
    s2: q2
    s3: dead

    [*] --> s0
    s0 --> s3: 0
    s0 --> s1: 1

    s1 --> s2: 0
    s1 --> s3: 1

    s2 --> s2: 0
    s2 --> s3: 1

    s3 --> s3: 0,1

    class s1 accepting
    class s0 start
    class s3 dead
```

4.

```mermaid
stateDiagram-v2
    direction LR
    classDef accepting fill:#4CAF50,stroke:#333,stroke-width:2px
    classDef start fill:#FFD700,stroke:#333,stroke-width:2px
    
    s0: q0
    s1: q1
    s2: q2
    s3: q3
    
    [*] --> s0
    
    s0 --> s0: b,c
    s0 --> s1: a
    
    s1 --> s0: a,c
    s1 --> s2: b
    
    s2 --> s0: a,b
    s2 --> s3: c
    
    s3 --> s3: a,b,c
    
    class s3 accepting
    class s0 start
```

5.

```mermaid
stateDiagram-v2
    direction LR
    classDef accepting fill:#4CAF50,stroke:#333,stroke-width:2px
    classDef start fill:#FFD700,stroke:#333,stroke-width:2px
    
    s0: q0
    s1: q1
    s2: q2
    s3: q3
    
    [*] --> s0
    
    s0 --> s0: a,b
    s0 --> s1: a
    
    s1 --> s2: b
    s1 --> s3: a
    
    s2 --> s0: a,b
    s3 --> s0: a,b
    
    class s2 accepting
    class s0 start
```
6.

non-regular.

_proof using Pumping Lemma_
- assume the language is regular, let $p$ be the pumping length.
- Consider string $s = a^n \cdot b^n$
- any way of diving $s=xyz$ where $\mid xy \mid \le p$ and $\mid y \mid \ge 0$ will results y contains only a'
- pumped wouldn't be in the language
q.e.d

7.

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
    
    s0 --> s0: a,b
    s0 --> s1: a
    
    s1 --> s2: a,b
    s2 --> s3: a,b
    s3 --> s4: a,b
    s4 --> s5: a,b
    s5 --> s0: a,b
    
    class s5 accepting
    class s0 start
```