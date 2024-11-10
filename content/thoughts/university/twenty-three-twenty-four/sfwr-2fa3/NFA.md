---
id: NFA
tags:
  - sfwr2fa3
date: "2024-01-30"
title: NFA
---
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
S &: \text{Start states},\quad S \subseteq Q \\\
F &: \text{Final states},\quad F \subseteq Q \\\
\end{align*}
$$

## examples

1. $\mathcal{L}(M) = \{ abxba \mid x \in \Sigma^{*}\}$
  ```mermaid
  stateDiagram-v2
    direction LR
    [*] --> 0
    0 --> 1 : a
    1 --> 2 : b
    2 --> 2 : a, b
    2 --> 3 : b
    3 --> 4 : a
    4 --> [*]
  ```

2. $\mathcal{L}(M) = \{ yx \mid x = 00 \lor x =11 \land  y \in \Sigma^{*}\}$
  ```mermaid
  stateDiagram-v2
    direction LR
    [*] --> 1
    1 --> 1 : 0,1
    1 --> 2 : 0
    2 --> 3 : 0
    3 --> [*]
    1 --> 4 : 1
    4 --> 3 : 1
  ```

## $\epsilon$ transition
```mermaid
graph LR

  s((*)) --> s1{{s1}} --"1"--> s2{{s2}} --"1"--> s3{{s3}} --"e"--> s4{{s4}}
  s1{{s1}} --"e"--> s4{{s4}}
  s1{{s1}} --"0"--> s1{{s1}}
  s3{{s3}} --"1"--> s3{{s3}}
```
![[thoughts/university/twenty-three-twenty-four/sfwr-2fa3/eps-nfa.jpeg]]

---

Given the following $M$
```mermaid
graph LR

  s((*)) --> s1{{s1}} --"1"--> s2{{s2}} --"1"--> s3{{s3}} --"e"--> s4{{s4}}
  s1{{s1}} --"e"--> s4{{s4}}
  s1{{s1}} --"0"--> s1{{s1}}
  s3{{s3}} --"1"--> s3{{s3}}
```

$\mathcal{L}(M) = \{0^n1^m \mid n \geq 0, m \neq 1 \space, x \in \Sigma^{*}\}$
