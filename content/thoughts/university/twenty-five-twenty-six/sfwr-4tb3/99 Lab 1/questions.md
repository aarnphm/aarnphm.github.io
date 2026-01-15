---
date: "2026-01-07"
description: more more more.
id: lab1
modified: 2026-01-15 11:30:19 GMT-05:00
tags:
  - sfwr4tb3
title: questions
---

## different derivation

> [!question]
>
> Consider again grammar $G_{0} = (T, N, P, S)$. Give two different derivations of `Dave runs`

grammar:

```text
S -> NP VP
NP -> PN
VP -> V
PN -> Dave | ...
V -> runs | ...
```

leftmost derivation:

$$
\begin{aligned}
S &\Rightarrow \text{NP} \; \text{VP} \\
  &\Rightarrow \text{PN} \; \text{VP} \\
  &\Rightarrow \text{Dave} \; \text{VP} \\
  &\Rightarrow \text{Dave} \; \text{V} \\
  &\Rightarrow \text{Dave} \; \text{runs}
\end{aligned}
$$

rightmost derivation:

$$
\begin{aligned}
S &\Rightarrow \text{NP} \; \text{VP} \\
  &\Rightarrow \text{NP} \; \text{V} \\
  &\Rightarrow \text{NP} \; \text{runs} \\
  &\Rightarrow \text{PN} \; \text{runs} \\
  &\Rightarrow \text{Dave} \; \text{runs}
\end{aligned}
$$

## equivalent languages

> [!question]
>
> Recall grammar $G_{2}$. Let $G_{2}^{'} = (T, N, P, S)$ with $T = \{a\}$, $N = \{S\}$ productions $P$:
>
> $$
> \begin{aligned}
> S &\to \epsilon \\
> S  &\to Sa
> \end{aligned}
> $$
>
> Prove that $L(G_{2}^{'}) = \{a^{n} \mid  n \ge 0\}$, which is the same as $L(G_{2})$

Note that for grammar $G_{2}$ has productions $P$:

$$
\begin{aligned}
S &\to \epsilon \\
S  &\to aS
\end{aligned}
$$

**prove $L(G_2') = \{a^n \mid n \ge 0\}$**

**part 1: $L(G_2') \subseteq \{a^n \mid n \ge 0\}$**

induction on derivation length $k$:

- **base** ($k=1$): only $S \Rightarrow \epsilon = a^0$
- **inductive**: assume all derivations of length $\le k$ produce $a^m$ for some $m$. for length $k+1$:
  - must start with $S \Rightarrow Sa$
  - remaining derivation from $S$ has length $k$, producing $a^m$ by IH
  - so $S \Rightarrow Sa \Rightarrow^* a^m a = a^{m+1}$

**part 2: $\{a^n \mid n \ge 0\} \subseteq L(G_2')$**

induction on $n$:

- **base** ($n=0$): $S \Rightarrow \epsilon$
- **inductive**: assume $S \Rightarrow^* a^k$. show $a^{k+1} \in L(G_2')$:

$$S \Rightarrow Sa \Rightarrow^* a^k a = a^{k+1}$$

where $S \Rightarrow^* a^k$ by IH.

**explicit derivation pattern for $a^n$:**

$$S \Rightarrow Sa \Rightarrow Saa \Rightarrow \cdots \Rightarrow Sa^n \Rightarrow \epsilon a^n = a^n$$

(apply $S \to Sa$ exactly $n$ times, then $S \to \epsilon$)

**equivalence to $G_2$:**

$G_2$: $S \to \epsilon \mid aS$ generates left-to-right, $G_2'$: $S \to \epsilon \mid Sa$ generates right-to-left—same language bc concatenation over single symbol is order-invariant. both produce exactly $a^*$.

## derivation in copy language

_context-sensitive grammar_

> [!question]
>
> Recall $G_{5} = (T, N, P, S)$ with productions:
>
> $$
> \begin{aligned}
> S &\to aAS \mid bBS \mid \epsilon \\
> Aa &\to aA \\
> Ab &\to bA \\
> Ba &\to aB \\
> Bb &\to bB \\
> AS &\to Sa \\
> BS &\to Sb
> \end{aligned}
> $$
>
> Give a derivation of `abbabb`

for `abbabb` where $w = \text{abb}$:

$$
\begin{aligned}
S &\Rightarrow aAS \\
  &\Rightarrow aAbBS \\
  &\Rightarrow aAbBbBS \\
  &\Rightarrow abABbBS && (Ab \to bA) \\
  &\Rightarrow abABbSb && (BS \to Sb) \\
  &\Rightarrow abAbBSb && (Bb \to bB) \\
  &\Rightarrow abAbSbb && (BS \to Sb) \\
  &\Rightarrow abbASbb && (Ab \to bA) \\
  &\Rightarrow abbSabb && (AS \to Sa) \\
  &\Rightarrow abbabb && (S \to \epsilon)
\end{aligned}
$$

another solution:
$$aAbBbBS \to abbABBS \to abbSabb \to abbabb$$

## ambiguity in English

> [!question]
>
> A well-known syntactically ambiguous English sentence is `Time flies like an arrow`. Explain the ambiguity

the ambiguity stems from part-of-speech flexibility:

| word  | possible categories           |
| ----- | ----------------------------- |
| time  | N (noun), V (verb)            |
| flies | N (plural noun), V (3sg verb) |
| like  | P (preposition), V (verb)     |

parse 1, canonical:

```
[S [NP time] [VP [V flies] [PP [P like] [NP an arrow]]]]
```

"time passes swiftly, as an arrow does"

parse 2, imperative:

```
[S [VP [V time] [NP flies] [PP [P like] [NP an arrow]]]]
```

"measure the speed of flies in the manner of an arrow" (or: using an arrow as your timing reference)

parse 3, entomological:

```
[S [NP time flies] [VP [V like] [NP an arrow]]]
```

"a species called 'time flies' is fond of arrows" (cf. "fruit flies like a banana")

the grammar is ambiguous bc it assigns multiple derivations to one surface form.
