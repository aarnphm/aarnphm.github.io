---
claude:
  session:
    cf21263a-2f1a-436c-9bd8-c71647135817: basic
date: "2026-01-12"
description: grammar et al.
id: results
modified: 2026-01-15 12:48:51 GMT-05:00
seealso:
  - "[[thoughts/NFA]]"
  - "[[thoughts/Context-Free Grammar]]"
tags:
  - sfwr4tb3
  - assignment
title: Language and grammar constructions
---

## A1

> [!notes]
>
> Recall grammar $G_{0}$. Define a new grammar $G_{0}^{'}$ by extending $G_{0}$:
>
> - A verb phrase is a verb phrase followed by a prepositional phrase.
> - A prepositional phrase is a preposition followed by a noun phrase.
> - Prepositions are `in` and `on`.
> - Nouns are `park` and `child`.

> [!question]
>
> Give the full definition of $G_{0}^{'}$! Is $G_{0}^{'}$ regular, context-free, or context-sensitive? Draw the parse tree of `the child eats a banana in the park`! [8 points]

for _the definition of_ $G_{0}^{'}$:

$$
G_{0}^{'} = (T', N', P', S)
$$

where:

- $T' = \{\text{Kevin}, \text{Dave}, \text{a}, \text{the}, \text{banana}, \text{apple}, \text{eats}, \text{runs}, \text{in}, \text{on}, \text{park}, \text{child}\}$
- $N' = \{S, NP, VP, PN, D, N, V, PP, P\}$
- Productions $P'$:

$$
\begin{aligned}
S &\to NP \space VP \\
NP &\to PN \mid D \space N \\
VP &\to V \mid V \space NP \mid VP \space PP \\
PP &\to P \space NP \\
PN &\to \text{Kevin} \mid \text{Dave} \\
D &\to \text{a} \mid \text{the} \\
N &\to \text{banana} \mid \text{apple} \mid \text{park} \mid \text{child} \\
V &\to \text{eats} \mid \text{runs} \\
P &\to \text{in} \mid \text{on}
\end{aligned}
$$

Now, $G_{0}^{'}$ is _context-free_, because every production has exactly one nonterminal on the LHS. It is NOT regular because regular grammars require productions of form $A \to \epsilon$, $A \to a$, or $A \to aB$ (right-linear), and we have $VP \to VP \space PP$ which has two nonterminals on the RHS.

**Parse tree for `the child eats a banana in the park`:**

![[thoughts/university/twenty-five-twenty-six/sfwr-4tb3/88 Assignment 1/img/parse_tree_a1.svg]]

_derivation:_

$$
\begin{aligned}
S &\Rightarrow NP \space VP \\
&\Rightarrow D \space N \space VP \\
&\Rightarrow \text{the} \space N \space VP \\
&\Rightarrow \text{the} \space \text{child} \space VP \\
&\Rightarrow \text{the} \space \text{child} \space VP \space PP \\
&\Rightarrow \text{the} \space \text{child} \space V \space NP \space PP \\
&\Rightarrow \text{the} \space \text{child} \space \text{eats} \space NP \space PP \\
&\Rightarrow \text{the} \space \text{child} \space \text{eats} \space D \space N \space PP \\
&\Rightarrow \text{the} \space \text{child} \space \text{eats} \space \text{a} \space \text{banana} \space PP \\
&\Rightarrow \text{the} \space \text{child} \space \text{eats} \space \text{a} \space \text{banana} \space P \space NP \\
&\Rightarrow \text{the} \space \text{child} \space \text{eats} \space \text{a} \space \text{banana} \space \text{in} \space NP \\
&\Rightarrow \text{the} \space \text{child} \space \text{eats} \space \text{a} \space \text{banana} \space \text{in} \space D \space N \\
&\Rightarrow \text{the} \space \text{child} \space \text{eats} \space \text{a} \space \text{banana} \space \text{in} \space \text{the} \space \text{park}
\end{aligned}
$$

## A2

> [!notes]
>
> Recall $G_{3} = (T, N, P, S)$ where $T = \{a,b,c\}, N = \{S\}$, and the production $P$ are
>
> $$
> \begin{aligned}
> S &\to b \\
> S  &\to aSc
> \end{aligned}
> $$

> [!question]
>
> Prove that $L(G_{3}) = \{a^{n}bc^{n} \mid n \ge 0\}$ [8 points]

_Proof._ We prove by inclusion in both directions. Note that by definition $L(G) = \{\chi \in T^{*} \mid S \Rightarrow^{+} \chi\}$

**Direction 1:** $L(G_3) \subseteq \{a^n b c^n \mid n \ge 0\}$

We show that for every $\chi \in T^*$ derivable from $S$, there exists $n \ge 0$ such that $\chi = a^n b c^n$. This is proved by induction on the length of derivations from $S$.

_Base._ A derivation of length 1 from $S$ can only use the production $S \to b$, yielding $\chi = b = a^0 b c^0$. The base case holds.

_Step._ Assume every string derivable from $S$ in $k$ steps is $a^i b c^i$ for some $i \ge 0$. Consider a derivation of $k+1$ steps. The first step must use $S \to aSc$ (since $S \to b$ gives a derivation of length 1). Thus:

$$
S \Rightarrow aSc \Rightarrow^k a\chi c
$$

where $\chi$ is derived from $S$ in $k$ steps. By the induction hypothesis, $\chi = a^i b c^i$ for some $i \ge 0$. Therefore:

$$
a \chi c = a \cdot a^i b c^i \cdot c = a^{i+1} b c^{i+1}
$$

which is of the required form.

**Direction 2:** $\{a^n b c^n \mid n \ge 0\} \subseteq L(G_3)$

We show that every string $a^n b c^n$ for $n \ge 0$ can be derived from $S$. This is proved by induction on $n$.

_Base._ For $n = 0$, we have $a^0 b c^0 = b$. By the production $S \to b$, we get $S \Rightarrow b$. The base case holds.

_Step._ Assume $a^n b c^n$ can be derived, i.e., $S \Rightarrow^+ a^n b c^n$. We show $a^{n+1} b c^{n+1}$ can also be derived:

$$
S \Rightarrow aSc \Rightarrow^+ a \cdot a^n b c^n \cdot c = a^{n+1} b c^{n+1}
$$

Since both inclusions hold, $L(G_3) = \{a^n b c^n \mid n \ge 0\}$. $\blacksquare$

## A3

> [!notes]
>
> Recall $G = (T, N, P, S)$ where $T = \{a,b\}, N = \{S\}$, and the production $P$ are
>
> $$
> \begin{aligned}
> S &\to \epsilon \\
> S &\to aSbS \\
> S  &\to bSaS
> \end{aligned}
> $$

> [!question]
>
> Show that $G$ is ambiguous by giving two derivations with different parse trees for `abab` [2 points]

**Derivation 1:** (rightmost `S` of root produces second `ab`)

$$
S \Rightarrow aSbS \Rightarrow abS \Rightarrow abaSbS \Rightarrow ababS \Rightarrow abab
$$

![[thoughts/university/twenty-five-twenty-six/sfwr-4tb3/88 Assignment 1/img/parse_tree_a3_1.svg]]

**Derivation 2:** (leftmost `S` of root uses `bSaS`)

$$
S \Rightarrow aSbS \Rightarrow abSaSbS \Rightarrow abaSbS \Rightarrow ababS \Rightarrow abab
$$

![[thoughts/university/twenty-five-twenty-six/sfwr-4tb3/88 Assignment 1/img/parse_tree_a3_2.svg]]

> [!question]
>
> What language does $G$ generate? Give a formal proof!
>
> _Hint_ use the notation $a\#\sigma$ for the number of ocurrence of $a$ in $\sigma$

_claim:_ $L(G) = \{\sigma \in \{a,b\}^* \mid a\#\sigma = b\#\sigma\}$ (strings with equal counts of `a` and `b`)

_proof._ By inclusion in both directions.

**Direction 1:** $L(G) \subseteq \{\sigma \mid a\#\sigma = b\#\sigma\}$

We show by induction on derivation length that every derivable string has equal counts of `a` and `b`.

_Base._ $S \Rightarrow \epsilon$. We have $a\#\epsilon = b\#\epsilon = 0$.

_Step._ Assume all strings derivable in $\le k$ steps satisfy $a\#\sigma = b\#\sigma$.

- For $S \Rightarrow aSbS$: the result is $a \sigma_1 b \sigma_2$ where $\sigma_1, \sigma_2$ are derivable from $S$.

  $$
  a\#(a\sigma_1 b\sigma_2) = 1 + a\#\sigma_1 + a\#\sigma_2
  $$

  $$
  b\#(a\sigma_1 b\sigma_2) = b\#\sigma_1 + 1 + b\#\sigma_2
  $$

  By induction hypothesis, $a\#\sigma_1 = b\#\sigma_1$ and $a\#\sigma_2 = b\#\sigma_2$, so the counts are equal. $\boxed{}$

- For $S \Rightarrow bSaS$: the result is $b \sigma_1 a \sigma_2$ where $\sigma_1, \sigma_2$ are derivable from $S$.

  $$
  a\#(b\sigma_1 a\sigma_2) = a\#\sigma_1 + 1 + a\#\sigma_2
  $$

  $$
  b\#(b\sigma_1 a\sigma_2) = 1 + b\#\sigma_1 + b\#\sigma_2
  $$

  By induction hypothesis, $a\#\sigma_1 = b\#\sigma_1$ and $a\#\sigma_2 = b\#\sigma_2$, so the counts are equal. $\boxed{}$

**Direction 2:** $\{\sigma \mid a\#\sigma = b\#\sigma\} \subseteq L(G)$

We prove by strong induction on $|\sigma|$ that every balanced string is derivable.

_Base._ $|\sigma| = 0$ implies $\sigma = \epsilon$. We have $S \Rightarrow \epsilon$.

_Step._ Assume all balanced strings $\tau$ with $|\tau| < |\sigma|$ are in $L(G)$.

Let $|\sigma| > 0$ with $a\#\sigma = b\#\sigma$. Then $\sigma$ starts with `a` or `b`.

**Case 1:** $\sigma = a\tau$ for some $\tau$.

Since $a\#\sigma = b\#\sigma$, we have $1 + a\#\tau = b\#\tau$, so $b\#\tau = a\#\tau + 1$.

Define $f(i) = a\#\tau_{[0:i]} - b\#\tau_{[0:i]}$ (the "balance" after reading $i$ symbols of $\tau$).

- $f(0) = 0$
- $f(|\tau|) = a\#\tau - b\#\tau = -1$

Since $f$ changes by $\pm 1$ at each step and goes from $0$ to $-1$, there exists a first index $j$ where $f(j) = -1$. At this position, $\tau_{j-1} = b$ and $\tau_{[0:j-1]}$ is balanced.

Therefore $\tau = \tau_1 b \tau_2$ where:

- $a\#\tau_1 = b\#\tau_1$ (balanced prefix)
- $a\#\tau_2 = b\#\tau_2$ (remaining must also balance)

So $\sigma = a \tau_1 b \tau_2$ with $|\tau_1| < |\sigma|$ and $|\tau_2| < |\sigma|$.

By induction hypothesis, $\tau_1, \tau_2 \in L(G)$. Using $S \Rightarrow aSbS$ with derivations for $\tau_1$ and $\tau_2$ gives $\sigma \in L(G)$. $\boxed{}$

**Case 2:** $\sigma = b\tau$ for some $\tau$.

Since $a\#\sigma = b\#\sigma$, we have $a\#\tau = 1 + b\#\tau$, so $a\#\tau = b\#\tau + 1$.

Define $g(i) = b\#\tau_{[0:i]} - a\#\tau_{[0:i]}$ (the "balance" after reading $i$ symbols of $\tau$).

- $g(0) = 0$
- $g(|\tau|) = b\#\tau - a\#\tau = -1$

Since $g$ changes by $\pm 1$ at each step and goes from $0$ to $-1$, there exists a first index $j$ where $g(j) = -1$. At this position, $\tau_{j-1} = a$ and $\tau_{[0:j-1]}$ is balanced.

Therefore $\tau = \tau_1 a \tau_2$ where:

- $a\#\tau_1 = b\#\tau_1$ (balanced prefix)
- $a\#\tau_2 = b\#\tau_2$ (remaining must also balance)

So $\sigma = b \tau_1 a \tau_2$ with $|\tau_1| < |\sigma|$ and $|\tau_2| < |\sigma|$.

By IH, $\tau_1, \tau_2 \in L(G)$. Using $S \Rightarrow bSaS$ with derivations for $\tau_1$ and $\tau_2$ gives $\sigma \in L(G)$. $\boxed{}$

**Conclusion:** $L(G) = \{\sigma \in \{a,b\}^* \mid a\#\sigma = b\#\sigma\}$. $\blacksquare$

## A4

> [!question]
>
> You ask ChatGPT:
>
> > Consider formal language theory. What is a monotonic grammar for the copy language?
>
> Below is ChatGPT's response. Argue why the given grammar is incorrect
>
> > [!notes]- **ChatGPT 6 January 2026, edited for formatting**
> >
> > In formal language theory, the **copy language** is usually defined as
> >
> > $$
> > L(\text{copy}) = \{w\space w \mid w \in \{a, b\}^{*}\}
> > $$
> >
> > This language is **not context-free**, but **is context-sensitive**, and therefore it can be generated by a **monotonic grammar**.
> >
> > ---
> >
> > ### Monotonic grammar (definition)
> >
> > A grammar $G = (V, \Sigma, P, S)$ is **monotonic** if for every production $\alpha \to \beta \in P$ we have $\abs{\alpha} \le \abs{\beta}$ except possibly for $S \to \epsilon$
> >
> > Monotonic grammars are equivalent in generative power to **context-sensitive grammars**.
> >
> > ---
> >
> > ### A monotonic grammar for the copy language
> >
> > Let
> >
> > - Terminals: $\Sigma = \{a, b\}$
> > - Nonterminals: $V = \{S, A, B\}$
> >
> > The productions are:
> >
> > #### Generating the first copy and marking it
> >
> > $$
> > \begin{aligned}
> > S &\to a\space S\space A \\
> > S &\to b\space S\space B \\
> > S &\to a\space  A \\
> > S &\to b\space  B
> > \end{aligned}
> > $$
> >
> > #### Copying symbols to the right
> >
> > $$
> > \begin{aligned}
> > A\space a &\to a\space A \\
> > A\space b &\to b\space A \\
> > B\space a &\to a\space B \\
> > B\space b &\to b\space B
> > \end{aligned}
> > $$
> >
> > #### Finalizing the copy
> >
> > $$
> > \begin{aligned}
> > A &\to a \\
> > B &\to b
> > \end{aligned}
> > $$
> >
> > ---
> >
> > ### Why this works
> >
> > - The first phase nondeterministically generates a string $w$ while placing marker $A$ or $B$ that remember each symbols.
> > - The context-sensitive rules "push" terminals past the markers, effectively copying the strings.
> > - Finally, the markers are rewritten as terminals, yielding exactly $w\space w$
> > - All production satisfy $|\text{LHS}| \le |\text{RHS}|$, so the grammar is **monotonic**
> >
> > ---
> >
> > ### Conclusion
> >
> > The copy language $\{w\space w\mid w \in\{a,b\}^{*}\}$ is **context-sensitive**, and the grammar above is a valid **monotonic grammar** that generates it.

ChatGPT's grammar is _incorrect_ here. It generates strings outside the copy language and fails to generate strings inside of the language.

_exhibit 1_: The grammar generates `abba`, which is NOT in $L(\text{copy})$.

_Derivation:_

$$
S \Rightarrow aSA \Rightarrow abBA \Rightarrow abBa \Rightarrow abba
$$

There is no $w \in \{a,b\}^*$ such that $ww = abba$. (If $|w| = 2$, then $w \in \{aa, ab, ba, bb\}$, giving $ww \in \{aaaa, abab, baba, bbbb\}$, none of which equal $abba$.)

_exhibit 2_: The grammar cannot generate `abab`, which IS in $L(\text{copy})$ (with $w = ab$).

To generate a string starting with `ab`, we must use:

$$
S \Rightarrow aSA \Rightarrow a \cdot bB \cdot A = abBA
$$

or

$$
S \Rightarrow aSA \Rightarrow a \cdot bSB \cdot A = abSBA \Rightarrow \ldots
$$

In all cases, the markers appear in LIFO order (last generated marker comes first). For $w = ab$, we generate markers in order $A$ then $B$, but they appear as $\ldots BA$ in the sentential form.

The context-sensitive rules ($Aa \to aA$, $Ab \to bA$, $Ba \to aB$, $Bb \to bB$) move markers _rightward_ past terminals. Since the markers are already at the rightmost position (no terminals follow them), these rules cannot apply.

Converting markers directly via $A \to a$ and $B \to b$ gives:

$$
abBA \Rightarrow abBa \Rightarrow abba \neq abab
$$

_note_: the recursive structure $S \to aSA$ and $S \to bSB$ defined by ChatGPT accumulates markers in stack order (LIFO), but reproducing the original string requires queue order (FIFO).

Compare with $G_6$, which uses a terminator symbol $T$ and rules $AT \to Ta$, $BT \to Tb$ that convert markers from _right to left_, preserving the original order. $\blacksquare$

## A5

Suppose we are given a tree structure and want to process each leaf of the tree one by one. The tree is given by nested tuples (or lists) in Python, like `((1,2),3,(4,(5,6)))`. The natural way is to traverse a tree recursively to visit the leaves in order, as in

```python
from collections.abc import Iterable


def printleaves(node):
  if isinstance(node, Iterable):
    for c in node:
      printleaves(c)
  else:
    print(node)


printleaves(((1, 2), 3, (4, (5, 6))))
```

The task is to define the Python function `leaves` as a generator that will return the leaves in sequence from left to right without explicitly constructing a sequence with the leaves:

```python
def leaves(tree):
  raise NotImplementedError
```

You can use the following for testing:

```python
from typing import Generator

l = leaves(((1, 2), 3, (4, (5, 6))))
assert isinstance(l, Generator)
assert list(leaves(((1, 2), 3, (4, (5, 6))))) == [1, 2, 3, 4, 5, 6]
assert sum(leaves(((1, 2), 3, (4, (5, 6))))) == 21
```

_solution_:

![[thoughts/university/twenty-five-twenty-six/sfwr-4tb3/88 Assignment 1/a5.py]]
