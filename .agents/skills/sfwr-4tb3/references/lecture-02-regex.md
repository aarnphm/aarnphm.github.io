---
name: lecture-02-regex
description: regular expressions, NFA, DFA, equality, and the classic "extend the regex language" question type
---

# lecture 02 — regular languages

## the regex AST (course-canonical)

```python
class RegEx:
  pass


class ε(RegEx):  # empty string
  def __repr__(self):
    return 'ε'


class Sym(RegEx):  # single symbol
  def __init__(self, a: str):
    self.a = a

  def __repr__(self):
    return self.a


class Choice(RegEx):  # E1 | E2
  def __init__(self, E1, E2):
    self.E1, self.E2 = E1, E2


class Conc(RegEx):  # E1 E2
  def __init__(self, E1, E2):
    self.E1, self.E2 = E1, E2


class Star(RegEx):  # E*
  def __init__(self, E):
    self.E = E
```

abstract syntax (EBNF): $E \to \texttt{'}\epsilon\texttt{'} \mid \Sigma \mid E \texttt{ '|' } E \mid E\,E \mid E \texttt{ '*'}$. precedence: $*$ binds tighter than concat, which binds tighter than $\mid$.

## thompson-style RegEx → NFA

(`RegExToFSA` in the notebook; fresh state ids come from the enclosing counter `QC`.)

- `ε()` → one state that is both initial and accepting, no transitions
- `Sym(a)` → two states $q \xrightarrow{a} r$; $q$ initial, $r$ accepting
- `Choice(E1, E2)` → build $A_1$, $A_2$ recursively; new state $q$ with $\epsilon$-transitions to both sets of initials; accepting = $A_1.F \cup A_2.F$
- `Conc(E1, E2)` → $A_1$ and $A_2$; $\epsilon$-transitions from every $q \in A_1.F$ to $A_2.I$; accepting = $A_2.F$
- `Star(E)` → $A$; $\epsilon$-transitions from every $q \in A.F$ back to $A.I$; accepting = $A.I \cup A.F$

## extending the regex — the recipe

whenever a question says "extend regex with X", do these three moves:

1. **add an AST class**: minimal `__init__` storing children, `__repr__` for display. follow the existing single-assignment-in-init idiom.
2. **add a case to `RegExToFSA.ToFSA`**: build children via recursive `ToFSA`, then stitch states with $\epsilon$-transitions and `merge` from the notebook.
3. **extend semantic helpers as needed**: $L(A)$ iterator, `equalRegEx`, or a nullability/derivatives function if the question asks for language equality.

## the specific case: exponentiation $E^n$

$E^n$ denotes $E$ concatenated with itself exactly $n$ times; $E^0 = \epsilon$, $E^{n+1} = E\,E^n$. variants:

- $E^{n}$ — exactly $n$ times
- $E^{n,m}$ / $E\{n,m\}$ — between $n$ and $m$ times
- $E^{n,}$ / $E\{n,\}$ — at least $n$ times (equivalent to $E^n \cdot E^*$)

### approach A (course canonical): helper function, no new AST class

this is how assignment 5's "RE with Counted Repetition" does it. the trick is that $E^n$ reduces to `Conc` of $E$ with itself $n$ times, so no new AST node is needed:

```python
def repeat(e, n):
  r = ε()
  for _ in range(n):
    r = Conc(r, e)
  return r


def repeatRange(e, lo, hi):
  r = repeat(e, lo)
  for i in range(lo + 1, hi + 1):
    r = Choice(r, repeat(e, i))
  return r
```

then in the parser for the extended regex syntax:

```python
elif sym == '{':
  nxt()
  n = integer()
  if sym == '}':
    nxt(); e = repeat(e, n)
  elif sym == ',':
    nxt()
    if sym == '}':
      nxt(); e = Conc(repeat(e, n), Star(e))   # {n,} → at least n
    else:
      m = integer()
      if sym == '}': nxt()
      else: raise Exception("'}' expected at " + str(pos))
      e = repeatRange(e, n, m)                  # {n,m} → between n and m
```

the beauty: `RegExToFSA` is unchanged. the existing `Conc`, `Choice`, `Star`, `ε` cases handle everything.

### approach B: explicit `Exp` AST class

only needed if the question asks for $E^n$ as an independent notion in the AST (e.g. for a specialised pretty-printer or a different semantic function):

```python
class Exp(RegEx):
  def __init__(self, E: RegEx, n: int):
    self.E, self.n = E, n

  def __repr__(self):
    return f'({self.E})^{self.n}'
```

FSA case:

```python
case Exp(E=E, n=n):
  if n == 0:
    return ToFSA(ε())
  A = ToFSA(E)
  for _ in range(n - 1):
    B = ToFSA(E)
    δ = merge(A.δ | B.δ, {q: {'ε': B.I} for q in A.F})
    A = FiniteStateAutomaton(A.Σ | B.Σ, A.Q | B.Q, A.I, δ, B.F)
  return A
```

three edge cases:

- **fresh states per copy**: each `ToFSA(E)` call bumps `QC`; don't clone an FSA
- **$n = 0$** must resolve to $\epsilon$, not a no-op FSA without an accepting state
- **alphabet union**: $A.\Sigma \cup B.\Sigma$, matters when $E$ uses a subset of $\Sigma$

### which approach does a practice-final question want

read the question text:

- if it says "extend the parser" → approach A (helper + parser change)
- if it says "extend the regex language" or "add a new construct" → approach B (new class + FSA case)
- if it shows `class Exp(RegEx): ...` already declared in the notebook → approach B, finish it
- if it gives an attribute grammar mentioning `repeat(e, n)` → approach A

most 4tb3 exponentiation questions take approach A because the course lives in "reduce to primitives" world.

## algebraic identities worth remembering

| law                   | expression                                          |
| :-------------------- | :-------------------------------------------------- |
| idempotence of choice | $E \mid E = E$                                      |
| absorption by empty   | $E \cdot \emptyset = \emptyset \cdot E = \emptyset$ |
| unit of concat        | $E \cdot \epsilon = \epsilon \cdot E = E$           |
| star of empty         | $\emptyset^* = \epsilon^* = \epsilon$               |
| star of star          | $(E^*)^* = E^*$                                     |
| distributivity        | $E \cdot (F \mid G) = E \cdot F \mid E \cdot G$     |

these show up in "simplify this regex" and "prove equality" questions.

## equivalence of DFAs

$\text{equiv}(A, B)$ works by simultaneously traversing both DFAs from their initial states; if the reached pair ever has one accepting and the other not, they differ. $\text{incl}(A, B)$ is the same idea but only fails when $A$ accepts and $B$ rejects.

## connecting to regular grammars

for every regex $E$, there is an equivalent regular grammar $G$ with one production per regex operator; the translation is mechanical (see the notebook's arden's-lemma-style elimination). likewise, for every DFA $A$ there is an equivalent regular grammar with one nonterminal per state.
