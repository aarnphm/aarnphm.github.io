---
date: '2026-02-03'
description: AST et al.
id: results
modified: 2026-02-03 14:32:02 GMT-05:00
tags:
  - sfwr4tb3
  - assignment
title: languages and regex
---

## A1

1. Assume the vocabulary is $\{a, b\}$. Give a regular expression that describes all sequences containing at least two consecutive occurrences of $a$, like $aa, baa, aaa, abaaaaba$

$$(a|b)^*aa(a|b)^*$$

2. Now give a regular expression of the complement of the above language (without using a complement operator), that is, those sequences over $\{ a, b\}$ that must not contaiin two or more consecutive occurrences of $a$ but still may have an arbitrary number of occurrences of $a$! [^note]

$$b^*(ab^+)^*a?$$

[^note]: Between any two $a$'s, there must be at least one $b$. Structure: optional $b$'s at start, then zero or more ($a$ followed by one-or-more $b$'s), then optionally a trailing $a$. Equivalently: $(b|ab)^*(\epsilon|a)$

3. Assume that the vocabulary is $\{a, b, c\}$. Give a regular expression that describes all sequences in which the number of $a$ symbols is divisible by three

$$((b|c)^*a(b|c)^*a(b|c)^*a)^*(b|c)^*$$

3-state automaton counting $a$'s mod 3. Each cycle through the inner group consumes exactly 3 $a$'s with arbitrary $b/c$ between them.

4. Now give a regular expression for sequences with the total number of $b$ and $c$ symbols being three!

$$a^*(b|c)a^*(b|c)a^*(b|c)a^*$$

Exactly 3 slots for $b$-or-$c$, with arbitrary $a$'s in the 4 gaps.

## A2

given regular grammar in EBNF for fractional numbers:

$$
\begin{aligned}
N &\to '0' N \mid\; \ldots \mid '9' N \mid '.' F \\
F &\to '0' D \mid\; \ldots \mid '9' D \\
D &\to '0' D \mid\; \ldots\;| '9' D \mid ''
\end{aligned}
$$

Given grammar ($d = 0|1|...|9$):

$$N \to dN \mid \texttt{.}F \quad F \to dD \quad D \to dD \mid \epsilon$$

$D = dD \mid \epsilon \xrightarrow{\text{Arden}} d^*$

$F = dD = d \cdot d^* \xrightarrow{aa^*=a^+} d^+$

$N = dN \mid \texttt{.}d^+ \xrightarrow{\text{Arden}} d^*\texttt{.}d^+$

## A3

Here is the Python code

```python
class RegEx:
  pass


class ε(RegEx):
  def __repr__(self):
    return 'ε'


class Sym(RegEx):
  def __init__(self, a: str):
    self.a = a

  def __repr__(self):
    return str(self.a)


class Choice(RegEx):
  def __init__(self, E1: RegEx, E2: RegEx):
    self.E1, self.E2 = E1, E2

  def __repr__(self):
    return '(' + str(self.E1) + '|' + str(self.E2) + ')'


class Conc(RegEx):
  def __init__(self, E1: RegEx, E2: RegEx):
    self.E1, self.E2 = E1, E2

  def __repr__(self):
    return '(' + str(self.E1) + str(self.E2) + ')'


class Star(RegEx):
  def __init__(self, E: RegEx):
    self.E = E

  def __repr__(self):
    return '(' + str(self.E) + ')*'


class fset(frozenset):
  def __repr__(self):
    return '{' + ', '.join(str(e) for e in self) + '}'


def wrap(a):
  import textwrap

  return '\\n'.join(textwrap.wrap(str(a), width=12))


TransFunc = dict[str, dict[str, set[str]]]


class FiniteStateAutomaton:
  Σ: set[str]  # set of symbols
  Q: set[str]  # set of states
  I: set[str]  # I ⊆ Q, the initial states,
  δ: TransFunc  # representing Q ↛ Σ ↛ 𝒫Q, the transition function
  F: set[str]  # F ⊆ Q, the finite states
  vars = ()  # for reduced FSAs, the names of the original variables

  def __init__(self, Σ, Q, I, δ, F):
    self.Σ, self.Q, self.I, self.δ, self.F = Σ, fset(Q), fset(I), δ, fset(F)

  def draw(self, trace=None):
    from graphviz import Digraph

    dot = Digraph(
      graph_attr={'rankdir': 'LR'},
      node_attr={
        'fontsize': '10',
        'fontname': 'Noto Sans',
        'margin': '0',
        'width': '0.25',
      },  # 'nodesep': '0.75', 'ranksep': '0.75'
      edge_attr={
        'fontsize': '10',
        'fontname': 'Noto Sans',
        'arrowsize': '0.5',
      },
    )  # 'weight': '5.0' # create a directed graph
    for q in self.I:
      dot.node('_' + str(q), label='', shape='none', height='.0', width='.0')
      dot.node(wrap(q), shape='circle')
      dot.edge('_' + str(q), wrap(q), len='.1')
    P = self.I | self.F
    for q in self.δ:
      P = P | {q}
      for a in self.δ[q]:
        dot.node(wrap(q), shape='circle')
        for r in self.δ[q][a]:
          dot.node(wrap(r), shape='circle')
          dot.edge(wrap(q), wrap(r), label=str(a))
          P = P | {r}
    for q in self.F:
      dot.node(wrap(q), shape='doublecircle')
    for q in self.Q - P:  # place all unreachable nodes to the right
      dot.node(wrap(q), shape='circle')
      for p in P:
        dot.edge(wrap(p), wrap(q), style='invis')  # , constraint='false'
    if trace:
      xlab = {}  # maps states to Graphviz external labels
      for i in range(0, len(trace), 2):
        xlab[trace[i]] = (
          xlab[trace[i]] + ', ' + str(i // 2)
          if trace[i] in xlab
          else str(i // 2)
        )
      for q in xlab:
        dot.node(
          wrap(q),
          xlabel='<<font color="royalblue">' + wrap(xlab[q]) + '</font>>',
        )
    return dot

  def writepdf(self, name, trace=None):
    open(name, 'wb').write(self.draw(trace).pipe(format='pdf'))

  def writesvg(self, name, trace=None):
    open(name, 'wb').write(self.draw(trace).pipe(format='svg'))

  def __repr__(self):
    return (
      ' '.join(str(q) for q in self.I)
      + '\n'
      + '\n'.join(
        str(q) + ' ' + str(a) + ' → ' + ', '.join(str(r) for r in self.δ[q][a])
        for q in self.δ
        for a in self.δ[q]
        if self.δ[q][a] != set()
      )
      + '\n'
      + ' '.join(str(f) for f in self.F)
      + '\n'
    )


def parseFSA(fsa: str) -> FiniteStateAutomaton:
  fl = [line for line in fsa.split('\n') if line != '']
  I = (
    set(fl[0].split()) if len(fl) > 0 else set()
  )  # second line: initial initial ...
  Σ, Q, δ, F = set(), set(), {}, set()
  for line in fl[1:]:  # all subsequent lines
    if '→' in line:  # source action → target
      l, r = line.split('→')
      p, a, q = l.split()[0], l.split()[1], r.split()[0]
      if p in δ:
        s = δ[p]
        s[a] = s[a] | {q} if a in s else {q}
      else:
        δ[p] = {a: {q}}
      Σ.add(a)
      Q.add(p)
      Q.add(q)
    else:  # a line without → is assumed to have the final states
      F = set(line.split()) if len(line) > 0 else set()  # final final ...
  return FiniteStateAutomaton(Σ, Q | I | F, I, δ, F)


def setunion(S: set[set]) -> set:
  return set.union(set(), *S)


def δ̂(δ: TransFunc, P: set[str], a: str) -> set[str]:
  return fset(setunion(δ[p][a] for p in P if p in δ if a in δ[p]))


def ε_closure(Q, δ) -> set:  #
  C, W = set(Q), Q  # as C is updated, a copy of Q is needed
  # invariant: C ∪ ε-closure W δ = ε-closure Q δ
  # variant: ε-closure Q δ - C
  while W != set():
    W = δ̂(δ, W, 'ε') - C
    C |= W
  return fset(C)


def accepts(A: FiniteStateAutomaton, α: str):
  W = ε_closure(A.I, A.δ)
  for a in α:
    W = ε_closure(δ̂(A.δ, W, a), A.δ)
  return W & A.F != set()


setattr(FiniteStateAutomaton, 'accepts', accepts)


def merge(γ: TransFunc, δ: TransFunc) -> TransFunc:
  return (
    {q: γ[q] for q in γ.keys() - δ.keys()}
    | {q: δ[q] for q in δ.keys() - γ.keys()}
    | {
      q: {
        a: γ[q].get(a, set()) | δ[q].get(a, set())
        for a in γ[q].keys() | δ[q].keys()
      }
      for q in γ.keys() & δ.keys()
    }
  )


def RegExToFSA(re) -> FiniteStateAutomaton:
  def ToFSA(re) -> FiniteStateAutomaton:
    nonlocal QC
    match re:
      case ε():
        q = QC
        QC += 1
        return FiniteStateAutomaton(set(), {q}, {q}, {}, {q})
      case Sym(a=a):
        q = QC
        QC += 1
        r = QC
        QC += 1
        return FiniteStateAutomaton({a}, {q, r}, {q}, {q: {a: {r}}}, {r})
      case Choice(E1=E1, E2=E2):
        A1, A2 = ToFSA(E1), ToFSA(E2)
        q = QC
        QC += 1
        δ = A1.δ | A2.δ | {q: {'ε': A1.I | A2.I}}
        return FiniteStateAutomaton(
          A1.Σ | A2.Σ, A1.Q | A2.Q | {q}, {q}, δ, A1.F | A2.F
        )
      case Conc(E1=E1, E2=E2):
        A1, A2 = ToFSA(E1), ToFSA(E2)
        δ = merge(A1.δ | A2.δ, {q: {'ε': A2.I} for q in A1.F})
        return FiniteStateAutomaton(A1.Σ | A2.Σ, A1.Q | A2.Q, A1.I, δ, A2.F)
      case Star(E=E):
        A = ToFSA(E)
        δ = merge(A.δ, {q: {'ε': A.I} for q in A.F})
        return FiniteStateAutomaton(A.Σ, A.Q, A.I, δ, A.I | A.F)
      case E:
        raise Exception(str(E) + ' not a regular expression')

  QC = 0
  return ToFSA(re)
```

1. Using the notation from the course notes, write a regular expression for identifiers: an identifier is a sequence of letters `abcdefghijklmnopqrstuvwxyz` and digits `0123456789` starting with a letter. You may use abbreviations:

   Test your answer by expressing it with Python constructors `ε`, `Sym`, `Choice`, `Conc`, `Star` and calling it `I`.

   ```python
   def choices(s: str) -> RegEx:
       result = Sym(s[0])
       for c in s[1:]:
           result = Choice(result, Sym(c))
       return result

   LETTERS = 'abcdefghijklmnopqrstuvwxyz'
   DIGITS = '0123456789'

   I = Conc(choices(LETTERS), Star(Choice(choices(LETTERS), choices(DIGITS))))
   ```

   Regex: $L(L|D)^*$ where $L$ = letter, $D$ = digit

   ```python
   A = RegExToFSA(I)
   ```

   ```python
   assert accepts(A, 'cloud7')
   assert accepts(A, 'if')
   assert accepts(A, 'b12')
   assert not accepts(A, '007')
   assert not accepts(A, '15b')
   assert not accepts(A, 'B12')
   assert not accepts(A, 'e-mail')
   ```

2. Using the notation from the course notes, write a regular expression for dollar amounts: A dollar amount must start with `$` and be followed by a non-empty sequence of digits. It may optionally be followed by `.` and exactly two digits for the cents. The separator `,` may be used for readability of the part before the `.`; if it is used, it must separate all groups of 3 digits.

   Test your answer by expressing it with Python constructors `ε`, `Sym`, `Choice`, `Conc`, `Star` and calling it `C`.

   ```python
   def plus(e: RegEx) -> RegEx: return Conc(e, Star(e))

   def opt(e: RegEx) -> RegEx:
       return Choice(ε(), e)

   def seq(*args: RegEx) -> RegEx:
       result = args[0]
       for e in args[1:]: result = Conc(result, e)
       return result

   def digit(): return choices(DIGITS)

   def dn(n: int) -> RegEx:
       if n == 1: return digit()
       return Conc(digit(), dn(n - 1))

   d_plus = plus(digit())
   d1_or_d2_or_d3 = Choice(digit(), Choice(dn(2), dn(3)))
   comma_d3 = Conc(Sym(','), dn(3))
   amount_with_commas = Conc(d1_or_d2_or_d3, plus(comma_d3))
   amount = Choice(d_plus, amount_with_commas)
   cents = opt(Conc(Sym('.'), dn(2)))

   C = seq(Sym('$'), amount, cents)
   ```

   Regex: $\$\;(d^+ \mid (d|dd|ddd)(,ddd)^+)\;(.dd)?$ where $d$ = digit

   ```python
   B = RegExToFSA(C)
   ```

   ```python
    assert accepts(B, '$27.04')
    assert accepts(B, '$11,222,333')
    assert accepts(B, '$0')
    assert not accepts(B, '27.04')
    assert not accepts(B, '$11222,333')
    assert not accepts(B, '$35.5')
    assert not accepts(B, '$9.409')
   ```

## A4

For your graduation party, you would like to invite all your friends to whom you have either an e-mail address or a telephone number. As you never had time to keep an address book, you like to search for these in all your files using `grep`. In the cells below, write `grep` commands. The `%%bash` cell magic runs the cell in the bash shell; the `%%capture output` cell magic captures the cell's output in the Python variable `output`.

1. E-mail addresses start with one or more upper case letters `A-Z`, lower case letters `a-z`, and symbols `+-._`, followed by the `@` sign and a domain. The domain is a sequence of subdomains separated by `.`, where each subdomain consists of a number of upper and lower case letters, digits, and the symbol `-`. There must be at least two subdomains (i.e. one `.`). The last subdomain, the top-level domain, must consist only of two to six upper or lower case letters. E-mail addresses must start at the beginning of a line or after a separator and end at the end of a line or a separator. Write a shell command using `grep` that, from the directory in which it is started, recursively visits all subdirectories and prints those lines of files that contain an e-mail address. Use `\b` and `\s` as separators and `grep -r` to recursively visit subdirectories.

   ```text
   %%capture output
   %%bash
   grep -rE '(^|\s)[A-Za-z+._-]+@([A-Za-z0-9-]+\.)+[A-Za-z]{2,6}(\s|$)' data/
   ```

   ```text
   assert str(output) == """data/03/friends.txt:abcd@abc.ca
    data/03/friends.txt:abcde@ab-BC.com
    data/02/other-friends.txt:ABCabc+-._@ancbd.ca
    data/02/other-friends.txt:ABCabc+-._@mcmaster.io.ca
    data/02/other-friends.txt:ABCabc+-._@school.image
    data/02/other-friends.txt:ABCabc+-._@school3-computer.image
    data/02/other-friends.txt:ABCabc+-._@school3-IT.image.tor.chrome.ca
    data/01/friends.txt:Marion Floyd (905 263-7740 jpflip@yahoo.com
    data/01/friends.txt:Cora Larson (905) 255-8305 frederic@yahoo.ca
    data/01/friends.txt:Van Craig 905) 608-2616 chunzi@aol.com
    data/01/friends.txt:Emilio Morrison (905) 2877753 cantu@sbcglobal.net
    data/01/friends.txt:Ismael Hanson (905) 755 9372 satch@hotmail.com
    data/01/friends.txt:Wayne Douglas (905)222-3316 tfinniga@verizon.net
    data/01/friends.txt:Tomas Carlson (905746-0359 ardagna@me.com
    data/01/friends.txt:Laurence Newman 9057803232 jaarnial@icloud.com
    data/01/friends.txt:Lori Sherman 905-543-7753 chaki@att.net
    data/01/friends.txt:Gladys Brock (539) 728-2363 lukka@icloud.com
    """
   ```

2. You are looking for telephone numbers in the `905` area for your party. Valid numbers are of the form `(905) 123 4567`, `(905) 1234567`, `905-123-4567`. However, `9051234567`, as well as `905) 123 4567`, `905-123 4567`, are not. Telephone addresses must start at a the beginning of a line or after a separator and must end at the end of a line or a separator. Write a shell command using grep that, from the directory in which it is started, recursively visits all subdirectories and prints those lines of files that contain a telephone number.
   ```bash
   %%capture output
   %%bash
   grep -rE '(^|\s)(\(905\) ([0-9]{3} [0-9]{4}|[0-9]{7})|905-[0-9]{3}-[0-9]{4})(\s|$)' data/
   ```

## A5

Extracting columns from CSV (sed). File `data/q.csv`:

```csv
a,b,c
d,e,f
gh,i,jkl
m n o,pq r,stuv1
```

1. Extract first column:

   ```bash
   %%capture output
   %%bash
   sed 's/,.*//' data/q.csv
   ```

   Pattern: `s/,.*//` - delete from first comma to end of line

   ```text
   assert str(output) == """a
   d
   gh
   m n o
   """
   ```

2. Extract second column:

   ```bash
   %%capture output
   %%bash
   sed 's/^[^,]*,//; s/,.*//' data/q.csv
   ```

   Pattern: `s/^[^,]*,//` deletes first column + comma, then `s/,.*//` deletes third column

   ```text
   assert str(output) == """b
   e
   i
   pq r
   """
   ```

3. Extract third column:

   ```bash
   %%capture output
   %%bash
   sed 's/.*,//' data/q.csv
   ```

   Pattern: `s/.*,//` - delete everything up to (and including) the LAST comma

   ```text
   assert str(output) == """c
   f
   jkl
   stuv1
   """
   ```

## A6

Lowercasing HTML `src` attribute values. File `data/q.html`:

```html
<img src="PiCtuRe.PnG "/>
<img src="PiCtuRe.PnG"></img>
<img src="PiCtuRe.PnG">alt</img>
<img src="PiCtuRe.PnG"> alt   </img>
<img src ="PiCtuRe.PnG" />
<img src = "PiCtuRe.PnG"/>
<img onclick="alert('Clicked!')" src = "PiCtuRe.PnG"/>
```

```bash
%%capture output
%%bash
sed -r 's/(src *= *")([^"]*)/\1\L\2/g' data/q.html
```

Pattern breakdown:

- `(src *= *")` captures `src` + optional spaces + `=` + optional spaces + opening `"`
- `([^"]*)` captures everything until closing quote (the value)
- `\1\L\2` outputs group 1, then lowercased group 2 (`\L` is GNU sed lowercase)

```text
assert str(output) == """<img src="picture.png "/>
<img src="picture.png"></img>
<img src="picture.png">alt</img>
<img src="picture.png"> alt   </img>
<img src ="picture.png" />
<img src = "picture.png"/>
<img onclick="alert('Clicked!')" src = "picture.png"/>
"""
```
