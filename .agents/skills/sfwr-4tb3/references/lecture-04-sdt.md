---
name: lecture-04-sdt
description: syntax-directed translation — attribute grammars, S- and L-attributed, synthesized/inherited attributes, AST construction, type checking
---

# lecture 04 — syntax-directed translation

(notebook title: "5. Syntax-Directed Translation". same content as this dir `04`.)

## attribute grammars

augment a CFG with **attributes** on symbols and **semantic rules** on productions. attributes carry values (types, evaluated results, ASTs, error messages). rules compute attribute values from other attributes.

notation: $\ll \text{attribute rule} \gg$ inside a production, using subscripts to disambiguate when a symbol appears multiple times.

$$
\text{expression}(e) \to \text{term}(e_1)\ \texttt{'+'}\ \text{term}(e_2) \quad \ll e := e_1 + e_2 \gg
$$

## synthesized vs inherited attributes

- **synthesized**: value flows _up_ the parse tree (child → parent). computed from children.
- **inherited**: value flows _down_ or _across_ (parent/siblings → child). The left-sibling restriction belongs to L-attributed grammars.

an attribute grammar is:

- **S-attributed** if all attributes are synthesized. easy to evaluate bottom-up (LR parsing).
- **L-attributed** if inherited attributes depend only on parent/left-siblings. evaluatable in a single left-to-right traversal. matches recursive descent naturally.

## evaluating arithmetic expressions (pattern)

$$
\begin{aligned}
\text{expression}(e) &\to \text{term}(e)\,\{\,\texttt{'+'}\,\text{term}(f)\ \ll e := e + f \gg \mid\,\texttt{'-'}\,\text{term}(f)\ \ll e := e - f \gg\,\} \\
\text{term}(e)       &\to \text{factor}(e)\,\{\,\texttt{'*'}\,\text{factor}(f)\ \ll e := e \cdot f \gg \mid\,\texttt{'/'}\,\text{factor}(f)\ \ll e := e / f \gg\,\} \\
\text{factor}(e)     &\to \text{number}(e) \mid \texttt{'('}\,\text{expression}(e)\,\texttt{')'}
\end{aligned}
$$

the attribute $e$ is synthesized — each function computes and returns $e$. in python:

```python
def expression():
  e = term()
  while sym in ['+', '-']:
    op = sym
    nxt()
    f = term()
    e = e + f if op == '+' else e - f
  return e
```

## type checking

grammar attributes carry types alongside values. compatibility rules go into the attribute rules:

$$
\begin{aligned}
\text{expression}(t) &\to \text{term}(t_1)\,\texttt{'+'}\,\text{term}(t_2) \\
&\quad \ll \text{if } t_1 = \texttt{int} \land t_2 = \texttt{int} \text{ then } t := \texttt{int} \\
&\quad\quad \text{else mark}(\texttt{'not int operands of +'}) \gg
\end{aligned}
$$

in code:

```python
def expression():
  x = term()
  while sym == '+':
    nxt()
    y = term()
    if x.tp != 'int' or y.tp != 'int':
      mark('not int operands of +')
    x = BinaryOp('+', x, y, tp='int')
  return x
```

the P0 parser carries types on every expression value and delegates compatibility to `compatible(xt, yt)`.

## infix-to-postfix (classic SDT exercise)

$$
\begin{aligned}
\text{expression} &\to \text{term}\,\{\,\texttt{'+'}\,\text{term}\ \ll \text{emit}(\texttt{'+'}) \gg \mid\,\texttt{'-'}\,\text{term}\ \ll \text{emit}(\texttt{'-'}) \gg\,\} \\
\text{term}       &\to \text{factor}\,\{\,\texttt{'*'}\,\text{factor}\ \ll \text{emit}(\texttt{'*'}) \gg \mid\,\texttt{'/'}\,\text{factor}\ \ll \text{emit}(\texttt{'/'}) \gg\,\} \\
\text{factor}     &\to \text{number}\ \ll \text{emit}(\text{number}) \gg \mid \texttt{'('}\,\text{expression}\,\texttt{')'}
\end{aligned}
$$

the emit happens _after_ the recursive call so operands are emitted before the operator. this is the fundamental SDT pattern: actions are placed where the semantic output belongs.

## AST construction

same structure as evaluation but the synthesized attribute is the tree, not the value:

$$
\text{expression}(e) \to \text{term}(e)\,\{\,\texttt{'+'}\,\text{term}(f)\ \ll e := \text{Add}(e, f) \gg\,\}
$$

the P0 compiler does this: `CGast`'s `genBinaryOp(op, x, y)` builds a `BinaryOp` node with $x$ and $y$ as children.

## L-attributed (inherited attributes)

example: count pairs of `a` in `aaba` — need to pass context down. a counter $c$ starts at $0$ and accumulates:

$$
\begin{aligned}
S(n)   &\to A(0, n) \\
A(c, n) &\to \texttt{'a'}\,\texttt{'a'}\,A(c+1, n) \mid \texttt{'b'}\,A(c, n) \mid \epsilon\ \ll n := c \gg
\end{aligned}
$$

the first argument ($c$) is inherited (parent → child); the second ($n$) is synthesized. in recursive descent you pass the inherited attribute as a function argument and return the synthesized one.

## combining phases

the P0 parser is a single-pass L-attributed SDT: it parses, type-checks, and emits code in one traversal. each `expression()` call returns a value that has a type, an address/register, and possibly emitted code. this is why error messages can point at exact positions — everything flows through the scanner state.
