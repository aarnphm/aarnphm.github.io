---
name: lecture-03-cfl
description: analysis of context-free languages — pushdown automata, top-down/bottom-up parsing, LL(k)/LR(k), recursive descent, FIRST/FOLLOW sets
---

# lecture 03 — analysis of context-free languages

(note: the notebook is numbered "4" internally; the directory is `03`. same content.)

## pushdown automata

a pushdown automaton (PDA) is a finite automaton + a stack. it recognizes exactly the context-free languages.

$\text{PDA} = (\Sigma, Q, \Gamma, I, \delta, F, Z_0)$:

- $\Sigma$ — input alphabet
- $Q$ — states
- $\Gamma$ — stack alphabet
- $I$ — initial states
- $\delta$ — transitions $(q, a \text{ or } \epsilon, Z) \to (q', \gamma)$ where $\gamma \in \Gamma^*$
- $F$ — accepting states
- $Z_0$ — initial stack symbol

intuition: the stack remembers nested structure (parens, block nesting).

## top-down vs bottom-up parsing

**top-down**: start from the start symbol, predict productions, try to match the input. lookahead decides which production to use. LL(k).

**bottom-up**: start from the input, shift tokens onto a stack, reduce stack prefixes to nonterminals using productions. LR(k), LALR(k), SLR(k).

| aspect         | top-down                         | bottom-up                       |
| :------------- | :------------------------------- | :------------------------------ |
| direction      | leftmost derivation              | rightmost derivation in reverse |
| decision point | when to expand a nonterminal     | when to reduce                  |
| implementation | recursive descent (easy by hand) | parse tables (generated)        |
| power          | $LL(k) \subset LR(k)$            | strictly more                   |
| left-recursion | cannot handle                    | fine                            |

## $LL(k)$ grammars

a grammar is $LL(k)$ if every step of leftmost derivation can be determined by looking at the current nonterminal and the next $k$ input symbols.

for $A \to \alpha \mid \beta$ to be $LL(1)$:

1. $\text{FIRST}(\alpha) \cap \text{FIRST}(\beta) = \emptyset$
2. if $\alpha \Rightarrow^* \epsilon$, then $\text{FIRST}(\beta) \cap \text{FOLLOW}(A) = \emptyset$

Apply the nullable/FOLLOW check symmetrically if $\beta$ is nullable; both alternatives being nullable creates a conflict.

these conditions let you pick the right production with one token of lookahead.

## FIRST and FOLLOW

$\text{FIRST}(\alpha)$ — the set of terminals that can begin a string derived from $\alpha$; includes $\epsilon$ if $\alpha \Rightarrow^* \epsilon$.

$\text{FOLLOW}(A)$ — the set of terminals that can immediately follow $A$ in any sentential form. $\text{FOLLOW}(S)$ includes `$` (end-of-input).

computation (iterate to fixpoint):

- $\text{FIRST}(a) = \{a\}$ for terminals
- $\text{FIRST}(A) = \bigcup \text{FIRST}(\alpha)$ for each $A \to \alpha$
- $\text{FIRST}(\alpha\,\beta)$: if $\epsilon \notin \text{FIRST}(\alpha)$ then $\text{FIRST}(\alpha)$, else $\text{FIRST}(\alpha) \cup \text{FIRST}(\beta)$ (minus $\epsilon$ from first, plus $\epsilon$ only if $\beta$ also nullable)

## $LR(k)$ grammars

$LR(k)$ parsers use $k$-symbol lookahead to decide whether to shift or reduce, and which production to reduce by. strictly more powerful than $LL(k)$ for the same $k$.

- $LR(1)$ tables are huge; $LALR(1)$ merges states to cut size
- tools: yacc/bison produce $LALR(1)$ parsers
- all deterministic CFLs can be parsed by $LR(1)$

## left recursion and elimination

direct left recursion $A \to A\,\alpha \mid \beta$ is fatal for top-down. rewrite as:

$$
\begin{aligned}
A  &\to \beta\,A' \\
A' &\to \alpha\,A' \mid \epsilon
\end{aligned}
$$

equivalently in EBNF: $A \to \beta\,\{ \alpha \}$. the shift from recursion to iteration is the essence of converting a recursive grammar for recursive-descent.

## recursive descent parsing

one parsing function per nonterminal. body follows the production, using `if sym in` $\text{FIRST}(\alpha)$ to pick between alternatives, `while sym in` $\text{FIRST}(\alpha)$ for EBNF $\{ \alpha \}$.

scanner invariant: `sym` holds the _next_ symbol; `nxt()` advances. each parsing function consumes its nonterminal and leaves `sym` at the first symbol after.

## dealing with ambiguity

options:

1. rewrite the grammar, such as separating matched and unmatched statements for dangling `else`; an eager nearest-`if` parser rule resolves the choice without making the original CFG unambiguous
2. use precedence declarations (in parser generators)
3. move to PEG (prioritized choice disambiguates)

## error recovery

- **panic mode**: skip tokens until a synchronizing symbol (`;`, `end`)
- **phrase-level**: insert/delete one token to continue
- **global**: find the minimum-edit correction

The lecture 05 `SC.mark('error message')` raises an exception. Error recovery strategies above are general techniques; check the target notebook before applying one to P0.
