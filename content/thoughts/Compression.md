---
date: '2024-02-07'
description: a [[/thoughts/reductionism|reductionist]] way to transfer data.
id: Compression
modified: 2025-11-05 07:06:08 GMT-05:00
tags:
  - math/linalg
  - math
  - technical
title: Compression
---

a form of [[thoughts/reductionism|reductionism]]--finding compact representations by identifying redundancy and pattern.

the high-level goal is to find an algebraic structure to permit a more economical encoding.

[[thoughts/Information Theory|Shannon's]] source coding theorem establishes that the [[thoughts/Entropy|entropy]] $H(X)$ of a source $X$ is the fundamental limit for lossless compression [^compression]

[^compression]: the entropy $H(X)$ represents the average number of bits needed per symbol to encode messages from source $X$. no lossless compression scheme can achieve an average code length below this bound.

$$
H(X) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)
$$

this bound is _asymptotic_:

- for sequences of length $n$, we can compress to $nH(X) + o(n)$ bits.
  - but what about individual strings?
    - [[thoughts/Kolmogorov-Arnold representation theorem|kolmogorov complexity]] $K(x)$ of a string $x$ is the length of the shortest program that outputs $x$.
    - this is incomputable in general

> compression algorithms are heuristic approximations to kolmogorov complexity.

## algebraic perspective

view data as elements of the **free monoid** $\Sigma^*$ over alphabet $\Sigma$, equipped with concatenation. a string $s \in \Sigma^*$ can be written as $s = s_1 s_2 \cdots s_n$ where $s_i \in \Sigma$.

compression seeks a compact **generating set** for representing strings.

- dictionary-based methods find frequently occurring substrings (monoid elements) and use them as generators.
- grammar-based methods discover a **free algebra** structure, expressing strings through production rules.

> [!math] syntactic monoid
>
> for a language $L \subseteq \Sigma^*$, the **syntactic congruence** $\equiv_L$ is defined by:
>
> $$
> x \equiv_L y \iff \forall u,v \in \Sigma^* : uxv \in L \leftrightarrow uyv \in L
> $$
>
> the quotient $\Sigma^* / \equiv_L$ is the **syntactic monoid** of $L$.

dictionary compression implicitly discovers elements of syntactic monoids--substrings that behave equivalently in context.

## dictionary-based compression

_a la zstd and lz77_

**zstandard** (zstd) exemplifies modern dictionary compression, combining lz77-style dictionary coding with finite state entropy (fse) for the entropy coding stage.

### lz77

_and syntactic structure_

represents strings as sequences of $\text{(offset, length, literal)}$ triples, encoding matches to earlier occurrences. formally, if string $s = s_1 \cdots s_n$, a dictionary $D$ of previously seen substrings allows factorization:

$$s = w_1 w_2 \cdots w_k \quad \text{where each } w_i \in D \cup \Sigma$$

this is finding a **basis** in the monoid $\Sigma^*$.

the lempel-ziv {{sidenotes[complexity]: the lz77 complexity counts the number of distinct factors in the greedy parsing. this connects to [[thoughts/Kolmogorov-Arnold representation theorem|kolmogorov complexity]], where LZ complexity is within logarithmic factors of $K(s)$.}} measures how many distinct factors appear, related to the combinatorial structure of $s$.

### finite state entropy

_or asymmetric numeral systems, by jarek duda_

after dictionary encoding, zstd uses **asymmetric numeral systems** (ans) [@duda2013ans] for entropy coding.

provides compression rates approaching arithmetic coding with speed rivaling huffman coding.

ans views encoding as a state machine:

- for alphabet $\Sigma$ with symbol frequencies $(f_s)_{s \in \Sigma}$, the encoder maintains states [^state] $x \in [L, 2L)$ for some precision $L$. encoding symbol $s$ updates state:
  $$
  x' = C(s, x \bmod f_s) + \lfloor x / f_s \rfloor \cdot L
  $$
- where $C(s, \cdot)$ is a bijection $\{0, \ldots, f_s-1\} \to$ the positions allocated to symbol $s$ in the state space.

[^state]: : the state transitions define a **finite automaton**. algebraically, this is an action of the free monoid $\Sigma^*$ on the state space $[L, 2L) \cap \mathbb{Z}$. the bijection $C$ ensures bijectivity, making decoding possible.

> [!important] ans redundancy [@kosolobov2022ansefficiency]
>
> for tabled ans (tans) with duda's "precise initialization",
> given string $a_1 \ldots a_n$ where symbol $a$ appears $f_a$ times and $n = 2^r$:
> $$\text{encoded length} = \sum_{a \in \Sigma} f_a \log \frac{n}{f_a} + O(\sigma + r)$$
>
> this refutes duda's conjecture of $O(\sigma/n^2)$ redundancy--the actual redundancy is $O(\sigma/n)$ per symbol.

the $O(\sigma + r)$ term arises from:

- $\sigma$ terms for alphabet encoding
- $r$ bits from the power-of-2 block structure

zstd's fse uses modular arithmetic: state updates involve division and modular operations, connecting to number theory and cyclic groups $\mathbb{Z}/L\mathbb{Z}$.

## grammar-based compression

grammar-based methods infer a [[thoughts/Context-Free Grammar|context-free grammar]] $G = (V, \Sigma, R, S)$ where:

- $V$ is non-terminal symbols
- $\Sigma$ is terminal alphabet
- $R \subseteq V \times (V \cup \Sigma)^*$ are production rules
- $S \in V$ is start symbol

the string is compressed by storing $G$ plus derivation information.

### sequitur algorithm

**sequitur** [@nevillmanning1997sequitur] constructs grammars {{sidenotes[online]: for string $abcab$, sequitur produces: $S \to AcA, A \to ab$. the digram $ab$ is factored into non-terminal $A$.}} in linear time, maintaining two invariants:

1. **digram uniqueness**: no pair of adjacent symbols occurs more than once
2. **rule utility**: every rule is used at least twice

when a repeated digram $ab$ appears, create rule $A \to ab$ and substitute both occurrences. this enforces digram uniqueness. rule utility is maintained by inlining single-use rules.

algebraically, sequitur discovers a **rewriting system**. each production rule $A \to \alpha$ is a rewrite. the grammar defines a quotient structure on $\Sigma^*$ where strings deriving the same parse tree are identified.

### re-pair algorithm

**re-pair** [@larsson1999repair] is an offline algorithm that recursively replaces the most frequent digram until no pair occurs twice. unlike sequitur, it doesn't maintain invariants during construction, potentially achieving better compression at higher computational cost.

both algorithms connect to **free algebras**: they find algebraic presentations $\langle \Sigma \mid R \rangle$ where $R$ are the grammar rules, expressing strings in terms of generators (non-terminals).

### recompression and approximation

@jez2013recompression introduced the **recompression technique**: given string of size $N$ and optimal grammar size $g$, construct a grammar of size $O(g \log(N/g))$ in linear time.

> [!abstract] grammar compression approximation
>
> there exists a linear-time algorithm producing a context-free grammar of size $O(g \log(N/g))$ for input string of size $N$, where $g$ is the size of the smallest grammar for the string.
>
> [@jez2013recompression]

recompression works by iteratively:

1. identifying pairs of symbols (blocks)
2. replacing occurrences with fresh symbols
3. reducing alphabet size through compression

this logarithmic approximation is the best known for grammar-based compression.

## formal language theory connections

compression algorithms operate on formal languages. the [[thoughts/DFA|deterministic finite automaton]] (dfa) recognizing a language $L$ has an associated syntactic monoid. for regular languages (recognized by dfas), the syntactic monoid is finite.

**algebraic automata theory** studies the connection between automata and monoids:

> [!math] definition: automaton as monoid action
> a dfa $(Q, \Sigma, \delta, q_0, F)$ induces an action of the free monoid $\Sigma^*$ on state set $Q$:
> $$\delta^* : Q \times \Sigma^* \to Q$$
> where $\delta^*(q, \epsilon) = q$ and $\delta^*(q, wa) = \delta(\delta^*(q, w), a)$.

this action factors through the syntactic monoid: $\Sigma^* \to M_L \to Q$ where $M_L$ is the syntactic monoid.

for compression, we care about languages with structure:

- **regular languages**: finite syntactic monoids, simple compression
- **context-free languages**: infinite monoids, grammar-based compression effective
- **context-sensitive and beyond**: more complex algebraic structures

## entropy coding as automaton

huffman, arithmetic, and ans are entropy coders. viewing them through automata:

**huffman coding** assigns fixed-length binary codes based on symbol frequencies. algebraically, it constructs a binary tree (free monoid on $\{0,1\}$) where symbols are leaves.

**arithmetic coding** represents the entire message as a subinterval of $[0,1)$, using rational arithmetic. the precision required grows with message length.

**ans** uses integer states and modular arithmetic, bridging huffman's simplicity and arithmetic's efficiency. the state machine perspective makes ans particularly elegant:

$$\text{encode}(x, s) = \text{bijection}(s, x \bmod f_s) + \lfloor x / f_s \rfloor \cdot L$$

decoding reverses this: given state $x$, determine symbol $s$ by finding which frequency interval $x$ falls into, then invert the bijection.

the bijection $C(s, \cdot)$ can be seen as a **permutation group** acting on symbol positions, though in practice it's typically the identity or a simple spreading function.

## connections to other fields

permeates through all aspects of computer science.

### kv compression

modern [[thoughts/Transformers|transformers]] use [[thoughts/KV compression|kv compression]] to reduce memory in attention mechanisms. techniques like h2o, snapkv identify important key-value pairs based on attention scores--a form of lossy compression exploiting statistical structure.

### measure theory and probability

entropy coding relies on [[thoughts/Lebesgue measure|measure theory]]. symbol frequencies define a probability measure $\mu$ on $\Sigma$, and entropy is the expected information:

$$
H(\mu) = -\int_{\Sigma} \log_2(\mu(\{s\})) \, d\mu(s) = -\sum_{s \in \Sigma} \mu(\{s\}) \log_2 \mu(\{s\})
$$

arithmetic coding directly implements this: interval sizes are proportional to probabilities, and the coding theorem guarantees near-optimal compression.

### [[thoughts/category theory|category theory]]

compression can be viewed categorically:

- objects: strings in $\Sigma^*$
- morphisms: compression algorithms as functors $F : \Sigma^* \to \mathcal{C}$ to some compressed representation category
- natural transformations: conversions between compression schemes

a lossless compressor is an injection $\Sigma^* \to \mathcal{C}$ with left inverse (decompressor). the size function defines a "norm" on objects, and compression seeks minimal representatives in equivalence classes.

## open questions and research directions

1. **optimal grammar inference**: finding the smallest grammar is np-hard. what algebraic properties of strings permit polynomial-time optimal compression?

2. **algebraic invariants**: are there algebraic invariants (homology groups, cohomology) that characterize compressibility?

3. **quantum compression**: how do quantum states compress? connections to [[thoughts/Entropy|entropy]] and entanglement measures?

4. **neural compression**: learned compression via autoencoders discovers latent algebraic structure. what is the relationship between neural network architectures and algebraic compression?

5. **streaming and online algorithms**: sequitur works online. can we develop online algorithms for other algebraic compression schemes with provable approximation guarantees?

6. **category-theoretic formalization**: develop a complete categorical framework for compression, characterizing lossless vs lossy, universal vs adaptive schemes.

## see also

- [[thoughts/Entropy|entropy]] - the fundamental limit
- [[thoughts/Information Theory|information theory]] - theoretical foundations
- [[thoughts/Context-Free Grammar|context-free grammars]] - grammar-based compression
- [[thoughts/DFA|deterministic finite automata]] - automata and monoids
- [[thoughts/KV compression|kv compression]] - applications in ml
- [[thoughts/reductionism|reductionism]] - philosophical connections
- [[thoughts/Kolmogorov-Arnold representation theorem|kolmogorov complexity]] - theoretical limit
