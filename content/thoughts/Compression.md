---
date: '2024-02-07'
description: lossless and lossy encodings that exploit redundancy, from entropy coders to dictionary and grammar methods.
id: Compression
modified: 2026-06-05 15:08:05 GMT-04:00
tags:
  - math/linalg
  - math
  - technical
title: Compression
---

a form of [[thoughts/reductionism|reductionism]]: describe the data with fewer bits by finding the redundancy in it, then store the description instead of the data.

a lossless compressor is an injective map $C$ from strings to bit strings, with a decompressor $D$ such that $D(C(x))=x$. injectivity already limits what any compressor can do. there are $2^n$ binary strings of length $n$, and only $2^n-1$ binary strings shorter than $n$. so no lossless compressor shortens every input. it can only shorten the inputs it expects, and it must lengthen some others. every compressor is therefore a bet on which data is likely.

## the entropy limit

[[thoughts/Information Theory|Shannon's]] source coding theorem says how good that bet can be. for a source that emits independent symbols with distribution $p$, the [[thoughts/Entropy|entropy]]

$$
H(X) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)
$$

is the minimum average number of bits per symbol. no uniquely decodable code can average fewer than $H(X)$ bits per symbol, and long enough blocks of $n$ symbols can be coded in $nH(X)+o(n)$ bits. a symbol with probability $p(x)$ ideally costs $-\log_2 p(x)$ bits: likely symbols get short codes, unlikely symbols get long ones.

the theorem is about a source, which is a probability model. a single file has no entropy by itself. for an individual string $x$, the matching idea is the kolmogorov complexity $K(x)$: the length of the shortest program that prints $x$. $K$ is incomputable. any real compressor still gives an upper bound: $K(x)\leq\lvert C(x)\rvert+c_D$, where $c_D$ is the length of the decompressor program. the bound can be very loose. the digits of $\pi$ have a short program, and a general-purpose compressor will not find it.

## entropy coders

an entropy coder turns a probability model into bits. three designs matter in practice.

**huffman coding** gives each symbol a prefix-free code word from a binary tree, built by repeatedly merging the two least likely symbols. it is the best code that assigns a whole number of bits to each symbol, and its average length $\bar\ell$ satisfies $H(X)\leq\bar\ell<H(X)+1$. the $+1$ hurts when one symbol dominates. with $p=(0.9,0.1)$, $H\approx0.469$ bits, but huffman still spends $1$ bit per symbol.

**arithmetic coding** removes the whole-bit restriction. it narrows an interval inside $[0,1)$ by the probability of each symbol in turn, and outputs enough bits to name a point in the final interval. a message $x$ costs at most $-\log_2 p(x)+2$ bits in total, so the overhead is shared across the message. the cost is speed: every symbol needs a multiplication and careful handling of finite precision.

**asymmetric numeral systems** (ans) [@duda2013ans] keep the state in one integer. the frequencies are quantized to integers $f_s$ that sum to $M$, and $B_s=\sum_{t<s}f_t$ is the start of the range $[B_s,B_s+f_s)$ for symbol $s$. the range variant (rans) encodes symbol $s$ into state $x$ as

$$
x' = M\left\lfloor \frac{x}{f_s} \right\rfloor + B_s + (x \bmod f_s).
$$

the state grows by a factor of about $M/f_s$, so $\log_2 x$ grows by about $-\log_2(f_s/M)$ bits, which is the ideal cost of $s$. to decode, the slot $x' \bmod M$ identifies $s$, and $x = f_s\lfloor x'/M\rfloor + (x' \bmod M) - B_s$ restores the old state.

a small example with $f_a=3$, $f_b=1$, $M=4$, $B_a=0$, $B_b=3$. start at $x=10$:

- encode $b$: $x'=4\cdot10+3+0=43$. the state grew by a factor of $4$, which is $2$ bits.
- encode $a$: $x'=4\cdot14+0+1=57$. the state grew by about $4/3$, which is $0.415$ bits.
- decode: $57 \bmod 4=1$ lies in $[0,3)$, so the symbol is $a$, and $x=3\cdot14+1-0=43$. then $43 \bmod 4=3$ gives $b$, and $x=1\cdot10+3-3=10$.

decoding returns symbols in reverse order, so encoders process the input backward. a real coder also renormalizes: when the state leaves an interval $[L,bL)$, it writes or reads its low bits so the integer stays in a fixed machine width.

tabled ans (tans) precomputes these transitions for every state in $[L,2L)$, so encoding and decoding become table lookups. this gives arithmetic-coding compression rates at speeds near huffman coding.

> [!important] how close tans gets [@kosolobov2022ansefficiency]
>
> take a string of length $n=2^r$ in which symbol $a$ occurs $f_a$ times, from an alphabet of $\sigma$ symbols. tans with duda's "precise initialization" encodes it in
>
> $$
> \sum_{a} f_a \log_2 \frac{n}{f_a} + O(\sigma + r)
> $$
>
> bits, not counting the frequency table. the first term is the empirical entropy. kosolobov gives strings that need the $\Omega(\sigma+r)$ excess. this refutes duda's conjecture of $O(\sigma/n^2)$ redundancy per symbol: outside the $r$-bit term, the excess per symbol is of order $\sigma/n$.

## dictionary methods

_a la lz77 and zstd_

entropy coders treat symbols one at a time. text and code repeat whole substrings, which a symbol model cannot see. lz77 [(ziv and lempel, 1977)](https://doi.org/10.1109/TIT.1977.1055714) replaces a repeat with a pointer to an earlier copy: a pair $(\text{offset},\text{length})$ that says "go back this far and copy this many symbols". the copy may overlap the text it produces. `abababab` becomes the literals `a`, `b`, then $(2,6)$: go back two symbols and copy six, one symbol at a time.

let $z$ be the number of phrases in the greedy lz77 parse. for stationary ergodic sources, the lz77 output per symbol approaches the entropy rate, without knowing the source in advance. for individual strings, $z$ never exceeds the size of the smallest grammar that generates the string ([rytter, 2003](<https://doi.org/10.1016/S0304-3975(02)00777-6>)), so lz77 finds at least as much repetition as any grammar does.

**zstandard** (zstd) parses with an lz77-style matcher, then entropy-codes the results. [RFC 8878](https://www.rfc-editor.org/rfc/rfc8878) specifies huffman coding for literal bytes and fse, a tans implementation, for the literal lengths, match lengths and offsets.

## grammar-based compression

a grammar compressor stores a context-free grammar that generates exactly one string: the input. such a grammar is also called a straight-line program. each rule has one right-hand side, and expanding the start symbol reproduces the string. the size of the grammar is the total length of its right-hand sides.

**sequitur** [@nevillmanning1997sequitur] builds the grammar {{sidenotes[online]: for string $abcab$, sequitur produces: $S \to AcA, A \to ab$. the digram $ab$ is factored into non-terminal $A$.}} in one left-to-right pass, in linear time. it maintains two invariants:

1. **digram uniqueness**: no pair of adjacent symbols occurs more than once in the grammar.
2. **rule utility**: every rule is used at least twice.

when a digram $ab$ appears a second time, sequitur creates $A \to ab$ and replaces both occurrences. when a rule drops to one use, sequitur inlines it.

**re-pair** [@larsson1999repair] works offline. it replaces the most frequent pair in the whole string with a new symbol, recounts, and repeats until no pair occurs twice. it also runs in linear time, but needs the whole input and more memory. because each step takes the globally most frequent pair, it usually produces smaller grammars than sequitur.

finding the smallest grammar is np-hard, and the best known polynomial-time algorithms are within a factor $O(\log(N/g))$ of optimal, where $N$ is the length of the string and $g$ the size of the smallest grammar ([charikar et al., 2005](https://doi.org/10.1109/TIT.2005.850116); rytter, 2003). @jez2013recompression reaches the same factor in linear time with **recompression**. it alternates two phases on the current string:

1. **block compression**: replace each maximal run $a^k$ with a fresh symbol.
2. **pair compression**: split the alphabet into a left part and a right part, then replace every pair $ab$ with $a$ on the left and $b$ on the right by a fresh symbol. the split guarantees that these pairs do not overlap.

each fresh symbol is a grammar rule, and the string shrinks by a constant factor in each round.

## connections

### kv compression

[[thoughts/Transformers|transformers]] keep a key and value vector for every past token. [[thoughts/KV compression|kv compression]] methods such as h2o and snapkv drop the entries that receive little attention. this is lossy: the bet is that a dropped token will not matter for later queries.

### [[thoughts/Kolmogorov-Arnold representation theorem|kolmogorov-arnold]]

this theorem shares a name with kolmogorov complexity and nothing else. it writes a continuous function of several variables with sums and compositions of one-variable functions. it is a representation result. it says nothing about the length of a description.

## see also

- [[thoughts/Entropy|entropy]]: the lower bound for lossless codes
- [[thoughts/Information Theory|information theory]]
- [[thoughts/Context-Free Grammar|context-free grammars]]: the rules that grammar compressors store
- [[thoughts/KV compression|kv compression]]
- [[thoughts/reductionism|reductionism]]
