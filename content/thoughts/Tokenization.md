---
id: Tokenization
tags:
  - seed
  - ml
description: how machine read different modalities
date: "2025-09-04"
modified: 2025-09-11 16:08:53 GMT-04:00
title: Tokenization
---

ASCII: 33 control characters and 95 printable characters

- 7 bit representations: $2^{7} = 128$ values
- 1 bit are wasted

> [!question]
>
> What is the binary representations of `Y` and `y`?

```python
bin(ord('Y'))
bin(ord('y'))
```

Extended ASCII for languages other than English

## utf-8

intuition: add more bits, where it is a variable-length encoding that uses sequences of 1-4 bytes

```python
'😋'.encode('utf-8')
```

> basic multilingual plane (most often used characters)

Unicode general assigns 21 bits to each codepoint.

## algorithm

Most tokenizers in modern day [[thoughts/Transformers|LLMs]] uses BPE. Others include WordPiece, SentencePiece. Also known as _subword tokenization_

- Characters?
  - discrete: follow UTF-8, which is the standard accross the web, but a lot less
- Words?
  - still discrete, but still expensive

> [!important]
>
> it balances vocabulary size and representation quality:
>
> - **Frequent words**: remain whole
> - **Rare words**: decompose into meaningful subwords
>
> Example: "annoyingly" → "annoying" + "ly"

![[thoughts/byte-pair encoding|BPE]]

> [!note] intuition
>
> Some techniques will add/remove certain rulesets to fine-tune BPE, such that it helps with overall quality/capabilities improvement
>
> i.e: Kimi K2, Qwen, R1, all adds additional policies to its tokenizer to help with higher Chinese tokens quality.

### WordPiece

used by: BERT, DistilBERT, Electra

Similar to BPE but chooses merges that **maximize training data likelihood** rather than frequency. Evaluates what is _lost_ by merging to ensure it's _worth it_.

### Unigram

> [!abstract] algorithm
>
> 1. Initialize large base vocabulary
> 2. Progressively remove symbols that least affect overall loss
> 3. Keep base characters for complete tokenization coverage

Loss function:

$$
\mathcal{L} = -\sum_{i=1}^{N} \log \left ( \sum_{x \in S(x_{i})} p(x) \right )
$$

where $S(x_i)$ represents all possible tokenizations of word $x_i$.

Multiple tokenization paths exist; algorithm selects most probable or samples according to probabilities.

### SentencePiece

used by: ALBERT, XLNet, Marian, T5

> [!important] notable contribution
>
> Treats input as raw stream including spaces, solving tokenization for languages without space-separated words.

Uses BPE or unigram on character stream. The "▁" symbol represents spaces, enabling simple decoding by concatenation and space replacement.

**Models using SentencePiece:** Combined with unigram algorithm in all Transformers implementations.

