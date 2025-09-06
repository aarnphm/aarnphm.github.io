---
id: Tokenization
tags:
  - seed
  - ml
description: how machine read different modalities
date: "2025-09-04"
modified: 2025-09-04 19:39:27 GMT-04:00
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

### BPE

alias: _byte-pair encoding_

## how do you train a tokenizer?
