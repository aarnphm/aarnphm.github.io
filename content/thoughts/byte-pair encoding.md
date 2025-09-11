---
published: "2006-07-04"
id: byte-pair encoding
author:
  - "[[Contributors to Wikimedia projects]]"
description: Algorithm for encoding text strings by iteratively replacing most frequent character pairs with new symbols. Used in data compression and LLM tokenization.
transclude:
  title: false
date: "2025-09-11"
created: "2025-09-11"
modified: 2025-09-11 16:08:23 GMT-04:00
tags:
  - seed
  - clippings
title: byte-pair encoding
source: https://en.wikipedia.org/wiki/Byte-pair_encoding
---

alias: _BPE, diagram coding_

1994, by Philip Gage for encoding strings of text into smaller strings by creating and using a translation table. A modified version is used in [[thoughts/LLMs|LLMs]] [[thoughts/Tokenization|tokenizers]].

## algorithm

> replacing the most common contiguous sequences of characters in a target text with unused 'placeholder' bytes. The iteration ends when no sequences can be found, leaving the target text effectively compressed.

### example

Suppose the data to be encoded is:

```
aaabdaaabac
```

The byte pair "aa" occurs most often, so it will be replaced by "Z":

```
ZabdZabac
Z=aa
```

Then replace "ab" with "Y":

```
ZYdZYac
Y=ab
Z=aa
```

Recursively replace "ZY" with "X":

```
XdXac
X=ZY
Y=ab
Z=aa
```

To decompress, perform replacements in reverse order.

## large language models

The modified BPE for language modeling encodes plaintext into "tokens" (natural numbers). All unique tokens found in a corpus are listed in a token vocabulary (e.g., size 100,256 for GPT-3.5 and GPT-4).

The algorithm initially treats unique characters as 1-character-long n-grams (initial tokens). Then successively, the most frequent pair of adjacent tokens is merged into a new, longer n-gram until a vocabulary of prescribed size is obtained.

### example

For "aaabdaaabac" with vocabulary size 6:

- Initial encoding: "0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 3" with vocabulary "a=0, b=1, d=2, c=3"
- Final encoding: "4, 5, 2, 4, 5, 0, 3" with vocabulary "a=0, b=1, d=2, c=3, aa=4, ab=5"

### byte-level

Byte-level converts text into UTF-8 first and treats it as a stream of bytes. This guarantees any UTF-8 encoded text can be encoded by the BPE. Used in BERT-like models (RoBERTa, BART, DeBERTa) and GPT-like models (GPT-2).

