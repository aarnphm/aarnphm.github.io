---
created: '2025-09-11'
date: '2025-09-11'
description: pair substitution in byte compression and subword tokenizers
id: byte-pair encoding
modified: 2026-06-05 15:08:26 GMT-04:00
published: '2006-07-04'
source: https://en.wikipedia.org/wiki/Byte-pair_encoding
tags:
  - seed
title: byte-pair encoding
transclude:
  title: false
---

alias: _BPE, digram coding_

Philip Gage introduced byte-pair encoding in 1994 as a lossless compression algorithm for byte data [@gage1994new]. Sennrich, Haddow, and Birch later adapted the merge procedure to learn subword vocabularies for [[thoughts/LLMs|language-model]] [[thoughts/Tokenization|tokenizers]] [@sennrich2016neural]. Both procedures replace a frequent adjacent pair with one new symbol. Gage uses the table to compress bytes. Sennrich uses the merge list to build a tokenizer vocabulary.

## compression

Gage's algorithm finds the most frequent adjacent byte pair in a block, replaces every occurrence with an unused byte, and records that substitution. It repeats while a frequent pair and an unused byte remain. The encoded block carries the substitution table needed to recover the original bytes.

### example

Suppose the data to be encoded is:

```
aaabdaaabac
```

The byte pair `aa` occurs most often, so replace it with the unused byte `Z`:

```
ZabdZabac
Z=aa
```

Then replace `ab` with `Y`:

```
ZYdZYac
Y=ab
Z=aa
```

Then replace `ZY` with `X`:

```
XdXac
X=ZY
Y=ab
Z=aa
```

Decompression expands `X`, then `Y`, then `Z`.

## subword tokenization

Sennrich, Haddow, and Birch adapted the merge procedure to learn a fixed subword vocabulary. Their version starts with each word split into characters plus an end-of-word marker. During training, the tokenizer repeatedly merges the most frequent adjacent symbol pair and stores the ordered merge rules. During encoding, it applies those learned rules to new text. It does not learn new pairs from the text being encoded.

The initial alphabet, merge budget, boundary rules, and special tokens determine the final token vocabulary. A model can also omit infrequent learned symbols from the vocabulary it uses.

### example

For `aaabdaaabac`, start with the four character tokens and learn `aa`, then `ab`:

- Initial encoding: "0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 3" with vocabulary "a=0, b=1, d=2, c=3"
- Final encoding: "4, 5, 2, 4, 5, 0, 3" with vocabulary "a=0, b=1, d=2, c=3, aa=4, ab=5"

### byte-level BPE

The released GPT-2 tokenizer first splits text with a regular expression. Within each piece, it maps UTF-8 bytes to reversible Unicode code points and applies ranked BPE merges [@radford2019language]. Its base alphabet has 256 byte values, so every UTF-8 string can be represented without an unknown token.

Original BERT uses a 30,000-token WordPiece vocabulary rather than byte-level BPE [@devlin2019bert]. RoBERTa, BART, and DeBERTa use byte-level BPE variants, so calling the whole family "BERT-like BPE" hides a real tokenizer boundary.
