---
date: '2025-09-04'
description: how text becomes bytes, pieces, and token IDs
id: Tokenization
modified: 2026-06-05 15:08:26 GMT-04:00
tags:
  - seed
  - ml
title: Tokenization
---

A language model receives token IDs. Text reaches the tokenizer after decoding into Unicode or as raw bytes. UTF-8 writes Unicode scalar values as bytes. The tokenizer groups text or bytes into vocabulary pieces and maps each piece to an integer.

## ascii

[ASCII](https://ecma-international.org/publications-and-standards/standards/ecma-6/) assigns 128 values, so each value fits in seven bits. Values `0` to `31` are control characters, and value `127` is `DEL`. Values `32` to `126` are the 95 printable characters, including space. An eight-bit representation of an ASCII value has a leading zero; ASCII does not define an eighth bit.

In ASCII, each Latin uppercase and lowercase pair differs by bit `0x20`. Python makes the representation visible:

```python
format(ord("Y"), "07b")  # '1011001'
format(ord("y"), "07b")  # '1111001'
```

"Extended ASCII" does not name one standard. It refers to several incompatible eight-bit encodings that reused values `128` to `255`.

## unicode and utf-8

Unicode defines a code space from `U+0000` through `U+10FFFF`. A Unicode scalar value is any code point in that range outside the surrogate range `U+D800` through `U+DFFF`. Unicode assigns abstract characters to code points inside the code space. A code point is still not the same thing as a visible character. One grapheme can contain several code points.

[UTF-8](https://www.rfc-editor.org/rfc/rfc3629) encodes each Unicode scalar value as one to four bytes. ASCII values keep their one-byte representation. Other values use a leading byte followed by continuation bytes.

```python
"😋".encode("utf-8")  # b'\xf0\x9f\x98\x8b'
```

## model tokens

The model and tokenizer share one fixed vocabulary. Changing the tokenizer changes the token IDs and therefore changes which embedding row the model reads.

A pure byte tokenizer can represent any UTF-8 byte stream with 256 base values, though long common strings then occupy many tokens. Byte-level BPE starts from those byte values and learns merges, keeping byte coverage while shortening frequent byte sequences. A whole-word vocabulary gives frequent words short sequences, though names, spelling variants, and new words either enlarge the vocabulary or require an unknown token. Subword tokenizers learn pieces between those two extremes. Frequent strings may stay whole, while rarer strings split into smaller pieces.

The exact split depends on the learned vocabulary and its boundary rules. `annoyingly` might be one token, several meaningful-looking pieces, or a sequence that cuts across morphemes. Token boundaries follow the learned vocabulary, so they need not match morphemes.

## subword methods

### byte-pair encoding

Subword BPE learns an ordered list of frequent pair merges and applies those merges to new text [@sennrich2016neural]. GPT-2 uses byte-level BPE with a 256-byte base vocabulary [@radford2019language].

![[thoughts/byte-pair encoding|BPE]]

### wordpiece

Original BERT uses a 30,000-token WordPiece vocabulary [@devlin2019bert]. Its [released tokenizer](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/tokenization.py#L300-L359) applies basic text processing, then scans each word with greedy longest-match-first search. A continuation piece begins with `##`, and a word becomes `[UNK]` when the vocabulary cannot cover the complete word. The paper specifies the vocabulary size and token format, while the released tokenizer specifies encoding.

### unigram

The unigram model starts with an oversized set of candidate pieces. It estimates a probability for each piece, removes pieces whose deletion changes the corpus loss least, and repeats until it reaches the target vocabulary [@kudo2018subword].

For an input string $\mathbf{x}$, let $\mathcal{S}(\mathbf{x})$ be its possible segmentations. If a segmentation $\mathbf{z}=(z_1,\ldots,z_m)$ uses independent piece probabilities, then:

$$
P(\mathbf{z}) = \prod_{j=1}^{m} p(z_j),
\qquad
\sum_{z\in V}p(z)=1
$$

The string probability sums over all valid segmentations:

$$
P(\mathbf{x}) = \sum_{\mathbf{z}\in\mathcal{S}(\mathbf{x})} P(\mathbf{z})
$$

Training minimizes the corpus negative log likelihood:

$$
\mathcal{L} = -\sum_{i=1}^{N}\log P\left(\mathbf{x}^{(i)}\right)
$$

Encoding can choose the most probable segmentation with Viterbi search. Subword regularization instead samples segmentations during training [@kudo2018subword].

### sentencepiece

SentencePiece is a tokenizer implementation. It can train either BPE or unigram models directly from raw sentences [@kudo2018sentencepiece]. It treats whitespace as a normal symbol, displayed as `▁`, so decoding can recover spaces by concatenating pieces and replacing that symbol [@kudo2018sentencepiece].
