---
date: '2025-09-07'
description: The merge-order and byte-storage contracts shared by this exercise's Python trainer and two encoders.
id: implementation
modified: 2026-09-16 09:15:11 GMT-04:00
tags:
  - seed
  - technical
  - ml
title: implementation of tokenization
---

The model is small enough to inspect: byte IDs, learned pairs, and the order in which those pairs were learned. The bugs came from letting training, storage, and encoding disagree about those three things.

## one merge order

The [[thoughts/tsfm/lecture-2-exercise/src/minibpe/training.py|trainer]] counts adjacent pairs, learns one pair, replaces it throughout the corpus, and counts again. Consider a corpus containing `abc` once and `bc` twice. The first merge is `bc`. The remaining `abc` is then represented as `a` followed by the token for `bc`; that pair is available for the next training step.

The old batch path selected `bc` and `ab` from the same counts. A left-to-right replacement could take `ab` inside `abc`, even though encoding gave `bc` the earlier rank. Recounting after each merge removes that disagreement. [Karpathy's basic BPE implementation](https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py) is a short reference for this training loop.

Both encoders choose the lowest-ranked pair currently present and resolve overlapping occurrences from left to right. The [[thoughts/tsfm/lecture-2-exercise/src/minibpe/impl.py|Python encoder]] uses a heap and linked indices. The [[thoughts/tsfm/lecture-2-exercise/rust/fastbpe/src/bpe.rs|Rust encoder]] scans the remaining sequence at each step, with worst-case time $O(n^2)$ for a pretoken of $n$ bytes. These data structures alone do not tell us which encoder is faster on a given workload.

Text first passes through the [GPT-2 pretokenization pattern](https://github.com/openai/gpt-2/blob/master/src/encoder.py). Its whitespace lookahead matters: in `a  b`, one space belongs to the whitespace piece and the other to the following word. The two implementations now use the same pattern. `encode_bytes` skips pretokenization and can merge across those boundaries.

## bytes survive storage

A token need not be valid UTF-8 by itself. The euro sign has bytes $[226,130,172]$; a learned token may cover only $[226,130]$. Decoding that fragment to text with replacement enabled discards the original bytes. Encoding the replacement character afterward cannot recover them.

`merges.txt` is the model's authoritative file. Each row records two existing IDs and their new ID. IDs determine merge rank, and loading expands each pair into its original bytes. The loader rejects malformed rows, repeated pairs, reused IDs, and references to undefined tokens. Unknown IDs also raise an error during decoding.

`save_pretrained` writes a decimal-byte `vocab.txt` for inspection. Loading reconstructs the vocabulary from merges, which also recovers old models whose display vocabulary contains replacement characters. Existing valid merge files remain usable. `decode_bytes` returns the exact concatenation; `decode` performs UTF-8 decoding afterward.

## checked limits

The [[thoughts/tsfm/lecture-2-exercise/index#checks|exercise checks]] cover learned merge order, ties, save/load, arbitrary bytes, Unicode, file training, malformed models, and partial UTF-8 tokens. Python/Rust parity tests require a built extension; skipped tests leave that comparison unverified.

Existing toy, TinyStories, and FineWeb merge files loaded and round-tripped a multilingual sample during this repair. This checks those files on sampled text. Large-corpus training and throughput measurements were outside the pass.

The dataset runner was tested through its Python function. A clean `uv sync` and the Fire command-line entry point were not exercised. The merge learner still rescans the corpus after every learned pair, so start with the small vocabulary in the exercise instructions.
