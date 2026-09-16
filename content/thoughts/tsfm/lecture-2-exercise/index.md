---
date: '2025-09-04'
description: Train a byte-level BPE model in Python and check its Python and Rust encoders against the same saved merges.
id: index
modified: 2026-09-16 09:15:11 GMT-04:00
tags:
  - ml
  - tsfm
title: tokenization and computation
---

see also [[thoughts/tsfm/2|notes]], [[thoughts/byte-pair encoding|BPE]], [[thoughts/tsfm/lecture-2-exercise/implementation|implementation]]

This exercise has a byte-level BPE trainer in Python and two encoders that load the same saved merge list. The Rust extension exposes encoding and decoding for saved models; training stays in Python.

The first 256 token IDs represent individual bytes. Each training step counts adjacent pairs in the current corpus, chooses the most frequent pair, and replaces its non-overlapping occurrences from left to right. Counts include each pretoken's frequency. Ties choose the lowest pair of token IDs, so multiprocessing completion order cannot decide the vocabulary.

A requested vocabulary size $V$ allows at most $V-256$ merges. Training can stop earlier when no pairs remain. The trainer recounts after every merge, which makes a small vocabulary useful for checking the exercise before spending time on a larger run.

## setup

Run these commands from this exercise's directory with Rust and uv available:

```bash
uv sync
```

The package uses Maturin to build the Rust extension. Its Cargo manifest declares a separate workspace, so Cargo commands stay within this exercise. After changing Rust, rebuild the extension before checking parity:

```bash
uv sync --reinstall-package minibpe
```

## download the training data

```bash
mkdir -p src/minibpe/data
wget -O src/minibpe/data/TinyStoriesV2-GPT4-train.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget -O src/minibpe/data/TinyStoriesV2-GPT4-valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

## train and inspect

```bash
uv run minibpe-train --dataset=tinygpt-train --proc=5 \
  --vocab_size=512 --output_dir=models/tinygpt-512
```

`proc` controls local-file pretokenization. Merge learning runs sequentially in Python. The former `fast` and `batch_size` training options have been removed: the Rust training calls had no implementation, and applying several competing pairs in one left-to-right pass changed the learned merge order.

For local files, `<|endoftext|>` separates training documents and is removed before counting pretokens. It does not receive a reserved token ID. Both encoders treat that spelling as ordinary input text.

```python
from minibpe import Tokenizer, TokenizerFast

python_model = Tokenizer.from_pretrained('models/tinygpt-512')
rust_model = TokenizerFast.from_pretrained('models/tinygpt-512')
text = 'café 你好 🦀'
ids = python_model.encode(text)
assert rust_model.encode(text) == ids
assert python_model.decode(ids) == rust_model.decode(ids) == text
```

Use `encode_bytes` and `decode_bytes` for arbitrary binary input. `decode` joins the token bytes before decoding UTF-8 and replaces invalid sequences with the replacement character. An individual token can contain only part of a character.

## checks

```bash
cargo fmt --manifest-path rust/fastbpe/Cargo.toml --check
cargo clippy --manifest-path rust/fastbpe/Cargo.toml \
  --all --benches --tests --examples --all-features
cargo test --manifest-path rust/fastbpe/Cargo.toml
uv run python -m unittest discover -s tests -v
```

The Python suite checks learned merges, save/load, arbitrary bytes, Unicode, file training, and agreement with a built Rust extension. It reports skipped parity tests when the extension is absent. Passing only the Python tests leaves Rust unverified.

![[thoughts/images/final-optimization-tsfm-tokenizers-from-scratch.webp]]

This earlier optimization image is retained as a record of the exercise. It does not establish throughput for the current code. See the implementation note for the checks run on the repaired version.
