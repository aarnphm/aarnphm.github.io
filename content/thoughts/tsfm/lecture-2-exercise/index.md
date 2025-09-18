---
id: index
tags:
  - seed
  - ml
  - tsfm
description: This exercise focuses on understanding and optimizing a Byte Pair Encoding (BPE) tokenizer.
date: "2025-09-04"
modified: 2025-09-19 12:21:20 GMT-04:00
noindex: true
title: tokenization and computation
---

see also [[thoughts/tsfm/2|notes]], [[thoughts/byte-pair encoding|BPE]]

![[thoughts/images/final-optimization-tsfm-tokenizers-from-scratch.webp]]

## download data for training tokenizers

```bash
cd src/minibpe/data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

## setup

make sure to have Rust, uv, setup.

```bash
# assuming you have a venv
uv pip install -e . -v
```

to run training:

```bash
minibpe-train -d tinygpt-train --proc 5 --batch_size 500
```
