---
id: index
tags:
  - seed
  - ml
description: This exercise focuses on understanding and optimizing a Byte Pair Encoding (BPE) tokenizer.
date: "2025-09-04"
modified: 2025-09-07 05:08:16 GMT+00:00
noindex: true
title: Tokenization and Computation
---

see also [[thoughts/tsfm/2|notes]], [[thoughts/Tokenization#BPE]]

TODO:

- [x] finish tasks
- [ ] 1B tokens
- [ ] Rust implementation

## download data for training tokenizers

```bash
cd tokenization/data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```
