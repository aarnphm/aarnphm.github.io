---
id: index
tags:
  - seed
  - ml
description: from scratch frfr
date: "2025-09-11"
modified: 2025-09-14 01:22:02 GMT-04:00
title: numpy implementation of Transformer
---

this directory contains a numpy implementation of transformer components with both forward and backward passes. the implementations are validated against pytorch to ensure gradient correctness.

## what you have

- `src/minigpt/np`: all numpy implementation, with `torch_primitives.py` the equivalent.

## setup

```bash
uv sync
# or create a venv and `pip install -e .`
```

Run the provided gradient checks:

```bash
pytest tests
```

## assignment

1. Add causal masking to the attention mechanism in this NumPy implementation for autoregressive language modeling.

2. Use the TinyStories dataset from Assignment 2 and a BPE tokenizer (either the one you trained previously or an existing tokenizer) to prepare tokenized training and validation splits.

3. Train a small Transformer language model end-to-end using only this NumPy implementation. Your training should include:
   - Token embedding and positional information
   - One or more Transformer blocks
   - A language modeling head that produces next-token logits
   - An appropriate training objective for next-token prediction

4. Report: Provide a brief summary of your training configuration, training/validation loss curves, and sample generations.

### constraints and expectations

- Implement causal masking within this NumPy stack.
- Use TinyStories from Assignment 2 and a BPE tokenizer (yours or an existing one).
- Keep your implementation self-contained in NumPy for the forward and backward passes.
- Ensure gradient checks continue to pass for unmasked cases.
- Provide clear instructions or scripts to reproduce your results.

### notes

- You may organize additional code (data loading, training loop, etc.) as you see fit in this directory.
- If you add new dependencies (e.g., for tokenization or data handling), document them and how to install.
