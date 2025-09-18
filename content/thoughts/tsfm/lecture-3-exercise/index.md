---
id: index
tags:
  - seed
  - ml
  - tsfm
description: from scratch frfr
date: "2025-09-11"
modified: 2025-09-19 12:21:42 GMT-04:00
title: numpy implementation of Transformer
---

this directory contains a numpy implementation of transformer components with both forward and backward passes.

the implementations are validated against pytorch to ensure gradient correctness.

see also [[thoughts/tsfm/lecture-3-exercise/reports|reports]]

## assignment

1. Add causal masking to the attention mechanism in this NumPy implementation for autoregressive language modeling.

2. Use the TinyStories dataset from Assignment 2 and a BPE tokenizer (either the one you trained previously or an existing tokenizer) to prepare tokenized training and validation splits.

3. Train a small Transformer language model end-to-end using only this NumPy implementation. Your training should include:
   - Token embedding and positional information
   - One or more Transformer blocks
   - A language modeling head that produces next-token logits
   - An appropriate training objective for next-token prediction

4. Report: Provide a brief summary of your training configuration, training/validation loss curves, and sample generations.

## constraints and expectations

- Implement causal masking within this NumPy stack.
- Use TinyStories from Assignment 2 and a BPE tokenizer (yours or an existing one).
- Keep your implementation self-contained in NumPy for the forward and backward passes.
- Ensure gradient checks continue to pass for unmasked cases.
- Provide clear instructions or scripts to reproduce your results.
