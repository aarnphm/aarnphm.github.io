---
id: index
tags:
  - seed
description: This exercise focuses on understanding and optimizing a Byte Pair Encoding (BPE) tokenizer.
date: "2025-09-04"
modified: 2025-09-04 21:17:32 GMT-04:00
title: Tokenization and Computation
---

## Byte Pair Encoding (BPE) Tokenizer

### Overview

The `tokenization/` directory contains a partial and inefficient implementation of a BPE tokenizer. Your task is to analyze the code, identify bugs and inefficiencies, and complete the implementation.

### Files

- `tokenization/tokenization.py`: An inefficient and partially incomplete implementation of the BPE training algorithm. You can test it on `tokenization/data/toy_data.txt`.
- `tokenization/EncoderDecoder.py`: An empty file where you will implement the encoder and decoder for your trained BPE tokenizer.
- `tokenization/data/toy_data.txt`: A small dataset for testing your tokenizer.
- `tokenization/data/TinyStoriesV2-GPT4-*.txt`: Larger datasets for training your tokenizer.
- `computation.py`: An example of a simple MLP forward pass which includes a docstring with commands to analyze memory consumption and log call stacks. You can use this as a reference for profiling `tokenization.py`.

### Your Tasks

1.  **Analyze `tokenization.py`:**
    - Read through the code to understand the current implementation.
    - Identify sources of inefficiency in terms of both speed and memory usage.
    - Find any potential bugs in the logic.
    - The existing code includes some basic memory profiling. How can you improve it to better understand performance bottlenecks? Refer to `computation.py` for more advanced profiling techniques.

2.  **Implement `EncoderDecoder.py`:**
    - Create a class in this file to handle encoding and decoding.
    - This class should load the vocabulary and merge rules generated from `train_bpe` in `tokenization.py`.
    - Implement an `encode` method that takes a string and returns a list of token IDs.
    - Implement a `decode` method that takes a list of token IDs and returns a string.

3.  **Train and Test your Tokenizer:**
    - Run the BPE training on `toy_data.txt` to start.
    - Once you are confident in your implementation, try training it on the larger `TinyStories` dataset.
    - Use your `EncoderDecoder` to encode and decode text and verify that it works correctly.

## Setup

- `uv venv`
- `uv sync`

## Download data for training tokenizers

- `cd tokenization/data`
- `wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt`
- `wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt`
