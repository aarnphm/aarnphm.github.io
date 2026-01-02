---
date: "2025-09-19"
description: and Modal examples
id: index
modified: 2026-01-02 02:23:18 GMT-05:00
tags:
  - ml
  - tsfm
title: Scaled dot-product attention profiling
---

This repo contains:

- `main.py`: naive attention and PyTorch SDPA (FlashAttention backend when available) with a small CLI and optional profiler traces.
- `triton_add.py`: a minimal Triton add kernel run remotely via Modal.

## Prerequisites

- Python 3.13+
- Recommended: `uv` for running with isolated, auto-resolved environments
- Optional: CUDA-capable GPU (for local GPU runs)

## Install (via uv)

No manual setup required; `uv run` will resolve and install deps from `pyproject.toml` on first use.

```bash
uv run python main.py --help
```

## Local usage: `main.py`

Get help:

```bash
uv run python main.py --help
```

Run naive attention on CUDA (if available):

```bash
uv run python main.py --B 4 --H 8 --S 1024 --D 64 --dtype fp16 --device cuda --iters 20 --warmup 10 --impl naive
```

Run SDPA/FlashAttention (PyTorch F.scaled_dot_product_attention):

```bash
uv run python main.py --B 4 --H 8 --S 1024 --D 64 --dtype fp16 --device cuda --iters 20 --warmup 10 --impl sdpa
```

CPU fallback example:

```bash
uv run python main.py --B 2 --H 4 --S 256 --D 64 --dtype fp32 --device cpu --iters 5 --warmup 2 --impl naive
```

Write PyTorch Profiler traces (viewable in TensorBoard and chrome://tracing):

```bash
uv run python main.py --logdir ./traces --iters 30 --warmup 10 --device cuda --impl sdpa
tensorboard --logdir ./traces
```

## Remote GPU (Modal)

Important: run this once before any Modal command to authenticate and set up:

```bash
uv run modal setup
```

Remote run `main.py` on an H100 (writes traces to a persisted Modal volume and a local copy under `/tmp`):

```bash
uv run modal run main.py --batch 4 --heads 8 --seq 1024 --dim 64 --dtype fp16 --seed 0 --warmup 10 --iters 20 --impl naive --label naive
uv run modal run main.py --batch 4 --heads 8 --seq 1024 --dim 64 --dtype fp16 --seed 0 --warmup 10 --iters 20 --impl sdpa  --label sdpa
```

## Remote Triton add kernel (`triton_add.py`)

Use the local entrypoint wrapper:

```bash
uv run modal run triton_add.py
```

Or invoke the remote function directly with args:

```bash
uv run modal run triton_add.py::triton_add_remote --size 98432 --block-size 1024 --seed 0
```

## notes

- Device selection: `--device cuda|cpu|mps`; if CUDA is requested but unavailable, the script falls back to CPU with a warning.
- SDPA path prefers the FlashAttention backend on CUDA when available.
