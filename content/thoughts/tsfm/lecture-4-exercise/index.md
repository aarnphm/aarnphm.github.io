---
date: '2025-09-19'
description: local and Modal traces for naive attention and PyTorch SDPA
id: index
modified: 2026-06-05 15:08:32 GMT-04:00
tags:
  - ml
  - tsfm
title: scaled dot-product attention profiling
---

This note documents `main.py`, which compares a direct PyTorch implementation of scaled dot-product attention with `torch.nn.functional.scaled_dot_product_attention`:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D}}\right)V.
$$

The tensors have shape $[B,H,S,D]$. The naive path materializes the $[B,H,S,S]$ score matrix, runs softmax, then multiplies by $V$. The SDPA path calls PyTorch SDPA with FlashAttention enabled.

## files

- `main.py` contains the naive and SDPA functions, a local profiler CLI, and a Modal entry point.
- `triton_add.py` is a separate vector-add exercise that checks a Triton kernel against `torch.add`.
- `sdpa_trace_*.json` and `nattn_trace_*.json` are saved profiler traces from earlier H100 and H200 runs.
- `pyproject.toml` declares NumPy, PyTorch, and Python 3.11 or newer. `.python-version` selects Python 3.13 for this directory.

## what the saved traces show

The naive H100 trace contains separate matrix multiplication and softmax kernels. The SDPA H100 trace contains `aten::_scaled_dot_product_flash_attention` and a fused FlashAttention CUDA kernel. The fused path avoids writing the full score and probability matrices to high-bandwidth memory.

The checked-in traces came from earlier runs. They use BF16 and 30 active profiler steps. Their shapes include $H=8,D=64$, $H=16,D=64$, and $H=8,D=128$. One trace records an H200. The current Modal configuration hardcodes an H100, so the H200 trace has no command path in the current script.

## local runs

`uv` resolves the dependencies declared in `pyproject.toml`.

```bash
uv run python main.py --help
```

Write a naive CUDA trace:

```bash
uv run python main.py --B 4 --H 8 --S 1024 --D 64 --dtype fp16 --device cuda --iters 20 --warmup 10 --impl naive --logdir ./traces/naive
```

Write an SDPA trace:

```bash
uv run python main.py --B 4 --H 8 --S 1024 --D 64 --dtype fp16 --device cuda --iters 20 --warmup 10 --impl sdpa --logdir ./traces/sdpa
```

The CLI does not print timing summaries or compare outputs. Without `--logdir`, it runs the profiler and exits without a saved result. View the saved traces with the PyTorch TensorBoard plugin:

```bash
uv run --with tensorboard --with torch-tb-profiler tensorboard --logdir ./traces
```

The SDPA function enables FlashAttention and disables PyTorch's math and memory-efficient alternatives. Unsupported devices, dtypes, or shapes can raise because the call has no execution-time fallback. The naive CPU path is the portable example:

```bash
uv run python main.py --B 2 --H 4 --S 256 --D 64 --dtype fp32 --device cpu --iters 5 --warmup 2 --impl naive --logdir ./traces/cpu
```

If CUDA is unavailable, requesting `--device cuda` falls back to CPU with a warning. An unavailable MPS request also falls back to CPU, but the current script does not print a warning for MPS.

## Modal runs

`modal` is imported by the scripts and is absent from `pyproject.toml`. Supply it through `uv`, then authenticate once:

```bash
uv run --with modal modal setup
```

Run both attention paths on the configured H100:

```bash
uv run --with modal modal run main.py --batch 4 --heads 8 --seq 1024 --dim 64 --dtype fp16 --seed 0 --warmup 10 --iters 20 --impl naive --label naive
uv run --with modal modal run main.py --batch 4 --heads 8 --seq 1024 --dim 64 --dtype fp16 --seed 0 --warmup 10 --iters 20 --impl sdpa --label sdpa
```

The remote image pins Python 3.11 and PyTorch 2.5.1. Local runs resolve PyTorch 2.8 or newer from `pyproject.toml`, so dispatcher and profiler versions differ. The Modal job stores traces in the `naive-attn-traces` volume and copies the newest JSON trace to `/tmp` on the local machine.

## Triton vector add

Run the local entry point:

```bash
uv run --with modal modal run triton_add.py
```

Or invoke the remote function with explicit arguments:

```bash
uv run --with modal modal run triton_add.py::triton_add_remote --size 98432 --block-size 1024 --seed 0
```

The function reports the maximum absolute difference between the Triton result and `torch.add`. It is a kernel and masking exercise, separate from the attention comparison.
