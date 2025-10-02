---
id: reports
tags:
  - technical
  - ml
  - vllm
description: and documentation of learning procedure.
date: "2025-09-13"
modified: 2025-09-15 18:00:42 GMT-04:00
title: assignment three reports.
---

## [[thoughts/optimization#ReLU]] run

```json
{
  "name": "hinterland-np",
  "tokenizer": "Qwen/Qwen3-Next-80B-A3B-Instruct",
  "d_model": 512,
  "n_heads": 4,
  "n_layers": 4,
  "d_ff": 2048,
  "vocab_size": 151669,
  "max_seq_len": 512,
  "weight_tying": true,
  "seed": 42,
  "lr": 0.0003,
  "betas": [0.9, 0.95],
  "weight_decay": 0.01,
  "grad_clip": 1.0,
  "batch_size": 16,
  "steps": 1000,
  "warmup_steps": 0,
  "eval_every": 50,
  "log_every": 10,
  "stride": 0,
  "prefetch": 8,
  "lr_min": 1e-6,
  "plateau_patience": 5,
  "plateau_factor": 0.5,
  "lr_cooldown": 0,
  "early_stop_patience": 10,
  "early_stop_min_delta": 0.001,
  "target_loss": null
}
```

![[thoughts/images/configuration-runs.webp]]

![[thoughts/images/inference-runs.webp]]

![[thoughts/images/plots.webp]]

## explanation

`TokenDataset`:

- memmaps a flat token stream from disk; serves random or sequential windows.
- Shapes: $x,y \in \text{int\_64}$ with shape `(B, S)`. y is x shifted by 1.

  Disk/memmap and metadata:

  ```text disableLineNumber=true
  [ <base>.tokens.bin ] <-- np.memmap(dtype=meta['dtype'], shape=(n*tokens,))
  |
  v
  tokens: t0 t1 t2 t3 t4 ... t_{N-1}
  ```

  Random window sampling (sample_batch):
  - Inputs: batch_size=B, seq_len=S, stride, rng
  - max_start = N - (S+1)
  - n_positions = max_start // stride + 1
  - starts = (rng.integers(0, n_positions, size=B)) \* stride
  - For each start s [^token-graph]:
    ![[thoughts/images/token-dataset-graph.webp]]

  Sequential iteration (sequential_batches)
  - starts = [0, stride, 2*stride, ...] up to max_start
  - yield in chunks of size B (drop_last optional)

`BatchPrefetcher`:

- Background thread fills a bounded queue with precomputed batches to overlap
  data prep and compute.
- Shapes: each queued item is a tuple (x, y) with x,y ∈ int64, (B, S). [^graph]
  ![[thoughts/images/graph-prefetcher.webp]]
- Stop logic: sets an event, attempts a final put to unblock consumers, then joins the worker thread.

[^token-graph]:
    ```text disableLineNumber=true
    seg = tokens[s : s + S + 1] = [t_s, t_{s+1}, ..., t_{s+S}]
    x_i = seg[:S]               = [t_s, ..., t_{s+S-1}]
    y_i = seg[1:]               = [t_{s+1}, ..., t_{s+S}]

    tokens:  ...  t_{s-1} | t_s  t_{s+1} t_{s+2} ... t_{s+S} | t_{s+S+1} ...
    [------- S+1 -------]
    x:           [------ S ------]
    y:              [------ S ------]
    ```

[^graph]:
    ```text disableLineNumber=true
    +-----------------------+        q.put((x,y))        +----------------------------+         q.get()         +-----------------------+
    |     Worker Thread     | -------------------------> |     Queue (maxsize=P)      | --------------------->  |      Main Thread      |
    | _worker():            |                            |  [(x,y), (x,y), ...]       |                         | train loop            |
    |  ds.sample_batch(...) |                            |  bounded producer/consumer |                         | prefetcher.next()     |
    +-----------------------+                            +----------------------------+                         +-----------------------+

    Stop sequence:
    [ prefetcher.stop() ] -> stop_event.set() -> q.put_nowait(dummy) -> thread.join()
    ```

---

## setup

> [!info]
>
> Environment:
>
> - Python 3.11
> - NumPy 2.3
> - PyTorch 2.8.0
> - HuggingFace `datasets/transformers` for data and tokenization.
> - FFN activation is GELU (tanh approximation).

- Quick checks
  - Run tests: `pytest -q` (expects 10 tests passing)
  - Smoke train (synthetic data): `python -m minigpt.np.smoke`
    - Example: `first: 5.5629, last: 2.3235` on CPU (20 steps)
- Training
  - Streaming TinyStories, memmapped tokens, async prefetch
  - `./run_train.sh [--steps 1000 --name run1 ...]`
  - See [[thoughts/tsfm/lecture-3-exercise/reports#data processing|data processing]] for pipeline details
- Inference
  - `./run_inference.sh` or `python -m minigpt.np.inference --prompt "..." --max_new_tokens 32`
- Loss plots
  - `./run_plot.sh [CKPT_DIR] [--width 120 --height 20]`
- Performance tips
  - `export OMP_NUM_THREADS=<cores>; export MKL_NUM_THREADS=<cores>`
  - Prefer larger batch/accumulation for bigger GEMMs

## profiling

trace through git commits for edification purposes.

### DDP CPU implementation

TODO

### [1ef6de57](https://github.com/aarnphm/aarnphm.github.io/commit/1ef6de57d5b911ce00bfad87443c63b554fab686)

- Add parameters counter, both simplex and tree view
- Add early stop and LR plateau implementation.
- Add ASCII plot for eval/train graph

### [758bba9d](https://github.com/aarnphm/aarnphm.github.io/commit/758bba9df1a2e7dcfa08e4239d737acef26c14ff)

- add resume checkpointing, saving optimizer states, plumbing logs
- remove unnecessary backward pass cache. Activation re-materialisation should only cache certain attention matrices, instead of everything.
- cleanup forward pass for inference logics.

### [d758a3e4](https://github.com/aarnphm/aarnphm.github.io/commit/d758a3e41a7b6da8c4d8f2770656a4774314b9f1)

- implemented more efficient [[thoughts/tsfm/lecture-3-exercise/reports#data processing]]
- implemented certain optimization [[thoughts/tsfm/lecture-3-exercise/reports#model implementation|observation]]
- cleanup test cases and one ops file for both numpy and torch equivalent.
- added `inference.py` with KVCache, top_k_top_p

### [212557eb](https://github.com/aarnphm/aarnphm.github.io/commit/212557ebffea31f1c5eaabe04c74a29d22ca7895)

#### data processing

- ✅ Pre-tokenize to disk + memmap. Stream TinyStories once, tokenize in big batches, write a single .bin of token IDs and .idx for document starts; training samples are contiguous chunks via random offsets.
- ✅ Sliding window with stride.
  - When packing tokens, use overlapping windows (seq_len, stride < seq_len) to maximize utilization.
- ✅ Async prefetch.
  - Run tokenization/packing in a background thread/process, push ready (x, y) into a bounded Queue, and pop from it in the train loop to hide preprocessing latency.

#### model implementation

- ✅ Will need to replace one-hot in cross-entropy.
  - Current `cross_entropy_logits` builds a full $N\times\;V$ one-hot and multiplies by `log(probs)`, which is suboptimal for large vocab size (i.e Qwen).
  - Compute loss/grad by direct indexing instead:
    - gather log_probs[np.arange(N), targets]
    - for grad: probs[np.arange(N), targets] -= 1 and scale by 1/N (or masked denom)
- ✅ 2D GEMMs for weight tying:
  - logits = x_f @ W_E.T does batched matmuls (B×S many small GEMMs).
  - flatten to (B·S, D) @ (D, V) then reshape back to (B, S, V) for 1 big GEMM.
  - Do the same in backward:
    - dX_f = dLogits2D @ W_E
    - dW_E_head = X2D.T @ dLogits2D
- ✅ speed up embedding backward.
  - avoid np.add.at directly on a huge (V, D) buffer:
    - Accumulate on unique tokens only, then scatter:
      - uniq, inv = np.unique(flat_ids, return_inverse=True)
      - tmp = np.zeros((len(uniq), D)); np.add.at(tmp, inv, flat_dOut)
      - dW = np.zeros((V, D)); dW[uniq] = tmp
- 🚧 Pre-flatten all residual linear ops to 2D GEMMs where possible
  - apply to the tied head path too
- 🚧 Reduce reshape+transpose copies. Right now there are a lot of those.
  - `views` when safe, and prefer a single reshape after a transpose
  - avoid mixed orders that force materialization.
- 🚧 Fuse simple ops:
  - fuse residual add + layer norm calls via cached forward/backward functions to reduce passes over memory
    - TODO: a fused residual+LN helper

#### system opts

- a multithreaded BLAS (MKL/OpenBLAS):
  ```bash
  export OMP_NUM_THREADS=<cores>
  export MKL_NUM_THREADS=<cores>
  ```
- Increase effective batch via gradient accumulation to feed bigger GEMMs if memory allows.
- Sparse optimizer for embeddings. Return sparse grads for W_E (rows + values) and upd

#### kernels

_(optional, but omitted for brevity)_

- JIT hotspots with Numba (LN backward, embedding scatter)
- CuPy and CUDA implementation

### [b88041b7](https://github.com/aarnphm/aarnphm.github.io/commit/b88041b7d6b1a493dcc1a3edd61ab456594f1782)

- initial implementation of the modular repository, with cleaning up from scaffolding.

### Activation: switch FFN to GELU

> [!note]
> We replaced ReLU with GELU in the feed‑forward network for better Transformer performance and smoother gradients.

- Change: swap ReLU → GELU in `src/minigpt/np/modular.py` FFN forward/backward and Torch equivalents.
  - Numpy: `ffn` now uses `gelu(z)`; `ffn_bwd` uses analytic derivative of the tanh‑approx GELU.
  - Torch: `torch_ffn_bwd` and `torch_block_bwd` use `F.gelu(..., approximate='tanh')`.
- GELU formula (tanh approx):
  - $\mathrm{gelu}(x) = \tfrac{1}{2} x \left(1 + \tanh\!\big(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\big)\right)$
- Rationale:
  - Standard for GPT‑like models; yields better training stability vs ReLU.
- Validation:
  - Unit tests comparing NumPy and PyTorch grads all pass locally (10/10).
