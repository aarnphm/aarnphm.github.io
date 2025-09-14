---
id: reports
tags:
  - seed
  - ml
  - vllm
date: "2025-09-13"
descriptions: and documentation of learning procedure.
modified: 2025-09-14 03:14:48 GMT-04:00
noindex: true
title: assignment three reports.
---

Tokenizer used: [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)

---

## appendix

## profiling

trace through git commits for edification purposes.

### [212557eb](https://github.com/aarnphm/aarnphm.github.io/commit/212557ebffea31f1c5eaabe04c74a29d22ca7895)

data processing:

- Pre-tokenize to disk + memmap. Stream TinyStories once, tokenize in big batches, write a single .bin of token IDs and .idx for document starts; training samples are contiguous chunks via random offsets.
- Sliding window with stride.
  - When packing tokens, use overlapping windows (seq_len, stride < seq_len) to maximize utilization and reduce “insufficient tokens, try again” loops.
- Async prefetch.
  - Run tokenization/packing in a background thread/process, push ready (x, y) into a bounded Queue, and pop from it in the train loop to hide preprocessing latency.

model implementation:

- Will need to replace one-hot in cross-entropy. Current `cross_entropy_logits` builds a full $N\times\;V$ one-hot and multiplies by `log(probs)` — devastating when V is ~150k. Compute loss/grad by direct indexing instead:
  - Gather log_probs[np.arange(N), targets]
  - For grad: probs[np.arange(N), targets] -= 1 and scale by 1/N (or masked denom)
- 2D GEMMs for weight tying:
  - logits = x_f @ W_E.T does batched matmuls (B×S many small GEMMs).
  - flatten to (B·S, D) @ (D, V) then reshape back to (B, S, V) for 1 big GEMM.
  - Do the same in backward:
    - dX_f = dLogits2D @ W_E
    - dW_E_head = X2D.T @ dLogits2D
- speed up embedding backward.
  - avoid np.add.at directly on a huge (V, D) buffer:
    - Accumulate on unique tokens only, then scatter:
      - uniq, inv = np.unique(flat_ids, return_inverse=True)
      - tmp = np.zeros((len(uniq), D)); np.add.at(tmp, inv, flat_dOut)
      - dW = np.zeros((V, D)); dW[uniq] = tmp
- Pre-flatten all residual linear ops to 2D GEMMs where possible
  - apply to the tied head path too
- Reduce reshape+transpose copies. Right now there are a lot of those.
  - `views` when safe, and prefer a single reshape after a transpose
  - avoid mixed orders that force materialization.
- Fuse simple ops:
  - fuse residual add + layer norm calls via cached forward/backward functions to reduce passes over memory
    - TODO: a fused residual+LN helper

system opts:

- a multithreaded BLAS (MKL/OpenBLAS):
  ```bash
  export OMP_NUM_THREADS=<cores>
  export MKL_NUM_THREADS=<cores>
  ```
- Increase effective batch via gradient accumulation to feed bigger GEMMs if memory allows.
- Sparse optimizer for embeddings. Return sparse grads for W_E (rows + values) and upd

kernels (optional, but omitted for brevity):

- JIT hotspots with Numba (LN backward, embedding scatter)
- CuPy and CUDA implementation

### [b88041b7](https://github.com/aarnphm/aarnphm.github.io/commit/b88041b7d6b1a493dcc1a3edd61ab456594f1782)

- initial implementation of the modular repository, with cleaning up from skafolding.
