"""
Example 10: Paged Attention with CuTe DSL Python API
Demonstrates: Python-native GPU kernel development using CUTLASS 4.x CuTe DSL

Requirements:
    pip install nvidia-cutlass-dsl torch

Note: This requires CUTLASS 4.0+ with Python DSL support (currently in beta)
"""

import torch
import numpy as np
import argparse
import time
from functools import lru_cache

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math
from cutlass.cute.runtime import from_dlpack

import triton
import triton.language as tl


def paged_attention_torch(
  query: torch.Tensor,  # [num_seqs, num_heads, head_dim]
  key_cache: torch.Tensor,  # [num_pages, num_heads, page_size, head_dim]
  value_cache: torch.Tensor,  # [num_pages, num_heads, page_size, head_dim]
  block_tables: torch.Tensor,  # [num_seqs, max_num_pages]
  context_lens: torch.Tensor,  # [num_seqs]
  scale: float = None,
) -> torch.Tensor:
  """
  Pure PyTorch implementation of paged attention for verification.

  This serves as a reference implementation to verify the correctness
  of optimized CUDA/CuTe kernels.
  """
  num_seqs, num_heads, head_dim = query.shape
  page_size = key_cache.shape[2]

  if scale is None:
    scale = 1.0 / np.sqrt(head_dim)

  output = torch.zeros_like(query)

  for seq_idx in range(num_seqs):
    context_len = context_lens[seq_idx].item()
    num_pages = (context_len + page_size - 1) // page_size

    # Gather keys and values from paged cache
    keys_list = []
    values_list = []

    for page_idx in range(num_pages):
      block_num = block_tables[seq_idx, page_idx].item()
      if block_num < 0:
        break

      page_start = page_idx * page_size
      page_end = min(page_start + page_size, context_len)
      tokens_in_page = page_end - page_start

      # Extract page from cache
      keys_list.append(key_cache[block_num, :, :tokens_in_page, :])
      values_list.append(value_cache[block_num, :, :tokens_in_page, :])

    if not keys_list:
      continue

    # Concatenate pages: [num_heads, context_len, head_dim]
    keys = torch.cat(keys_list, dim=1)
    values = torch.cat(values_list, dim=1)

    # Compute attention: Q @ K^T
    # query[seq_idx]: [num_heads, head_dim]
    # keys: [num_heads, context_len, head_dim]
    scores = torch.einsum('hd,hkd->hk', query[seq_idx], keys) * scale  # [num_heads, context_len]

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)  # [num_heads, context_len]

    # Attention @ V
    out = torch.einsum('hk,hkd->hd', attn_weights, values)  # [num_heads, head_dim]

    output[seq_idx] = out

  return output


@triton.jit
def paged_attention_triton_kernel(
  q_ptr,
  k_ptr,
  v_ptr,
  block_table_ptr,
  context_lens_ptr,
  out_ptr,
  stride_q_seq,
  stride_q_head,
  stride_q_d,
  stride_k_page,
  stride_k_head,
  stride_k_token,
  stride_k_d,
  stride_v_page,
  stride_v_head,
  stride_v_token,
  stride_v_d,
  stride_o_seq,
  stride_o_head,
  stride_o_d,
  stride_bt_seq,
  num_seqs,
  num_heads,
  scale,
  MAX_PAGES: tl.constexpr,
  PAGE_SIZE: tl.constexpr,
  HEAD_DIM: tl.constexpr,
):
  pid = tl.program_id(0)
  seq_idx = pid // num_heads
  head_idx = pid % num_heads

  if seq_idx >= num_seqs:
    return

  context_len = tl.load(context_lens_ptr + seq_idx)
  if context_len <= 0:
    return

  token_offsets = tl.arange(0, PAGE_SIZE)
  dim_offsets = tl.arange(0, HEAD_DIM)

  q_ptrs = q_ptr + seq_idx * stride_q_seq + head_idx * stride_q_head + dim_offsets * stride_q_d
  q = tl.load(q_ptrs, mask=dim_offsets < HEAD_DIM, other=0.0).to(tl.float32)

  row_max = tl.full((1,), -float('inf'), dtype=tl.float32)
  row_sum = tl.zeros((1,), dtype=tl.float32)
  acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
  scale_f = tl.full((1,), scale, dtype=tl.float32)

  num_pages_seq = (context_len + PAGE_SIZE - 1) // PAGE_SIZE

  for page_idx in range(MAX_PAGES):
    within_range = page_idx < num_pages_seq
    block_num = tl.load(
      block_table_ptr + seq_idx * stride_bt_seq + page_idx,
      mask=within_range,
      other=0,
    )

    page_start = page_idx * PAGE_SIZE
    tokens_left = context_len - page_start
    tokens_left = tl.maximum(tokens_left, 0)
    tokens_in_page = tl.minimum(tokens_left, PAGE_SIZE)

    page_active = within_range & (tokens_in_page > 0)
    block_num = tl.where(page_active, block_num, 0)

    k_base = k_ptr + block_num * stride_k_page + head_idx * stride_k_head
    v_base = v_ptr + block_num * stride_v_page + head_idx * stride_v_head

    token_mask = page_active & (token_offsets < tokens_in_page)

    k_ptrs = k_base + token_offsets[:, None] * stride_k_token + dim_offsets[None, :] * stride_k_d
    v_ptrs = v_base + token_offsets[:, None] * stride_v_token + dim_offsets[None, :] * stride_v_d

    k = tl.load(k_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)

    scores = tl.sum(k * q[None, :], axis=1) * scale_f
    scores = tl.where(token_mask, scores, -float('inf'))

    page_max = tl.max(scores, axis=0)
    new_max = tl.maximum(row_max, page_max)
    exp_prev = tl.exp(row_max - new_max)

    scores_exp = tl.exp(scores - new_max)
    scores_exp = tl.where(token_mask, scores_exp, 0.0)

    row_sum = row_sum * exp_prev + tl.sum(scores_exp, axis=0)
    acc = acc * exp_prev + tl.sum(scores_exp[:, None] * v, axis=0)
    row_max = new_max

  inv_row_sum = 1.0 / tl.maximum(row_sum, 1e-9)
  acc = acc * inv_row_sum

  out_ptrs = out_ptr + seq_idx * stride_o_seq + head_idx * stride_o_head + dim_offsets * stride_o_d
  tl.store(out_ptrs, acc.to(tl.float32), mask=dim_offsets < HEAD_DIM)


def paged_attention_triton(
  query: torch.Tensor,
  key_cache: torch.Tensor,
  value_cache: torch.Tensor,
  block_tables: torch.Tensor,
  context_lens: torch.Tensor,
  scale: float,
) -> torch.Tensor:
  query_contig = query.contiguous()
  key_contig = key_cache.contiguous()
  value_contig = value_cache.contiguous()
  block_tables_contig = block_tables.contiguous()
  context_lens_contig = context_lens.contiguous()

  num_seqs, num_heads, head_dim = query_contig.shape
  page_size = key_contig.shape[2]
  max_num_pages = block_tables_contig.shape[1]

  output = torch.zeros_like(query_contig)

  stride_q_seq, stride_q_head, stride_q_d = query_contig.stride()
  stride_k_page, stride_k_head, stride_k_token, stride_k_d = key_contig.stride()
  stride_v_page, stride_v_head, stride_v_token, stride_v_d = value_contig.stride()
  stride_o_seq, stride_o_head, stride_o_d = output.stride()
  stride_bt_seq, _ = block_tables_contig.stride()

  total_ctas = num_seqs * num_heads

  paged_attention_triton_kernel[(total_ctas,)](
    query_contig,
    key_contig,
    value_contig,
    block_tables_contig,
    context_lens_contig,
    output,
    stride_q_seq,
    stride_q_head,
    stride_q_d,
    stride_k_page,
    stride_k_head,
    stride_k_token,
    stride_k_d,
    stride_v_page,
    stride_v_head,
    stride_v_token,
    stride_v_d,
    stride_o_seq,
    stride_o_head,
    stride_o_d,
    stride_bt_seq,
    num_seqs,
    num_heads,
    scale,
    MAX_PAGES=max_num_pages,
    PAGE_SIZE=page_size,
    HEAD_DIM=head_dim,
  )

  torch.cuda.synchronize()
  return output


@cute.kernel
def paged_attention_cute_kernel(
  output: cute.Tensor,
  query: cute.Tensor,
  key_cache: cute.Tensor,
  value_cache: cute.Tensor,
  block_tables: cute.Tensor,
  context_lens: cute.Tensor,
  scale: cutlass.Float32,
  num_seqs: cutlass.Int32,
  num_heads: cutlass.Int32,
  head_dim: cutlass.Int32,
  page_size: cutlass.Int32,
  max_num_pages: cutlass.Int32,
):
  """
  Baseline CuTe DSL implementation of paged attention.

  This is a simplified version demonstrating the CuTe DSL Python API.
  A full implementation would include:
  - Shared memory tiling
  - Online softmax with streaming updates
  - Warp-level optimizations
  - TMA (Tensor Memory Accelerator) usage on Hopper
  """

  # Grid: (num_heads, num_seqs, 1), Block: (1, 1, 1)
  head_idx, seq_idx, _ = cute.arch.block_idx()

  if seq_idx < num_seqs and head_idx < num_heads:
    context_len = context_lens[seq_idx]
    num_pages_seq = (context_len + page_size - 1) // page_size

    # Cache query vector for this CTA in registers
    q_local = cute.make_fragment((head_dim,), cutlass.Float32)
    acc = cute.make_fragment((head_dim,), cutlass.Float32)

    for d in range(head_dim):
      q_local[d] = query[seq_idx, head_idx, d]
      acc[d] = cutlass.Float32(0.0)

    # Online softmax statistics
    row_max = float('-inf')
    row_sum = 0.0

    # Process each page
    for page_idx in range(max_num_pages):
      page_is_active = page_idx < num_pages_seq
      block_num = block_tables[seq_idx, page_idx] if page_is_active else -1
      page_is_active = page_is_active and (block_num >= 0)

      page_start = page_idx * page_size
      page_end = min(page_start + page_size, context_len)
      tokens_in_page = page_end - page_start if page_is_active else 0

      # Compute attention scores for tokens in this page
      for token_idx in range(page_size):
        token_is_active = page_is_active and (token_idx < tokens_in_page)
        if token_is_active:
          qk_score = 0.0

          for d in range(head_dim):
            qk_score += q_local[d] * key_cache[block_num, head_idx, token_idx, d]

          qk_score *= scale

          # Online softmax update
          prev_max = row_max
          row_max = max(row_max, qk_score)

          exp_prev = 0.0 if prev_max == float('-inf') else cute_math.exp(prev_max - row_max)
          exp_curr = cute_math.exp(qk_score - row_max)

          row_sum = row_sum * exp_prev + exp_curr

          # Scale accumulator
          for d in range(head_dim):
            acc[d] *= exp_prev

          # Add attention * V
          attn_weight = exp_curr

          for d in range(head_dim):
            acc[d] += attn_weight * value_cache[block_num, head_idx, token_idx, d]

    # Final normalization and write output
    inv_row_sum = 0.0 if row_sum == 0.0 else 1.0 / row_sum

    for d in range(head_dim):
      output[seq_idx, head_idx, d] = acc[d] * inv_row_sum


@cute.jit
def paged_attention_cute_launch(
  output: cute.Tensor,
  query: cute.Tensor,
  key_cache: cute.Tensor,
  value_cache: cute.Tensor,
  block_tables: cute.Tensor,
  context_lens: cute.Tensor,
  scale: cutlass.Float32,
  num_seqs: cutlass.Int32,
  num_heads: cutlass.Int32,
  head_dim: cutlass.Int32,
  page_size: cutlass.Int32,
  max_num_pages: cutlass.Int32,
):
  """Launch wrapper for the baseline scalar CuTe kernel."""

  paged_attention_cute_kernel(
    output,
    query,
    key_cache,
    value_cache,
    block_tables,
    context_lens,
    scale,
    num_seqs,
    num_heads,
    head_dim,
    page_size,
    max_num_pages,
  ).launch(grid=(num_heads, num_seqs, 1), block=(1, 1, 1))


@lru_cache(maxsize=None)
def build_paged_attention_cute_optimized_launch(
  head_dim: int,
  page_size: int,
  max_num_pages: int,
  tile_tokens: int,
):
  """Bake compile-time tile constants into an optimized CuTe kernel."""

  head_dim_const = head_dim
  page_size_const = page_size
  max_pages_const = max_num_pages
  tile_tokens_const = max(1, min(tile_tokens, page_size_const))
  tiles_per_page_const = (page_size_const + tile_tokens_const - 1) // tile_tokens_const

  @cute.kernel
  def paged_attention_cute_kernel_optimized(
    output: cute.Tensor,
    query: cute.Tensor,
    key_cache: cute.Tensor,
    value_cache: cute.Tensor,
    block_tables: cute.Tensor,
    context_lens: cute.Tensor,
    scale: cutlass.Float32,
    num_seqs: cutlass.Int32,
    num_heads: cutlass.Int32,
  ):
    head_idx, seq_idx, _ = cute.arch.block_idx()

    if seq_idx < num_seqs and head_idx < num_heads:
      context_len = context_lens[seq_idx]
      num_pages_seq = (context_len + page_size_const - 1) // page_size_const

      q_vec_half = cute.make_fragment((head_dim_const,), cutlass.Float16)
      q_vec = cute.make_fragment((head_dim_const,), cutlass.Float32)
      acc = cute.make_fragment((head_dim_const,), cutlass.Float32)

      for d in range(head_dim_const):
        q_vec_half[d] = query[seq_idx, head_idx, d]
        q_vec[d] = cutlass.Float32(q_vec_half[d])
        acc[d] = cutlass.Float32(0.0)

      row_max = float('-inf')
      row_sum = 0.0

      for page_idx in range(max_pages_const):
        page_is_active = page_idx < num_pages_seq
        block_num = block_tables[seq_idx, page_idx] if page_is_active else -1
        page_is_active = page_is_active and (block_num >= 0)

        page_start = page_idx * page_size_const
        page_end = min(page_start + page_size_const, context_len)
        tokens_in_page = page_end - page_start if page_is_active else 0

        if page_is_active and tokens_in_page > 0:
          for tile_idx in range(tiles_per_page_const):
            tile_base = tile_idx * tile_tokens_const
            tokens_remaining = tokens_in_page - tile_base
            valid_tokens = tile_tokens_const if tokens_remaining >= tile_tokens_const else max(tokens_remaining, 0)

            if valid_tokens > 0:
              for token_offset in range(tile_tokens_const):
                token_is_valid = token_offset < valid_tokens

                if token_is_valid:
                  token_idx = tile_base + token_offset
                  qk_score = cutlass.Float32(0.0)

                  for d in range(head_dim_const):
                    k_val_half = key_cache[block_num, head_idx, token_idx, d]
                    q_val = q_vec[d]
                    qk_score += q_val * cutlass.Float32(k_val_half)

                  qk_score *= scale

                  prev_max = row_max
                  row_max = max(row_max, qk_score)

                  exp_prev = 0.0 if prev_max == float('-inf') else cute_math.exp(prev_max - row_max)
                  exp_curr = cute_math.exp(qk_score - row_max)

                  row_sum = row_sum * exp_prev + exp_curr

                  for d in range(head_dim_const):
                    v_val_half = value_cache[block_num, head_idx, token_idx, d]
                    acc[d] = acc[d] * exp_prev + exp_curr * cutlass.Float32(v_val_half)

      inv_row_sum = 0.0 if row_sum == 0.0 else 1.0 / row_sum

      for d in range(head_dim_const):
        output[seq_idx, head_idx, d] = cutlass.Float16(acc[d] * inv_row_sum)

  @cute.jit
  def paged_attention_cute_optimized_launch(
    output: cute.Tensor,
    query: cute.Tensor,
    key_cache: cute.Tensor,
    value_cache: cute.Tensor,
    block_tables: cute.Tensor,
    context_lens: cute.Tensor,
    scale: cutlass.Float32,
    num_seqs: cutlass.Int32,
    num_heads: cutlass.Int32,
  ):
    paged_attention_cute_kernel_optimized(
      output,
      query,
      key_cache,
      value_cache,
      block_tables,
      context_lens,
      scale,
      num_seqs,
      num_heads,
    ).launch(grid=(num_heads, num_seqs, 1), block=(1, 1, 1))

  return paged_attention_cute_optimized_launch


def paged_attention_cute_reference(
  query: torch.Tensor,
  key_cache: torch.Tensor,
  value_cache: torch.Tensor,
  block_tables: torch.Tensor,
  context_lens: torch.Tensor,
  scale: float,
) -> torch.Tensor:
  """Run the baseline CuTe kernel and return the output tensor."""

  output = torch.zeros_like(query)

  output_ = from_dlpack(output, assumed_align=16)
  query_ = from_dlpack(query, assumed_align=16)
  key_cache_ = from_dlpack(key_cache, assumed_align=16)
  value_cache_ = from_dlpack(value_cache, assumed_align=16)
  block_tables_ = from_dlpack(block_tables, assumed_align=16)
  context_lens_ = from_dlpack(context_lens, assumed_align=16)

  launcher = cute.compile(
    paged_attention_cute_launch,
    output_,
    query_,
    key_cache_,
    value_cache_,
    block_tables_,
    context_lens_,
    cutlass.Float32(scale),
    cutlass.Int32(query.shape[0]),
    cutlass.Int32(query.shape[1]),
    cutlass.Int32(query.shape[2]),
    cutlass.Int32(key_cache.shape[2]),
    cutlass.Int32(block_tables.shape[1]),
  )

  launcher(
    output_,
    query_,
    key_cache_,
    value_cache_,
    block_tables_,
    context_lens_,
    cutlass.Float32(scale),
    cutlass.Int32(query.shape[0]),
    cutlass.Int32(query.shape[1]),
    cutlass.Int32(query.shape[2]),
    cutlass.Int32(key_cache.shape[2]),
    cutlass.Int32(block_tables.shape[1]),
  )
  torch.cuda.synchronize()
  return output


_OPT_KERNEL_CACHE = {}


def paged_attention_cute_optimized(
  query: torch.Tensor,
  key_cache: torch.Tensor,
  value_cache: torch.Tensor,
  block_tables: torch.Tensor,
  context_lens: torch.Tensor,
  scale: float,
  tile_tokens: int = 32,
) -> torch.Tensor:
  """Run the FP16 tiled CuTe kernel and return the output tensor (fp32)."""

  query_fp16 = query.to(torch.float16)
  key_cache_fp16 = key_cache.to(torch.float16)
  value_cache_fp16 = value_cache.to(torch.float16)
  output_fp16 = torch.zeros_like(query_fp16)

  output_ = from_dlpack(output_fp16, assumed_align=16)
  query_ = from_dlpack(query_fp16, assumed_align=16)
  key_cache_ = from_dlpack(key_cache_fp16, assumed_align=16)
  value_cache_ = from_dlpack(value_cache_fp16, assumed_align=16)
  block_tables_ = from_dlpack(block_tables, assumed_align=16)
  context_lens_ = from_dlpack(context_lens, assumed_align=16)

  head_dim = query.shape[2]
  page_size = key_cache.shape[2]
  max_num_pages = block_tables.shape[1]
  tile_tokens = max(1, min(tile_tokens, page_size))

  launch = build_paged_attention_cute_optimized_launch(head_dim, page_size, max_num_pages, tile_tokens)

  cache_key = (
    query.shape[0],
    query.shape[1],
    head_dim,
    page_size,
    max_num_pages,
    tile_tokens,
  )

  compiled = _OPT_KERNEL_CACHE.get(cache_key)

  if compiled is None:
    compiled = cute.compile(
      launch,
      output_,
      query_,
      key_cache_,
      value_cache_,
      block_tables_,
      context_lens_,
      cutlass.Float32(scale),
      cutlass.Int32(query.shape[0]),
      cutlass.Int32(query.shape[1]),
    )
    _OPT_KERNEL_CACHE[cache_key] = compiled

  compiled(
    output_,
    query_,
    key_cache_,
    value_cache_,
    block_tables_,
    context_lens_,
    cutlass.Float32(scale),
    cutlass.Int32(query.shape[0]),
    cutlass.Int32(query.shape[1]),
  )
  torch.cuda.synchronize()

  return output_fp16.to(torch.float32)


# Profiling workflow:
#   nsys profile --trace=cuda,osrt --sample=cpu --gpu-metrics-device=all \\
#     python content/lectures/420/examples/cute_paged_attention.py --seq-len 4096 --num-seqs 4 --num-heads 16 --head-dim 128
#   ncu --set full --metrics sm__inst_executed_pipe_tensor_op_hmma.sum,smsp__warps_active.avg \\
#     python content/lectures/420/examples/cute_paged_attention.py --seq-len 4096 --num-seqs 4 --num-heads 16 --head-dim 128 --tile-tokens 64

def test_paged_attention(
  verbosity: int = 0,
  num_seqs: int = 2,
  seq_len: int = 32,
  num_heads: int = 4,
  head_dim: int = 64,
  page_size: int = 16,
  tile_tokens: int = 32,
):
  # Problem size
  max_num_pages = (seq_len + page_size - 1) // page_size

  # Context lengths for each sequence (all at max for this test)
  context_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device='cuda')

  # Calculate total pages needed
  total_pages = num_seqs * max_num_pages

  # Block tables (maps logical pages to physical pages)
  block_tables = torch.zeros((num_seqs, max_num_pages), dtype=torch.int32, device='cuda')
  for seq_idx in range(num_seqs):
    pages_for_seq = max_num_pages
    for page_idx in range(pages_for_seq):
      block_tables[seq_idx, page_idx] = seq_idx * max_num_pages + page_idx

  # Allocate tensors
  query = torch.randn(num_seqs, num_heads, head_dim, dtype=torch.float32, device='cuda')
  key_cache = torch.randn(total_pages, num_heads, page_size, head_dim, dtype=torch.float32, device='cuda')
  value_cache = torch.randn(total_pages, num_heads, page_size, head_dim, dtype=torch.float32, device='cuda')

  scale = 1.0 / np.sqrt(head_dim)
  total_tokens = int(context_lens.sum().item())
  total_flops = 4.0 * head_dim * num_heads * total_tokens

  def tflops_from_ms(milliseconds):
    if milliseconds is None or milliseconds <= 0:
      return None
    return total_flops / ((milliseconds / 1000.0) * 1e12)

  print('\nConfiguration:')
  print(f'  Sequences: {num_seqs}')
  print(f'  Heads: {num_heads}')
  print(f'  Head dimension: {head_dim}')
  print(f'  Page size: {page_size} tokens')
  print(f'  Sequence length: {seq_len} tokens')
  print(f'  Total pages: {total_pages}')

  # Run PyTorch reference
  print('\n' + '-' * 70)
  print('Running PyTorch reference implementation...')
  print('-' * 70)

  if verbosity >= 1:
    print(f'Benchmarking with {100 if seq_len <= 128 else 20} iterations (3 warmup runs)')

  # Warmup
  for _ in range(3):
    _ = paged_attention_torch(query, key_cache, value_cache, block_tables, context_lens, scale)
  torch.cuda.synchronize()

  # Time PyTorch implementation (fewer iterations for larger inputs)
  num_iterations = 100 if seq_len <= 128 else 20
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(num_iterations):
    output_ref = paged_attention_torch(query, key_cache, value_cache, block_tables, context_lens, scale)
  torch.cuda.synchronize()
  pytorch_time = (time.perf_counter() - start) / num_iterations * 1000  # Convert to ms

  print(f'Output shape: {output_ref.shape}')
  if verbosity >= 1:
    print('Output sample (seq=0, head=0, first 8 dims):')
    print(f'  {output_ref[0, 0, :8].cpu().numpy()}')

  triton_time = None
  triton_max_diff = None
  triton_mean_diff = None
  output_triton = None

  print('\n' + '-' * 70)
  print('Running Triton kernel...')
  print('-' * 70)

  for _ in range(3):
    output_triton = paged_attention_triton(query, key_cache, value_cache, block_tables, context_lens, scale)
  torch.cuda.synchronize()

  start = time.perf_counter()
  for _ in range(num_iterations):
    output_triton = paged_attention_triton(query, key_cache, value_cache, block_tables, context_lens, scale)
  torch.cuda.synchronize()
  triton_time = (time.perf_counter() - start) / num_iterations * 1000

  triton_max_diff = torch.max(torch.abs(output_triton - output_ref)).item()
  triton_mean_diff = torch.mean(torch.abs(output_triton - output_ref)).item()

  if verbosity >= 1:
    print('Triton output sample (seq=0, head=0, first 8 dims):')
    print(f'  {output_triton[0, 0, :8].cpu().numpy()}')

  print('\n' + '-' * 70)
  print('Running CuTe DSL kernels...')
  print('-' * 70)

  cute_time = None
  cute_time_opt = None
  baseline_max_diff = None
  opt_max_diff = None
  baseline_mean_diff = None
  opt_mean_diff = None

  try:
    # Baseline (FP32 scalar kernel)
    output_cute = torch.zeros_like(query)
    output_cute_ = from_dlpack(output_cute, assumed_align=16)
    query_ = from_dlpack(query, assumed_align=16)
    key_cache_ = from_dlpack(key_cache, assumed_align=16)
    value_cache_ = from_dlpack(value_cache, assumed_align=16)
    block_tables_ = from_dlpack(block_tables, assumed_align=16)
    context_lens_ = from_dlpack(context_lens, assumed_align=16)

    paged_attention_baseline = cute.compile(
      paged_attention_cute_launch,
      output_cute_,
      query_,
      key_cache_,
      value_cache_,
      block_tables_,
      context_lens_,
      cutlass.Float32(scale),
      cutlass.Int32(num_seqs),
      cutlass.Int32(num_heads),
      cutlass.Int32(head_dim),
      cutlass.Int32(page_size),
      cutlass.Int32(max_num_pages),
    )

    for _ in range(3):
      paged_attention_baseline(
        output_cute_,
        query_,
        key_cache_,
        value_cache_,
        block_tables_,
        context_lens_,
        cutlass.Float32(scale),
        cutlass.Int32(num_seqs),
        cutlass.Int32(num_heads),
        cutlass.Int32(head_dim),
        cutlass.Int32(page_size),
        cutlass.Int32(max_num_pages),
      )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
      paged_attention_baseline(
        output_cute_,
        query_,
        key_cache_,
        value_cache_,
        block_tables_,
        context_lens_,
        cutlass.Float32(scale),
        cutlass.Int32(num_seqs),
        cutlass.Int32(num_heads),
        cutlass.Int32(head_dim),
        cutlass.Int32(page_size),
        cutlass.Int32(max_num_pages),
      )
    torch.cuda.synchronize()
    cute_time = (time.perf_counter() - start) / num_iterations * 1000

    # Optimized (FP16 tiled kernel)
    query_fp16 = query.to(torch.float16)
    key_cache_fp16 = key_cache.to(torch.float16)
    value_cache_fp16 = value_cache.to(torch.float16)
    output_opt_fp16 = torch.zeros_like(query_fp16)

    output_opt_ = from_dlpack(output_opt_fp16, assumed_align=16)
    query_fp16_ = from_dlpack(query_fp16, assumed_align=16)
    key_cache_fp16_ = from_dlpack(key_cache_fp16, assumed_align=16)
    value_cache_fp16_ = from_dlpack(value_cache_fp16, assumed_align=16)

    opt_launch = build_paged_attention_cute_optimized_launch(head_dim, page_size, max_num_pages, tile_tokens)
    paged_attention_opt = cute.compile(
      opt_launch,
      output_opt_,
      query_fp16_,
      key_cache_fp16_,
      value_cache_fp16_,
      block_tables_,
      context_lens_,
      cutlass.Float32(scale),
      cutlass.Int32(num_seqs),
      cutlass.Int32(num_heads),
    )

    for _ in range(3):
      paged_attention_opt(
        output_opt_,
        query_fp16_,
        key_cache_fp16_,
        value_cache_fp16_,
        block_tables_,
        context_lens_,
        cutlass.Float32(scale),
        cutlass.Int32(num_seqs),
        cutlass.Int32(num_heads),
      )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
      paged_attention_opt(
        output_opt_,
        query_fp16_,
        key_cache_fp16_,
        value_cache_fp16_,
        block_tables_,
        context_lens_,
        cutlass.Float32(scale),
        cutlass.Int32(num_seqs),
        cutlass.Int32(num_heads),
      )
    torch.cuda.synchronize()
    cute_time_opt = (time.perf_counter() - start) / num_iterations * 1000

    output_cute_opt = output_opt_fp16.to(torch.float32)

    print('Baseline CuTe output shape:', output_cute.shape)
    if verbosity >= 1:
      print('  sample:', output_cute[0, 0, :8].cpu().numpy())
    print('Optimized CuTe output shape:', output_cute_opt.shape)
    if verbosity >= 1:
      print('  sample:', output_cute_opt[0, 0, :8].cpu().numpy())

    baseline_max_diff = torch.max(torch.abs(output_cute - output_ref)).item()
    baseline_mean_diff = torch.mean(torch.abs(output_cute - output_ref)).item()
    opt_max_diff = torch.max(torch.abs(output_cute_opt - output_ref)).item()
    opt_mean_diff = torch.mean(torch.abs(output_cute_opt - output_ref)).item()

  except Exception as e:
    print(f'Error running CuTe DSL kernels: {e}')
    print('Note: CuTe DSL is in beta and may require specific CUDA/driver versions')

  if any(v is not None for v in (baseline_max_diff, opt_max_diff, triton_max_diff)):
    print('\n' + '-' * 70)
    print('Verification vs. PyTorch reference')
    print('-' * 70)
    tolerance = 1e-3
    diffs = []
    if baseline_max_diff is not None:
      print(f'Baseline CuTe max |Δ|: {baseline_max_diff:.6f} (mean {baseline_mean_diff:.6f})')
      diffs.append(baseline_max_diff)
    if opt_max_diff is not None:
      print(f'Optimized CuTe max |Δ|: {opt_max_diff:.6f} (mean {opt_mean_diff:.6f})')
      diffs.append(opt_max_diff)
    if triton_max_diff is not None:
      print(f'Triton kernel max |Δ|: {triton_max_diff:.6f} (mean {triton_mean_diff:.6f})')
      diffs.append(triton_max_diff)
    if diffs:
      status = 'PASSED' if all(diff < tolerance for diff in diffs) else 'FAILED'
      print(f'Comparison status: {status} (tolerance {tolerance})')

  print('\n' + '-' * 70)
  print('Performance Comparison')
  print('-' * 70)
  pytorch_tflops = tflops_from_ms(pytorch_time)
  if pytorch_tflops is not None:
    print(f'PyTorch reference: {pytorch_time:.3f} ms ({pytorch_tflops:.3f} TFLOP/s)')
  else:
    print(f'PyTorch reference: {pytorch_time:.3f} ms')
  if triton_time is not None:
    triton_tflops = tflops_from_ms(triton_time)
    line = f'Triton kernel:     {triton_time:.3f} ms'
    if triton_tflops is not None:
      line += f' ({triton_tflops:.3f} TFLOP/s, speedup {pytorch_time / triton_time:.2f}x)'
    else:
      line += f' (speedup {pytorch_time / triton_time:.2f}x)'
    print(line)
  if cute_time is not None:
    cute_tflops = tflops_from_ms(cute_time)
    line = f'CuTe baseline:     {cute_time:.3f} ms'
    if cute_tflops is not None:
      line += f' ({cute_tflops:.3f} TFLOP/s, speedup {pytorch_time / cute_time:.2f}x)'
    else:
      line += f' (speedup {pytorch_time / cute_time:.2f}x)'
    print(line)
  if cute_time_opt is not None:
    cute_opt_tflops = tflops_from_ms(cute_time_opt)
    speedups = [f'speedup vs PyTorch {pytorch_time / cute_time_opt:.2f}x']
    if cute_time is not None:
      speedups.append(f'vs baseline {cute_time / cute_time_opt:.2f}x')
    speedups_text = ', '.join(speedups)
    line = f'CuTe optimized:    {cute_time_opt:.3f} ms'
    if cute_opt_tflops is not None:
      line += f' ({cute_opt_tflops:.3f} TFLOP/s, {speedups_text})'
    else:
      line += f' ({speedups_text})'
    print(line)

  # Show memory efficiency benefits
  print('\n' + '=' * 70)
  print('Memory Efficiency Analysis')
  print('=' * 70)

  # Use actual sequence length as max, and simulate 25% utilization as average
  max_len = seq_len
  avg_len = max(seq_len // 4, page_size)

  print(f'\nScenario: {num_seqs} sequences with max_len={max_len}, avg_len={avg_len}')
  print(
    f'KV cache element size: {num_heads} heads × {head_dim} dim × 4 bytes = {num_heads * head_dim * 4} bytes/token'
  )

  # Traditional pre-allocated cache
  traditional_memory = num_seqs * max_len * num_heads * head_dim * 4 * 2  # 2 for K and V
  print('\nTraditional (pre-allocated):')
  print(f'  Memory: {num_seqs} seqs × {max_len} tokens × {num_heads * head_dim * 4 * 2} bytes')
  print(f'  Total: {traditional_memory / 1024 / 1024:.2f} MB')
  print(f'  Utilization: {avg_len / max_len * 100:.1f}%')
  print(f'  Wasted: {(max_len - avg_len) * num_seqs * num_heads * head_dim * 4 * 2 / 1024 / 1024:.2f} MB')

  # Paged attention
  pages_needed = num_seqs * ((avg_len + page_size - 1) // page_size)
  paged_memory = pages_needed * page_size * num_heads * head_dim * 4 * 2
  print('\nPaged attention:')
  print(f'  Pages needed: {pages_needed} pages × {page_size} tokens')
  print(f'  Memory: {paged_memory / 1024 / 1024:.2f} MB')
  print(f'  Savings: {(traditional_memory - paged_memory) / traditional_memory * 100:.1f}%')
  print(f'  Fragmentation: < {page_size} tokens per sequence')

  if verbosity >= 2:
    print('\n' + '=' * 70)
    print('Key Concepts Demonstrated')
    print('=' * 70)
    print("""
1. KV cache paging: Fixed-size pages eliminate memory fragmentation
2. Block tables: Logical-to-physical page mapping enables flexibility
3. Online softmax: Streaming computation with running max/sum statistics
4. Memory efficiency: Allocate only what's needed, share common prefixes
5. Dynamic growth: Add pages as sequences grow without reallocation

For production use, see:
- vLLM: https://github.com/vllm-project/vllm
- Flash Attention: https://github.com/Dao-AILab/flash-attention
- CUTLASS Examples: https://github.com/NVIDIA/cutlass/tree/main/examples/python
""")


def main():
  parser = argparse.ArgumentParser(description='Paged Attention: PyTorch vs CuTe DSL')
  parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (-v for version info, -vv for all details)')
  parser.add_argument('--seq-len', type=int, default=None, help='Sequence length (default: run multiple sizes)')
  parser.add_argument('--num-seqs', type=int, default=2, help='Number of sequences (default: 2)')
  parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads (default: 4)')
  parser.add_argument('--head-dim', type=int, default=64, help='Head dimension (default: 64)')
  parser.add_argument('--page-size', type=int, default=16, help='Page size in tokens (default: 16)')
  parser.add_argument('--tile-tokens', type=int, default=32, help='Optimized CuTe kernel tile size (default: 32)')
  args = parser.parse_args()

  if not torch.cuda.is_available():
    print('Error: CUDA not available')
    exit(1)

  if args.verbose >= 1:
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA version: {torch.version.cuda}')
    print()

  print('=' * 70)
  print('Paged Attention: PyTorch, CuTe DSL')
  print('=' * 70)

  # Run tests with different sequence lengths
  if args.seq_len is not None:
    # Single test with specified size
      test_paged_attention(
        verbosity=args.verbose,
        num_seqs=args.num_seqs,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        tile_tokens=args.tile_tokens,
      )
  else:
    # Run multiple sizes: small for correctness, large for performance
    test_configs = [
      {'seq_len': 32, 'label': 'Small (correctness check)'},
      {'seq_len': 4096, 'label': 'Large (performance test)'},
    ]

    for i, config in enumerate(test_configs):
      if i > 0:
        print('\n\n')
      print(f"Test {i + 1}/{len(test_configs)}: {config['label']}")
      print()
      test_paged_attention(
        verbosity=args.verbose,
        num_seqs=args.num_seqs,
        seq_len=config['seq_len'],
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        tile_tokens=args.tile_tokens,
      )

if __name__ == '__main__': main()
