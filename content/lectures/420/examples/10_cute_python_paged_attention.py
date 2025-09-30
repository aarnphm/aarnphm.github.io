"""
Example 10: Paged Attention with CuTe DSL Python API
Demonstrates: Python-native GPU kernel development using CUTLASS 4.x CuTe DSL

Requirements:
    pip install nvidia-cutlass-dsl torch

Note: This requires CUTLASS 4.0+ with Python DSL support (currently in beta)
"""

import torch
import numpy as np

try:
  import cutlass
  import cutlass.cute as cute
  import cutlass.cute.math as cute_math
  from cutlass.cute.runtime import from_dlpack

  CUTE_AVAILABLE = True
except ImportError:
  print('Warning: nvidia-cutlass-dsl not installed')
  print('Install with: pip install nvidia-cutlass-dsl')
  CUTE_AVAILABLE = False


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


if CUTE_AVAILABLE:

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
    head_dim: cutlass.Constexpr,
    page_size: cutlass.Constexpr,
    max_num_pages: cutlass.Constexpr,
  ):
    """
    CuTe DSL implementation of paged attention.

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


def test_paged_attention():
  print('=' * 70)
  print('Testing Paged Attention: PyTorch Reference vs CuTe DSL')
  print('=' * 70)

  # Problem size
  num_seqs = 2
  num_heads = 4
  head_dim = 64
  page_size = 16
  max_num_pages = 4

  # Context lengths for each sequence
  context_lens = torch.tensor([32, 24], dtype=torch.int32, device='cuda')

  # Block tables (maps logical pages to physical pages)
  block_tables = torch.tensor(
    [
      [0, 1, -1, -1],  # Seq 0: 32 tokens = 2 pages (pages 0, 1)
      [2, 3, -1, -1],  # Seq 1: 24 tokens = 2 pages (pages 2, 3)
    ],
    dtype=torch.int32,
    device='cuda',
  )

  total_pages = 4

  # Allocate tensors
  query = torch.randn(num_seqs, num_heads, head_dim, dtype=torch.float32, device='cuda')
  key_cache = torch.randn(total_pages, num_heads, page_size, head_dim, dtype=torch.float32, device='cuda')
  value_cache = torch.randn(total_pages, num_heads, page_size, head_dim, dtype=torch.float32, device='cuda')

  scale = 1.0 / np.sqrt(head_dim)

  print('\nConfiguration:')
  print(f'  Sequences: {num_seqs}')
  print(f'  Heads: {num_heads}')
  print(f'  Head dimension: {head_dim}')
  print(f'  Page size: {page_size} tokens')
  print(f'  Context lengths: {context_lens.cpu().tolist()}')
  print(f'  Total pages: {total_pages}')

  # Run PyTorch reference
  print('\n' + '-' * 70)
  print('Running PyTorch reference implementation...')
  print('-' * 70)

  output_ref = paged_attention_torch(query, key_cache, value_cache, block_tables, context_lens, scale)

  print(f'Output shape: {output_ref.shape}')
  print('Output sample (seq=0, head=0, first 8 dims):')
  print(f'  {output_ref[0, 0, :8].cpu().numpy()}')

  # Run CuTe DSL kernel if available
  if CUTE_AVAILABLE:
    print('\n' + '-' * 70)
    print('Running CuTe DSL kernel...')
    print('-' * 70)

    output_cute = torch.zeros_like(query)

    # Compile and launch kernel
    grid = (num_heads, num_seqs, 1)
    block = (1, 1, 1)

    try:
      # Use cute.jit to compile and cache
      @cute.jit
      def launch_paged_attention():
        paged_attention_cute_kernel(
          from_dlpack(output_cute),
          from_dlpack(query),
          from_dlpack(key_cache),
          from_dlpack(value_cache),
          from_dlpack(block_tables),
          from_dlpack(context_lens),
          np.float32(scale),
          num_seqs,
          num_heads,
          head_dim,
          page_size,
          max_num_pages,
        ).launch(grid=grid, block=block)

      launch_paged_attention()

      print(f'Output shape: {output_cute.shape}')
      print('Output sample (seq=0, head=0, first 8 dims):')
      print(f'  {output_cute[0, 0, :8].cpu().numpy()}')

      # Compare results
      print('\n' + '-' * 70)
      print('Verification')
      print('-' * 70)

      max_diff = torch.max(torch.abs(output_cute - output_ref)).item()
      mean_diff = torch.mean(torch.abs(output_cute - output_ref)).item()

      print(f'Max absolute difference: {max_diff:.6f}')
      print(f'Mean absolute difference: {mean_diff:.6f}')

      tolerance = 1e-3
      if max_diff < tolerance:
        print(f'✓ PASSED (tolerance: {tolerance})')
      else:
        print(f'✗ FAILED (tolerance: {tolerance})')

    except Exception as e:
      print(f'Error running CuTe DSL kernel: {e}')
      print('Note: CuTe DSL is in beta and may require specific CUDA/driver versions')

  else:
    print('\n' + '-' * 70)
    print('CuTe DSL not available - showing PyTorch reference only')
    print('-' * 70)

  # Show memory efficiency benefits
  print('\n' + '=' * 70)
  print('Paged Attention Benefits')
  print('=' * 70)

  max_len = 2048
  avg_len = 512

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


if __name__ == '__main__':
  if not torch.cuda.is_available():
    print('Error: CUDA not available')
    exit(1)

  print(f'PyTorch version: {torch.__version__}')
  print(f'CUDA available: {torch.cuda.is_available()}')
  print(f'CUDA version: {torch.version.cuda}')

  if CUTE_AVAILABLE:
    print('CuTe DSL available: Yes')
    print('  Install: pip install nvidia-cutlass-dsl')
  else:
    print('CuTe DSL available: No (falling back to PyTorch reference)')

  test_paged_attention()
