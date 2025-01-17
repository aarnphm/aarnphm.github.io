// Optimized Paged Attention with CuTe
// Highlights:
//  - One-shot CTA per (sequence, head)
//  - CuTe tiled copies for vectorized gmem->smem staging
//  - Online softmax with lossless max/sum tracking
//  - Half-precision math with float accumulation

#include "common.cuh"
#include <cute/algorithm/for_each.hpp>
#include <cute/container/array.hpp>
#include <cute/layout.hpp>

using namespace cute;

namespace {

__device__ inline float warp_allreduce_max(float val) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

__device__ inline float warp_allreduce_sum(float val) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

}

template<int HEAD_DIM, int PAGE_SIZE, int CTA_TOKENS, int CTA_THREADS>
__global__ void paged_attention_cute(
  half * __restrict__ out,
  const half * __restrict__ query,
  const half * __restrict__ key_cache,
  const half * __restrict__ value_cache,
  const int * __restrict__ block_tables,
  const int * __restrict__ context_lens,
  int num_seqs,
  int num_heads,
  int max_num_pages) {
  constexpr int WarpSize = 32;
  static_assert(CTA_THREADS % WarpSize == 0, "CTA_THREADS must be warp multiple");
  static_assert(HEAD_DIM % 2 == 0, "HEAD_DIM must be even for half2 path");
  static_assert(CTA_TOKENS % 16 == 0, "CTA_TOKENS must tile page nicely");
  static_assert(HEAD_DIM % WarpSize == 0, "HEAD_DIM must be warp divisible");
  constexpr int PerLane = HEAD_DIM / WarpSize;

  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int lane = threadIdx.x & (WarpSize - 1);

  if (seq_idx >= num_seqs || head_idx >= num_heads) {
    return;
  }

  const int context_len = context_lens[seq_idx];
  if (context_len == 0) {
    return;
  }

  const int num_pages = (context_len + PAGE_SIZE - 1) / PAGE_SIZE;

  extern __shared__ half smem[];
  half *smem_k = smem;
  half *smem_v = smem + PAGE_SIZE * HEAD_DIM;

  const int q_offset = (seq_idx * num_heads + head_idx) * HEAD_DIM;
  cute::array<float, HEAD_DIM> q_reg;
  for_each(make_range(Int<0>{}, Int<HEAD_DIM>{}), [&](auto d) {
    int idx = d;
    q_reg[idx] = __half2float(query[q_offset + idx]);
  });

  const float inv_sqrt_dim = rsqrtf(static_cast<float>(HEAD_DIM));

  float row_max = -CUDART_INF_F;
  float row_sum = 0.f;
  cute::array<float, PerLane> acc_frag;
  for (int i = 0; i < PerLane; ++i) {
    acc_frag[i] = 0.f;
  }

  for (int page_idx = 0; page_idx < num_pages; ++page_idx) {
    const int block_number = block_tables[seq_idx * max_num_pages + page_idx];
    const int page_start = page_idx * PAGE_SIZE;
    const int page_end = min(page_start + PAGE_SIZE, context_len);
    const int tokens_in_page = page_end - page_start;

    const int kv_offset = (block_number * num_heads + head_idx) * PAGE_SIZE * HEAD_DIM;
    const int vec_width = HEAD_DIM / 2;
    const int total_vecs = PAGE_SIZE * vec_width;
    const __half2 *gK_vec = reinterpret_cast<const __half2 *>(key_cache + kv_offset);
    const __half2 *gV_vec = reinterpret_cast<const __half2 *>(value_cache + kv_offset);
    __half2 *sK_vec = reinterpret_cast<__half2 *>(smem_k);
    __half2 *sV_vec = reinterpret_cast<__half2 *>(smem_v);

    for (int idx = threadIdx.x; idx < total_vecs; idx += CTA_THREADS) {
      int token = idx / vec_width;
      if (token < tokens_in_page) {
        sK_vec[idx] = gK_vec[idx];
        sV_vec[idx] = gV_vec[idx];
      } else {
        sK_vec[idx] = __halves2half2(__float2half(0.f), __float2half(0.f));
        sV_vec[idx] = __halves2half2(__float2half(0.f), __float2half(0.f));
      }
    }

    __syncthreads();

    for (int tile_start = 0; tile_start < tokens_in_page; tile_start += CTA_TOKENS) {
      const int tile_tokens = min(CTA_TOKENS, tokens_in_page - tile_start);

      for (int t = 0; t < tile_tokens; ++t) {
        const half *k_row = &smem_k[(tile_start + t) * HEAD_DIM];
        const half *v_row = &smem_v[(tile_start + t) * HEAD_DIM];

        float thread_dot = 0.f;
        for (int i = 0; i < PerLane; ++i) {
          int dim = lane + i * WarpSize;
          float k_val = __half2float(k_row[dim]);
          thread_dot += q_reg[dim] * k_val;
        }
        float qk = warp_allreduce_sum(thread_dot);
        qk *= inv_sqrt_dim;

        float new_max = row_max;
        float exp_scale = 0.f;
        float exp_val = 0.f;
        float new_sum = row_sum;

        if (lane == 0) {
          new_max = fmaxf(row_max, qk);
          exp_scale = (row_sum == 0.f) ? 0.f : __expf(row_max - new_max);
          exp_val = __expf(qk - new_max);
          new_sum = row_sum * exp_scale + exp_val;
          row_max = new_max;
          row_sum = new_sum;
        }

        new_max = __shfl_sync(0xffffffff, new_max, 0);
        exp_scale = __shfl_sync(0xffffffff, exp_scale, 0);
        exp_val = __shfl_sync(0xffffffff, exp_val, 0);
        row_sum = __shfl_sync(0xffffffff, row_sum, 0);
        row_max = new_max;

        for (int i = 0; i < PerLane; ++i) {
          int dim = lane + i * WarpSize;
          float v_val = __half2float(v_row[dim]);
          acc_frag[i] = acc_frag[i] * exp_scale + exp_val * v_val;
        }
      }
    }

    __syncthreads();
  }

  if (row_sum == 0.f) {
    return;
  }

  const float inv_sum = 1.f / row_sum;
  const int out_offset = (seq_idx * num_heads + head_idx) * HEAD_DIM;

  for (int i = 0; i < PerLane; ++i) {
    int dim = lane + i * WarpSize;
    out[out_offset + dim] = __float2half(acc_frag[i] * inv_sum);
  }
}

constexpr int HEAD_DIM = 128;
constexpr int PAGE_SIZE = 128;
constexpr int CTA_TOKENS = 64;
constexpr int CTA_THREADS = 32;

void launch_paged_attention(
  half *out,
  const half *query,
  const half *key_cache,
  const half *value_cache,
  const int *block_tables,
  const int *context_lens,
  int num_seqs,
  int num_heads,
  int max_num_pages,
  cudaStream_t stream) {
  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(CTA_THREADS);
  size_t shared_bytes = 2 * PAGE_SIZE * HEAD_DIM * sizeof(half);
  paged_attention_cute<HEAD_DIM, PAGE_SIZE, CTA_TOKENS, CTA_THREADS>
    <<<grid, block, shared_bytes, stream>>>(
      out,
      query,
      key_cache,
      value_cache,
      block_tables,
      context_lens,
      num_seqs,
      num_heads,
      max_num_pages);
}

int main() {
  constexpr int num_seqs = 4;
  constexpr int num_heads = 8;
  constexpr int max_pages = 16;

  size_t out_bytes = num_seqs * num_heads * HEAD_DIM * sizeof(half);
  size_t q_bytes = out_bytes;
  size_t cache_bytes = max_pages * num_heads * PAGE_SIZE * HEAD_DIM * sizeof(half);
  size_t table_bytes = num_seqs * max_pages * sizeof(int);
  size_t ctx_bytes = num_seqs * sizeof(int);

  half *h_out = (half *)malloc(out_bytes);
  half *h_query = (half *)malloc(q_bytes);
  half *h_key = (half *)malloc(cache_bytes);
  half *h_value = (half *)malloc(cache_bytes);
  int *h_table = (int *)malloc(table_bytes);
  int *h_ctx = (int *)malloc(ctx_bytes);

  init_array(h_query, q_bytes / sizeof(half), __float2half(1.f));
  init_array(h_key, cache_bytes / sizeof(half), __float2half(1.f));
  init_array(h_value, cache_bytes / sizeof(half), __float2half(1.f));
  for (int s = 0; s < num_seqs; ++s) {
    h_ctx[s] = PAGE_SIZE * (s + 1);
    for (int p = 0; p < max_pages; ++p) {
      h_table[s * max_pages + p] = p;
    }
  }

  half *d_out, *d_query, *d_key, *d_value;
  int *d_table, *d_ctx;
  CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
  CUDA_CHECK(cudaMalloc(&d_query, q_bytes));
  CUDA_CHECK(cudaMalloc(&d_key, cache_bytes));
  CUDA_CHECK(cudaMalloc(&d_value, cache_bytes));
  CUDA_CHECK(cudaMalloc(&d_table, table_bytes));
  CUDA_CHECK(cudaMalloc(&d_ctx, ctx_bytes));

  CUDA_CHECK(cudaMemcpy(d_query, h_query, q_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_key, h_key, cache_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_value, h_value, cache_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_table, h_table, table_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ctx, h_ctx, ctx_bytes, cudaMemcpyHostToDevice));

  launch_paged_attention(d_out, d_query, d_key, d_value, d_table, d_ctx,
                         num_seqs, num_heads, max_pages, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));

  free(h_out);
  free(h_query);
  free(h_key);
  free(h_value);
  free(h_table);
  free(h_ctx);

  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_query));
  CUDA_CHECK(cudaFree(d_key));
  CUDA_CHECK(cudaFree(d_value));
  CUDA_CHECK(cudaFree(d_table));
  CUDA_CHECK(cudaFree(d_ctx));

  return 0;
}
