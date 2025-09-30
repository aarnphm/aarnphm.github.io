// Example 9: Simplified Paged Attention with CuTe
// Demonstrates: KV cache paging, block tables, online softmax, CuTe tensor operations

#include "common.cuh"
#include <cute/tensor.hpp>

using namespace cute;

// Simplified paged attention kernel
// Each sequence's KV cache is split into fixed-size pages (blocks)
template<int HEAD_DIM, int PAGE_SIZE, int BLOCK_SIZE>
__global__ void paged_attention_kernel(
    float* __restrict__ out,              // [num_seqs, num_heads, head_dim]
    const float* __restrict__ query,      // [num_seqs, num_heads, head_dim]
    const float* __restrict__ key_cache,  // [num_pages, num_heads, page_size, head_dim]
    const float* __restrict__ value_cache,// [num_pages, num_heads, page_size, head_dim]
    const int* __restrict__ block_tables, // [num_seqs, max_num_pages]
    const int* __restrict__ context_lens, // [num_seqs]
    int num_seqs, int num_heads, int max_num_pages) {

  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int tid = threadIdx.x;

  if (seq_idx >= num_seqs || head_idx >= num_heads)
    return;

  const int context_len = context_lens[seq_idx];
  const int num_pages = (context_len + PAGE_SIZE - 1) / PAGE_SIZE;

  // Shared memory for current page's K and V
  __shared__ float smem_K[PAGE_SIZE][HEAD_DIM];
  __shared__ float smem_V[PAGE_SIZE][HEAD_DIM];

  // Load query for this sequence and head
  float q[HEAD_DIM / BLOCK_SIZE];
  const int q_offset = (seq_idx * num_heads + head_idx) * HEAD_DIM;

  #pragma unroll
  for (int i = 0; i < HEAD_DIM / BLOCK_SIZE; i++) {
    int d = tid + i * BLOCK_SIZE;
    if (d < HEAD_DIM) {
      q[i] = query[q_offset + d];
    }
  }

  // Online softmax statistics
  float row_max = -INFINITY;
  float row_sum = 0.0f;
  float acc[HEAD_DIM / BLOCK_SIZE] = {0.0f};

  // Process each page
  for (int page_idx = 0; page_idx < num_pages; page_idx++) {
    // Get physical page number from block table
    const int block_number = block_tables[seq_idx * max_num_pages + page_idx];

    // Compute tokens in this page
    int page_start = page_idx * PAGE_SIZE;
    int page_end = min(page_start + PAGE_SIZE, context_len);
    int tokens_in_page = page_end - page_start;

    // Load K page into shared memory
    const int k_page_offset = (block_number * num_heads + head_idx) * PAGE_SIZE * HEAD_DIM;

    for (int t = 0; t < tokens_in_page; t += BLOCK_SIZE / HEAD_DIM) {
      int token = t + tid / HEAD_DIM;
      int dim = tid % HEAD_DIM;

      if (token < tokens_in_page && dim < HEAD_DIM) {
        smem_K[token][dim] = key_cache[k_page_offset + token * HEAD_DIM + dim];
      }
    }

    // Load V page into shared memory
    const int v_page_offset = (block_number * num_heads + head_idx) * PAGE_SIZE * HEAD_DIM;

    for (int t = 0; t < tokens_in_page; t += BLOCK_SIZE / HEAD_DIM) {
      int token = t + tid / HEAD_DIM;
      int dim = tid % HEAD_DIM;

      if (token < tokens_in_page && dim < HEAD_DIM) {
        smem_V[token][dim] = value_cache[v_page_offset + token * HEAD_DIM + dim];
      }
    }

    __syncthreads();

    // Compute attention scores for this page
    float scores[PAGE_SIZE];

    for (int t = 0; t < tokens_in_page; t++) {
      float qk = 0.0f;

      #pragma unroll
      for (int i = 0; i < HEAD_DIM / BLOCK_SIZE; i++) {
        int d = tid + i * BLOCK_SIZE;
        if (d < HEAD_DIM) {
          qk += q[i] * smem_K[t][d];
        }
      }

      // Warp reduction to sum qk across threads
      #pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        qk += __shfl_down_sync(0xffffffff, qk, offset);
      }

      if (tid == 0) {
        scores[t] = qk / sqrtf(float(HEAD_DIM));
      }
    }

    __syncthreads();

    // Broadcast scores to all threads
    if (tid == 0) {
      for (int t = 0; t < tokens_in_page; t++) {
        // Online softmax update
        float prev_max = row_max;
        row_max = fmaxf(row_max, scores[t]);

        float exp_prev = expf(prev_max - row_max);
        float exp_curr = expf(scores[t] - row_max);

        row_sum = row_sum * exp_prev + exp_curr;
        scores[t] = exp_curr;

        // Update accumulator
        float scale = exp_prev;
        for (int i = 0; i < HEAD_DIM / BLOCK_SIZE; i++) {
          acc[i] *= scale;
        }
      }
    }

    __syncthreads();

    // Accumulate attention * V
    for (int t = 0; t < tokens_in_page; t++) {
      float attn_weight = (tid == 0) ? scores[t] : 0.0f;

      #pragma unroll
      for (int i = 0; i < HEAD_DIM / BLOCK_SIZE; i++) {
        int d = tid + i * BLOCK_SIZE;
        if (d < HEAD_DIM) {
          acc[i] += attn_weight * smem_V[t][d];
        }
      }
    }

    __syncthreads();
  }

  // Final normalization and write output
  if (tid == 0) {
    #pragma unroll
    for (int i = 0; i < HEAD_DIM / BLOCK_SIZE; i++) {
      acc[i] /= row_sum;
    }
  }

  __syncthreads();

  const int out_offset = (seq_idx * num_heads + head_idx) * HEAD_DIM;

  #pragma unroll
  for (int i = 0; i < HEAD_DIM / BLOCK_SIZE; i++) {
    int d = tid + i * BLOCK_SIZE;
    if (d < HEAD_DIM) {
      out[out_offset + d] = acc[i];
    }
  }
}

// Helper to visualize block table structure
void print_block_table_structure() {
  printf("\n=== Paged Attention Architecture ===\n\n");
  printf("Logical KV Cache -> Physical Pages via Block Table\n\n");

  printf("Example: 2 sequences, page_size=16 tokens\n\n");

  printf("Sequence 0 (48 tokens = 3 pages):\n");
  printf("  Logical:  [0..15] [16..31] [32..47]\n");
  printf("  Physical: Page 0   Page 7   Page 12\n");
  printf("  Block table[0] = [0, 7, 12]\n\n");

  printf("Sequence 1 (32 tokens = 2 pages):\n");
  printf("  Logical:  [0..15] [16..31]\n");
  printf("  Physical: Page 5   Page 9\n");
  printf("  Block table[1] = [5, 9]\n\n");

  printf("Benefits:\n");
  printf("1. No memory fragmentation (fixed page size)\n");
  printf("2. Dynamic growth (allocate pages as needed)\n");
  printf("3. Page sharing (common prefixes can share pages)\n");
  printf("4. Memory efficiency (no pre-allocation of max length)\n\n");
}

void test_paged_attention() {
  printf("\n=== Testing Simplified Paged Attention ===\n");

  const int num_seqs = 2;
  const int num_heads = 4;
  const int head_dim = 64;
  const int page_size = 16;
  const int max_num_pages = 4;

  // Sequence lengths
  int h_context_lens[num_seqs] = {32, 24};  // Seq0: 32 tokens, Seq1: 24 tokens

  // Block tables (simplified: sequential allocation)
  int h_block_tables[num_seqs * max_num_pages] = {
      0, 1, -1, -1,  // Seq0: uses pages 0, 1
      2, 3, -1, -1   // Seq1: uses pages 2, 3 (only needs 2 pages)
  };

  // Calculate total pages needed
  int total_pages = 4;

  // Allocate host memory
  float *h_query = (float*)malloc(num_seqs * num_heads * head_dim * sizeof(float));
  float *h_output = (float*)malloc(num_seqs * num_heads * head_dim * sizeof(float));

  // Initialize query
  init_array(h_query, num_seqs * num_heads * head_dim, 1.0f);

  // Allocate device memory
  float *d_query, *d_output, *d_key_cache, *d_value_cache;
  int *d_block_tables, *d_context_lens;

  CUDA_CHECK(cudaMalloc(&d_query, num_seqs * num_heads * head_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, num_seqs * num_heads * head_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_key_cache, total_pages * num_heads * page_size * head_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_value_cache, total_pages * num_heads * page_size * head_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_block_tables, num_seqs * max_num_pages * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_context_lens, num_seqs * sizeof(int)));

  // Initialize caches with dummy data
  CUDA_CHECK(cudaMemset(d_key_cache, 0, total_pages * num_heads * page_size * head_dim * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_value_cache, 0, total_pages * num_heads * page_size * head_dim * sizeof(float)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_query, h_query, num_seqs * num_heads * head_dim * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_block_tables, h_block_tables, num_seqs * max_num_pages * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_context_lens, h_context_lens, num_seqs * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Launch kernel
  dim3 grid(num_heads, num_seqs);
  dim3 block(256);

  GpuTimer timer;
  timer.start();

  paged_attention_kernel<head_dim, page_size, 256><<<grid, block>>>(
      d_output, d_query, d_key_cache, d_value_cache,
      d_block_tables, d_context_lens,
      num_seqs, num_heads, max_num_pages);

  timer.stop();
  CUDA_CHECK(cudaGetLastError());

  // Copy result back
  CUDA_CHECK(cudaMemcpy(h_output, d_output, num_seqs * num_heads * head_dim * sizeof(float),
                        cudaMemcpyDeviceToHost));

  printf("Kernel execution time: %.3f ms\n", timer.elapsed());
  printf("Configuration:\n");
  printf("  Sequences: %d\n", num_seqs);
  printf("  Heads: %d\n", num_heads);
  printf("  Head dimension: %d\n", head_dim);
  printf("  Page size: %d tokens\n", page_size);
  printf("  Context lengths: [%d, %d]\n", h_context_lens[0], h_context_lens[1]);

  // Cleanup
  free(h_query);
  free(h_output);
  CUDA_CHECK(cudaFree(d_query));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_key_cache));
  CUDA_CHECK(cudaFree(d_value_cache));
  CUDA_CHECK(cudaFree(d_block_tables));
  CUDA_CHECK(cudaFree(d_context_lens));

  printf("\nTest completed successfully!\n");
}

int main() {
  printf("=== CuTe Paged Attention ===\n");
  print_device_info();

  print_block_table_structure();
  test_paged_attention();

  printf("\n=== Key Concepts ===\n");
  printf("1. KV cache paging: Split cache into fixed-size pages (blocks)\n");
  printf("2. Block table: Maps logical sequence positions to physical pages\n");
  printf("3. Online softmax: Streaming computation with running max and sum\n");
  printf("4. Memory efficiency: Eliminates fragmentation, enables dynamic growth\n");
  printf("5. Page sharing: Multiple sequences can share common prefix pages\n");
  printf("\n");
  printf("This is a simplified version. Full implementation includes:\n");
  printf("- TMA for async memory copies\n");
  printf("- Warp specialization (producer/consumer)\n");
  printf("- Software pipelining\n");
  printf("- Optimized attention score computation\n");

  return 0;
}
