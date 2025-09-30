// Example 5: Parallel Reduction
// Demonstrates: Shared memory, synchronization, dynamic shared memory allocation

#include "common.cuh"
#include <stdio.h>

// Reduction kernel using shared memory
__global__ void reduce_sum(const half *input, half *output, int N) {
  extern __shared__ half smem[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  smem[tid] = (idx < N) ? input[idx] : __float2half(0.0f);
  __syncthreads();

  // Reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] = __hadd(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }

  // Write result for this block
  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}

// Optimized reduction with less divergence
__global__ void reduce_sum_optimized(const half *input, half *output, int N) {
  extern __shared__ half smem[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  // Load and perform first level of reduction during global load
  half sum = __float2half(0.0f);
  if (idx < N)
    sum = __hadd(sum, input[idx]);
  if (idx + blockDim.x < N)
    sum = __hadd(sum, input[idx + blockDim.x]);

  smem[tid] = sum;
  __syncthreads();

  // Reduction in shared memory with sequential addressing
  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      smem[tid] = __hadd(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }

  // Unroll last warp (no __syncthreads needed within warp)
  if (tid < 32) {
    volatile half *vsmem = smem;
    if (blockDim.x >= 64)
      vsmem[tid] = __hadd(vsmem[tid], vsmem[tid + 32]);
    if (blockDim.x >= 32)
      vsmem[tid] = __hadd(vsmem[tid], vsmem[tid + 16]);
    if (blockDim.x >= 16)
      vsmem[tid] = __hadd(vsmem[tid], vsmem[tid + 8]);
    if (blockDim.x >= 8)
      vsmem[tid] = __hadd(vsmem[tid], vsmem[tid + 4]);
    if (blockDim.x >= 4)
      vsmem[tid] = __hadd(vsmem[tid], vsmem[tid + 2]);
    if (blockDim.x >= 2)
      vsmem[tid] = __hadd(vsmem[tid], vsmem[tid + 1]);
  }

  if (tid == 0) {
    output[blockIdx.x] = smem[0];
  }
}

// CPU reference
half reduce_sum_cpu(const half *data, int N) {
  half sum = __float2half(0.0f);
  for (int i = 0; i < N; i++) {
    sum = __hadd(sum, data[i]);
  }
  return sum;
}

int main() {
  printf("=== Parallel Reduction ===\n");
  print_device_info();

  const int N = 1 << 24; // 16M elements
  const size_t bytes = N * sizeof(half);

  // Allocate host memory
  half *h_input = (half *)malloc(bytes);

  // Initialize with ones for easy verification
  for (int i = 0; i < N; i++) {
    h_input[i] = __float2half(1.0f);
  }

  // Allocate device memory
  half *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));

  const int threads = 256;
  const int blocks_basic = (N + threads - 1) / threads;
  const int blocks_optimized = (N + threads * 2 - 1) / (threads * 2);

  CUDA_CHECK(cudaMalloc(&d_output, blocks_basic * sizeof(half)));

  CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

  // Test 1: Basic reduction
  {
    const size_t shared_mem = threads * sizeof(half);

    GpuTimer timer;
    timer.start();

    // First pass: reduce to blocks_basic elements
    reduce_sum<<<blocks_basic, threads, shared_mem>>>(d_input, d_output, N);

    // If needed, recursively reduce the output
    int remaining = blocks_basic;
    half *d_temp = d_output;
    while (remaining > 1) {
      int next_blocks = (remaining + threads - 1) / threads;
      reduce_sum<<<next_blocks, threads, shared_mem>>>(d_temp, d_temp,
                                                        remaining);
      remaining = next_blocks;
    }

    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    half h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_output, sizeof(half),
                          cudaMemcpyDeviceToHost));

    printf("\nBasic reduction:\n");
    printf("  Time: %.3f ms\n", timer.elapsed());
    printf("  Bandwidth: %.2f GB/s\n", bytes / (timer.elapsed() * 1e6));
    printf("  Result: %.0f (expected: %d)\n", __half2float(h_result), N);
    printf("  Accuracy: %s\n", (fabs(__half2float(h_result) - N) < 1.0f) ? "PASSED" : "FAILED");
  }

  // Test 2: Optimized reduction
  {
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaMalloc(&d_output, blocks_optimized * sizeof(half)));

    const size_t shared_mem = threads * sizeof(half);

    GpuTimer timer;
    timer.start();

    reduce_sum_optimized<<<blocks_optimized, threads, shared_mem>>>(
        d_input, d_output, N);

    int remaining = blocks_optimized;
    half *d_temp = d_output;
    while (remaining > 1) {
      int next_blocks = (remaining + threads * 2 - 1) / (threads * 2);
      reduce_sum_optimized<<<next_blocks, threads, shared_mem>>>(d_temp, d_temp,
                                                                  remaining);
      remaining = next_blocks;
    }

    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    half h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_output, sizeof(half),
                          cudaMemcpyDeviceToHost));

    printf("\nOptimized reduction:\n");
    printf("  Time: %.3f ms\n", timer.elapsed());
    printf("  Bandwidth: %.2f GB/s\n", bytes / (timer.elapsed() * 1e6));
    printf("  Result: %.0f (expected: %d)\n", __half2float(h_result), N);
    printf("  Accuracy: %s\n", (fabs(__half2float(h_result) - N) < 1.0f) ? "PASSED" : "FAILED");
  }

  // CPU reference
  half cpu_result = reduce_sum_cpu(h_input, N);
  printf("\nCPU reference: %.0f\n", __half2float(cpu_result));

  // Cleanup
  free(h_input);
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  return 0;
}
