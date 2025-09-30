// Example 1: Basic Vector Addition
// Demonstrates: Basic kernel launch, thread indexing, memory management

#include "common.cuh"
#include <stdio.h>

__global__ void vector_add(const float *a, const float *b, float *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

// CPU reference implementation
void vector_add_cpu(const float *a, const float *b, float *c, int N) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  printf("=== Vector Addition ===\n");
  print_device_info();

  const int N = 1 << 20; // 1M elements
  const size_t bytes = N * sizeof(float);

  // Allocate host memory
  float *h_a = (float *)malloc(bytes);
  float *h_b = (float *)malloc(bytes);
  float *h_c = (float *)malloc(bytes);
  float *h_c_ref = (float *)malloc(bytes);

  // Initialize input arrays
  init_array(h_a, N, 10.0f);
  init_array(h_b, N, 10.0f);

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  // Copy inputs to device
  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  // Launch configuration
  int threads_per_block = 256;
  int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

  printf("Launch config: %d blocks × %d threads = %d total threads\n",
         blocks_per_grid, threads_per_block, blocks_per_grid * threads_per_block);

  // Launch kernel
  GpuTimer timer;
  timer.start();
  vector_add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, N);
  timer.stop();
  CUDA_CHECK(cudaGetLastError());

  // Copy result back
  CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

  printf("Kernel execution time: %.3f ms\n", timer.elapsed());
  printf("Throughput: %.2f GB/s\n",
         (3 * bytes) / (timer.elapsed() * 1e6)); // 2 reads + 1 write

  // Verify result
  vector_add_cpu(h_a, h_b, h_c_ref, N);
  bool correct = verify_results(h_c, h_c_ref, N);
  printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

  // Cleanup
  free(h_a);
  free(h_b);
  free(h_c);
  free(h_c_ref);
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return 0;
}
