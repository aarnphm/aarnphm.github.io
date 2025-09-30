// Example 2: Vectorized Memory Access
// Demonstrates: half2 vectorized loads, memory bandwidth optimization

#include "common.cuh"
#include <stdio.h>

// Scalar loads (baseline)
__global__ void vector_add_scalar(const half *a, const half *b, half *c,
                                   int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    c[idx] = __hadd(a[idx], b[idx]);
  }
}

// Vectorized loads using half2
__global__ void vector_add_vectorized(const half *a, const half *b, half *c,
                                      int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int vec_idx = idx * 2; // Each thread processes 2 elements

  if (vec_idx + 1 < N) {
    // Load 2 halfs (4 bytes) in single instruction
    half2 a_vec = *reinterpret_cast<const half2 *>(&a[vec_idx]);
    half2 b_vec = *reinterpret_cast<const half2 *>(&b[vec_idx]);

    // Compute
    half2 c_vec = __hadd2(a_vec, b_vec);

    // Store 2 halfs in single instruction
    *reinterpret_cast<half2 *>(&c[vec_idx]) = c_vec;
  }
}

// CPU reference
void vector_add_cpu(const half *a, const half *b, half *c, int N) {
  for (int i = 0; i < N; i++) {
    c[i] = __hadd(a[i], b[i]);
  }
}

int main() {
  printf("=== Vectorized Memory Access ===\n");
  print_device_info();

  const int N = 1 << 24; // 16M elements (must be multiple of 2)
  const size_t bytes = N * sizeof(half);

  // Allocate host memory
  half *h_a = (half *)malloc(bytes);
  half *h_b = (half *)malloc(bytes);
  half *h_c_scalar = (half *)malloc(bytes);
  half *h_c_vectorized = (half *)malloc(bytes);
  half *h_c_ref = (half *)malloc(bytes);

  // Initialize
  init_array(h_a, N, __float2half(10.0f));
  init_array(h_b, N, __float2half(10.0f));

  // Allocate device memory
  half *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  // Test 1: Scalar loads
  {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    GpuTimer timer;
    timer.start();
    vector_add_scalar<<<blocks, threads>>>(d_a, d_b, d_c, N);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_c_scalar, d_c, bytes, cudaMemcpyDeviceToHost));

    printf("\nScalar loads:\n");
    printf("  Time: %.3f ms\n", timer.elapsed());
    printf("  Bandwidth: %.2f GB/s\n", (3 * bytes) / (timer.elapsed() * 1e6));
  }

  // Test 2: Vectorized loads
  {
    int threads = 256;
    int blocks = ((N / 2) + threads - 1) / threads; // N/2 because each thread handles 2 elements

    GpuTimer timer;
    timer.start();
    vector_add_vectorized<<<blocks, threads>>>(d_a, d_b, d_c, N);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_c_vectorized, d_c, bytes, cudaMemcpyDeviceToHost));

    printf("\nVectorized loads (half2):\n");
    printf("  Time: %.3f ms\n", timer.elapsed());
    printf("  Bandwidth: %.2f GB/s\n", (3 * bytes) / (timer.elapsed() * 1e6));
  }

  // Verify results
  vector_add_cpu(h_a, h_b, h_c_ref, N);
  bool scalar_correct = verify_results(h_c_scalar, h_c_ref, N, __float2half(1e-3f));
  bool vectorized_correct = verify_results(h_c_vectorized, h_c_ref, N, __float2half(1e-3f));

  printf("\nVerification:\n");
  printf("  Scalar: %s\n", scalar_correct ? "PASSED" : "FAILED");
  printf("  Vectorized: %s\n", vectorized_correct ? "PASSED" : "FAILED");

  // Cleanup
  free(h_a);
  free(h_b);
  free(h_c_scalar);
  free(h_c_vectorized);
  free(h_c_ref);
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return 0;
}
