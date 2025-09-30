// Example 2: Vectorized Memory Access
// Demonstrates: float4 vectorized loads, memory bandwidth optimization

#include "common.cuh"
#include <stdio.h>

// Scalar loads (baseline)
__global__ void vector_add_scalar(const float *a, const float *b, float *c,
                                   int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

// Vectorized loads using float4
__global__ void vector_add_vectorized(const float *a, const float *b, float *c,
                                      int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int vec_idx = idx * 4; // Each thread processes 4 elements

  if (vec_idx + 3 < N) {
    // Load 4 floats (16 bytes) in single instruction
    float4 a_vec = *reinterpret_cast<const float4 *>(&a[vec_idx]);
    float4 b_vec = *reinterpret_cast<const float4 *>(&b[vec_idx]);

    // Compute
    float4 c_vec;
    c_vec.x = a_vec.x + b_vec.x;
    c_vec.y = a_vec.y + b_vec.y;
    c_vec.z = a_vec.z + b_vec.z;
    c_vec.w = a_vec.w + b_vec.w;

    // Store 4 floats in single instruction
    *reinterpret_cast<float4 *>(&c[vec_idx]) = c_vec;
  }
}

// CPU reference
void vector_add_cpu(const float *a, const float *b, float *c, int N) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  printf("=== Vectorized Memory Access ===\n");
  print_device_info();

  const int N = 1 << 24; // 16M elements (must be multiple of 4)
  const size_t bytes = N * sizeof(float);

  // Allocate host memory
  float *h_a = (float *)malloc(bytes);
  float *h_b = (float *)malloc(bytes);
  float *h_c_scalar = (float *)malloc(bytes);
  float *h_c_vectorized = (float *)malloc(bytes);
  float *h_c_ref = (float *)malloc(bytes);

  // Initialize
  init_array(h_a, N, 10.0f);
  init_array(h_b, N, 10.0f);

  // Allocate device memory
  float *d_a, *d_b, *d_c;
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
    int blocks = ((N / 4) + threads - 1) / threads; // N/4 because each thread handles 4 elements

    GpuTimer timer;
    timer.start();
    vector_add_vectorized<<<blocks, threads>>>(d_a, d_b, d_c, N);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_c_vectorized, d_c, bytes, cudaMemcpyDeviceToHost));

    printf("\nVectorized loads (float4):\n");
    printf("  Time: %.3f ms\n", timer.elapsed());
    printf("  Bandwidth: %.2f GB/s\n", (3 * bytes) / (timer.elapsed() * 1e6));
  }

  // Verify results
  vector_add_cpu(h_a, h_b, h_c_ref, N);
  bool scalar_correct = verify_results(h_c_scalar, h_c_ref, N);
  bool vectorized_correct = verify_results(h_c_vectorized, h_c_ref, N);

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
