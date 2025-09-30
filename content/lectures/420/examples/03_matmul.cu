// Example 3: Matrix Multiplication Optimization Progression
// Demonstrates: Naive → Tiled with shared memory → Memory coalescing

#include "common.cuh"
#include <stdio.h>

// Naive matrix multiplication: C = A * B
// Each thread computes one element of C
__global__ void matmul_naive(const float *A, const float *B, float *C, int M,
                             int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

// Tiled matrix multiplication with shared memory
#define TILE_SIZE 32

__global__ void matmul_tiled(const float *A, const float *B, float *C, int M,
                              int N, int K) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  // Loop over tiles
  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
    // Load tile into shared memory
    int a_col = t * TILE_SIZE + threadIdx.x;
    int b_row = t * TILE_SIZE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads();

    // Compute partial product for this tile
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// CPU reference implementation
void matmul_cpu(const float *A, const float *B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

int main() {
  printf("=== Matrix Multiplication ===\n");
  print_device_info();

  // Matrix dimensions (M×K) * (K×N) = (M×N)
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;

  const size_t bytes_A = M * K * sizeof(float);
  const size_t bytes_B = K * N * sizeof(float);
  const size_t bytes_C = M * N * sizeof(float);

  // Allocate host memory
  float *h_A = (float *)malloc(bytes_A);
  float *h_B = (float *)malloc(bytes_B);
  float *h_C_naive = (float *)malloc(bytes_C);
  float *h_C_tiled = (float *)malloc(bytes_C);
  float *h_C_ref = (float *)malloc(bytes_C);

  // Initialize
  init_array(h_A, M * K, 1.0f);
  init_array(h_B, K * N, 1.0f);

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

  // Test 1: Naive implementation
  {
    dim3 threads(32, 32);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    GpuTimer timer;
    timer.start();
    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytes_C, cudaMemcpyDeviceToHost));

    float flops = 2.0f * M * N * K; // 2 ops per multiply-add
    float gflops = (flops / 1e9) / (timer.elapsed() / 1000.0f);

    printf("\nNaive implementation:\n");
    printf("  Time: %.3f ms\n", timer.elapsed());
    printf("  Performance: %.2f GFLOP/s\n", gflops);
  }

  // Test 2: Tiled with shared memory
  {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    GpuTimer timer;
    timer.start();
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, bytes_C, cudaMemcpyDeviceToHost));

    float flops = 2.0f * M * N * K;
    float gflops = (flops / 1e9) / (timer.elapsed() / 1000.0f);

    printf("\nTiled with shared memory:\n");
    printf("  Time: %.3f ms\n", timer.elapsed());
    printf("  Performance: %.2f GFLOP/s\n", gflops);
  }

  // CPU reference (only for small matrices due to speed)
  if (M <= 512 && N <= 512 && K <= 512) {
    printf("\nComputing CPU reference...\n");
    matmul_cpu(h_A, h_B, h_C_ref, M, N, K);

    bool naive_correct = verify_results(h_C_naive, h_C_ref, M * N, 1e-3f);
    bool tiled_correct = verify_results(h_C_tiled, h_C_ref, M * N, 1e-3f);

    printf("\nVerification:\n");
    printf("  Naive: %s\n", naive_correct ? "PASSED" : "FAILED");
    printf("  Tiled: %s\n", tiled_correct ? "PASSED" : "FAILED");
  } else {
    printf("\nSkipping CPU verification (matrix too large)\n");
    // Just verify naive vs tiled
    bool consistent = verify_results(h_C_naive, h_C_tiled, M * N, 1e-3f);
    printf("Naive vs Tiled consistency: %s\n", consistent ? "PASSED" : "FAILED");
  }

  // Cleanup
  free(h_A);
  free(h_B);
  free(h_C_naive);
  free(h_C_tiled);
  free(h_C_ref);
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return 0;
}
