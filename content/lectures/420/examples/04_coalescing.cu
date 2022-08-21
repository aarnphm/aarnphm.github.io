// Example 4: Memory Coalescing
// Demonstrates: Coalesced vs uncoalesced memory access patterns

#include "common.cuh"
#include <stdio.h>

// Coalesced access: consecutive threads read consecutive addresses
__global__ void transpose_coalesced(const half *input, half *output, int width,
                                     int height) {
  __shared__ half tile[32][33]; // +1 to avoid bank conflicts

  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;

  // Coalesced read from input
  if (x < width && y < height) {
    tile[threadIdx.y][threadIdx.x] = input[y * width + x];
  }

  __syncthreads();

  // Coalesced write to output (transposed coordinates)
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;

  if (x < height && y < width) {
    output[y * height + x] = tile[threadIdx.x][threadIdx.y];
  }
}

// Uncoalesced access: naive transpose
__global__ void transpose_naive(const half *input, half *output, int width,
                                 int height) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;

  if (x < width && y < height) {
    // This causes uncoalesced writes!
    output[x * height + y] = input[y * width + x];
  }
}

// CPU reference
void transpose_cpu(const half *input, half *output, int width, int height) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      output[x * height + y] = input[y * width + x];
    }
  }
}

int main() {
  printf("=== Memory Coalescing Demonstration ===\n");
  print_device_info();

  const int width = 4096;
  const int height = 4096;
  const size_t bytes = width * height * sizeof(half);

  // Allocate host memory
  half *h_input = (half *)malloc(bytes);
  half *h_output_naive = (half *)malloc(bytes);
  half *h_output_coalesced = (half *)malloc(bytes);
  half *h_output_ref = (half *)malloc(bytes);

  // Initialize
  init_array(h_input, width * height, __float2half(100.0f));

  // Allocate device memory
  half *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));

  CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

  dim3 threads(32, 32);
  dim3 blocks((width + 31) / 32, (height + 31) / 32);

  // Test 1: Naive (uncoalesced)
  {
    GpuTimer timer;
    timer.start();
    transpose_naive<<<blocks, threads>>>(d_input, d_output, width, height);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(
        cudaMemcpy(h_output_naive, d_output, bytes, cudaMemcpyDeviceToHost));

    printf("\nNaive transpose (uncoalesced):\n");
    printf("  Time: %.3f ms\n", timer.elapsed());
    printf("  Bandwidth: %.2f GB/s\n", (2 * bytes) / (timer.elapsed() * 1e6));
  }

  // Test 2: Coalesced with shared memory
  {
    GpuTimer timer;
    timer.start();
    transpose_coalesced<<<blocks, threads>>>(d_input, d_output, width, height);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_output_coalesced, d_output, bytes,
                          cudaMemcpyDeviceToHost));

    printf("\nCoalesced transpose (with shared memory):\n");
    printf("  Time: %.3f ms\n", timer.elapsed());
    printf("  Bandwidth: %.2f GB/s\n", (2 * bytes) / (timer.elapsed() * 1e6));
  }

  // CPU reference
  transpose_cpu(h_input, h_output_ref, width, height);

  // Verify
  bool naive_correct = verify_results(h_output_naive, h_output_ref,
                                      width * height, __float2half(1e-3f));
  bool coalesced_correct = verify_results(h_output_coalesced, h_output_ref,
                                          width * height, __float2half(1e-3f));

  printf("\nVerification:\n");
  printf("  Naive: %s\n", naive_correct ? "PASSED" : "FAILED");
  printf("  Coalesced: %s\n", coalesced_correct ? "PASSED" : "FAILED");

  // Cleanup
  free(h_input);
  free(h_output_naive);
  free(h_output_coalesced);
  free(h_output_ref);
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  return 0;
}
