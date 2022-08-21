// Example 7: CuTe Swizzling for Bank Conflict Avoidance
// Demonstrates: Swizzle layouts, shared memory optimization

#include "common.cuh"
#include <cute/tensor.hpp>

using namespace cute;

// Matrix transpose WITHOUT swizzling (has bank conflicts)
template<int TILE_M, int TILE_N>
__global__ void transpose_no_swizzle(half const* A, half* B, int M, int N) {
  __shared__ half smem[TILE_M][TILE_N];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x * TILE_N;
  int by = blockIdx.y * TILE_M;

  // Load into shared memory (coalesced)
  if (by + ty < M && bx + tx < N) {
    smem[ty][tx] = A[(by + ty) * N + (bx + tx)];
  }
  __syncthreads();

  // Write to global (coalesced, but shared memory reads have conflicts!)
  int out_x = blockIdx.y * TILE_M + tx;
  int out_y = blockIdx.x * TILE_N + ty;
  if (out_x < M && out_y < N) {
    B[out_y * M + out_x] = smem[tx][ty];  // Bank conflicts here!
  }
}

// Matrix transpose WITH CuTe swizzling (no bank conflicts)
template<int TILE_M, int TILE_N>
__global__ void transpose_with_swizzle(half const* A, half* B, int M, int N) {
  // CuTe swizzle layout: Swizzle<3,3,3> for 128-byte swizzle
  using SmemLayout = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<Int<TILE_M>, Int<TILE_N>>,
             Stride<Int<TILE_N>, Int<1>>>{}));

  __shared__ __align__(16) half smem_data[TILE_M * TILE_N];
  auto smem_layout = SmemLayout{};
  auto smem = make_tensor(make_smem_ptr(smem_data), smem_layout);

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x * TILE_N;
  int by = blockIdx.y * TILE_M;

  // Load into shared memory with swizzled layout
  if (by + ty < M && bx + tx < N) {
    smem(ty, tx) = A[(by + ty) * N + (bx + tx)];
  }
  __syncthreads();

  // Write to global (no bank conflicts due to swizzling!)
  int out_x = blockIdx.y * TILE_M + tx;
  int out_y = blockIdx.x * TILE_N + ty;
  if (out_x < M && out_y < N) {
    B[out_y * M + out_x] = smem(tx, ty);
  }
}

// Demonstrate different swizzle patterns
__global__ void swizzle_demo() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("\n=== Swizzle Pattern Comparison ===\n\n");

    // No swizzle
    auto no_swizzle = make_layout(make_shape(Int<32>{}, Int<32>{}),
                                   make_stride(Int<32>{}, Int<1>{}));
    printf("No swizzle layout:\n");
    print(no_swizzle);
    printf("\n\n");

    // 32-byte swizzle (Swizzle<2,0,3>)
    auto swizzle_32 = composition(Swizzle<2, 0, 3>{}, no_swizzle);
    printf("32-byte swizzle <2,0,3>:\n");
    print(swizzle_32);
    printf("\n\n");

    // 128-byte swizzle (Swizzle<3,3,3>)
    auto swizzle_128 = composition(Swizzle<3, 3, 3>{}, no_swizzle);
    printf("128-byte swizzle <3,3,3>:\n");
    print(swizzle_128);
    printf("\n\n");

    // Show how indices map
    printf("Index mapping for 128-byte swizzle:\n");
    printf("Logical (0,0) -> Physical: %d\n", int(swizzle_128(0, 0)));
    printf("Logical (0,1) -> Physical: %d\n", int(swizzle_128(0, 1)));
    printf("Logical (1,0) -> Physical: %d\n", int(swizzle_128(1, 0)));
    printf("Logical (1,1) -> Physical: %d\n", int(swizzle_128(1, 1)));
  }
}

void test_transpose(int M, int N) {
  printf("\n=== Transpose Performance Comparison (%dx%d) ===\n", M, N);

  const size_t bytes = M * N * sizeof(half);
  half *h_A = (half*)malloc(bytes);
  half *h_B_no_swizzle = (half*)malloc(bytes);
  half *h_B_swizzle = (half*)malloc(bytes);

  init_array(h_A, M * N, __float2half(100.0f));

  half *d_A, *d_B;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

  const int TILE = 32;
  dim3 threads(TILE, TILE);
  dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  // Test 1: No swizzle
  {
    GpuTimer timer;
    timer.start();
    transpose_no_swizzle<TILE, TILE><<<blocks, threads>>>(d_A, d_B, M, N);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_B_no_swizzle, d_B, bytes, cudaMemcpyDeviceToHost));

    printf("\nNo swizzle:\n");
    printf("  Time: %.3f ms\n", timer.elapsed());
    printf("  Bandwidth: %.2f GB/s\n", (2 * bytes) / (timer.elapsed() * 1e6));
  }

  // Test 2: With swizzle
  {
    GpuTimer timer;
    timer.start();
    transpose_with_swizzle<TILE, TILE><<<blocks, threads>>>(d_A, d_B, M, N);
    timer.stop();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_B_swizzle, d_B, bytes, cudaMemcpyDeviceToHost));

    printf("\nWith CuTe swizzle:\n");
    printf("  Time: %.3f ms\n", timer.elapsed());
    printf("  Bandwidth: %.2f GB/s\n", (2 * bytes) / (timer.elapsed() * 1e6));
  }

  // Verify correctness
  bool no_swizzle_correct = true;
  bool swizzle_correct = true;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float expected = __half2float(h_A[i * N + j]);
      if (fabs(__half2float(h_B_no_swizzle[j * M + i]) - expected) > 1e-2)
        no_swizzle_correct = false;
      if (fabs(__half2float(h_B_swizzle[j * M + i]) - expected) > 1e-2)
        swizzle_correct = false;
    }
  }

  printf("\nVerification:\n");
  printf("  No swizzle: %s\n", no_swizzle_correct ? "PASSED" : "FAILED");
  printf("  With swizzle: %s\n", swizzle_correct ? "PASSED" : "FAILED");

  free(h_A);
  free(h_B_no_swizzle);
  free(h_B_swizzle);
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
}

int main() {
  printf("=== CuTe Swizzling and Bank Conflict Avoidance ===\n");
  print_device_info();

  // Show swizzle patterns
  swizzle_demo<<<1, 1>>>();
  CUDA_CHECK(cudaDeviceSynchronize());

  // Test transpose with different sizes
  test_transpose(4096, 4096);

  printf("\n=== Key Concepts ===\n");
  printf("1. Swizzle<B,M,S>: B=bits for swizzle width, M=mask bits, S=shift bits\n");
  printf("2. Swizzle<3,3,3>: 128-byte swizzle for FP16/BF16 tensors\n");
  printf("3. composition(Swizzle{}, Layout{}): Apply swizzle to layout\n");
  printf("4. Bank conflicts occur with column-major access to row-major shared memory\n");
  printf("5. Swizzling permutes addresses to spread accesses across banks\n");
  printf("\n");
  printf("Expected: 1.3-1.6× speedup with swizzling on Hopper\n");

  return 0;
}
