// Example 8: CuTe Hierarchical Tiling with local_tile and local_partition
// Demonstrates: Decomposing work across grid, block, and thread hierarchy

#include "common.cuh"
#include <cute/tensor.hpp>

using namespace cute;

// GEMM using CuTe hierarchical tiling
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_cute_tiled(float const* A_ptr, float const* B_ptr,
                                 float* C_ptr, int M, int N, int K) {
  // Define shared memory tiles
  __shared__ float smem_A[TILE_M][TILE_K];
  __shared__ float smem_B[TILE_K][TILE_N];

  // Global tensor layouts
  auto A_layout = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
  auto B_layout = make_layout(make_shape(K, N), make_stride(N, Int<1>{}));
  auto C_layout = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));

  auto A = make_tensor(make_gmem_ptr(A_ptr), A_layout);
  auto B = make_tensor(make_gmem_ptr(B_ptr), B_layout);
  auto C = make_tensor(make_gmem_ptr(C_ptr), C_layout);

  // Shared memory tensor layouts
  auto sA_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
                                make_stride(Int<TILE_K>{}, Int<1>{}));
  auto sB_layout = make_layout(make_shape(Int<TILE_K>{}, Int<TILE_N>{}),
                                make_stride(Int<TILE_N>{}, Int<1>{}));

  auto sA = make_tensor(make_smem_ptr(smem_A), sA_layout);
  auto sB = make_tensor(make_smem_ptr(smem_B), sB_layout);

  // CTA (thread block) tiling
  auto cta_tile = make_tile(Int<TILE_M>{}, Int<TILE_N>{}, Int<TILE_K>{});
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

  // Partition global tensors by CTA
  auto gA_cta = local_tile(A, cta_tile, cta_coord, Step<_1, X, _1>{});
  auto gB_cta = local_tile(B, cta_tile, cta_coord, Step<X, _1, _1>{});
  auto gC_cta = local_tile(C, cta_tile, cta_coord, Step<_1, _1, X>{});

  // Thread layout within block (32x8 = 256 threads)
  auto thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}),
                                 make_stride(Int<8>{}, Int<1>{}));
  int tid = threadIdx.x;

  // Partition shared memory by threads
  auto sA_thr = local_partition(sA, thr_layout, tid);
  auto sB_thr = local_partition(sB, thr_layout, tid);

  // Accumulator
  float acc = 0.0f;

  // Main loop over K dimension
  int num_k_tiles = (K + TILE_K - 1) / TILE_K;

  for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
    // Load A tile from global to shared memory
    for (int i = 0; i < size(sA_thr); i++) {
      if (get<0>(gA_cta(k_tile).shape()) > 0 && get<1>(gA_cta(k_tile).shape()) > 0) {
        int m_idx = i / TILE_K;
        int k_idx = i % TILE_K;
        if (m_idx < TILE_M && k_idx < TILE_K) {
          sA_thr(i) = gA_cta(k_tile)(m_idx, k_idx);
        }
      }
    }

    // Load B tile from global to shared memory
    for (int i = 0; i < size(sB_thr); i++) {
      if (get<0>(gB_cta(k_tile).shape()) > 0 && get<1>(gB_cta(k_tile).shape()) > 0) {
        int k_idx = i / TILE_N;
        int n_idx = i % TILE_N;
        if (k_idx < TILE_K && n_idx < TILE_N) {
          sB_thr(i) = gB_cta(k_tile)(k_idx, n_idx);
        }
      }
    }

    __syncthreads();

    // Compute: each thread computes one element
    int m_local = threadIdx.x / 8;
    int n_local = threadIdx.x % 8;

    if (m_local < TILE_M && n_local < TILE_N) {
      for (int k = 0; k < TILE_K; k++) {
        acc += sA(m_local, k) * sB(k, n_local);
      }
    }

    __syncthreads();
  }

  // Write result
  int m_global = blockIdx.x * TILE_M + threadIdx.x / 8;
  int n_global = blockIdx.y * TILE_N + threadIdx.x % 8;

  if (m_global < M && n_global < N) {
    C(m_global, n_global) = acc;
  }
}

// Demonstrate tiling concepts
__global__ void tiling_demo() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("\n=== Hierarchical Tiling Demonstration ===\n\n");

    // Global tensor: 16×16 matrix
    auto global_shape = make_shape(Int<16>{}, Int<16>{});
    auto global_layout = make_layout(global_shape, make_stride(Int<16>{}, Int<1>{}));

    printf("1. Global layout (16×16 matrix):\n");
    print(global_layout);
    printf("\n\n");

    // Tile into 4×4 blocks
    auto cta_tile = make_tile(Int<4>{}, Int<4>{});
    auto tiled = logical_divide(global_layout, cta_tile);

    printf("2. After tiling into 4×4 blocks:\n");
    print(tiled);
    printf("\n\n");

    // Create a coordinate for block (1,1)
    auto cta_coord = make_coord(1, 1);
    printf("3. Select block at coordinate (1,1):\n");
    printf("   This block spans elements [4:8, 4:8]\n\n");

    // Thread partitioning within block
    auto thread_layout = make_layout(make_shape(Int<2>{}, Int<2>{}));
    printf("4. Thread layout within block (2×2 threads):\n");
    print(thread_layout);
    printf("\n\n");

    printf("Key concepts:\n");
    printf("- local_tile: Extract a tile from global tensor at given CTA coordinate\n");
    printf("- local_partition: Distribute tile elements across threads\n");
    printf("- Step<_1,X,_1>: Partition mode 0 and 2, broadcast mode 1\n");
  }
}

void test_gemm_cute(int M, int N, int K) {
  printf("\n=== GEMM with CuTe Tiling (%dx%dx%d) ===\n", M, N, K);

  const size_t bytes_A = M * K * sizeof(float);
  const size_t bytes_B = K * N * sizeof(float);
  const size_t bytes_C = M * N * sizeof(float);

  float *h_A = (float*)malloc(bytes_A);
  float *h_B = (float*)malloc(bytes_B);
  float *h_C = (float*)malloc(bytes_C);
  float *h_C_ref = (float*)malloc(bytes_C);

  init_array(h_A, M * K, 1.0f);
  init_array(h_B, K * N, 1.0f);

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

  const int TILE_M = 32;
  const int TILE_N = 32;
  const int TILE_K = 32;

  dim3 threads(256);  // 32×8 threads
  dim3 blocks((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

  GpuTimer timer;
  timer.start();
  gemm_cute_tiled<TILE_M, TILE_N, TILE_K><<<blocks, threads>>>(
      d_A, d_B, d_C, M, N, K);
  timer.stop();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

  float flops = 2.0f * M * N * K;
  float gflops = (flops / 1e9) / (timer.elapsed() / 1000.0f);

  printf("Time: %.3f ms\n", timer.elapsed());
  printf("Performance: %.2f GFLOP/s\n", gflops);

  // CPU reference (for small sizes)
  if (M <= 256 && N <= 256 && K <= 256) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
          sum += h_A[i * K + k] * h_B[k * N + j];
        }
        h_C_ref[i * N + j] = sum;
      }
    }

    bool correct = verify_results(h_C, h_C_ref, M * N, 1e-3f);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
  }

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
}

int main() {
  printf("=== CuTe Hierarchical Tiling ===\n");
  print_device_info();

  // Show tiling concepts
  tiling_demo<<<1, 1>>>();
  CUDA_CHECK(cudaDeviceSynchronize());

  // Test GEMM with different sizes
  test_gemm_cute(256, 256, 256);
  test_gemm_cute(512, 512, 512);

  printf("\n=== Key CuTe Tiling Concepts ===\n");
  printf("1. make_tile(M, N, K): Create tile shape\n");
  printf("2. local_tile(tensor, tile, coord, step): Extract CTA tile\n");
  printf("3. local_partition(tensor, layout, tid): Partition across threads\n");
  printf("4. Step<_1,X,_1>: Control which modes are partitioned vs broadcast\n");
  printf("5. Hierarchical decomposition: Grid → CTA → Thread\n");

  return 0;
}
