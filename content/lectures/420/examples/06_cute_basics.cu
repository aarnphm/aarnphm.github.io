// Example 6: CuTe DSL Basics
// Demonstrates: Layout algebra, tensor creation, indexing

#include "common.cuh"
#include <cute/tensor.hpp>

using namespace cute;

// Simple kernel using CuTe layouts
__global__ void vector_add_cute(half const* A_ptr, half const* B_ptr,
                                 half* C_ptr, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    // Create CuTe layouts
    auto shape = make_shape(N);
    auto stride = make_stride(1);  // Contiguous
    auto layout = make_layout(shape, stride);

    // Create tensors with global memory pointers
    auto A = make_tensor(make_gmem_ptr(A_ptr), layout);
    auto B = make_tensor(make_gmem_ptr(B_ptr), layout);
    auto C = make_tensor(make_gmem_ptr(C_ptr), layout);

    // Access elements using CuTe indexing
    C(idx) = A(idx) + B(idx);
  }
}

// Demonstrate 2D layout operations
__global__ void matrix_copy_cute(half const* src, half* dst, int M, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    // 2D layout: Shape (M, N), Stride (N, 1) for row-major
    auto layout = make_layout(make_shape(M, N), make_stride(N, 1));

    auto src_tensor = make_tensor(make_gmem_ptr(src), layout);
    auto dst_tensor = make_tensor(make_gmem_ptr(dst), layout);

    // 2D indexing
    dst_tensor(row, col) = src_tensor(row, col);
  }
}

// Demonstrate layout composition
__global__ void layout_composition_demo(half const* input, half* output,
                                        int M, int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid == 0) {
    // Create base layout
    auto base_layout = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));

    // Create tiled layout: divide into 4×4 tiles
    auto tiled_layout = logical_divide(base_layout, make_tile(Int<4>{}, Int<4>{}));

    // Print layout information (only thread 0)
    printf("Base layout: ");
    print(base_layout);
    printf("\n");

    printf("Tiled layout: ");
    print(tiled_layout);
    printf("\n");

    printf("Base layout size: %d\n", int(size(base_layout)));
    printf("Tiled layout size: %d\n", int(size(tiled_layout)));
  }
}

void test_vector_add() {
  printf("\n=== Test 1: Vector Addition with CuTe ===\n");

  const int N = 1024;
  const size_t bytes = N * sizeof(half);

  half *h_A = (half*)malloc(bytes);
  half *h_B = (half*)malloc(bytes);
  half *h_C = (half*)malloc(bytes);

  init_array(h_A, N, __float2half(10.0f));
  init_array(h_B, N, __float2half(10.0f));

  half *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  vector_add_cute<<<blocks, threads>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // Verify
  bool correct = true;
  for (int i = 0; i < N; i++) {
    float expected = __half2float(h_A[i]) + __half2float(h_B[i]);
    if (fabs(__half2float(h_C[i]) - expected) > 1e-2) {
      correct = false;
      break;
    }
  }

  printf("Result: %s\n", correct ? "PASSED" : "FAILED");

  free(h_A);
  free(h_B);
  free(h_C);
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
}

void test_matrix_copy() {
  printf("\n=== Test 2: Matrix Copy with 2D Layout ===\n");

  const int M = 128;
  const int N = 128;
  const size_t bytes = M * N * sizeof(half);

  half *h_src = (half*)malloc(bytes);
  half *h_dst = (half*)malloc(bytes);

  init_array(h_src, M * N, __float2half(100.0f));

  half *d_src, *d_dst;
  CUDA_CHECK(cudaMalloc(&d_src, bytes));
  CUDA_CHECK(cudaMalloc(&d_dst, bytes));

  CUDA_CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

  dim3 threads(16, 16);
  dim3 blocks((N + 15) / 16, (M + 15) / 16);

  matrix_copy_cute<<<blocks, threads>>>(d_src, d_dst, M, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost));

  bool correct = verify_results(h_dst, h_src, M * N);
  printf("Result: %s\n", correct ? "PASSED" : "FAILED");

  free(h_src);
  free(h_dst);
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));
}

void test_layout_composition() {
  printf("\n=== Test 3: Layout Composition ===\n");

  half *d_dummy;
  CUDA_CHECK(cudaMalloc(&d_dummy, 1024 * sizeof(half)));

  layout_composition_demo<<<1, 32>>>(d_dummy, d_dummy, 16, 16);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_dummy));
}

int main() {
  printf("=== CuTe DSL Basics ===\n");
  print_device_info();

  test_vector_add();
  test_matrix_copy();
  test_layout_composition();

  printf("\n=== Key CuTe Concepts Demonstrated ===\n");
  printf("1. Layout creation: make_layout(shape, stride)\n");
  printf("2. Tensor creation: make_tensor(pointer, layout)\n");
  printf("3. Indexing: tensor(i) or tensor(i, j)\n");
  printf("4. Layout composition: logical_divide for tiling\n");
  printf("5. Compile-time sizes: Int<N>{} for static shapes\n");

  return 0;
}
