#include <cuda_runtime.h>

__global__ void apply_values_kernel(
    const float* __restrict__ S,  // [n, n]
    const float* __restrict__ V,  // [n, d_v]
    float* __restrict__ O,        // [n, d_v]
    int n, int d_v) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;  // row in S / O
  int t = blockIdx.y * blockDim.y + threadIdx.y;  // value dim
  if (i >= n || t >= d_v) return;

  float acc = 0.f;
  for (int j = 0; j < n; ++j) acc += S[i*n + j] * V[j*d_v + t];
  O[i*d_v + t] = acc;
}

extern "C" void apply_values(const float* S, const float* V, float* O, int n, int d_v) {
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (d_v + block.y - 1) / block.y);
  apply_values_kernel<<<grid, block>>>(S, V, O, n, d_v);
}

