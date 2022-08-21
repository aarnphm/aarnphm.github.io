#include <cuda_runtime.h>
#include <cmath>

__global__ void qk_scores_kernel(
    const float* __restrict__ Q,  // [n, d]
    const float* __restrict__ K,  // [n, d]
    float* __restrict__ S,        // [n, n]
    int n, int d, float inv_sqrt_d) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;  // query index i
  int col = blockIdx.y * blockDim.y + threadIdx.y;  // key   index j
  if (row >= n || col >= n) return;

  const float* q = Q + row * d;
  const float* k = K + col * d;

  float acc = 0.f;
  for (int t = 0; t < d; ++t) acc += q[t] * k[t];

  S[row * n + col] = acc * inv_sqrt_d;
}

extern "C" void qk_scores(const float* Q, const float* K, float* S, int n, int d) {
  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
  float inv_sqrt_d = 1.f / std::sqrtf((float)d);
  qk_scores_kernel<<<grid, block>>>(Q, K, S, n, d, inv_sqrt_d);
}
