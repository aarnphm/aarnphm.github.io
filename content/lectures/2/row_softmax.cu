#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void row_softmax_kernel(float* __restrict__ S, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float m = -FLT_MAX;
  for (int j = 0; j < n; ++j) m = fmaxf(m, S[i*n + j]);

  float sum = 0.f;
  for (int j = 0; j < n; ++j) {
    float e = expf(S[i*n + j] - m);
    S[i*n + j] = e;
    sum += e;
  }

  float inv = 1.f / fmaxf(sum, 1e-12f);
  for (int j = 0; j < n; ++j) S[i*n + j] *= inv;
}

extern "C" void row_softmax(float* S, int n) {
  int block = 256;
  int grid  = (n + block - 1) / block;
  row_softmax_kernel<<<grid, block>>>(S, n);
}
