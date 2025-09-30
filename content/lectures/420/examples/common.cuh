#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Timing utilities
class GpuTimer {
  cudaEvent_t start_event, stop_event;

public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
  }

  ~GpuTimer() {
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
  }

  void start() { CUDA_CHECK(cudaEventRecord(start_event, 0)); }

  void stop() { CUDA_CHECK(cudaEventRecord(stop_event, 0)); }

  float elapsed() {
    float elapsed_ms;
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    return elapsed_ms;
  }
};

// Initialize array with random values
template <typename T> void init_array(T *arr, size_t n, T max_val = 1.0) {
  for (size_t i = 0; i < n; i++) {
    arr[i] = static_cast<T>(rand()) / RAND_MAX * max_val;
  }
}

// Verify results
template <typename T>
bool verify_results(const T *result, const T *expected, size_t n,
                    T tolerance = 1e-5) {
  for (size_t i = 0; i < n; i++) {
    if (fabs(result[i] - expected[i]) > tolerance) {
      fprintf(stderr, "Mismatch at index %zu: got %f, expected %f\n", i,
              (float)result[i], (float)expected[i]);
      return false;
    }
  }
  return true;
}

// Print device properties
void print_device_info() {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  printf("Device: %s\n", prop.name);
  printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("  Total Global Memory: %.2f GB\n",
         prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
  printf("  Shared Memory per Block: %.2f KB\n",
         prop.sharedMemPerBlock / 1024.0);
  printf("  Registers per Block: %d\n", prop.regsPerBlock);
  printf("  Warp Size: %d\n", prop.warpSize);
  printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("  Max Thread Dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0],
         prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("  Max Grid Dimensions: (%d, %d, %d)\n", prop.maxGridSize[0],
         prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
  printf("\n");
}
