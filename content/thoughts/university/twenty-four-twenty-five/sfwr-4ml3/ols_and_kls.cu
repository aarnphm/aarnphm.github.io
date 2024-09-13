// CUDA OLS and Kernelized Least Squares Regression
#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>

// Error checking macros
#define CUDA_CHECK(err)                                                        \
  if (err != cudaSuccess) {                                                    \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line "      \
              << __LINE__ << std::endl;                                        \
    exit(EXIT_FAILURE);                                                        \
  }

#define CUBLAS_CHECK(err)                                                      \
  if (err != CUBLAS_STATUS_SUCCESS) {                                          \
    std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl;             \
    exit(EXIT_FAILURE);                                                        \
  }

#define CUSOLVER_CHECK(err)                                                    \
  if (err != CUSOLVER_STATUS_SUCCESS) {                                        \
    std::cerr << "cuSolver Error at line " << __LINE__ << std::endl;           \
    exit(EXIT_FAILURE);                                                        \
  }

// Generate data on the host
void generate_data(int n, std::vector<double> &X, std::vector<double> &Y) {
  X.resize(n);
  Y.resize(n);
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  for (int i = 0; i < n; ++i) {
    X[i] = static_cast<double>(std::rand()) / RAND_MAX * 5.0;
  }

  std::sort(X.begin(), X.end());

  for (int i = 0; i < n; ++i) {
    double xi = X[i];
    double yi = 10 + std::pow(xi - 0.1, 3) - 5 * std::pow(xi - 0.5, 2) +
                10 * xi + 5 * sin(5 * xi);
    double noise = 0.1 * (static_cast<double>(std::rand()) / RAND_MAX - 0.5);
    Y[i] = yi + noise;
  }
}

// Kernel to augment data by adding a column of ones
__global__ void augment_data_kernel(const double *X, double *X_aug, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    X_aug[idx * 2] = X[idx];  // Original data
    X_aug[idx * 2 + 1] = 1.0; // Augmented column of ones
  }
}

// Compute the Gaussian kernel matrix
__global__ void gaussian_kernel_kernel(const double *X1, const double *X2,
                                       double *K, int n1, int n2,
                                       double sigma) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n1 * n2) {
    int i = idx / n2;
    int j = idx % n2;
    double diff = X1[i] - X2[j];
    K[i * n2 + j] = exp(-(diff * diff) / (sigma * sigma));
  }
}

// Solve OLS regression
void solve_ols(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle,
               double *d_X_train, double *d_Y_train, double *d_X_test,
               double *d_Y_LS_train, double *d_Y_LS_test, int n_train,
               int n_test, int d, double alpha) {
  // Compute XtX = X_train^T * X_train
  double *d_XtX;
  CUDA_CHECK(cudaMalloc(&d_XtX, d * d * sizeof(double)));

  const double alpha_blas = 1.0;
  const double beta_blas = 0.0;

  CUBLAS_CHECK(cublasDsyrk(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, d,
                           n_train, &alpha_blas, d_X_train, n_train, &beta_blas,
                           d_XtX, d));

  // Add regularization term alpha * I
  std::vector<double> h_XtX(d * d, 0);
  CUDA_CHECK(cudaMemcpy(h_XtX.data(), d_XtX, d * d * sizeof(double),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < d; ++i) {
    h_XtX[i * d + i] += alpha;
  }

  CUDA_CHECK(cudaMemcpy(d_XtX, h_XtX.data(), d * d * sizeof(double),
                        cudaMemcpyHostToDevice));

  // Compute XtY = X_train^T * Y_train
  double *d_XtY;
  CUDA_CHECK(cudaMalloc(&d_XtY, d * sizeof(double)));

  CUBLAS_CHECK(cublasDgemv(cublasHandle, CUBLAS_OP_T, n_train, d, &alpha_blas,
                           d_X_train, n_train, d_Y_train, 1, &beta_blas, d_XtY,
                           1));

  // Solve (XtX + alpha*I) * W = XtY
  int *devInfo;
  CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));

  int workspace_size = 0;
  CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
      cusolverHandle, CUBLAS_FILL_MODE_UPPER, d, d_XtX, d, &workspace_size));

  double *workspace;
  CUDA_CHECK(cudaMalloc(&workspace, workspace_size * sizeof(double)));

  // Cholesky factorization
  CUSOLVER_CHECK(cusolverDnDpotrf(cusolverHandle, CUBLAS_FILL_MODE_UPPER, d,
                                  d_XtX, d, workspace, workspace_size,
                                  devInfo));

  int devInfo_h = 0;
  CUDA_CHECK(
      cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (devInfo_h != 0) {
    std::cerr << "Cholesky factorization failed\n";
    exit(EXIT_FAILURE);
  }

  // Solve for W
  double *d_W;
  CUDA_CHECK(cudaMalloc(&d_W, d * sizeof(double)));
  CUDA_CHECK(
      cudaMemcpy(d_W, d_XtY, d * sizeof(double), cudaMemcpyDeviceToDevice));

  CUSOLVER_CHECK(cusolverDnDpotrs(cusolverHandle, CUBLAS_FILL_MODE_UPPER, d, 1,
                                  d_XtX, d, d_W, d, devInfo));

  CUDA_CHECK(
      cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (devInfo_h != 0) {
    std::cerr << "Solving linear system failed\n";
    exit(EXIT_FAILURE);
  }

  // Compute Y_LS_train = X_train * W
  CUBLAS_CHECK(cublasDgemv(cublasHandle, CUBLAS_OP_N, n_train, d, &alpha_blas,
                           d_X_train, n_train, d_W, 1, &beta_blas, d_Y_LS_train,
                           1));

  // Compute Y_LS_test = X_test * W
  CUBLAS_CHECK(cublasDgemv(cublasHandle, CUBLAS_OP_N, n_test, d, &alpha_blas,
                           d_X_test, n_test, d_W, 1, &beta_blas, d_Y_LS_test,
                           1));

  // Cleanup
  CUDA_CHECK(cudaFree(d_XtX));
  CUDA_CHECK(cudaFree(d_XtY));
  CUDA_CHECK(cudaFree(d_W));
  CUDA_CHECK(cudaFree(devInfo));
  CUDA_CHECK(cudaFree(workspace));
}

// Solve Kernelized Least Squares
void solve_kernel_ls(cublasHandle_t cublasHandle,
                     cusolverDnHandle_t cusolverHandle, double *d_K_train,
                     double *d_Y_train, double *d_K_test, double *d_Y_LS_train,
                     double *d_Y_LS_test, int n_train, int n_test,
                     double alpha) {
  // Compute (K_train + alpha * I)
  std::vector<double> h_K_train(n_train * n_train);
  CUDA_CHECK(cudaMemcpy(h_K_train.data(), d_K_train,
                        n_train * n_train * sizeof(double),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < n_train; ++i) {
    h_K_train[i * n_train + i] += alpha;
  }

  CUDA_CHECK(cudaMemcpy(d_K_train, h_K_train.data(),
                        n_train * n_train * sizeof(double),
                        cudaMemcpyHostToDevice));

  // Solve (K_train + alpha * I) * a = Y_train
  int *devInfo;
  CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));

  int workspace_size = 0;
  CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
      cusolverHandle, CUBLAS_FILL_MODE_UPPER, n_train, d_K_train, n_train,
      &workspace_size));

  double *workspace;
  CUDA_CHECK(cudaMalloc(&workspace, workspace_size * sizeof(double)));

  // Cholesky factorization
  CUSOLVER_CHECK(cusolverDnDpotrf(cusolverHandle, CUBLAS_FILL_MODE_UPPER,
                                  n_train, d_K_train, n_train, workspace,
                                  workspace_size, devInfo));

  int devInfo_h = 0;
  CUDA_CHECK(
      cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (devInfo_h != 0) {
    std::cerr << "Cholesky factorization failed\n";
    exit(EXIT_FAILURE);
  }

  // Solve for a
  double *d_a;
  CUDA_CHECK(cudaMalloc(&d_a, n_train * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_a, d_Y_train, n_train * sizeof(double),
                        cudaMemcpyDeviceToDevice));

  CUSOLVER_CHECK(cusolverDnDpotrs(cusolverHandle, CUBLAS_FILL_MODE_UPPER,
                                  n_train, 1, d_K_train, n_train, d_a, n_train,
                                  devInfo));

  CUDA_CHECK(
      cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (devInfo_h != 0) {
    std::cerr << "Solving linear system failed\n";
    exit(EXIT_FAILURE);
  }

  // Compute Y_LS_train = K_train * a
  const double alpha_blas = 1.0;
  const double beta_blas = 0.0;
  CUBLAS_CHECK(cublasDgemv(cublasHandle, CUBLAS_OP_N, n_train, n_train,
                           &alpha_blas, d_K_train, n_train, d_a, 1, &beta_blas,
                           d_Y_LS_train, 1));

  // Compute Y_LS_test = K_test^T * a
  CUBLAS_CHECK(cublasDgemv(cublasHandle, CUBLAS_OP_T, n_train, n_test,
                           &alpha_blas, d_K_test, n_train, d_a, 1, &beta_blas,
                           d_Y_LS_test, 1));

  // Cleanup
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(devInfo));
  CUDA_CHECK(cudaFree(workspace));
}

// Compute Mean Squared Error
double compute_mse(cublasHandle_t cublasHandle, double *d_Y_pred,
                   double *d_Y_true, int n) {
  double *d_diff;
  CUDA_CHECK(cudaMalloc(&d_diff, n * sizeof(double)));

  const double alpha_blas = -1.0;
  CUDA_CHECK(cudaMemcpy(d_diff, d_Y_pred, n * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CUBLAS_CHECK(
      cublasDaxpy(cublasHandle, n, &alpha_blas, d_Y_true, 1, d_diff, 1));

  double mse = 0.0;
  CUBLAS_CHECK(cublasDnrm2(cublasHandle, n, d_diff, 1, &mse));
  mse = (mse * mse) / n;

  CUDA_CHECK(cudaFree(d_diff));

  return mse;
}

// Main function
int main() {
  int n_train = 50;
  int n_test = 1000;
  double alpha = 0.1;
  double sigma = 0.1;

  // Generate data
  std::vector<double> h_X_train, h_Y_train;
  std::vector<double> h_X_test, h_Y_test;
  generate_data(n_train, h_X_train, h_Y_train);
  generate_data(n_test, h_X_test, h_Y_test);

  // Allocate device memory
  double *d_X_train, *d_Y_train, *d_X_test, *d_Y_test;
  CUDA_CHECK(cudaMalloc(&d_X_train, n_train * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_Y_train, n_train * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_X_test, n_test * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_Y_test, n_test * sizeof(double)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_X_train, h_X_train.data(), n_train * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Y_train, h_Y_train.data(), n_train * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_X_test, h_X_test.data(), n_test * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Y_test, h_Y_test.data(), n_test * sizeof(double),
                        cudaMemcpyHostToDevice));

  // Augment data for non-homogenous line
  // Add ones column
  double *d_X_train_aug;
  double *d_X_test_aug;
  CUDA_CHECK(cudaMalloc(&d_X_train_aug, n_train * 2 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_X_test_aug, n_test * 2 * sizeof(double)));

  // Kernel to augment data
  int blockSize = 256;
  int gridSize_train = (n_train + blockSize - 1) / blockSize;
  int gridSize_test = (n_test + blockSize - 1) / blockSize;

  augment_data_kernel<<<gridSize_train, blockSize>>>(d_X_train, d_X_train_aug,
                                                     n_train);
  augment_data_kernel<<<gridSize_test, blockSize>>>(d_X_test, d_X_test_aug,
                                                    n_test);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Initialize cuBLAS and cuSolver handles
  cublasHandle_t cublasHandle;
  cusolverDnHandle_t cusolverHandle;
  CUBLAS_CHECK(cublasCreate(&cublasHandle));
  CUSOLVER_CHECK(cusolverDnCreate(&cusolverHandle));

  // Allocate memory for predictions
  double *d_Y_LS_train;
  double *d_Y_LS_test;
  CUDA_CHECK(cudaMalloc(&d_Y_LS_train, n_train * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_Y_LS_test, n_test * sizeof(double)));

  // Solve OLS for non-homogenous line (d = 2)
  std::cout << "Non-homogenous line/hyperplane:" << std::endl;
  solve_ols(cublasHandle, cusolverHandle, d_X_train_aug, d_Y_train,
            d_X_test_aug, d_Y_LS_train, d_Y_LS_test, n_train, n_test, 2, alpha);

  // Compute MSE
  double mse_train =
      compute_mse(cublasHandle, d_Y_LS_train, d_Y_train, n_train);
  double mse_test = compute_mse(cublasHandle, d_Y_LS_test, d_Y_test, n_test);

  std::cout << "Mean Squared Error (MSE) of train data: " << mse_train
            << std::endl;
  std::cout << "Mean Squared Error (MSE) of test data: " << mse_test
            << std::endl;

  // Kernelized Least Squares with Gaussian Kernel
  // Compute K_train and K_test
  double *d_K_train;
  double *d_K_test;
  CUDA_CHECK(cudaMalloc(&d_K_train, n_train * n_train * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_K_test, n_train * n_test * sizeof(double)));

  int gridSize_K_train = (n_train * n_train + blockSize - 1) / blockSize;
  gaussian_kernel_kernel<<<gridSize_K_train, blockSize>>>(
      d_X_train, d_X_train, d_K_train, n_train, n_train, sigma);
  CUDA_CHECK(cudaDeviceSynchronize());

  int gridSize_K_test = (n_train * n_test + blockSize - 1) / blockSize;
  gaussian_kernel_kernel<<<gridSize_K_test, blockSize>>>(
      d_X_train, d_X_test, d_K_test, n_train, n_test, sigma);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Solve Kernelized Least Squares
  std::cout << "\nKernelized Least Squares with Gaussian Kernel:" << std::endl;
  solve_kernel_ls(cublasHandle, cusolverHandle, d_K_train, d_Y_train, d_K_test,
                  d_Y_LS_train, d_Y_LS_test, n_train, n_test, alpha);

  // Compute MSE
  mse_train = compute_mse(cublasHandle, d_Y_LS_train, d_Y_train, n_train);
  mse_test = compute_mse(cublasHandle, d_Y_LS_test, d_Y_test, n_test);

  std::cout << "Mean Squared Error (MSE) of train data: " << mse_train
            << std::endl;
  std::cout << "Mean Squared Error (MSE) of test data: " << mse_test
            << std::endl;

  // Cleanup
  CUDA_CHECK(cudaFree(d_X_train));
  CUDA_CHECK(cudaFree(d_Y_train));
  CUDA_CHECK(cudaFree(d_X_test));
  CUDA_CHECK(cudaFree(d_Y_test));
  CUDA_CHECK(cudaFree(d_X_train_aug));
  CUDA_CHECK(cudaFree(d_X_test_aug));
  CUDA_CHECK(cudaFree(d_Y_LS_train));
  CUDA_CHECK(cudaFree(d_Y_LS_test));
  CUDA_CHECK(cudaFree(d_K_train));
  CUDA_CHECK(cudaFree(d_K_test));
  CUBLAS_CHECK(cublasDestroy(cublasHandle));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolverHandle));

  return 0;
}
