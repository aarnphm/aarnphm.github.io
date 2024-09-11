#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <iostream>

// CUDA kernel for Gaussian kernel computation
__global__ void gaussian_kernel_kernel(double* X1, double* X2, double* K, int m, int n, double sigma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        double diff = X1[i] - X2[j];
        K[i + j*m] = exp(-diff * diff / (2 * sigma * sigma));
    }
}

// Function to compute Gaussian kernel matrix on GPU
void gaussian_kernel(double* X1, double* X2, double* K, int m, int n, double sigma) {
    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    gaussian_kernel_kernel<<<grid, block>>>(X1, X2, K, m, n, sigma);
    cudaDeviceSynchronize();
}

// Function to solve least squares problem on GPU using cuBLAS and cuSOLVER
void solve_ls(double* X, double* y, double* w, int m, int n, double alpha) {
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);

    double *d_X, *d_y, *d_w, *d_XT, *d_XTX, *d_XTy;
    cudaMalloc(&d_X, m*n*sizeof(double));
    cudaMalloc(&d_y, m*sizeof(double));
    cudaMalloc(&d_w, n*sizeof(double));
    cudaMalloc(&d_XT, n*m*sizeof(double));
    cudaMalloc(&d_XTX, n*n*sizeof(double));
    cudaMalloc(&d_XTy, n*sizeof(double));

    cudaMemcpy(d_X, X, m*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, m*sizeof(double), cudaMemcpyHostToDevice);

    double alpha1 = 1.0, beta1 = 0.0;
    cublasDgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha1, d_X, m, &beta1, d_X, m, d_XT, n);
    cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, m, &alpha1, d_XT, n, d_X, m, &beta1, d_XTX, n);

    double *d_reg;
    cudaMalloc(&d_reg, n*n*sizeof(double));
    cudaMemset(d_reg, 0, n*n*sizeof(double));
    thrust::device_vector<double> regDiagVec(n, alpha);
    thrust::copy(regDiagVec.begin(), regDiagVec.end(), thrust::device_pointer_cast(d_reg));
    cublasDaxpy(cublasHandle, n*n, &alpha1, d_reg, 1, d_XTX, 1);

    cublasDgemv(cublasHandle, CUBLAS_OP_N, n, m, &alpha1, d_XT, n, d_y, 1, &beta1, d_XTy, 1);

    int *d_info, info;
    cudaMalloc(&d_info, sizeof(int));
    cusolverDnDpotrf(cusolverHandle, CUBLAS_FILL_MODE_LOWER, n, d_XTX, n, d_w, n, d_info);
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    if (info == 0) {
        cusolverDnDpotrs(cusolverHandle, CUBLAS_FILL_MODE_LOWER, n, 1, d_XTX, n, d_XTy, n, d_info);
        cudaMemcpy(w, d_XTy, n*sizeof(double), cudaMemcpyDeviceToHost);
    } else {
        std::cout << "Error: Cholesky factorization failed" << std::endl;
    }

    cudaFree(d_X); cudaFree(d_y); cudaFree(d_w); cudaFree(d_XT); cudaFree(d_XTX); cudaFree(d_XTy);
    cudaFree(d_reg); cudaFree(d_info);
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);
}

// Function to solve kernel least squares problem on GPU
void solve_kernel_ls(double* K, double* y, double* a, int n, double alpha) {
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);

    double *d_K, *d_y, *d_a, *d_reg;
    cudaMalloc(&d_K, n*n*sizeof(double));
    cudaMalloc(&d_y, n*sizeof(double));
    cudaMalloc(&d_a, n*sizeof(double));
    cudaMalloc(&d_reg, n*n*sizeof(double));

    cudaMemcpy(d_K, K, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemset(d_reg, 0, n*n*sizeof(double));
    thrust::device_vector<double> regDiagVec(n, alpha);
    thrust::copy(regDiagVec.begin(), regDiagVec.end(), thrust::device_pointer_cast(d_reg));
    double alpha1 = 1.0;
    cublasDaxpy(cublasHandle, n*n, &alpha1, d_reg, 1, d_K, 1);

    int *d_info, info;
    cudaMalloc(&d_info, sizeof(int));
    cusolverDnDpotrf(cusolverHandle, CUBLAS_FILL_MODE_LOWER, n, d_K, n, d_a, n, d_info);
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    if (info == 0) {
        cusolverDnDpotrs(cusolverHandle, CUBLAS_FILL_MODE_LOWER, n, 1, d_K, n, d_y, n, d_info);
        cudaMemcpy(a, d_y, n*sizeof(double), cudaMemcpyDeviceToHost);
    } else {
        std::cout << "Error: Cholesky factorization failed" << std::endl;
    }

    cudaFree(d_K); cudaFree(d_y); cudaFree(d_a); cudaFree(d_reg); cudaFree(d_info);
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);
}

// CUDA kernel for data generation
__global__ void generate_data_kernel(double* X, double* y, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        X[idx] = 5.0 * curand_uniform_double(&state);
        double x = X[idx];
        y[idx] = 10 + (x - 0.1) * (x - 0.1) * (x - 0.1) - 5 * (x - 0.5) * (x - 0.5) + 10 * x + 5 * sin(5 * x);
        y[idx] += 0.1 * curand_uniform_double(&state);
    }
}

// Function to generate random training/test data
void generate_data(double* X, double* y, int n) {
    double *d_X, *d_y;
    cudaMalloc(&d_X, n * sizeof(double));
    cudaMalloc(&d_y, n * sizeof(double));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    unsigned long long seed = 1234ULL;

    generate_data_kernel<<<numBlocks, blockSize>>>(d_X, d_y, n, seed);

    cudaMemcpy(X, d_X, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_y);

    thrust::sort_by_key(thrust::host, X, X + n, y);
}

int main() {
    int n_train = 50;
    int n_test = 1000;
    double alpha = 0.1;
    double sigma = 0.1;

    double *X_train, *y_train, *X_test, *y_test;
    X_train = (double*)malloc(n_train * sizeof(double));
    y_train = (double*)malloc(n_train * sizeof(double));
    X_test = (double*)malloc(n_test * sizeof(double));
    y_test = (double*)malloc(n_test * sizeof(double));

    generate_data(X_train, y_train, n_train);
    generate_data(X_test, y_test, n_test);

    // Standard OLS
    double *w = (double*)malloc(2 * sizeof(double));

    double *X_train_aug = (double*)malloc(2 * n_train * sizeof(double));
    double *X_test_aug = (double*)malloc(2 * n_test * sizeof(double));
    for (int i = 0; i < n_train; i++) {
        X_train_aug[i] = X_train[i];
        X_train_aug[i + n_train] = 1.0;
    }
    for (int i = 0; i < n_test; i++) {
        X_test_aug[i] = X_test[i];
        X_test_aug[i + n_test] = 1.0;
    }

    solve_ls(X_train_aug, y_train, w, n_train, 2, alpha);

    double *y_train_pred = (double*)malloc(n_train * sizeof(double));
    double *y_test_pred = (double*)malloc(n_test * sizeof(double));
    for (int i = 0; i < n_train; i++) {
        y_train_pred[i] = w[0] * X_train[i] + w[1];
    }
    for (int i = 0; i < n_test; i++) {
        y_test_pred[i] = w[0] * X_test[i] + w[1];
    }

    // Kernelized OLS
    double *K_train = (double*)malloc(n_train * n_train * sizeof(double));
    double *K_test = (double*)malloc(n_train * n_test * sizeof(double));
    gaussian_kernel(X_train, X_train, K_train, n_train, n_train, sigma);
    gaussian_kernel(X_train, X_test, K_test, n_train, n_test, sigma);

    double *a = (double*)malloc(n_train * sizeof(double));
    solve_kernel_ls(K_train, y_train, a, n_train, alpha);

    double *y_train_pred_ker = (double*)malloc(n_train * sizeof(double));
    double *y_test_pred_ker = (double*)malloc(n_test * sizeof(double));

    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha1 = 1.0, beta1 = 0.0;
    cublasDgemv(handle, CUBLAS_OP_N, n_train, n_train, &alpha1, K_train, n_train, a, 1, &beta1, y_train_pred_ker, 1);
    cublasDgemv(handle, CUBLAS_OP_T, n_train, n_test, &alpha1, K_test, n_test, a, 1, &beta1, y_test_pred_ker, 1);
    cublasDestroy(handle);

    free(X_train); free(y_train); free(X_test); free(y_test);
    free(X_train_aug); free(X_test_aug); free(w);
    free(y_train_pred); free(y_test_pred);
    free(K_train); free(K_test); free(a);
    free(y_train_pred_ker); free(y_test_pred_ker);
}
