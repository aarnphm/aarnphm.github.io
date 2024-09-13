from __future__ import annotations
import matplotlib.pyplot as plt, numpy as np, numpy.typing as npt


# Solves the Least Squares problem for given X, Y.
# alpha is the regularization coefficient
# Output is the estimated y for both X and X_all
def solve_ols(
  X_train: npt.NDArray[np.float64], Y_train: npt.NDArray[np.float64],
  X_test: npt.NDArray[np.float64], alpha: float
):
  beta = alpha * np.identity(np.shape(X_train)[1])
  W = np.dot(
    np.linalg.pinv(np.dot(X_train.T, X_train) + beta),
    np.dot(X_train.T, Y_train)
  )
  # print("Optimal W is ", W.flatten())
  return np.dot(X_train, W), np.dot(X_test, W)


# Solves the Kernelized Regularized Least Squares problem for given K_train, Y.
# K_train is an n-by-n kernel matrix for train data (n := #training_points)
# K_test is an m-by-d kernel matrix (m:= #test_points, n := #training_points)
# It assumes that K_train is invertible
# alpha is the regularization coefficient
# Output is the estimated y for both X and X_all
def solve_kernel_ls(
  K_train: npt.NDArray[np.float64], Y_train: npt.NDArray[np.float64],
  K_test: npt.NDArray[np.float64], alpha: float
):
  a = np.dot(np.linalg.pinv(K_train + alpha * np.identity(np.shape(X_train)[0])), Y_train)
  return np.dot(K, a), np.dot(K_test.T, a)


# If X1 and X2 are the same, then it just computes the kernel matrix of X1
# In general, it computes the kernel matrix corresponding to rows of X1 and rows of X2
# sigma is the bandwidth of the kernel
def gaussian_kernel(X1: npt.NDArray[np.float64], X2: npt.NDArray[np.float64], sigma: float):
  m, n = np.shape(X2)[0], np.shape(X1)[0]
  # vectorized form of computing the kernel matrix
  norms_X1 = np.sum(np.multiply(X1, X1), axis=1)
  norms_X2 = np.sum(np.multiply(X2, X2), axis=1)
  K0 = np.tile(norms_X2, (n, 1)) + np.tile(norms_X1, (m, 1)).T - 2 * (X1 @ X2.T)
  return np.power(np.exp(-1.0 / sigma**2), K0)


def run_ols(
  X_train: npt.NDArray[np.float64], Y_train: npt.NDArray[np.float64],
  X_test: npt.NDArray[np.float64], Y_test: npt.NDArray[np.float64],
  alpha: float,
  plot_X_train: npt.NDArray[np.float64], plot_X_test: npt.NDArray[np.float64],
  description: str,
  kernel: bool = False,
):
  if kernel == False:
    Y_LS_train, Y_LS_test = solve_ols(X_train, Y_train, X_test, alpha)
  else:
    Y_LS_train, Y_LS_test = solve_kernel_ls(X_train, Y_train, X_test, alpha)

  print('Mean Squarred Error (MSE) of train data:', np.square(np.linalg.norm(Y_LS_train - Y_train)) / Y_train.size)
  print('Mean Squarred Error (MSE) of test data:', np.square(np.linalg.norm(Y_LS_test - Y_test)) / Y_test.size)


# generate n data points based on a combination of sinosuidal and polynomial functions
def generate_data(n: int):
  X = np.random.rand(n, 1) * 5
  X = np.sort(X, axis=0)
  Y = 10 + (X - 0.1) * (X - 0.1) * (X - 0.1) - 5 * (X - 0.5) * (X - 0.5) + 10 * X + 5 * np.sin(5 * X)
  # Adding noise
  Y += 0.1 * np.random.randn(n, 1)
  return X, Y


if __name__ == '__main__':
  # Number of training and test points.
  n_train = 50
  n_test = 1000

  # Regularizaion Coefficient in RLS (lambda in the class)
  alpha = 0.1

  # Bandiwidth of the Gaussian Kernel
  sigma = 0.1

  # Generating train and test data.
  X_train, Y_train = generate_data(n_train)
  X_test, Y_test = generate_data(n_test)

  # Homogenous line/hyperplane (goes through the origin)
  run_ols(X_train, Y_train, X_test, Y_test, alpha, X_train, X_test, 'Homogenous line')

  # Non-homogenous line/hyperplane
  # First we augment the data with an all 1 column/feature
  X_augmented_train = np.concatenate((X_train, np.ones((n_train, 1))), axis=1)
  X_augmented_test = np.concatenate((X_test, np.ones((n_test, 1))), axis=1)

  # OLS on the augmented data.
  run_ols(X_augmented_train, Y_train, X_augmented_test, Y_test, alpha, X_train, X_test, 'Nonhomogenous line')

  # Kernelized Least Squares with Gaussian Kernel
  K = gaussian_kernel(X_train, X_train, sigma)
  K_test = gaussian_kernel(X_train, X_test, sigma)
  run_ols(K, Y_train, K_test, Y_test, alpha, X_train, X_test, 'Gaussian Kernel', True)
