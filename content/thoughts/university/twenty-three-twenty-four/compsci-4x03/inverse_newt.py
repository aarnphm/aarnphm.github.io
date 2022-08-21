import os
import numpy as np


def matrix_inverse_newt(A, tol=1e-9, max_iter=100):
  n = A.shape[0]
  I = np.eye(n)
  Xk = A.T / (np.linalg.norm(A, 1) * np.linalg.norm(A, np.inf))
  for _ in range(max_iter):
    Rk = I - np.dot(A, Xk)
    Xk_new = Xk + np.dot(Xk, Rk)
    # Stopping criterion based on the norm of the residual matrix
    if np.linalg.norm(Rk) < tol:
      break
    Xk = Xk_new
  return Xk


def tests(A):
  A_inv = matrix_inverse_newt(A)
  # Let's matrix_inverse_newt to the true inverse computed by numpy's built-in function
  A_inv_true = np.linalg.inv(A)
  # Compare the two inverses
  print("Inverse using Newton's method:\n", A_inv)
  print('True Inverse:\n', A_inv_true)
  print('Difference:\n', A_inv - A_inv_true)


def errors(A):
  A_inv = matrix_inverse_newt(A)
  # Recalculate the inverse and get the error at each iteration
  A_inv_newton, errors_newton = matrix_inverse_newt_err(A)
  # Now we will check for quadratic convergence by calculating the ratio of errors
  ratios = []
  for i in range(1, len(errors_newton) - 1):
    ratios.append(errors_newton[i + 1] / errors_newton[i] ** 2)
  print(ratios)
  return ratios


def matrix_inverse_newt_err(A, tol=1e-9, max_iter=100):
  n = A.shape[0]
  I = np.eye(n)
  A_inv_true = np.linalg.inv(A)  # True inverse for error calculation
  Xk = A.T / (np.linalg.norm(A, 1) * np.linalg.norm(A, np.inf))
  errors = []  # List to track errors over iterations
  for _ in range(max_iter):
    Rk = I - np.dot(A, Xk)
    Xk_new = Xk + np.dot(Xk, Rk)
    # Calculate and store the current error
    current_error = np.linalg.norm(Xk_new - A_inv_true)
    errors.append(current_error)
    # Stopping criterion based on the norm of the residual matrix
    if current_error < tol:
      break
    Xk = Xk_new
  return Xk, errors


if __name__ == '__main__':
  # Test the function with a random matrix
  np.random.seed(0)  # Seed for reproducibility
  n = 4  # Size of the matrix
  A = np.random.rand(n, n)
  tests(A) if os.getenv('ERROR', str(False)).lower() == 'false' else errors(A)
