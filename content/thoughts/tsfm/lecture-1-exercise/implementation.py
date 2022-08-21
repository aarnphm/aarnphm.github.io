"""
Matrix multiplication forward and backward pass implementation.

And test correctness with PyTorch and JAX frameworks
"""

from __future__ import annotations

import numpy as np, torch

try:
  import jax, jax.numpy as jnp
  from jax import config as jax_config

  jax_config.update('jax_enable_x64', True)
  JAX_AVAILABLE = True
except Exception:
  JAX_AVAILABLE = False


# A: (m, k), B: (k, n)
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray: return np.dot(A, B)


# Y = X @ W
# (bs, output_dim) = (bs, input_dim) @ (input_dim, output_dim)
def forward(X: np.ndarray, W: np.ndarray) -> np.ndarray: return matmul(X, W)


# given a `Y = X @ W`, with $X \in \mathbb{R}^{N\times D}, W \in \mathbb{R}^{D\times M}, Y\in \mathbb{R}^{N\times M}$, the [[thoughts/Vector calculus#Jacobian matrix|Jacobian]] is:
#
# $$
# dW = X^{T}dY, dW = dY W^{T}
# $$
def backward(X: np.ndarray, W: np.ndarray, dY: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  # apriori: assume dims are matched, otherwise uncomment this check. Usually we don't want to check for assert here.
  # assert X.ndim == W.ndim == dY.ndim == 2
  # N, D = X.shape
  # D2, M = W.shape
  # N2, M2 = dY.shape
  # assert D == D2 and N == N2 and M== M2
  return X.T @ dY, dY @ W.T


def check() -> None:
  """
  Main function to test the forward and backward implementations.

  Creates random matrices, performs forward and backward passes manually,
  then validates the results against PyTorch's autograd.
  """
  # Set random seed for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)

  # Create random input and weight matrices
  batch_size, input_dim, output_dim = 500, 1000, 100
  X = np.random.randn(batch_size, input_dim)
  W = np.random.randn(input_dim, output_dim)

  # Manual forward pass
  Y = forward(X, W)

  # Create random gradient for backward pass
  dY = np.random.randn(batch_size, output_dim)

  # Manual backward pass
  dW_manual, dX_manual = backward(X, W, dY)

  # PyTorch validation
  X_torch = torch.tensor(X, dtype=torch.float64, requires_grad=True)
  W_torch = torch.tensor(W, dtype=torch.float64, requires_grad=True)
  dY_torch = torch.tensor(dY, dtype=torch.float64)

  # PyTorch forward pass
  Y_torch = torch.matmul(X_torch, W_torch)

  # PyTorch backward pass
  Y_torch.backward(dY_torch)

  # Extract gradients
  assert W_torch.grad is not None, 'W_torch.grad should not be None after backward()'
  assert X_torch.grad is not None, 'X_torch.grad should not be None after backward()'
  dW_torch = W_torch.grad.detach().numpy()
  dX_torch = X_torch.grad.detach().numpy()

  if JAX_AVAILABLE:
    X_jax = jnp.asarray(X)
    W_jax = jnp.asarray(W)
    dY_jax = jnp.asarray(dY)

    # Forward
    Y_jax = jnp.matmul(X_jax, W_jax)

    def f(X_, W_):
      return jnp.matmul(X_, W_)

    Y_val, vjp_fun = jax.vjp(f, X_jax, W_jax)
    dX_jax, dW_jax = vjp_fun(dY_jax)

    Y_jax_np = np.asarray(Y_jax)
    dW_jax_np = np.asarray(dW_jax)
    dX_jax_np = np.asarray(dX_jax)

  # Compare results
  print('Gradient comparisons:')
  print(f'dW matches PyTorch: {np.allclose(dW_manual, dW_torch, rtol=1e-10)}')
  print(f'dX matches PyTorch: {np.allclose(dX_manual, dX_torch, rtol=1e-10)}')

  # Print maximum absolute differences
  print('\nMaximum absolute differences:')
  print(f'dW max diff: {np.max(np.abs(dW_manual - dW_torch))}')
  print(f'dX max diff: {np.max(np.abs(dX_manual - dX_torch))}')

  if JAX_AVAILABLE:
    print('\nJAX forward comparison:')
    print(f'Y (NumPy vs JAX) matches: {np.allclose(Y, Y_jax_np, rtol=1e-10, atol=0)}')

    print('\nGradient comparisons vs JAX:')
    print(f'dW (manual vs JAX) matches: {np.allclose(dW_manual, dW_jax_np, rtol=1e-10, atol=0)}')
    print(f'dX (manual vs JAX) matches: {np.allclose(dX_manual, dX_jax_np, rtol=1e-10, atol=0)}')

    print('\nMaximum absolute differences (manual vs JAX):')
    print(f'dW max diff: {np.max(np.abs(dW_manual - dW_jax_np))}')
    print(f'dX max diff: {np.max(np.abs(dX_manual - dX_jax_np))}')
  else:
    print('\n[JAX not available] Skipping JAX validation. Install jax & jaxlib to enable this check.')

if __name__ == "__main__":check()
