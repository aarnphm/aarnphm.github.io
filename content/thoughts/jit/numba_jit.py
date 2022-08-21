#!/usr/bin/env python3
"""
This demonstrates actual just-in-time compilation where Python bytecode
is analyzed, translated to LLVM IR, optimized, and compiled to native
machine code.

Compare with minimal_jit.py which shows pure Python bytecode optimization.
"""

import time
import timeit
import numpy as np
from numba import njit, prange


# =============================================================================
# fibonacci benchmarks: recursive vs iterative vs JIT-compiled
# =============================================================================


def fibonacci_naive(n: int) -> int:
  """Naive recursive fibonacci - O(2^n) without memoization"""
  if n <= 1:
    return n
  return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


def fibonacci_iterative(n: int) -> int:
  """Iterative fibonacci - O(n) algorithmic improvement"""
  if n <= 1:
    return n
  a, b = 0, 1
  for _ in range(2, n + 1):
    a, b = b, a + b
  return b


@njit  # nopython mode: pure LLVM compilation, no Python interpreter
def fibonacci_jit(n: int) -> int:
  """JIT-compiled fibonacci - same algorithm as naive, but compiled to native code"""
  if n <= 1:
    return n
  return fibonacci_jit(n - 1) + fibonacci_jit(n - 2)


@njit
def fibonacci_jit_iterative(n: int) -> int:
  """JIT-compiled iterative - combines algorithmic improvement + compilation"""
  if n <= 1:
    return n
  a, b = 0, 1
  for _ in range(2, n + 1):
    a, b = b, a + b
  return b


# =============================================================================
# DCT benchmarks: numpy vs numba (parallel + SIMD)
# =============================================================================


def dct_numpy(x: np.ndarray) -> np.ndarray:
  """Discrete Cosine Transform using NumPy - calls optimized BLAS/MKL"""
  N = len(x)
  result = np.zeros(N)
  for k in range(N):
    sum_val = 0.0
    for n in range(N):
      sum_val += x[n] * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    result[k] = sum_val
  return result


@njit
def dct_jit(x: np.ndarray) -> np.ndarray:
  """JIT-compiled DCT - sequential, SIMD-optimized by LLVM"""
  N = len(x)
  result = np.zeros(N)
  for k in range(N):
    sum_val = 0.0
    for n in range(N):
      sum_val += x[n] * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    result[k] = sum_val
  return result


@njit(parallel=True)
def dct_jit_parallel(x: np.ndarray) -> np.ndarray:
  """JIT-compiled DCT with automatic parallelization via prange"""
  N = len(x)
  result = np.zeros(N)
  for k in prange(N):  # parallel loop
    sum_val = 0.0
    for n in range(N):
      sum_val += x[n] * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    result[k] = sum_val
  return result


# =============================================================================
# matrix multiplication: demonstrate SIMD and loop optimization
# =============================================================================


def matmul_python(A: np.ndarray, B: np.ndarray) -> np.ndarray:
  """Pure Python matrix multiplication - extremely slow"""
  M, K = A.shape
  K2, N = B.shape
  assert K == K2
  C = np.zeros((M, N))
  for i in range(M):
    for j in range(N):
      for k in range(K):
        C[i, j] += A[i, k] * B[k, j]
  return C


@njit
def matmul_jit(A: np.ndarray, B: np.ndarray) -> np.ndarray:
  """JIT-compiled matmul - LLVM applies SIMD vectorization"""
  M, K = A.shape
  K2, N = B.shape
  assert K == K2
  C = np.zeros((M, N))
  for i in range(M):
    for j in range(N):
      for k in range(K):
        C[i, j] += A[i, k] * B[k, j]
  return C


@njit(parallel=True, fastmath=True)
def matmul_jit_parallel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
  """Parallel JIT matmul with fastmath (relaxed IEEE 754 compliance)"""
  M, K = A.shape
  K2, N = B.shape
  assert K == K2
  C = np.zeros((M, N))
  for i in prange(M):  # parallelize outer loop
    for j in range(N):
      for k in range(K):
        C[i, j] += A[i, k] * B[k, j]
  return C


# =============================================================================
# benchmarking utilities
# =============================================================================


def benchmark(func, *args, warmup=3, number=10, name=None):
  """
  Benchmark a function with proper warmup and timing isolation

  Args:
    func: function to benchmark
    args: arguments to pass to func
    warmup: number of warmup iterations (triggers JIT compilation)
    number: number of timed iterations
    name: optional name for display

  Returns:
    tuple: (result, compile_time, avg_execution_time)
  """
  name = name or func.__name__

  # warmup phase: trigger JIT compilation and cache warming
  compile_start = time.perf_counter()
  for _ in range(warmup):
    result = func(*args)
  compile_time = time.perf_counter() - compile_start

  # timed execution: measure steady-state performance
  exec_time = timeit.timeit(lambda: func(*args), number=number) / number

  return result, compile_time, exec_time


def compare_implementations(implementations, args_list, test_name):
  """
  Compare multiple implementations with proper benchmarking

  Args:
    implementations: list of (name, func) tuples
    args_list: list of argument tuples for each implementation
    test_name: descriptive name for the benchmark
  """
  print(f'\n{"=" * 80}')
  print(f'{test_name}')
  print(f'{"=" * 80}\n')

  results = []
  for (name, func), args in zip(implementations, args_list):
    result, compile_time, exec_time = benchmark(func, *args, name=name)
    results.append((name, result, compile_time, exec_time))

    print(f'{name:30} | compile: {compile_time * 1000:8.3f}ms | execute: {exec_time * 1e6:10.3f}µs')

  # verify correctness
  print('\nCorrectness check:')
  reference = results[0][1]
  for name, result, _, _ in results:
    if isinstance(result, np.ndarray):
      match = np.allclose(result, reference, rtol=1e-5)
    else:
      match = result == reference
    status = '✓' if match else '✗'
    print(f'  {status} {name}')

  # speedup analysis
  print('\nSpeedup (vs first implementation):')
  baseline_time = results[0][3]
  for name, _, _, exec_time in results[1:]:
    speedup = baseline_time / exec_time
    print(f'  {name:30} | {speedup:6.2f}x')


# =============================================================================
# LLVM IR inspection (requires numba with debug enabled)
# =============================================================================


def show_llvm_ir(func, *args):
  """
  Display LLVM IR generated for a JIT-compiled function

  This shows the intermediate representation before native code generation
  """
  print(f'\nLLVM IR for {func.__name__}:')
  print('-' * 80)

  # trigger compilation with specific types
  sig = func.inspect_types()
  print(sig)

  # get LLVM IR (requires eager compilation)
  try:
    llvm_ir = func.inspect_llvm(args)
    print(llvm_ir)
  except Exception as e:
    print(f'Could not retrieve LLVM IR: {e}')
    print('(Run with NUMBA_DUMP_LLVM=1 environment variable to see IR)')


# =============================================================================
# main benchmarks
# =============================================================================


def run_fibonacci_benchmark():
  """Compare fibonacci implementations"""
  n = 30  # small enough for naive recursive

  implementations = [
    ('fibonacci_naive', fibonacci_naive),
    ('fibonacci_iterative', fibonacci_iterative),
    ('fibonacci_jit', fibonacci_jit),
    ('fibonacci_jit_iterative', fibonacci_jit_iterative),
  ]

  args_list = [(n,)] * len(implementations)

  compare_implementations(implementations, args_list, 'Fibonacci(30) Benchmark')

  # show compilation effect on large n
  print('\n' + '=' * 80)
  print('Fibonacci(35) - larger problem size')
  print('=' * 80 + '\n')

  # skip naive (too slow)
  large_impls = implementations[1:]
  large_args = [(35,)] * len(large_impls)
  compare_implementations(large_impls, large_args, 'Fibonacci(35)')


def run_dct_benchmark():
  """Compare DCT implementations"""
  signal_size = 1024
  signal = np.random.random(signal_size).astype(np.float64)

  implementations = [('dct_numpy', dct_numpy), ('dct_jit', dct_jit), ('dct_jit_parallel', dct_jit_parallel)]

  args_list = [(signal,)] * len(implementations)

  compare_implementations(implementations, args_list, f'DCT(n={signal_size}) Benchmark')


def run_matmul_benchmark():
  """Compare matrix multiplication implementations"""
  size = 200
  A = np.random.random((size, size)).astype(np.float64)
  B = np.random.random((size, size)).astype(np.float64)

  implementations = [
    ('numpy.dot (BLAS)', lambda A, B: np.dot(A, B)),
    ('matmul_jit', matmul_jit),
    ('matmul_jit_parallel', matmul_jit_parallel),
  ]

  args_list = [(A, B)] * len(implementations)

  compare_implementations(implementations, args_list, f'Matrix Multiply ({size}x{size}) Benchmark')

  print('\nNote: NumPy uses optimized BLAS (MKL/OpenBLAS) which is hard to beat')
  print("Numba JIT is competitive but doesn't match hand-tuned BLAS for large matrices")


def demonstrate_compilation_overhead():
  """Show compilation time vs execution time tradeoff"""
  print('\n' + '=' * 80)
  print('JIT Compilation Overhead Analysis')
  print('=' * 80 + '\n')

  n = 25

  # measure compilation time (first call)
  compile_start = time.perf_counter()
  result1 = fibonacci_jit(n)
  compile_time = time.perf_counter() - compile_start

  # measure cached execution (subsequent calls)
  exec_times = []
  for _ in range(100):
    start = time.perf_counter()
    result = fibonacci_jit(n)
    exec_times.append(time.perf_counter() - start)

  avg_exec = np.mean(exec_times)

  print(f'First call (compile + execute): {compile_time * 1000:.3f}ms')
  print(f'Subsequent calls (cached):      {avg_exec * 1e6:.3f}µs')
  print(f'Compilation overhead:           {compile_time / avg_exec:.0f}x execution time')
  print(f'Break-even point:               {int(compile_time / avg_exec)} executions')
  print(f'Conclusion: JIT compilation pays off after ~{int(compile_time / avg_exec)} calls')


def main():
  # fibonacci benchmarks
  run_fibonacci_benchmark()

  # DCT benchmarks
  run_dct_benchmark()

  # matrix multiplication benchmarks
  run_matmul_benchmark()

  # compilation overhead analysis
  demonstrate_compilation_overhead()


if __name__ == '__main__':
  raise SystemExit(main())
