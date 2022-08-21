#!/usr/bin/env python3
"""
minimal blas: TinyCJIT vs Compiler (adaptive) vs Numba

implements basic blas operations comparing three strategies:
- TinyCJIT: fast single-pass compilation
- Compiler: adaptive (chooses TinyCJIT or IRCompiler based on complexity)
- Numba: LLVM-based optimization

benchmarks compile time and runtime to highlight tradeoffs.
"""

from __future__ import annotations
import time
from ctypes import POINTER, c_float, c_int

import numpy as np
import numba

from minimal_jit import TinyCJIT
from compiler import Compiler

# =============================================================================
# TinyCJIT implementations
# =============================================================================

jit_c = TinyCJIT(verbose=False)

# =============================================================================
# Compiler (adaptive) implementations
# =============================================================================

# adaptive compiler: chooses TinyCJIT or IRCompiler based on complexity
jit_auto = Compiler(mode='auto', verbose=False, complexity_threshold=10)


@jit_c(restype=None, argtypes=[c_float, POINTER(c_float), POINTER(c_float), c_int])
def saxpy_c(a, x, y, n):
  """y = a*x + y (single-precision a*x plus y)"""
  for i in range(n):
    y[i] = a * x[i] + y[i]


@jit_c(restype=c_float, argtypes=[POINTER(c_float), POINTER(c_float), c_int])
def dot_c(x, y, n):
  """dot product"""
  total = 0.0
  for i in range(n):
    total += x[i] * y[i]
  return total


@jit_c(restype=None, argtypes=[POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int])
def gemv_c(A, x, y, m, n):
  """y = A @ x (matrix-vector product, A is m×n row-major)"""
  for i in range(m):
    total = 0.0
    for j in range(n):
      total += A[i * n + j] * x[j]
    y[i] = total


@jit_c(restype=None, argtypes=[POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int])
def gemm_c(A, B, C, m, n, k):
  """C = A @ B (matrix-matrix, A is m×k, B is k×n, C is m×n, all row-major)"""
  for i in range(m):
    for j in range(n):
      total = 0.0
      for p in range(k):
        total += A[i * k + p] * B[p * n + j]
      C[i * n + j] = total


# hand-optimized blocked gemm with aggressive compiler flags
jit_c_opt = TinyCJIT(verbose=False, extra_cflags=['-O3', '-march=native', '-ffast-math', '-funroll-loops', '-fPIC'])


@jit_c_opt(
  restype=None,
  argtypes=[POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int],
  headers=['#define BLOCK_SIZE 32'],
)
def gemm_c_blocked(A, B, C, m, n, k):
  """blocked gemm with cache tiling"""
  for i in range(0, m, 32):
    for j in range(0, n, 32):
      for p in range(0, k, 32):
        i_max = min(i + 32, m)
        j_max = min(j + 32, n)
        p_max = min(p + 32, k)
        for ii in range(i, i_max):
          for jj in range(j, j_max):
            total = C[ii * n + jj]
            for pp in range(p, p_max):
              total += A[ii * k + pp] * B[pp * n + jj]
            C[ii * n + jj] = total


# =============================================================================
# Compiler (adaptive) implementations
# =============================================================================


@jit_auto(restype=None, argtypes=[c_float, POINTER(c_float), POINTER(c_float), c_int])
def saxpy_auto(a, x, y, n):
  """y = a*x + y (single-precision a*x plus y)"""
  for i in range(n):
    y[i] = a * x[i] + y[i]


@jit_auto(restype=c_float, argtypes=[POINTER(c_float), POINTER(c_float), c_int])
def dot_auto(x, y, n):
  """dot product"""
  total = 0.0
  for i in range(n):
    total += x[i] * y[i]
  return total


@jit_auto(restype=None, argtypes=[POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int])
def gemv_auto(A, x, y, m, n):
  """y = A @ x (matrix-vector product, A is m×n row-major)"""
  for i in range(m):
    total = 0.0
    for j in range(n):
      total += A[i * n + j] * x[j]
    y[i] = total


@jit_auto(restype=None, argtypes=[POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int])
def gemm_auto(A, B, C, m, n, k):
  """C = A @ B (matrix-matrix, A is m×k, B is k×n, C is m×n, all row-major)"""
  for i in range(m):
    for j in range(n):
      total = 0.0
      for p in range(k):
        total += A[i * k + p] * B[p * n + j]
      C[i * n + j] = total


# hand-optimized blocked gemm with adaptive compiler (higher complexity -> likely optimized)
jit_auto_opt = Compiler(mode='auto', verbose=False, complexity_threshold=15)


@jit_auto_opt(
  restype=None,
  argtypes=[POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int],
  headers=['#define BLOCK_SIZE 32'],
)
def gemm_auto_blocked(A, B, C, m, n, k):
  """blocked gemm with cache tiling"""
  for i in range(0, m, 32):
    for j in range(0, n, 32):
      for p in range(0, k, 32):
        i_max = min(i + 32, m)
        j_max = min(j + 32, n)
        p_max = min(p + 32, k)
        for ii in range(i, i_max):
          for jj in range(j, j_max):
            total = C[ii * n + jj]
            for pp in range(p, p_max):
              total += A[ii * k + pp] * B[pp * n + jj]
            C[ii * n + jj] = total


# =============================================================================
# Numba implementations
# =============================================================================


@numba.jit(nopython=True, fastmath=True)
def saxpy_numba(a: float, x: np.ndarray, y: np.ndarray, n: int) -> None:
  """y = a*x + y"""
  for i in range(n):
    y[i] = a * x[i] + y[i]


@numba.jit(nopython=True, fastmath=True)
def dot_numba(x: np.ndarray, y: np.ndarray, n: int) -> float:
  """dot product"""
  total = 0.0
  for i in range(n):
    total += x[i] * y[i]
  return total


@numba.jit(nopython=True, fastmath=True)
def gemv_numba(A: np.ndarray, x: np.ndarray, y: np.ndarray, m: int, n: int) -> None:
  """y = A @ x"""
  for i in range(m):
    total = 0.0
    for j in range(n):
      total += A[i, j] * x[j]
    y[i] = total


@numba.jit(nopython=True, fastmath=True)
def gemm_numba(A: np.ndarray, B: np.ndarray, C: np.ndarray, m: int, n: int, k: int) -> None:
  """C = A @ B"""
  for i in range(m):
    for j in range(n):
      total = 0.0
      for p in range(k):
        total += A[i, p] * B[p, j]
      C[i, j] = total


# =============================================================================
# benchmarking
# =============================================================================


def benchmark_op(
  name: str, c_func, auto_func, numba_func, setup_fn, n_warmup: int = 3, n_runs: int = 10
) -> None:
  """benchmark an operation across three implementations"""
  print(f'\n{"=" * 80}')
  print(f'Benchmark: {name}')
  print(f'{"=" * 80}')

  # setup data
  args_c, args_numba = setup_fn()

  # warmup and measure TinyCJIT compile time
  compile_start = time.perf_counter()
  for _ in range(n_warmup):
    c_func(*args_c)
  compile_time_c = (time.perf_counter() - compile_start) / n_warmup

  # warmup and measure Compiler (adaptive) compile time
  compile_start = time.perf_counter()
  for _ in range(n_warmup):
    auto_func(*args_c)
  compile_time_auto = (time.perf_counter() - compile_start) / n_warmup

  # warmup and measure Numba compile time
  compile_start = time.perf_counter()
  for _ in range(n_warmup):
    numba_func(*args_numba)
  compile_time_numba = (time.perf_counter() - compile_start) / n_warmup

  # benchmark TinyCJIT runtime
  times_c = []
  for _ in range(n_runs):
    start = time.perf_counter()
    c_func(*args_c)
    times_c.append(time.perf_counter() - start)
  runtime_c = sum(times_c) / len(times_c)

  # benchmark Compiler (adaptive) runtime
  times_auto = []
  for _ in range(n_runs):
    start = time.perf_counter()
    auto_func(*args_c)
    times_auto.append(time.perf_counter() - start)
  runtime_auto = sum(times_auto) / len(times_auto)

  # benchmark Numba runtime
  times_numba = []
  for _ in range(n_runs):
    start = time.perf_counter()
    numba_func(*args_numba)
    times_numba.append(time.perf_counter() - start)
  runtime_numba = sum(times_numba) / len(times_numba)

  # benchmark pure numpy (reference)
  if 'reference' in setup_fn.__name__:
    ref_func = setup_fn.__globals__.get(f'{setup_fn.__name__.replace("_setup", "_ref")}')
    if ref_func:
      times_ref = []
      for _ in range(n_runs):
        start = time.perf_counter()
        ref_func(*args_numba)
        times_ref.append(time.perf_counter() - start)
      runtime_ref = sum(times_ref) / len(times_ref)
    else:
      runtime_ref = 0.0
  else:
    runtime_ref = 0.0

  # report
  print('\nCompile time:')
  print(f'  TinyCJIT:   {compile_time_c * 1000:6.2f} ms')
  print(f'  Compiler:   {compile_time_auto * 1000:6.2f} ms')
  print(f'  Numba:      {compile_time_numba * 1000:6.2f} ms')
  print(f'  Strategy:   {auto_func._compiler_strategy}')

  print(f'\nRuntime (average of {n_runs} runs):')
  print(f'  TinyCJIT:   {runtime_c * 1e6:8.2f} µs')
  print(f'  Compiler:   {runtime_auto * 1e6:8.2f} µs')
  print(f'  Numba:      {runtime_numba * 1e6:8.2f} µs')

  # find winner
  times = {'TinyCJIT': runtime_c, 'Compiler': runtime_auto, 'Numba': runtime_numba}
  winner = min(times.items(), key=lambda x: x[1])
  print(f'  Winner:     {winner[0]}')
  for name, time_val in times.items():
    if name != winner[0]:
      ratio = time_val / winner[1]
      print(f'              ({winner[0]} is {ratio:.2f}x faster than {name})')

  if runtime_ref > 0:
    print(f'  NumPy ref:  {runtime_ref * 1e6:8.2f} µs')


# =============================================================================
# setup functions
# =============================================================================


def saxpy_setup(n: int = 1_000_000) -> tuple:
  """setup data for saxpy"""
  # TinyCJIT (numpy arrays)
  a_val = 2.5
  x_c = np.random.rand(n).astype(np.float32)
  y_c = np.random.rand(n).astype(np.float32)

  # Numba (numpy arrays)
  x_numba = x_c.copy()
  y_numba = y_c.copy()

  return (a_val, x_c, y_c, n), (a_val, x_numba, y_numba, n)


def dot_setup(n: int = 1_000_000) -> tuple:
  """setup data for dot product"""
  x_c = np.random.rand(n).astype(np.float32)
  y_c = np.random.rand(n).astype(np.float32)

  x_numba = x_c.copy()
  y_numba = y_c.copy()

  return (x_c, y_c, n), (x_numba, y_numba, n)


def gemv_setup(m: int = 2000, n: int = 2000) -> tuple:
  """setup data for matrix-vector"""
  # TinyCJIT (row-major flat array)
  A_c = np.random.rand(m * n).astype(np.float32)
  x_c = np.random.rand(n).astype(np.float32)
  y_c = np.zeros(m, dtype=np.float32)

  # Numba (2D array)
  A_numba = A_c.reshape(m, n)
  x_numba = x_c.copy()
  y_numba = np.zeros(m, dtype=np.float32)

  return (A_c, x_c, y_c, m, n), (A_numba, x_numba, y_numba, m, n)


def gemm_setup(m: int = 256, n: int = 256, k: int = 256) -> tuple:
  """setup data for matrix-matrix"""
  # TinyCJIT (row-major flat arrays)
  A_c = np.random.rand(m * k).astype(np.float32)
  B_c = np.random.rand(k * n).astype(np.float32)
  C_c = np.zeros(m * n, dtype=np.float32)

  # Numba (2D arrays)
  A_numba = A_c.reshape(m, k)
  B_numba = B_c.reshape(k, n)
  C_numba = np.zeros((m, n), dtype=np.float32)

  return (A_c, B_c, C_c, m, n, k), (A_numba, B_numba, C_numba, m, n, k)


# =============================================================================
# main
# =============================================================================


def main():
  print('=' * 80)
  print('Minimal BLAS: TinyCJIT vs Compiler (adaptive) vs Numba')
  print('=' * 80)
  print('\nThree compilation strategies:')
  print('- TinyCJIT: fast single-pass (0.1-1ms compile)')
  print('- Compiler: adaptive (chooses TinyCJIT or IRCompiler)')
  print('- Numba: LLVM-based optimization\n')

  # saxpy: bandwidth-bound, should be competitive
  benchmark_op('SAXPY (y = a*x + y, n=1M)', saxpy_c, saxpy_auto, saxpy_numba, lambda: saxpy_setup(1_000_000))

  # dot product: reduction, more compute
  benchmark_op('DOT (dot product, n=1M)', dot_c, dot_auto, dot_numba, lambda: dot_setup(1_000_000))

  # gemv: moderate compute intensity
  benchmark_op(
    'GEMV (matrix-vector, 2000×2000)', gemv_c, gemv_auto, gemv_numba, lambda: gemv_setup(2000, 2000)
  )

  # gemm: compute-bound, numba should win
  benchmark_op(
    'GEMM naive (matrix-matrix, 256×256×256)', gemm_c, gemm_auto, gemm_numba, lambda: gemm_setup(256, 256, 256)
  )

  # gemm blocked: hand-optimized with cache tiling
  benchmark_op(
    'GEMM blocked (matrix-matrix, 256×256×256)',
    gemm_c_blocked,
    gemm_auto_blocked,
    gemm_numba,
    lambda: gemm_setup(256, 256, 256),
  )

  print('\n' + '=' * 80)
  print('Summary:')
  print('=' * 80)
  print(f'Compiler strategy stats:')
  print(f'  Fast (TinyCJIT):    {jit_auto.stats["fast"]} functions')
  print(f'  Optimized (IR):     {jit_auto.stats["optimized"]} functions')

  print('\n' + '=' * 80)
  print('Generated C source for SAXPY (TinyCJIT):')
  print('=' * 80)
  print(saxpy_c.c_source)

  print('\n' + '=' * 80)
  print('Generated C source for blocked GEMM (TinyCJIT):')
  print('=' * 80)
  print(gemm_c_blocked.c_source)


if __name__ == '__main__':
  raise SystemExit(main())
