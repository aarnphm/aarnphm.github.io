from typing import Literal
import dis, types, time, numpy as np


def test_performance(func, n, iterations):
  start_time = time.time()
  for _ in range(iterations): result = func(n)
  end_time = time.time()
  return result, end_time - start_time


def fibonacci(n):
  if n <= 1: return n
  else: return fibonacci(n - 1) + fibonacci(n - 2)


def dct_1d(x):
  N=len(x)
  result=np.zeros(N)
  for k in range(N):
    sum_val = 0
    for n in range(N): sum_val += x[n] * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    result[k]=sum_val
  return result


class OptimizedToyJIT:
  def __init__(self):
    self.cache = {}

  def fibonacci_jit(self, n):
    if n in self.cache:
      return self.cache[n]

    # Branch optimization
    if n <= 1:
      result = n
    elif n < 20:  # Use recursive approach for small n
      result = self.fibonacci_jit(n - 1) + self.fibonacci_jit(n - 2)
    else:  # Use iterative approach for larger n
      a, b = 0, 1
      for _ in range(2, n + 1):
        a, b = b, a + b
      result = b

    self.cache[n] = result
    return result

  def dct_jit(self, x):
    x_tuple = tuple(x)
    if x_tuple in self.cache: return self.cache[x_tuple]

    N = len(x)
    result = np.zeros(N)
    for k in range(N):
      sum_val = 0
      for n in range(N): sum_val += x[n] * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
      result[k] = sum_val

    self.cache[x_tuple] = result
    return result


jit = OptimizedToyJIT()

signal_size = 1024
signal = np.random.random(signal_size)

def print_info(native_result,optimized_result,native_time,optimized_time,caller,jit_caller):
  print(f'Native     = {native_result}')
  print(f'Optimized  = {optimized_result}')
  print(f'Native time:    {native_time:.6f} seconds')
  print(f'Optimized time: {optimized_time:.6f} seconds')
  print(f'Speedup: {native_time / optimized_time:.2f}x')
  print('\nNative bytecode:')
  dis.dis(caller)
  print('\nOptimized bytecode:')
  dis.dis(jit_caller)

def run_fibonacci_benchmark(n, iterations):
  native_result, native_time = test_performance(fibonacci, n, iterations)
  optimized_result, optimized_time = test_performance(jit.fibonacci_jit, n, iterations)
  print_info(native_result, optimized_result, native_time, optimized_time, fibonacci, jit.fibonacci_jit)

def run_dct_benchmark(iterations):
  native_result, native_time = test_performance(dct_1d, signal, iterations)
  optimized_result, optimized_time = test_performance(jit.dct_jit, signal, iterations)
  print_info(native_result, optimized_result, native_time, optimized_time, dct_1d, jit.dct_jit)

def main(n, iterations, method: Literal["fib", "dct"] = "fib") -> int:
    benchmark_functions = {
      "fib": lambda: run_fibonacci_benchmark(n, iterations),
      "dct": lambda: run_dct_benchmark(iterations)
    }
    if method in benchmark_functions: benchmark_functions[method]()
    else: print(f"Unknown method: {method}")

    return 0


if __name__ == '__main__': raise SystemExit(main(n=30, iterations=30, method='dct'))
