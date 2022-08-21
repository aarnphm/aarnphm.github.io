#!/usr/bin/env python3
"""
unified compiler interface: TinyCJIT + IRCompiler

adaptive compilation strategy:
- fast path (TinyCJIT): simple functions, minimal overhead, 0.1-1ms compile
- optimized path (IRCompiler): complex functions, type inference, optimizations, 10-50ms compile

user can force strategy via mode='fast' or mode='optimized', or let the compiler
decide based on function complexity heuristics.

demonstrates real-world JIT design: multiple compilation tiers with different
optimization levels, similar to V8 (Ignition → TurboFan) or PyPy (tracing JIT).
"""

from __future__ import annotations
import ast
import inspect
import textwrap
from typing import Any, Callable, Literal

from minimal_jit import TinyCJIT
from ir_compiler import IRCompiler


# =============================================================================
# compilation strategy heuristics
# =============================================================================


def _estimate_complexity(func: Callable[..., Any]) -> int:
  """estimate function complexity from AST"""
  try:
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
  except:
    return 0

  # count nodes as complexity metric
  complexity = 0

  class ComplexityCounter(ast.NodeVisitor):
    def __init__(self):
      self.count = 0

    def visit_For(self, node: ast.For) -> None:
      self.count += 5  # loops are expensive
      self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
      self.count += 5
      self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
      self.count += 1
      self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
      self.count += 2
      self.generic_visit(node)

  counter = ComplexityCounter()
  counter.visit(tree)
  return counter.count


# =============================================================================
# unified compiler
# =============================================================================


class Compiler:
  """
  unified JIT compiler with adaptive compilation

  usage:
    jit = Compiler(mode='auto')  # choose strategy automatically
    jit = Compiler(mode='fast')  # always use TinyCJIT
    jit = Compiler(mode='optimized')  # always use IRCompiler

    @jit(restype=c_float, argtypes=[c_float, c_float])
    def add(a, b):
      return a + b
  """

  def __init__(
    self,
    mode: Literal['auto', 'fast', 'optimized'] = 'auto',
    verbose: bool = False,
    complexity_threshold: int = 10,
  ):
    self.mode = mode
    self.verbose = verbose
    self.complexity_threshold = complexity_threshold
    self.fast_compiler = TinyCJIT(verbose=verbose)
    self.opt_compiler = IRCompiler(verbose=verbose, optimize=True)
    self.stats = {'fast': 0, 'optimized': 0}

  def __call__(self, **kwargs) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """decorator interface matching TinyCJIT/IRCompiler"""
    restype = kwargs.get('restype')
    argtypes = kwargs.get('argtypes', [])
    headers = kwargs.get('headers', [])

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
      # decide compilation strategy
      if self.mode == 'fast':
        strategy = 'fast'
      elif self.mode == 'optimized':
        strategy = 'optimized'
      else:  # auto
        complexity = _estimate_complexity(func)
        strategy = 'optimized' if complexity >= self.complexity_threshold else 'fast'

      self.stats[strategy] += 1

      if self.verbose:
        print(f'[Compiler] compiling {func.__name__} with strategy={strategy}')

      # dispatch to appropriate compiler
      if strategy == 'fast':
        compiled = self.fast_compiler(restype=restype, argtypes=argtypes, headers=headers)(func)
      else:
        compiled = self.opt_compiler(restype=restype, argtypes=argtypes, headers=headers)(func)

      # attach metadata
      compiled._compiler_strategy = strategy  # type: ignore
      compiled._compiler_stats = self.stats  # type: ignore
      return compiled

    return decorator


# =============================================================================
# demo: compare strategies
# =============================================================================


def demo_compiler():
  """demonstrate adaptive compilation with both strategies"""
  import ctypes
  import time

  print('=' * 80)
  print('Unified Compiler Demo: TinyCJIT vs IRCompiler')
  print('=' * 80)

  # auto mode: compiler chooses based on complexity
  jit_auto = Compiler(mode='auto', verbose=True, complexity_threshold=10)

  # simple function → should use TinyCJIT
  @jit_auto(restype=ctypes.c_float, argtypes=[ctypes.c_float, ctypes.c_float])
  def simple_add(a, b):
    """simple add: complexity < 10"""
    return a + b

  # complex function → should use IRCompiler
  @jit_auto(restype=ctypes.c_float, argtypes=[ctypes.c_float, ctypes.c_float])
  def complex_compute(a, b):
    """complex compute: complexity >= 10"""
    tmp1 = a + b
    tmp2 = tmp1 * 2.0
    tmp3 = tmp2 - a
    tmp4 = tmp3 / b
    tmp5 = tmp4 + a
    tmp6 = tmp5 * b
    tmp7 = tmp6 - tmp1
    result = tmp7 + tmp2
    return result

  print('\n' + '=' * 80)
  print('Testing:')
  print('=' * 80)

  # test simple
  result1 = simple_add(3.0, 4.0)
  print(f'simple_add(3.0, 4.0) = {result1:.1f} (strategy: {simple_add._compiler_strategy})')

  # test complex
  result2 = complex_compute(3.0, 4.0)
  print(f'complex_compute(3.0, 4.0) = {result2:.1f} (strategy: {complex_compute._compiler_strategy})')

  print('\n' + '=' * 80)
  print('Compiler statistics:')
  print('=' * 80)
  print(f'Fast path (TinyCJIT):      {jit_auto.stats["fast"]} functions')
  print(f'Optimized path (IRCompiler): {jit_auto.stats["optimized"]} functions')

  # benchmark compile time
  print('\n' + '=' * 80)
  print('Compile time comparison:')
  print('=' * 80)

  jit_fast = Compiler(mode='fast', verbose=False)
  jit_opt = Compiler(mode='optimized', verbose=False)

  def benchmark_func(a, b):
    return (a + b) * 2.0

  # TinyCJIT compile time
  start = time.perf_counter()
  fast_fn = jit_fast(restype=ctypes.c_float, argtypes=[ctypes.c_float, ctypes.c_float])(benchmark_func)
  fast_time = time.perf_counter() - start

  # IRCompiler compile time
  start = time.perf_counter()
  opt_fn = jit_opt(restype=ctypes.c_float, argtypes=[ctypes.c_float, ctypes.c_float])(benchmark_func)
  opt_time = time.perf_counter() - start

  print(f'TinyCJIT:    {fast_time * 1000:6.2f} ms')
  print(f'IRCompiler:  {opt_time * 1000:6.2f} ms')
  print(f'Ratio:       {opt_time / fast_time:.1f}x (IRCompiler / TinyCJIT)')


if __name__ == '__main__':
  demo_compiler()
