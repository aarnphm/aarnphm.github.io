#!/usr/bin/env python3
"""
trace-to-native demo

record the hot path, turn it into C, compile it, and execute the native stub.
this mirrors the "trace, lower, execute" loop used by meta-tracing engines, but
keeps the output readable: the generated C is written to stdout when DEBUG=2.
"""

from __future__ import annotations

import ast
import dis
import inspect
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence

import ctypes


_CTYPE_TO_C = {
  ctypes.c_float: 'float',
  ctypes.c_double: 'double',
  ctypes.c_int: 'int',
  ctypes.c_uint32: 'uint32_t',
  ctypes.c_int32: 'int32_t',
  ctypes.c_size_t: 'size_t',
  ctypes.c_int64: 'int64_t',
  ctypes.c_uint64: 'uint64_t',
  ctypes.c_uint8: 'uint8_t',
  ctypes.c_int8: 'int8_t',
  ctypes.c_void_p: 'void*',
}


def _is_pointer(tp: type | None) -> bool:
  return isinstance(tp, type) and issubclass(tp, ctypes._Pointer)


def _ctype_to_cstring(tp: type | None) -> str:
  if tp is None:
    return 'void'
  if _is_pointer(tp):
    return f'{_ctype_to_cstring(tp._type_)}*'  # type: ignore[attr-defined]
  mapped = _CTYPE_TO_C.get(tp)
  if mapped:
    return mapped
  raise TypeError(f'unsupported ctypes type {tp!r}')


@dataclass
class TraceSegment:
  index_var: str
  start: str
  stop: str
  body: str


def _detect_compiler() -> str:
  for candidate in ('clang', 'gcc', 'cc'):
    path = shutil.which(candidate)
    if path:
      return path
  raise RuntimeError('no C compiler (clang/gcc/cc) found on PATH')


def _shared_suffix() -> str:
  if sys.platform == 'darwin':
    return '.dylib'
  if sys.platform.startswith('win'):
    return '.dll'
  return '.so'


def describe(func: Callable[..., Any]) -> None:
  print('\n' + '=' * 80)
  print(f'python hot loop: {func.__name__}')
  print('=' * 80)
  print(textwrap.dedent(inspect.getsource(func)))
  print('bytecode disassembly:\n')
  dis.dis(func)


class LoopTracer(ast.NodeVisitor):
  def __init__(self) -> None:
    self.segments: list[TraceSegment] = []
    self.expr = ExpressionEmitter()

  def visit_For(self, node: ast.For) -> None:
    if not isinstance(node.target, ast.Name):
      raise NotImplementedError('only range-based for loops are supported')
    start, stop = self._parse_range(node.iter)
    body_lines = []
    emitter = StatementEmitter(self.expr)
    for stmt in node.body:
      body_lines.append(emitter.emit(stmt))
    body = '\n'.join(line for line in body_lines if line)
    self.segments.append(TraceSegment(node.target.id, start, stop, body))

  def _parse_range(self, call: ast.AST) -> tuple[str, str]:
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
      raise NotImplementedError('loop must be range(...) based')
    args = call.args
    if len(args) == 1:
      return '0', self.expr.emit(args[0])
    if len(args) == 2:
      return self.expr.emit(args[0]), self.expr.emit(args[1])
    raise NotImplementedError('range step currently unsupported')


class ExpressionEmitter(ast.NodeVisitor):
  def emit(self, node: ast.AST) -> str:
    return self.visit(node)

  def visit_Name(self, node: ast.Name) -> str:
    return node.id

  def visit_Constant(self, node: ast.Constant) -> str:
    if isinstance(node.value, bool):
      return '1' if node.value else '0'
    if isinstance(node.value, (int, float)):
      return repr(node.value)
    raise NotImplementedError(f'constant {node.value!r} unsupported')

  def visit_BinOp(self, node: ast.BinOp) -> str:
    op = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}.get(type(node.op))
    if op is None:
      raise NotImplementedError(f'binary op {ast.dump(node.op)} unsupported')
    return f'({self.emit(node.left)} {op} {self.emit(node.right)})'

  def visit_Subscript(self, node: ast.Subscript) -> str:
    return f'{self.emit(node.value)}[{self.emit(node.slice)}]'

  def visit_Index(self, node: ast.Index) -> str:  # pragma: no cover - py<3.9
    return self.emit(node.value)

  def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
    if isinstance(node.op, ast.USub):
      return f'(-{self.emit(node.operand)})'
    if isinstance(node.op, ast.UAdd):
      return f'(+{self.emit(node.operand)})'
    raise NotImplementedError(f'unary op {ast.dump(node.op)} unsupported')

  def generic_visit(self, node: ast.AST) -> str:  # pragma: no cover - debug
    raise NotImplementedError(f'expression {ast.dump(node)} unsupported')


class StatementEmitter:
  def __init__(self, expr: ExpressionEmitter) -> None:
    self.expr = expr

  def emit(self, stmt: ast.stmt) -> str:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
      target = stmt.targets[0]
      if isinstance(target, ast.Subscript):
        return f'{self.expr.emit(target)} = {self.expr.emit(stmt.value)};'
      if isinstance(target, ast.Name):
        return f'float {target.id} = {self.expr.emit(stmt.value)};'
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
      call = stmt.value
      if isinstance(call.func, ast.Name):
        args = ', '.join(self.expr.emit(arg) for arg in call.args)
        return f'{call.func.id}({args});'
    if isinstance(stmt, ast.AugAssign):
      op = {ast.Add: '+=', ast.Sub: '-=', ast.Mult: '*=', ast.Div: '/='}.get(type(stmt.op))
      if op is None:
        raise NotImplementedError('augmented assign unsupported')
      return f'{self.expr.emit(stmt.target)} {op} {self.expr.emit(stmt.value)};'
    if isinstance(stmt, ast.Return):
      if stmt.value is None:
        return 'return;'
      return f'return {self.expr.emit(stmt.value)};'
    raise NotImplementedError(f'statement {ast.dump(stmt)} unsupported')


class CTraceCompiler:
  def __init__(self, compiler: str | None = None, verbose: bool = False) -> None:
    self.compiler = compiler or _detect_compiler()
    self.work_dir = Path(tempfile.gettempdir()) / 'tracing_cjit'
    self.work_dir.mkdir(parents=True, exist_ok=True)
    env_debug = int(os.environ.get('DEBUG', '0')) if os.environ.get('DEBUG', '0').isdigit() else 0
    self.debug = max(env_debug, 1 if verbose else 0)

  def compile(self, func: Callable[..., Any], restype: type | None, argtypes: Sequence[type]) -> Callable:
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    tracer = LoopTracer()
    tracer.visit(tree)
    if not tracer.segments:
      raise ValueError('no supported loops detected to trace')

    print('\n[TracingJIT] recorded trace:')
    for seg in tracer.segments:
      print(f'  for {seg.index_var} in [{seg.start}, {seg.stop}):')
      for line in seg.body.splitlines():
        print(f'    {line}')

    c_body = []
    for seg in tracer.segments:
      c_body.append(f'for (int {seg.index_var} = {seg.start}; {seg.index_var} < {seg.stop}; {seg.index_var}++) {{')
      c_body.append(textwrap.indent(seg.body, '  '))
      c_body.append('}')

    arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
    arg_decls = [f'{_ctype_to_cstring(tp)} {name}' for name, tp in zip(arg_names, argtypes)]
    ret_decl = _ctype_to_cstring(restype)
    c_src = '\n'.join([
      '#include <stddef.h>',
      '#include <stdint.h>',
      '#include <math.h>',
      '',
      f'__attribute__((visibility("default"))) {ret_decl} {func.__name__}({", ".join(arg_decls)}) {{',
      textwrap.indent('\n'.join(c_body), '  '),
      '}',
      '',
    ])

    if self.debug >= 2:
      print('\n[TracingJIT] generated C:\n' + c_src)

    key = hashlib.sha256(c_src.encode('utf-8')).hexdigest()[:16]
    c_path = self.work_dir / f'{func.__name__}_{key}.c'
    so_path = self.work_dir / f'{func.__name__}_{key}{_shared_suffix()}'
    if not so_path.exists():
      c_path.write_text(c_src)
      cmd = [self.compiler, str(c_path), '-shared', '-fPIC', '-O3', '-o', str(so_path), '-lm']
      if sys.platform == 'darwin':
        cmd.insert(1, '-dynamiclib')
      if self.debug >= 1:
        print('[TracingJIT]', ' '.join(cmd))
      subprocess.run(cmd, check=True)
    elif self.debug >= 1:
      print(f'[TracingJIT] reuse cached {so_path}')

    lib = ctypes.CDLL(str(so_path))
    native = getattr(lib, func.__name__)
    native.argtypes = list(argtypes)
    native.restype = restype
    return native


def demo_tracing_vector_add() -> None:
  if sys.platform.startswith('win'):
    print('tracing demo skipped on Windows (expects clang/gcc toolchain)')
    return

  def python_loop(a, b, out, n):
    for i in range(n):
      out[i] = a[i] + b[i]

  describe(python_loop)

  compiler = CTraceCompiler()
  native = compiler.compile(
    python_loop,
    restype=None,
    argtypes=[
      ctypes.POINTER(ctypes.c_float),
      ctypes.POINTER(ctypes.c_float),
      ctypes.POINTER(ctypes.c_float),
      ctypes.c_int,
    ],
  )

  for n in (100_000, 1_000_000, 8_000_000):
    Array = ctypes.c_float * n
    a = Array(*[float(i) for i in range(n)])
    b = Array(*[float(n - i) for i in range(n)])
    out_py = Array(*([0.0] * n))
    out_native = Array(*([0.0] * n))
    ptr_a = ctypes.cast(a, ctypes.POINTER(ctypes.c_float))
    ptr_b = ctypes.cast(b, ctypes.POINTER(ctypes.c_float))
    ptr_out = ctypes.cast(out_native, ctypes.POINTER(ctypes.c_float))

    def python_exec():
      for i in range(n):
        out_py[i] = a[i] + b[i]

    start = time.perf_counter()
    python_exec()
    python_time = time.perf_counter() - start

    start = time.perf_counter()
    native(ptr_a, ptr_b, ptr_out, n)
    native_time = time.perf_counter() - start

    max_error = max(abs(out_py[i] - out_native[i]) for i in range(n))
    speedup = python_time / native_time if native_time else float('inf')

    print('\n' + '=' * 80)
    print('TracingJIT vector add')
    print(f'elements       : {n:,}')
    print(f'python loop    : {python_time:.6f} s')
    print(f'native kernel  : {native_time:.6f} s')
    print(f'speedup        : {speedup:,.2f}x')
    print(f'max abs error  : {max_error}')


def main() -> None:
  demo_tracing_vector_add()


if __name__ == '__main__':
  raise SystemExit(main())
