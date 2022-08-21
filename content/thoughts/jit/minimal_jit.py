#!/usr/bin/env python3
"""
tinygrad-inspired jit demo

this note keeps things tangible: take a tiny python kernel, show its bytecode,
lower it to C, and execute the compiled version. ``DEBUG=1`` prints the compiler
command; ``DEBUG=2`` also dumps the generated C source.
"""

from __future__ import annotations

import ast
import dis
import functools
import hashlib
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence

import ctypes
import numpy as np


# ---------------------------------------------------------------------------
# helpers to inspect the python side
# ---------------------------------------------------------------------------


def describe_python_kernel(func: Callable[..., Any]) -> None:
  """Print the source and disassembly for ``func``."""

  print('\n' + '=' * 80)
  print(f'python kernel: {func.__name__}')
  print('=' * 80)
  try:
    src = textwrap.dedent(inspect.getsource(func))
  except OSError:
    src = '<source unavailable>'
  print(src)

  print('bytecode disassembly:\n')
  dis.dis(func)


# ---------------------------------------------------------------------------
# TinyCJIT implementation
# ---------------------------------------------------------------------------


_BINOP_MAP = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.Mod: '%'}
_AUGMENTED_MAP = {ast.Add: '+=', ast.Sub: '-=', ast.Mult: '*=', ast.Div: '/='}
_UNARY_MAP = {ast.UAdd: '+', ast.USub: '-'}


def _detect_compiler() -> str:
  for candidate in ('clang', 'gcc', 'cc'):
    path = shutil.which(candidate)
    if path:
      return path
  raise RuntimeError('no suitable C compiler found (tried clang/gcc/cc)')


def _shared_suffix() -> str:
  if sys.platform == 'darwin':
    return '.dylib'
  if sys.platform.startswith('win'):
    return '.dll'
  return '.so'


def _is_pointer(tp: type) -> bool:
  return isinstance(tp, type) and issubclass(tp, ctypes._Pointer)


def _deref(tp: type) -> type:
  while _is_pointer(tp):
    tp = tp._type_  # type: ignore[attr-defined]
  return tp


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


def _ctype_to_cstring(tp: type | None) -> str:
  if tp is None:
    return 'void'
  if _is_pointer(tp):
    return f'{_ctype_to_cstring(tp._type_)}*'  # type: ignore[attr-defined]
  mapped = _CTYPE_TO_C.get(tp)
  if mapped:
    return mapped
  raise TypeError(f'unsupported ctypes type {tp!r}')


class _ExprEmitter(ast.NodeVisitor):
  def __init__(self):
    super().__init__()

  def visit_Name(self, node: ast.Name) -> str:
    return node.id

  def visit_Constant(self, node: ast.Constant) -> str:
    if isinstance(node.value, bool):
      return '1' if node.value else '0'
    if isinstance(node.value, (int, float)):
      return repr(node.value)
    raise NotImplementedError(f'unsupported constant {node.value!r}')

  def visit_Subscript(self, node: ast.Subscript) -> str:
    base = self.visit(node.value)
    if isinstance(node.slice, ast.Slice):
      raise NotImplementedError('slice syntax not supported in kernels')
    index = self.visit(node.slice)
    return f'{base}[{index}]'

  def visit_BinOp(self, node: ast.BinOp) -> str:
    if isinstance(node.op, ast.Pow):
      return f'pow({self.visit(node.left)}, {self.visit(node.right)})'
    op = _BINOP_MAP.get(type(node.op))
    if not op:
      raise NotImplementedError(f'binary op {ast.dump(node.op)} not supported')
    return f'({self.visit(node.left)} {op} {self.visit(node.right)})'

  def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
    op = _UNARY_MAP.get(type(node.op))
    if op is None:
      raise NotImplementedError(f'unary op {ast.dump(node.op)} not supported')
    return f'({op}{self.visit(node.operand)})'

  def visit_Call(self, node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
      # intrinsics: min, max, abs, sqrt, etc.
      func_name = node.func.id
      args = ', '.join(self.visit(arg) for arg in node.args)
      # map Python functions to C equivalents
      c_name = {'min': 'fminf', 'max': 'fmaxf', 'abs': 'fabsf', 'sqrt': 'sqrtf'}.get(func_name, func_name)
      return f'{c_name}({args})'
    raise NotImplementedError(f'unsupported call: {ast.dump(node)}')

  def visit_Index(self, node: ast.Index) -> str:  # pragma: no cover - py<3.9
    return self.visit(node.value)

  def generic_visit(self, node: ast.AST) -> str:  # pragma: no cover - debug
    raise NotImplementedError(f'unsupported expression: {ast.dump(node)}')


class _CBodyGenerator(ast.NodeVisitor):
  def __init__(self):
    super().__init__()
    self.lines: list[str] = []
    self.indent = 1
    self.locals: set[str] = set()
    self.expr = _ExprEmitter()

  def emit(self, text: str) -> None:
    self.lines.append('  ' * self.indent + text)

  def generate(self, body: Sequence[ast.stmt]) -> str:
    for stmt in body:
      self.visit(stmt)
    return '\n'.join(self.lines) + ('\n' if self.lines else '')

  def visit_For(self, node: ast.For) -> None:
    if not isinstance(node.target, ast.Name):
      raise NotImplementedError('loop target must be a simple name')
    start, stop, step = self._parse_range(node.iter)
    idx = node.target.id
    self.emit(f'for (int {idx} = {start}; {idx} < {stop}; {idx} += {step}) {{')
    self.indent += 1
    for stmt in node.body:
      self.visit(stmt)
    self.indent -= 1
    self.emit('}')

  def _parse_range(self, node: ast.AST) -> tuple[str, str, str]:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'range':
      raise NotImplementedError('only range(...) loops are supported')
    args = node.args
    if len(args) == 1:
      return '0', self.expr.visit(args[0]), '1'
    if len(args) == 2:
      return self.expr.visit(args[0]), self.expr.visit(args[1]), '1'
    if len(args) == 3:
      return self.expr.visit(args[0]), self.expr.visit(args[1]), self.expr.visit(args[2])
    raise NotImplementedError('range() with more than three arguments is unsupported')

  def visit_Assign(self, node: ast.Assign) -> None:
    if len(node.targets) != 1:
      raise NotImplementedError('chained assignment unsupported')
    target = node.targets[0]
    expr = self.expr.visit(node.value)
    if isinstance(target, ast.Name):
      if target.id not in self.locals:
        self.locals.add(target.id)
        self.emit(f'float {target.id} = {expr};')  # treat temps as float
      else:
        self.emit(f'{target.id} = {expr};')
    elif isinstance(target, ast.Subscript):
      self.emit(f'{self.expr.visit(target)} = {expr};')
    else:
      raise NotImplementedError('assignment target not supported')

  def visit_AugAssign(self, node: ast.AugAssign) -> None:
    op = _AUGMENTED_MAP.get(type(node.op))
    if op is None:
      raise NotImplementedError('augmented assignment not supported')
    lhs = self.expr.visit(node.target)
    rhs = self.expr.visit(node.value)
    self.emit(f'{lhs} {op} {rhs};')

  def visit_Return(self, node: ast.Return) -> None:
    if node.value is None:
      self.emit('return;')
    else:
      self.emit(f'return {self.expr.visit(node.value)};')

  def visit_Expr(self, node: ast.Expr) -> None:
    # skip docstrings
    if isinstance(node.value, (ast.Constant, ast.Str)):
      return
    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
      call = f'{node.value.func.id}(' + ', '.join(self.expr.visit(arg) for arg in node.value.args) + ');'
      self.emit(call)
      return
    raise NotImplementedError('only function-call expressions are allowed')

  def generic_visit(self, node: ast.AST) -> None:  # pragma: no cover
    raise NotImplementedError(f'unsupported statement: {ast.dump(node)}')


class TinyCJIT:
  """Lower a constrained Python kernel to native code via clang/gcc."""

  def __init__(
    self,
    compiler: str | None = None,
    work_dir: Path | None = None,
    verbose: bool = False,
    extra_cflags: Sequence[str] | None = None,
    link_flags: Sequence[str] | None = None,
  ) -> None:
    self.compiler = compiler or _detect_compiler()
    self.work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir()) / 'minimal_cjit'
    self.work_dir.mkdir(parents=True, exist_ok=True)
    self.extra_cflags = list(extra_cflags) if extra_cflags is not None else ['-O3', '-std=c11', '-fPIC']
    self.link_flags = list(link_flags) if link_flags is not None else ['-lm']
    env_level = int(os.environ.get('DEBUG', '0')) if os.environ.get('DEBUG', '0').isdigit() else 0
    self.debug_level = max(env_level, 1 if verbose else 0)

  def __call__(
    self, *, restype: type | None = None, argtypes: Sequence[type], headers: Sequence[str] | None = None
  ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    headers = list(headers) if headers is not None else []

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
      return self._compile_function(func, restype, argtypes, headers)

    return decorator

  # compilation ----------------------------------------------------------
  def _compile_function(
    self, func: Callable[..., Any], restype: type | None, argtypes: Sequence[type], headers: Sequence[str]
  ) -> Callable[..., Any]:
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
    if func_def is None:
      raise ValueError('unable to recover function definition')
    if len(func_def.args.args) != len(argtypes):
      raise ValueError('argtypes length must match python signature')

    arg_decls = [f'{_ctype_to_cstring(tp)} {py_arg.arg}' for py_arg, tp in zip(func_def.args.args, argtypes)]
    body_src = _CBodyGenerator().generate(func_def.body)

    includes = ['#include <stddef.h>', '#include <stdint.h>', '#include <math.h>'] + list(headers)
    ret_decl = _ctype_to_cstring(restype)
    fn_decl = f'__attribute__((visibility("default"))) {ret_decl} {func_def.name}({", ".join(arg_decls)})'
    module_src = '\n'.join(includes) + '\n\n' + fn_decl + ' {\n' + body_src + '}\n'

    if self.debug_level >= 2:
      print('\n[TinyCJIT] generated C source:\n' + module_src)

    key = hashlib.sha256(module_src.encode('utf-8')).hexdigest()[:16]
    c_path = self.work_dir / f'{func_def.name}_{key}.c'
    so_path = self.work_dir / f'{func_def.name}_{key}{_shared_suffix()}'
    if not so_path.exists():
      c_path.write_text(module_src)
      self._invoke_compiler(c_path, so_path)
    elif self.debug_level >= 1:
      print(f'[TinyCJIT] reuse cached {so_path}')

    library = ctypes.CDLL(str(so_path))
    c_callable = getattr(library, func_def.name)
    c_callable.argtypes = list(argtypes)
    c_callable.restype = restype

    @functools.wraps(func)
    def wrapper(*args):
      if len(args) != len(argtypes):
        raise TypeError(f'expected {len(argtypes)} arguments, got {len(args)}')
      converted: list[Any] = []
      keepalive: list[Any] = []
      for value, ctype_expected in zip(args, argtypes):
        converted_arg, keep = self._prepare_argument(value, ctype_expected)
        converted.append(converted_arg)
        if keep is not None:
          keepalive.append(keep)
      return c_callable(*converted)

    wrapper.c_func = c_callable  # type: ignore[attr-defined]
    wrapper.c_source = module_src  # type: ignore[attr-defined]
    wrapper.c_library_path = so_path  # type: ignore[attr-defined]
    wrapper.compiler = self.compiler  # type: ignore[attr-defined]
    return wrapper

  def _invoke_compiler(self, c_path: Path, so_path: Path) -> None:
    cmd = [self.compiler, str(c_path), '-o', str(so_path)]
    flags = list(self.extra_cflags)
    if sys.platform == 'darwin':
      flags.insert(0, '-dynamiclib')
    elif sys.platform.startswith('win'):
      flags.insert(0, '-shared')
      flags.append('-lmsvcrt')
    else:
      flags.insert(0, '-shared')
    cmd[1:1] = flags
    if sys.platform != 'win32':
      cmd.extend(self.link_flags)
    if self.debug_level >= 1:
      print('[TinyCJIT]', ' '.join(cmd))
    subprocess.run(cmd, check=True)

  def _prepare_argument(self, value: Any, ctype_expected: type) -> tuple[Any, Any | None]:
    if _is_pointer(ctype_expected):
      if isinstance(value, np.ndarray):
        arr = value if value.flags['C_CONTIGUOUS'] else np.ascontiguousarray(value)
        return arr.ctypes.data_as(ctype_expected), arr
      if isinstance(value, ctypes.Array):
        return ctypes.cast(value, ctype_expected), value
      if isinstance(value, ctypes._Pointer):  # type: ignore[arg-type]
        return value, value
      raise TypeError(f'cannot convert {value!r} to {ctype_expected}')
    return ctype_expected(value), None


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


def demo_cjit_vector_add() -> None:
  if sys.platform.startswith('win'):
    print('TinyCJIT demo skipped on Windows (requires clang/gcc toolchain).')
    return

  jit = TinyCJIT()

  def python_kernel(a, b, out, n):
    for i in range(n):
      out[i] = a[i] + b[i]

  describe_python_kernel(python_kernel)

  vector_add = jit(
    restype=None,
    argtypes=[
      ctypes.POINTER(ctypes.c_float),
      ctypes.POINTER(ctypes.c_float),
      ctypes.POINTER(ctypes.c_float),
      ctypes.c_int,
    ],
  )(python_kernel)

  for n in (100_000, 1_000_000, 8_000_000):
    Array = ctypes.c_float * n
    a = Array(*[float(i) for i in range(n)])
    b = Array(*[float(n - i) for i in range(n)])
    out_py = Array(*([0.0] * n))
    out_native = Array(*([0.0] * n))

    def python_impl(inp_a, inp_b, out_arr, count):
      for idx in range(count):
        out_arr[idx] = inp_a[idx] + inp_b[idx]

    start = time.perf_counter()
    python_impl(a, b, out_py, n)
    python_time = time.perf_counter() - start

    start = time.perf_counter()
    vector_add(a, b, out_native, n)
    native_time = time.perf_counter() - start

    max_error = max(abs(out_py[i] - out_native[i]) for i in range(n))
    speedup = python_time / native_time if native_time else float('inf')

    print('\n' + '=' * 80)
    print('TinyCJIT vector add')
    print(f'elements       : {n:,}')
    print(f'python loop    : {python_time:.6f} s')
    print(f'native kernel  : {native_time:.6f} s')
    print(f'speedup        : {speedup:,.2f}x')
    print(f'max abs error  : {max_error}')


def main() -> None:
  demo_cjit_vector_add()


if __name__ == '__main__':
  raise SystemExit(main())
