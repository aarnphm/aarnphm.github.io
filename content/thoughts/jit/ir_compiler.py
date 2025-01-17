#!/usr/bin/env python3
"""
IR-based JIT compiler using ir infrastructure

demonstrates full compilation pipeline:
1. Python AST → IR lowering
2. Type inference
3. Optimization passes (constant folding, DCE)
4. IR → C code generation
5. C compilation + ctypes binding

educational comparison to TinyCJIT:
- TinyCJIT: fast single-pass AST → C
- IRCompiler: slower multi-pass with optimizations
"""

from __future__ import annotations
import ast
import ctypes
import functools
import hashlib
import inspect
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ir import (
  BasicBlock,
  BinOp,
  Call,
  Const,
  ConstantFolding,
  CCodegen,
  DeadCodeElimination,
  IRFunction,
  IRType,
  IRValue,
  Jump,
  Load,
  PhiElimination,
  Ret,
  Store,
  TypeInference,
)


# =============================================================================
# AST → IR lowering
# =============================================================================


class IRBuilder:
  """lower Python AST to IR"""

  def __init__(self, func_name: str, verbose: bool = False):
    self.func = IRFunction(func_name)
    self.current_bb: BasicBlock | None = None
    self.verbose = verbose
    self.value_counter = 0
    self.bb_counter = 0
    self.sym_table: dict[str, IRValue] = {}  # variable name → current SSA value

  def fresh_value(self, prefix: str = 'v') -> IRValue:
    """generate fresh SSA value name"""
    self.value_counter += 1
    return IRValue(f'{prefix}{self.value_counter}')

  def fresh_bb(self, prefix: str = 'bb') -> BasicBlock:
    """generate fresh basic block"""
    self.bb_counter += 1
    return self.func.add_block(f'{prefix}{self.bb_counter}')

  def lower_function(
    self, func_def: ast.FunctionDef, param_types: list[IRType], ret_type: IRType
  ) -> IRFunction:
    """lower function AST to IR"""
    # create parameter values
    for arg, ty in zip(func_def.args.args, param_types):
      param_val = IRValue(arg.arg, ty)
      self.func.params.append(param_val)
      self.sym_table[arg.arg] = param_val

    self.func.ret_type = ret_type

    # create entry block
    self.current_bb = self.func.add_block('entry')

    # lower body
    for stmt in func_def.body:
      self.lower_stmt(stmt)

    # add implicit return if function doesn't end with one
    assert self.current_bb
    if not self.current_bb.instrs or not isinstance(self.current_bb.instrs[-1], Ret):
      self.current_bb.append(Ret(value=None))

    return self.func

  def lower_stmt(self, node: ast.stmt) -> None:
    """lower statement to IR"""
    if isinstance(node, ast.For):
      self.lower_for(node)
    elif isinstance(node, ast.Assign):
      self.lower_assign(node)
    elif isinstance(node, ast.AugAssign):
      self.lower_aug_assign(node)
    elif isinstance(node, ast.Return):
      self.lower_return(node)
    elif isinstance(node, ast.Expr):
      # skip docstrings
      if not isinstance(node.value, (ast.Constant, ast.Str)):
        self.lower_expr(node.value)
    else:
      raise NotImplementedError(f'unsupported statement: {ast.dump(node)}')

  def lower_for(self, node: ast.For) -> None:
    """lower for loop to IR basic blocks with proper phi nodes"""
    if not isinstance(node.target, ast.Name):
      raise NotImplementedError('loop target must be simple name')

    # parse range(start, stop, step)
    if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name):
      raise NotImplementedError('only range loops supported')
    if node.iter.func.id != 'range':
      raise NotImplementedError('only range loops supported')

    args = node.iter.args
    if len(args) == 1:
      start_val = self.emit_const(0)
      stop_val = self.lower_expr(args[0])
      step_val = self.emit_const(1)
    elif len(args) == 2:
      start_val = self.lower_expr(args[0])
      stop_val = self.lower_expr(args[1])
      step_val = self.emit_const(1)
    elif len(args) == 3:
      start_val = self.lower_expr(args[0])
      stop_val = self.lower_expr(args[1])
      step_val = self.lower_expr(args[2])
    else:
      raise NotImplementedError('range with >3 arguments not supported')

    # mark loop induction variable as int
    start_val.ty = IRType.INT
    stop_val.ty = IRType.INT
    step_val.ty = IRType.INT

    # create blocks: loop_header, loop_body, loop_exit
    loop_header = self.fresh_bb('loop_header')
    loop_body = self.fresh_bb('loop_body')
    loop_exit = self.fresh_bb('loop_exit')

    # save entry block name for phi node
    assert self.current_bb
    entry_bb_name = self.current_bb.name

    # current block jumps to header
    self.current_bb.append(Jump(target=loop_header.name))
    self.current_bb.succs.append(loop_header.name)
    loop_header.preds.append(self.current_bb.name)

    # loop header: create phi nodes for induction variable and loop-carried vars
    self.current_bb = loop_header
    from ir import Phi

    # snapshot symbol table before loop to identify loop-carried variables
    pre_loop_symbols = dict(self.sym_table)

    iv = self.fresh_value('iv')
    iv.ty = IRType.INT

    # phi node for induction variable
    iv_phi = Phi(result=iv, incoming=[(start_val, entry_bb_name)])
    self.current_bb.append(iv_phi)
    self.sym_table[node.target.id] = iv

    # save header block to add phi nodes later
    saved_loop_header = loop_header

    # check condition: iv < stop
    cond = self.fresh_value('cond')
    cond.ty = IRType.BOOL
    self.current_bb.append(BinOp(result=cond, op='lt', lhs=iv, rhs=stop_val))

    # branch to body or exit
    from ir import Br

    self.current_bb.append(Br(cond=cond, true_bb=loop_body.name, false_bb=loop_exit.name))
    self.current_bb.succs.extend([loop_body.name, loop_exit.name])
    loop_body.preds.append(loop_header.name)
    loop_exit.preds.append(loop_header.name)

    # loop body
    self.current_bb = loop_body
    for stmt in node.body:
      self.lower_stmt(stmt)  # may change current_bb (e.g., nested loops)

    # FIX: after processing loop body, current_bb may have changed
    # (e.g., nested loop leaves us in inner loop's exit block)
    # Track the actual block where we create the back edge
    back_edge_bb = self.current_bb

    # TODO: loop-carried phi nodes for reductions
    # Currently disabled due to issues with nested loops causing infinite compilation loops
    # The dot product test (test_dot.py) shows this works for simple cases,
    # but nested loops and complex control flow need more sophisticated handling
    #
    # For now, reductions (dot product, gemv) will be incorrect with IRCompiler
    # Use TinyCJIT for these cases which handles them correctly

    # increment IV and jump back to header (in actual current block)
    next_iv = self.fresh_value('iv_next')
    next_iv.ty = IRType.INT
    back_edge_bb.append(BinOp(result=next_iv, op='add', lhs=iv, rhs=step_val))

    # add back edge to IV phi node - use actual block name, not loop_body assumption
    iv_phi.incoming.append((next_iv, back_edge_bb.name))

    back_edge_bb.append(Jump(target=loop_header.name))
    back_edge_bb.succs.append(loop_header.name)
    loop_header.preds.append(back_edge_bb.name)

    # continue after loop
    self.current_bb = loop_exit

  def lower_assign(self, node: ast.Assign) -> None:
    """lower assignment to IR"""
    if len(node.targets) != 1:
      raise NotImplementedError('chained assignment not supported')

    value = self.lower_expr(node.value)
    target = node.targets[0]

    if isinstance(target, ast.Name):
      # scalar assignment: create new SSA value
      self.sym_table[target.id] = value
    elif isinstance(target, ast.Subscript):
      # array store
      base = self.lower_expr(target.value)
      index = self.lower_expr(target.slice)
      assert self.current_bb
      self.current_bb.append(Store(ptr=base, index=index, value=value))
    else:
      raise NotImplementedError('unsupported assignment target')

  def lower_aug_assign(self, node: ast.AugAssign) -> None:
    """lower augmented assignment (+=, etc.)"""
    target = node.target
    rhs = self.lower_expr(node.value)

    if isinstance(target, ast.Name):
      lhs = self.sym_table.get(target.id)
      if not lhs:
        raise ValueError(f'undefined variable: {target.id}')
      result = self.fresh_value(target.id)
      op = self._ast_op_to_ir(node.op)
      assert self.current_bb
      self.current_bb.append(BinOp(result=result, op=op, lhs=lhs, rhs=rhs))
      self.sym_table[target.id] = result
    elif isinstance(target, ast.Subscript):
      # ptr[idx] += val  →  tmp = load ptr[idx]; tmp2 = tmp + val; store ptr[idx] = tmp2
      base = self.lower_expr(target.value)
      index = self.lower_expr(target.slice)
      tmp = self.fresh_value('tmp')
      assert self.current_bb
      self.current_bb.append(Load(result=tmp, ptr=base, index=index))
      op = self._ast_op_to_ir(node.op)
      tmp2 = self.fresh_value('tmp')
      self.current_bb.append(BinOp(result=tmp2, op=op, lhs=tmp, rhs=rhs))
      self.current_bb.append(Store(ptr=base, index=index, value=tmp2))
    else:
      raise NotImplementedError('unsupported augmented assignment target')

  def lower_return(self, node: ast.Return) -> None:
    """lower return statement"""
    assert self.current_bb
    if node.value:
      val = self.lower_expr(node.value)
      self.current_bb.append(Ret(value=val))
    else:
      self.current_bb.append(Ret())

  def lower_expr(self, node: ast.expr) -> IRValue:
    """lower expression to IR, returns SSA value"""
    if isinstance(node, ast.Name):
      val = self.sym_table.get(node.id)
      if not val:
        raise ValueError(f'undefined variable: {node.id}')
      return val

    if isinstance(node, ast.Constant):
      return self.emit_const(node.value)

    if isinstance(node, ast.BinOp):
      lhs = self.lower_expr(node.left)
      rhs = self.lower_expr(node.right)
      result = self.fresh_value('binop')
      op = self._ast_op_to_ir(node.op)
      assert self.current_bb
      self.current_bb.append(BinOp(result=result, op=op, lhs=lhs, rhs=rhs))
      return result

    if isinstance(node, ast.Subscript):
      # array load
      base = self.lower_expr(node.value)
      index = self.lower_expr(node.slice)
      result = self.fresh_value('load')
      assert self.current_bb
      self.current_bb.append(Load(result=result, ptr=base, index=index))
      return result

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
      # intrinsic call (min, max, etc.)
      func_name = node.func.id
      args = [self.lower_expr(arg) for arg in node.args]
      result = self.fresh_value('call')
      assert self.current_bb
      self.current_bb.append(Call(result=result, func=func_name, args=args))
      return result

    raise NotImplementedError(f'unsupported expression: {ast.dump(node)}')

  def emit_const(self, value: Any) -> IRValue:
    """emit constant value"""
    result = self.fresh_value('const')
    if isinstance(value, int):
      result.ty = IRType.INT
    elif isinstance(value, float):
      result.ty = IRType.FLOAT
    assert self.current_bb
    self.current_bb.append(Const(result=result, const_value=value))
    return result

  def _ast_op_to_ir(self, op: ast.operator) -> str:
    """convert AST operator to IR opcode"""
    op_map = {
      ast.Add: 'add',
      ast.Sub: 'sub',
      ast.Mult: 'mul',
      ast.Div: 'div',
      ast.Mod: 'mod',
      ast.Lt: 'lt',
      ast.LtE: 'le',
      ast.Gt: 'gt',
      ast.GtE: 'ge',
      ast.Eq: 'eq',
      ast.NotEq: 'ne',
    }
    ir_op = op_map.get(type(op))
    if not ir_op:
      raise NotImplementedError(f'unsupported operator: {type(op).__name__}')
    return ir_op


# =============================================================================
# IR-based JIT compiler
# =============================================================================


def _ctypes_to_irtype(ct: type) -> IRType:
  """convert ctypes type to IRType"""
  if ct == ctypes.c_int or ct == ctypes.c_int32:
    return IRType.INT
  if ct == ctypes.c_float:
    return IRType.FLOAT
  if hasattr(ct, '_type_'):  # pointer type
    elem = ct._type_  # type: ignore
    if elem == ctypes.c_int or elem == ctypes.c_int32:
      return IRType.PTR_INT
    if elem == ctypes.c_float:
      return IRType.PTR_FLOAT
  return IRType.UNKNOWN


class IRCompiler:
  """JIT compiler using IR-based optimization pipeline"""

  def __init__(self, verbose: bool = False, optimize: bool = True):
    self.verbose = verbose
    self.optimize = optimize
    self.work_dir = Path(tempfile.gettempdir()) / 'ir_compiler'
    self.work_dir.mkdir(parents=True, exist_ok=True)

  def __call__(
    self, *, restype: type | None = None, argtypes: Sequence[type], headers: Sequence[str] | None = None
  ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    headers = list(headers) if headers is not None else []

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
      return self._compile(func, restype, argtypes, headers)

    return decorator

  def _compile(
    self, func: Callable[..., Any], restype: type | None, argtypes: Sequence[type], headers: Sequence[str]
  ) -> Callable[..., Any]:
    """compile Python function to native code via IR"""
    if self.verbose:
      print(f'[IRCompiler._compile] starting compilation of {func.__name__}')

    # parse source
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
    if not func_def:
      raise ValueError('unable to find function definition')

    # lower AST → IR
    param_types = [_ctypes_to_irtype(ct) for ct in argtypes]
    ret_ir_type = _ctypes_to_irtype(restype) if restype else IRType.UNKNOWN
    builder = IRBuilder(func_def.name, verbose=self.verbose)
    ir_func = builder.lower_function(func_def, param_types, ret_ir_type)

    if self.verbose:
      print('\n[IRCompiler] Lowered IR:')
      print(ir_func)

    # type inference
    typer = TypeInference()
    typer.infer(ir_func)

    if self.verbose:
      print('\n[IRCompiler] After type inference:')
      print(ir_func)

    # optimization passes
    if self.optimize:
      folder = ConstantFolding()
      folder.run(ir_func)
      dce = DeadCodeElimination()
      dce.run(ir_func)

      if self.verbose:
        print('\n[IRCompiler] After optimizations:')
        print(ir_func)

    # phi elimination (convert out of SSA for C codegen)
    phi_elim = PhiElimination()
    phi_elim.run(ir_func)

    if self.verbose:
      print('\n[IRCompiler] After phi elimination:')
      print(ir_func)

    # generate C code
    codegen = CCodegen()
    c_code = codegen.generate(ir_func)

    # fix C variable names (remove % prefix from SSA names, but not params)
    param_names = {p.name for p in ir_func.params}
    lines = []
    for line in c_code.split('\n'):
      # replace %name with v_name, except for parameters
      for pname in param_names:
        line = line.replace(f'%{pname}', pname)
      line = line.replace('%', 'v_')
      lines.append(line)
    c_code = '\n'.join(lines)

    # wrap in includes + proper signature
    includes = ['#include <stddef.h>', '#include <stdint.h>', '#include <math.h>'] + list(headers)
    full_src = '\n'.join(includes) + '\n\n' + c_code

    if self.verbose:
      print('\n[IRCompiler] Generated C code:')
      print(full_src)

    # compile and link
    key = hashlib.sha256(full_src.encode()).hexdigest()[:16]
    c_path = self.work_dir / f'{func_def.name}_{key}.c'
    so_path = self.work_dir / f'{func_def.name}_{key}.dylib'

    if not so_path.exists():
      c_path.write_text(full_src)
      self._invoke_compiler(c_path, so_path)

    # load and bind
    lib = ctypes.CDLL(str(so_path))
    c_func = getattr(lib, func_def.name)
    c_func.argtypes = list(argtypes)
    c_func.restype = restype

    # wrap with argument conversion (same as TinyCJIT)
    @functools.wraps(func)
    def wrapper(*args):
      converted = []
      keepalive = []
      for value, ctype_expected in zip(args, argtypes):
        conv, keep = self._prepare_arg(value, ctype_expected)
        converted.append(conv)
        if keep is not None:
          keepalive.append(keep)
      return c_func(*converted)

    wrapper.ir_func = ir_func  # type: ignore
    wrapper.c_source = full_src  # type: ignore
    return wrapper

  def _invoke_compiler(self, c_path: Path, so_path: Path) -> None:
    """compile C code to shared object"""
    import shutil

    compiler = shutil.which('clang') or shutil.which('gcc') or shutil.which('cc')
    if not compiler:
      raise RuntimeError('no C compiler found')

    cmd = [compiler, '-O3', '-fPIC', '-shared', str(c_path), '-o', str(so_path), '-lm']
    if sys.platform == 'darwin':
      cmd.insert(1, '-dynamiclib')

    if self.verbose:
      print('[IRCompiler]', ' '.join(cmd))

    subprocess.run(cmd, check=True)

  def _prepare_arg(self, value: Any, ctype: type) -> tuple[Any, Any | None]:
    """prepare argument for ctypes call"""
    if hasattr(ctype, '_type_'):  # pointer
      if isinstance(value, np.ndarray):
        arr = value if value.flags['C_CONTIGUOUS'] else np.ascontiguousarray(value)
        return arr.ctypes.data_as(ctype), arr
      if isinstance(value, ctypes.Array):
        return ctypes.cast(value, ctype), value
    return ctype(value), None


# =============================================================================
# demo
# =============================================================================


def demo_ir_compiler():
  """demonstrate IR-based compilation with optimizations"""
  print('=' * 80)
  print('IR-based JIT Compiler Demo')
  print('=' * 80)

  compiler = IRCompiler(verbose=True, optimize=True)

  # simple example without loops
  @compiler(restype=ctypes.c_float, argtypes=[ctypes.c_float, ctypes.c_float])
  def add_mul(a, b):
    """compute (a + b) * 2"""
    tmp = a + b
    return tmp * 2.0

  # test it
  print('\n' + '=' * 80)
  print('Test add_mul:')
  print('=' * 80)
  result = add_mul(3.0, 4.0)
  print(f'add_mul(3.0, 4.0) = {result}')
  print(f'Expected: {(3.0 + 4.0) * 2.0}')


if __name__ == '__main__':
  demo_ir_compiler()
