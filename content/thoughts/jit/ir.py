#!/usr/bin/env python3
"""
simple IR for educational JIT compiler

demonstrates:
- SSA form (static single assignment)
- basic blocks and control flow graph
- type inference via constraint solving
- simple optimization passes (constant folding, DCE)
- lowering to C

pedagogical focus: show the mechanics of a real compiler IR without the
complexity of LLVM. each pass is ~50 lines, readable, hackable.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# =============================================================================
# IR types
# =============================================================================


class IRType(Enum):
  """IR type system"""

  INT = auto()
  FLOAT = auto()
  BOOL = auto()
  PTR_INT = auto()
  PTR_FLOAT = auto()
  UNKNOWN = auto()

  def is_pointer(self) -> bool:
    return self in (IRType.PTR_INT, IRType.PTR_FLOAT)

  def deref(self) -> IRType:
    return {IRType.PTR_INT: IRType.INT, IRType.PTR_FLOAT: IRType.FLOAT}.get(self, IRType.UNKNOWN)

  def to_c(self) -> str:
    return {
      IRType.INT: 'int',
      IRType.FLOAT: 'float',
      IRType.BOOL: 'int',
      IRType.PTR_INT: 'int*',
      IRType.PTR_FLOAT: 'float*',
    }[self]


# =============================================================================
# IR instructions (SSA form)
# =============================================================================


@dataclass(unsafe_hash=True)
class IRValue:
  """SSA value with unique name and type"""

  name: str
  ty: IRType = field(default=IRType.UNKNOWN, hash=False, compare=False)

  def __str__(self) -> str:
    return f'%{self.name}'


@dataclass
class IRInstr:
  """base class for IR instructions"""

  result: IRValue | None = None

  def uses(self) -> list[IRValue]:
    """values used by this instruction"""
    return []

  def replace_use(self, old: IRValue, new: IRValue) -> None:
    """replace uses of old with new"""
    pass


@dataclass
class BinOp(IRInstr):
  """binary operation: result = lhs op rhs"""

  op: str = ''
  lhs: IRValue | None = None
  rhs: IRValue | None = None

  def uses(self) -> list[IRValue]:
    return [v for v in [self.lhs, self.rhs] if v]

  def replace_use(self, old: IRValue, new: IRValue) -> None:
    if self.lhs == old:
      self.lhs = new
    if self.rhs == old:
      self.rhs = new

  def __str__(self) -> str:
    return f'{self.result} = {self.op} {self.lhs}, {self.rhs}'


@dataclass
class UnaryOp(IRInstr):
  """unary operation: result = op operand"""

  op: str = ''
  operand: IRValue | None = None

  def uses(self) -> list[IRValue]:
    return [self.operand] if self.operand else []

  def replace_use(self, old: IRValue, new: IRValue) -> None:
    if self.operand == old:
      self.operand = new

  def __str__(self) -> str:
    return f'{self.result} = {self.op} {self.operand}'


@dataclass
class Call(IRInstr):
  """function call: result = func(args)"""

  func: str = ''
  args: list[IRValue] = field(default_factory=list)

  def uses(self) -> list[IRValue]:
    return self.args

  def replace_use(self, old: IRValue, new: IRValue) -> None:
    self.args = [new if a == old else a for a in self.args]

  def __str__(self) -> str:
    args_str = ', '.join(str(a) for a in self.args)
    return f'{self.result} = call {self.func}({args_str})'


@dataclass
class Load(IRInstr):
  """load from memory: result = ptr[index]"""

  ptr: IRValue | None = None
  index: IRValue | None = None

  def uses(self) -> list[IRValue]:
    return [v for v in [self.ptr, self.index] if v]

  def replace_use(self, old: IRValue, new: IRValue) -> None:
    if self.ptr == old:
      self.ptr = new
    if self.index == old:
      self.index = new

  def __str__(self) -> str:
    return f'{self.result} = load {self.ptr}[{self.index}]'


@dataclass
class Store(IRInstr):
  """store to memory: ptr[index] = value"""

  ptr: IRValue | None = None
  index: IRValue | None = None
  value: IRValue | None = None

  def uses(self) -> list[IRValue]:
    return [v for v in [self.ptr, self.index, self.value] if v]

  def replace_use(self, old: IRValue, new: IRValue) -> None:
    if self.ptr == old:
      self.ptr = new
    if self.index == old:
      self.index = new
    if self.value == old:
      self.value = new

  def __str__(self) -> str:
    return f'store {self.ptr}[{self.index}] = {self.value}'


@dataclass
class Phi(IRInstr):
  """phi node: result = phi [val1, bb1], [val2, bb2], ..."""

  incoming: list[tuple[IRValue, str]] = field(default_factory=list)  # (value, block_name)

  def uses(self) -> list[IRValue]:
    return [v for v, _ in self.incoming]

  def replace_use(self, old: IRValue, new: IRValue) -> None:
    self.incoming = [(new if v == old else v, bb) for v, bb in self.incoming]

  def __str__(self) -> str:
    pairs = ', '.join(f'[{v}, {bb}]' for v, bb in self.incoming)
    return f'{self.result} = phi {pairs}'


@dataclass
class Br(IRInstr):
  """conditional branch: br cond, true_bb, false_bb"""

  cond: IRValue | None = None
  true_bb: str = ''
  false_bb: str = ''

  def uses(self) -> list[IRValue]:
    return [self.cond] if self.cond else []

  def replace_use(self, old: IRValue, new: IRValue) -> None:
    if self.cond == old:
      self.cond = new

  def __str__(self) -> str:
    if self.cond:
      return f'br {self.cond}, {self.true_bb}, {self.false_bb}'
    return f'br {self.true_bb}'


@dataclass
class Jump(IRInstr):
  """unconditional jump: jump target_bb"""

  target: str = ''

  def __str__(self) -> str:
    return f'jump {self.target}'


@dataclass
class Ret(IRInstr):
  """return: ret value"""

  value: IRValue | None = None

  def uses(self) -> list[IRValue]:
    return [self.value] if self.value else []

  def replace_use(self, old: IRValue, new: IRValue) -> None:
    if self.value == old:
      self.value = new

  def __str__(self) -> str:
    return f'ret {self.value}' if self.value else 'ret'


@dataclass
class Const(IRInstr):
  """constant: result = const value"""

  const_value: Any = None

  def __str__(self) -> str:
    return f'{self.result} = const {self.const_value}'


# =============================================================================
# basic blocks and CFG
# =============================================================================


@dataclass
class BasicBlock:
  """basic block: sequence of instructions with single entry/exit"""

  name: str
  instrs: list[IRInstr] = field(default_factory=list)
  preds: list[str] = field(default_factory=list)  # predecessor block names
  succs: list[str] = field(default_factory=list)  # successor block names

  def append(self, instr: IRInstr) -> None:
    self.instrs.append(instr)

  def __str__(self) -> str:
    lines = [f'{self.name}:']
    for instr in self.instrs:
      lines.append(f'  {instr}')
    return '\n'.join(lines)


@dataclass
class IRFunction:
  """IR function: control flow graph of basic blocks"""

  name: str
  params: list[IRValue] = field(default_factory=list)
  blocks: dict[str, BasicBlock] = field(default_factory=dict)
  ret_type: IRType = IRType.UNKNOWN

  def add_block(self, name: str) -> BasicBlock:
    bb = BasicBlock(name)
    self.blocks[name] = bb
    return bb

  def __str__(self) -> str:
    lines = [f'function {self.name}({", ".join(str(p) for p in self.params)}):']
    for bb in self.blocks.values():
      lines.append(str(bb))
    return '\n'.join(lines)


# =============================================================================
# type inference via constraint solving
# =============================================================================


class TypeInference:
  """infer types for IR values via unification"""

  def __init__(self):
    self.constraints: list[tuple[IRValue, IRValue]] = []  # v1 == v2
    self.type_assigns: dict[IRValue, IRType] = {}  # v : type

  def infer(self, func: IRFunction) -> None:
    """infer types for all values in function"""
    # collect constraints from instructions
    for bb in func.blocks.values():
      for instr in bb.instrs:
        self._collect_constraints(instr)

    # propagate param types
    for param in func.params:
      if param.ty != IRType.UNKNOWN:
        self.type_assigns[param] = param.ty

    # solve constraints via unification
    changed = True
    while changed:
      changed = False
      for v1, v2 in self.constraints:
        t1 = self.type_assigns.get(v1, v1.ty)
        t2 = self.type_assigns.get(v2, v2.ty)
        if t1 != IRType.UNKNOWN and t2 == IRType.UNKNOWN:
          self.type_assigns[v2] = t1
          v2.ty = t1
          changed = True
        elif t2 != IRType.UNKNOWN and t1 == IRType.UNKNOWN:
          self.type_assigns[v1] = t2
          v1.ty = t2
          changed = True

    # assign inferred types back to values
    for v, ty in self.type_assigns.items():
      v.ty = ty

  def _collect_constraints(self, instr: IRInstr) -> None:
    """collect type constraints from instruction"""
    if isinstance(instr, BinOp):
      # result has same type as operands
      if instr.result and instr.lhs:
        self.constraints.append((instr.result, instr.lhs))
      if instr.result and instr.rhs:
        self.constraints.append((instr.result, instr.rhs))
      # for int ops, mark as int
      if instr.lhs and instr.lhs.ty == IRType.INT:
        self.type_assigns[instr.lhs] = IRType.INT
    elif isinstance(instr, Load):
      # result has type of pointer element
      if instr.result and instr.ptr and instr.ptr.ty.is_pointer():
        self.type_assigns[instr.result] = instr.ptr.ty.deref()
    elif isinstance(instr, Const):
      # constant type determined by value
      if instr.result and isinstance(instr.const_value, int):
        self.type_assigns[instr.result] = IRType.INT
        instr.result.ty = IRType.INT
      elif instr.result and isinstance(instr.const_value, float):
        self.type_assigns[instr.result] = IRType.FLOAT
        instr.result.ty = IRType.FLOAT


# =============================================================================
# optimization passes
# =============================================================================


class ConstantFolding:
  """fold constant expressions at IR level"""

  def run(self, func: IRFunction) -> bool:
    """returns True if any changes made"""
    changed = False
    const_vals: dict[IRValue, Any] = {}

    # collect constant values
    for bb in func.blocks.values():
      for instr in bb.instrs:
        if isinstance(instr, Const) and instr.result:
          const_vals[instr.result] = instr.const_value

    # fold binary operations on constants
    for bb in func.blocks.values():
      new_instrs = []
      for instr in bb.instrs:
        if isinstance(instr, BinOp) and instr.lhs in const_vals and instr.rhs in const_vals:
          lhs_val = const_vals[instr.lhs]
          rhs_val = const_vals[instr.rhs]
          result_val = self._eval_binop(instr.op, lhs_val, rhs_val)
          if result_val is not None and instr.result:
            new_instrs.append(Const(result=instr.result, const_value=result_val))
            const_vals[instr.result] = result_val
            changed = True
            continue
        new_instrs.append(instr)
      bb.instrs = new_instrs

    return changed

  def _eval_binop(self, op: str, lhs: Any, rhs: Any) -> Any:
    try:
      if op == 'add':
        return lhs + rhs
      if op == 'sub':
        return lhs - rhs
      if op == 'mul':
        return lhs * rhs
      if op == 'div':
        return lhs / rhs
    except:
      pass
    return None


class DeadCodeElimination:
  """remove instructions that compute unused values"""

  def run(self, func: IRFunction) -> bool:
    """returns True if any changes made"""
    # find live values (used by other instructions or returned)
    live: set[IRValue] = set()
    for bb in func.blocks.values():
      for instr in bb.instrs:
        live.update(instr.uses())
        if isinstance(instr, Ret) and instr.value:
          live.add(instr.value)

    # remove instructions computing dead values
    changed = False
    for bb in func.blocks.values():
      new_instrs = []
      for instr in bb.instrs:
        if instr.result and instr.result not in live and not isinstance(instr, (Store, Ret, Br, Jump)):
          changed = True
          continue
        new_instrs.append(instr)
      bb.instrs = new_instrs

    return changed


class PhiElimination:
  """eliminate phi nodes by converting out of SSA form"""

  def __init__(self):
    self.const_counter = 0

  def run(self, func: IRFunction) -> bool:
    """eliminate all phi nodes, returns True if any changes made"""
    changed = False

    for bb_name, bb in func.blocks.items():
      new_instrs = []
      for instr in bb.instrs:
        if isinstance(instr, Phi):
          changed = True
          # for each incoming value, insert assignment at end of predecessor block
          for value, pred_name in instr.incoming:
            pred_bb = func.blocks[pred_name]
            # create const 0
            zero = self._make_zero(value.ty)
            # insert assignment before terminator (last instruction should be jump/br)
            if pred_bb.instrs and isinstance(pred_bb.instrs[-1], (Jump, Br, Ret)):
              # insert const and assign before terminator
              pred_bb.instrs.insert(-1, Const(result=zero, const_value=0))
              assign = BinOp(result=instr.result, op='add', lhs=value, rhs=zero)
              pred_bb.instrs.insert(-1, assign)
            else:
              # no terminator, just append
              pred_bb.instrs.append(Const(result=zero, const_value=0))
              assign = BinOp(result=instr.result, op='add', lhs=value, rhs=zero)
              pred_bb.instrs.append(assign)
        else:
          new_instrs.append(instr)
      bb.instrs = new_instrs

    return changed

  def _make_zero(self, ty: IRType) -> IRValue:
    """create a fresh zero constant"""
    self.const_counter += 1
    zero = IRValue(f'phi_zero_{self.const_counter}', ty)
    return zero


# =============================================================================
# C code generation
# =============================================================================


class CCodegen:
  """lower IR to C code"""

  def __init__(self):
    self.indent = 1

  def generate(self, func: IRFunction) -> str:
    """generate C code from IR function"""
    lines = []

    # function signature
    ret_ty = func.ret_type.to_c() if func.ret_type != IRType.UNKNOWN else 'void'
    params = ', '.join(f'{p.ty.to_c()} {p.name}' for p in func.params)
    lines.append(f'{ret_ty} {func.name}({params}) {{')

    # declare SSA values as locals
    declared: set[str] = {p.name for p in func.params}
    for bb in func.blocks.values():
      for instr in bb.instrs:
        if instr.result and instr.result.name not in declared:
          lines.append(f'  {instr.result.ty.to_c()} {instr.result};')
          declared.add(instr.result.name)

    # emit basic blocks
    for bb in func.blocks.values():
      lines.append(f'{bb.name}:')
      for instr in bb.instrs:
        lines.append('  ' + self._codegen_instr(instr))

    lines.append('}')
    return '\n'.join(lines)

  def _codegen_instr(self, instr: IRInstr) -> str:
    """generate C code for single instruction"""
    if isinstance(instr, BinOp):
      op_map = {
        'add': '+',
        'sub': '-',
        'mul': '*',
        'div': '/',
        'mod': '%',
        'lt': '<',
        'le': '<=',
        'gt': '>',
        'ge': '>=',
        'eq': '==',
        'ne': '!=',
      }
      c_op = op_map.get(instr.op, instr.op)
      return f'{instr.result} = {instr.lhs} {c_op} {instr.rhs};'
    if isinstance(instr, UnaryOp):
      return f'{instr.result} = {instr.op}{instr.operand};'
    if isinstance(instr, Call):
      # map Python intrinsics to C equivalents
      c_func_map = {'min': 'fminf', 'max': 'fmaxf', 'abs': 'fabsf', 'sqrt': 'sqrtf'}
      c_func = c_func_map.get(instr.func, instr.func)
      args = ', '.join(str(a) for a in instr.args)
      return f'{instr.result} = {c_func}({args});'
    if isinstance(instr, Load):
      return f'{instr.result} = {instr.ptr}[{instr.index}];'
    if isinstance(instr, Store):
      return f'{instr.ptr}[{instr.index}] = {instr.value};'
    if isinstance(instr, Const):
      return f'{instr.result} = {instr.const_value};'
    if isinstance(instr, Ret):
      return f'return {instr.value};' if instr.value else 'return;'
    if isinstance(instr, Jump):
      return f'goto {instr.target};'
    if isinstance(instr, Br):
      return f'if ({instr.cond}) goto {instr.true_bb}; else goto {instr.false_bb};'
    if isinstance(instr, Phi):
      # phi nodes eliminated during CFG linearization, shouldn't reach here
      return f'/* phi node: {instr} */'
    return f'/* unknown: {instr} */'


# =============================================================================
# demo: build IR by hand and run passes
# =============================================================================


def demo_simple_ir():
  """demonstrate IR construction, type inference, optimization"""
  # build simple function: result = (a + b) * 2
  func = IRFunction('example')

  # parameters
  a = IRValue('a', IRType.INT)
  b = IRValue('b', IRType.INT)
  func.params = [a, b]
  func.ret_type = IRType.INT

  # entry block
  entry = func.add_block('entry')
  tmp1 = IRValue('tmp1')
  entry.append(BinOp(result=tmp1, op='add', lhs=a, rhs=b))

  two = IRValue('two')
  entry.append(Const(result=two, const_value=2))

  result = IRValue('result')
  entry.append(BinOp(result=result, op='mul', lhs=tmp1, rhs=two))
  entry.append(Ret(value=result))

  print('=' * 80)
  print('Original IR:')
  print('=' * 80)
  print(func)

  # type inference
  print('\n' + '=' * 80)
  print('After type inference:')
  print('=' * 80)
  typer = TypeInference()
  typer.infer(func)
  print(func)

  # constant folding
  print('\n' + '=' * 80)
  print('After constant folding:')
  print('=' * 80)
  folder = ConstantFolding()
  folder.run(func)
  print(func)

  # dead code elimination
  print('\n' + '=' * 80)
  print('After dead code elimination:')
  print('=' * 80)
  dce = DeadCodeElimination()
  dce.run(func)
  print(func)

  # C codegen
  print('\n' + '=' * 80)
  print('Generated C code:')
  print('=' * 80)
  codegen = CCodegen()
  c_code = codegen.generate(func)
  print(c_code)


if __name__ == '__main__':
  demo_simple_ir()
