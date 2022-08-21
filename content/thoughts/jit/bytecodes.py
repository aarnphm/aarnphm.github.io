#!/usr/bin/env python3
"""
examples for  compiler optimization techniques using Python bytecode manipulation.
Each example shows:
  1. Original Python code
  2. Original bytecode (dis.dis output)
  3. Transformation logic
  4. Optimized bytecode
  5. Correctness verification
  6. Performance comparison
"""

import dis
import time
import types
import bytecode as bc


def print_header(title):
  """Print section header"""
  print('\n' + '=' * 80)
  print(title)
  print('=' * 80)


# =============================================================================
# Example 1: Code Object Inspection
# =============================================================================


def example_code_object_inspection():
  """Demonstrate code object structure"""
  print_header('Example 1: Code Object Inspection')

  def sample_function(x, y):
    """Sample function for inspection"""
    z = x + y
    return z * 2

  code = sample_function.__code__

  print('\nCode Object Attributes:')
  print(f'  co_name:        {code.co_name}')
  print(f'  co_argcount:    {code.co_argcount}')
  print(f'  co_nlocals:     {code.co_nlocals}')
  print(f'  co_varnames:    {code.co_varnames}')
  print(f'  co_consts:      {code.co_consts}')
  print(f'  co_names:       {code.co_names}')
  print(f'  co_code (raw):  {code.co_code[:20]}... ({len(code.co_code)} bytes)')

  print('\nDisassembly:')
  dis.dis(sample_function)


# =============================================================================
# Example 2: Constant Folding
# =============================================================================


def example_constant_folding():
  """Demonstrate constant folding optimization"""
  print_header('Example 2: Constant Folding')

  def original():
    """Compute with constants that can be folded"""
    a = 10 + 20  # foldable: compile-time
    b = 5 * 4  # foldable: compile-time
    c = a + b  # not foldable: a, b are variables
    return c

  print('\nOriginal Python code:')
  print('  a = 10 + 20')
  print('  b = 5 * 4')
  print('  c = a + b')
  print('  return c')

  print('\nPython 3.11+ bytecode (already optimized by compiler):')
  dis.dis(original)

  print("\nNote: Python's peephole optimizer already folded 10+20→30 and 5*4→20")
  print("To demonstrate the optimization, we'll manually construct unoptimized bytecode:\n")

  # manually construct unoptimized bytecode
  print('Manually constructed bytecode (before constant folding):')
  print('  Computes: (10 + 20) + (5 * 4)')

  unoptimized = bc.Bytecode([
    bc.Instr('RESUME', 0),
    bc.Instr('LOAD_CONST', 10),
    bc.Instr('LOAD_CONST', 20),
    bc.Instr('BINARY_OP', 0),  # ADD
    bc.Instr('LOAD_CONST', 5),
    bc.Instr('LOAD_CONST', 4),
    bc.Instr('BINARY_OP', 5),  # MULTIPLY
    bc.Instr('BINARY_OP', 0),  # ADD
    bc.Instr('RETURN_VALUE'),
  ])
  unoptimized.name = 'unoptimized'

  unoptimized_func = types.FunctionType(unoptimized.to_code(), {}, 'unoptimized')

  dis.dis(unoptimized_func)

  # apply constant folding
  new_instrs = []
  i = 0
  instrs = list(unoptimized)

  while i < len(instrs):
    instr = instrs[i]

    # pattern: LOAD_CONST, LOAD_CONST, BINARY_OP
    if (
      i + 2 < len(instrs)
      and isinstance(instr, bc.Instr)
      and instr.name == 'LOAD_CONST'
      and isinstance(instrs[i + 1], bc.Instr)
      and instrs[i + 1].name == 'LOAD_CONST'
      and isinstance(instrs[i + 2], bc.Instr)
      and instrs[i + 2].name == 'BINARY_OP'
    ):
      const1 = instr.arg
      const2 = instrs[i + 1].arg
      op_arg = instrs[i + 2].arg

      # BINARY_OP arg encoding in Python 3.11+
      # 0=ADD, 5=MULTIPLY, etc
      ops = {
        0: lambda x, y: x + y,  # ADD
        5: lambda x, y: x * y,  # MULTIPLY
      }

      if op_arg in ops:
        result = ops[op_arg](const1, const2)
        new_instrs.append(bc.Instr('LOAD_CONST', result))
        i += 3
        continue

    new_instrs.append(instr)
    i += 1

  optimized = bc.Bytecode(new_instrs)
  optimized.name = 'optimized'

  optimized_func = types.FunctionType(optimized.to_code(), {}, 'optimized')

  print('\nOptimized bytecode (after constant folding):')
  dis.dis(optimized_func)

  print(f'\nUnoptimized result: {unoptimized_func()}')
  print(f'Optimized result: {optimized_func()}')
  print(f'Results match: {unoptimized_func() == optimized_func()}')
  print(f'\nInstructions reduced: {len(instrs)} → {len(list(optimized))}')


# =============================================================================
# Example 3: Dead Code Elimination
# =============================================================================


def example_dead_code_elimination():
  """Demonstrate dead code elimination"""
  print_header('Example 3: Dead Code Elimination')

  def original(x):
    """Function with unreachable code"""
    if x > 0:
      return x * 2
    else:
      return -x
    # unreachable code below
    y = 100
    z = y + 50
    return z

  print('\nOriginal bytecode:')
  dis.dis(original)

  # transform: remove code after RETURN_VALUE
  code = bc.Bytecode.from_code(original.__code__)
  new_instrs = []
  seen_returns = 0
  keep_instructions = True

  for instr in code:
    if keep_instructions:
      new_instrs.append(instr)
      if isinstance(instr, bc.Instr) and instr.name == 'RETURN_VALUE':
        seen_returns += 1
        # after seeing 2 returns (if and else branches), remaining code is dead
        # stop adding instructions but keep labels to avoid broken references
        if seen_returns >= 2:
          keep_instructions = False
    else:
      # keep labels and metadata, skip instructions
      if not isinstance(instr, bc.Instr):
        new_instrs.append(instr)

  # create new bytecode and copy metadata from original
  optimized_bytecode = bc.Bytecode(new_instrs)
  optimized_bytecode.argcount = code.argcount
  optimized_bytecode.argnames = code.argnames
  optimized_bytecode.name = code.name

  optimized_code = optimized_bytecode.to_code()
  optimized = types.FunctionType(
    optimized_code, original.__globals__, original.__name__, original.__defaults__, original.__closure__
  )

  print('\nOptimized bytecode (dead code removed):')
  dis.dis(optimized)

  print(f'\nOriginal(5): {original(5)}')
  print(f'Optimized(5): {optimized(5)}')
  print(f'Results match: {original(5) == optimized(5)}')


# =============================================================================
# Example 4: Loop Unrolling (Simple)
# =============================================================================


def example_loop_unrolling():
  """Demonstrate manual loop unrolling"""
  print_header('Example 4: Loop Unrolling (Conceptual)')

  def original():
    """Loop with known small iteration count"""
    result = 0
    for i in range(4):  # fixed, small loop
      result += i * 2
    return result

  def manually_unrolled():
    """Manually unrolled version"""
    result = 0
    result += 0 * 2  # i=0
    result += 1 * 2  # i=1
    result += 2 * 2  # i=2
    result += 3 * 2  # i=3
    return result

  print('\nOriginal (with loop):')
  dis.dis(original)

  print('\nManually unrolled:')
  dis.dis(manually_unrolled)

  # benchmark
  iterations = 1000000

  start = time.perf_counter()
  for _ in range(iterations):
    r1 = original()
  time_orig = time.perf_counter() - start

  start = time.perf_counter()
  for _ in range(iterations):
    r2 = manually_unrolled()
  time_unrolled = time.perf_counter() - start

  print(f'\nOriginal result: {r1}')
  print(f'Unrolled result: {r2}')
  print(f'Results match: {r1 == r2}')

  print(f'\nPerformance ({iterations:,} iterations):')
  print(f'  Original:  {time_orig:.4f}s')
  print(f'  Unrolled:  {time_unrolled:.4f}s')
  print(f'  Speedup:   {time_orig / time_unrolled:.2f}x')


# =============================================================================
# Example 5: Function Inlining
# =============================================================================


def example_function_inlining():
  """Demonstrate function inlining concept"""
  print_header('Example 5: Function Inlining (Conceptual)')

  def helper(x):
    return x * 2

  def original(a, b):
    """Uses helper function"""
    return helper(a) + helper(b)

  def inlined(a, b):
    """Manually inlined version"""
    return a * 2 + b * 2

  print('\nOriginal (with function calls):')
  dis.dis(original)

  print('\nInlined (calls eliminated):')
  dis.dis(inlined)

  # benchmark
  iterations = 1000000

  start = time.perf_counter()
  for _ in range(iterations):
    r1 = original(5, 10)
  time_orig = time.perf_counter() - start

  start = time.perf_counter()
  for _ in range(iterations):
    r2 = inlined(5, 10)
  time_inlined = time.perf_counter() - start

  print(f'\nOriginal result: {r1}')
  print(f'Inlined result: {r2}')
  print(f'Results match: {r1 == r2}')

  print(f'\nPerformance ({iterations:,} iterations):')
  print(f'  Original:  {time_orig:.4f}s')
  print(f'  Inlined:   {time_inlined:.4f}s')
  print(f'  Speedup:   {time_orig / time_inlined:.2f}x')


# =============================================================================
# Example 6: Pattern-Based Optimization (Identity Operations)
# =============================================================================


def example_identity_operations():
  """Demonstrate identity operation elimination"""
  print_header('Example 6: Identity Operation Elimination')

  def original(x):
    """Code with identity operations"""
    a = x + 0  # identity: x + 0 = x
    b = a * 1  # identity: x * 1 = x
    c = b - 0  # identity: x - 0 = x
    return c

  print('\nOriginal bytecode:')
  dis.dis(original)

  # transform: eliminate identity operations
  code = bc.Bytecode.from_code(original.__code__)
  new_instrs = []
  i = 0

  while i < len(code):
    instr = code[i]

    # pattern: LOAD_*, LOAD_CONST, BINARY_OP
    if (
      i + 2 < len(code)
      and isinstance(code[i + 1], bc.Instr)
      and code[i + 1].name == 'LOAD_CONST'
      and isinstance(code[i + 2], bc.Instr)
    ):
      const = code[i + 1].arg
      op = code[i + 2].name

      # x + 0 or x - 0 → x
      if const == 0 and op in ('BINARY_ADD', 'BINARY_SUBTRACT'):
        new_instrs.append(instr)  # keep the LOAD
        i += 3
        continue

      # x * 1 → x
      if const == 1 and op == 'BINARY_MULTIPLY':
        new_instrs.append(instr)
        i += 3
        continue

    new_instrs.append(instr)
    i += 1

  # create new bytecode and copy metadata from original
  optimized_bytecode = bc.Bytecode(new_instrs)
  optimized_bytecode.argcount = code.argcount
  optimized_bytecode.argnames = code.argnames
  optimized_bytecode.name = code.name

  optimized_code = optimized_bytecode.to_code()
  optimized = types.FunctionType(
    optimized_code, original.__globals__, original.__name__, original.__defaults__, original.__closure__
  )

  print('\nOptimized bytecode:')
  dis.dis(optimized)

  print(f'\nOriginal(42): {original(42)}')
  print(f'Optimized(42): {optimized(42)}')
  print(f'Results match: {original(42) == optimized(42)}')


# =============================================================================
# Example 7: Concrete vs Abstract Bytecode
# =============================================================================


def example_concrete_vs_abstract():
  """Demonstrate concrete vs abstract bytecode representations"""
  print_header('Example 7: Concrete vs Abstract Bytecode')

  def sample(x):
    if x > 0:
      return x
    else:
      return -x

  print('\nConcrete bytecode (raw bytes, offsets):')
  dis.dis(sample)

  print('\nAbstract bytecode (bytecode library, labels):')
  abstract = bc.Bytecode.from_code(sample.__code__)
  for i, instr in enumerate(abstract):
    print(f'  {i:3d}: {instr}')

  print('\nAdvantages of abstract representation:')
  print('  • Labels instead of byte offsets (easier to manipulate)')
  print('  • Automatic offset calculation when converting back')
  print('  • Can insert/remove instructions without manual offset updates')
  print('  • Clear distinction between instructions, labels, and metadata')


# =============================================================================
# Main
# =============================================================================


def main():
  example_code_object_inspection()
  example_constant_folding()
  example_dead_code_elimination()
  example_loop_unrolling()
  example_function_inlining()
  example_identity_operations()
  example_concrete_vs_abstract()


if __name__ == '__main__':
  raise SystemExit(main())
