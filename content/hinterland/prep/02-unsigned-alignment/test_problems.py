"""Test harness for module 02: unsigned semantics + data alignment.

python3 test_problems.py                              # tests problems.py
PRACTICE_MODULE=solutions python3 test_problems.py    # tests solutions.py
"""

import importlib
import itertools
import os
import random
import sys

MOD = importlib.import_module(os.environ.get('PRACTICE_MODULE', 'problems'))

CASES = []


def case(fn):
  CASES.append((fn.__name__, fn))
  return fn


def eq(got, want, label=''):
  if got != want:
    raise AssertionError(f'{label} got {got!r}, want {want!r}')


def expect_raises(exc, fn, *args):
  try:
    fn(*args)
  except exc:
    return
  raise AssertionError(f'expected {exc.__name__}, got no exception')


def _ref_layout(fields):
  offsets, cursor, align = {}, 0, 1
  for name, size, falign in fields:
    cursor = (cursor + falign - 1) & ~(falign - 1)
    offsets[name] = cursor
    cursor += size
    align = max(align, falign)
  return offsets, (cursor + align - 1) & ~(align - 1), align


@case
def align_up_canonical():
  vectors = [
    (0, 1, 0),
    (0, 8, 0),
    (1, 8, 8),
    (7, 8, 8),
    (8, 8, 8),
    (9, 8, 16),
    (13, 8, 16),
    (5, 1, 5),
    (0x1000, 0x1000, 0x1000),
    (0x1001, 0x1000, 0x2000),
    (2**63 - 1, 64, 2**63),
  ]
  for x, a, want in vectors:
    eq(MOD.align_up(x, a), want, f'align_up({x:#x}, {a:#x})')


@case
def align_down_canonical():
  vectors = [
    (0, 8, 0),
    (7, 8, 0),
    (8, 8, 8),
    (9, 8, 8),
    (15, 8, 8),
    (16, 8, 16),
    (5, 1, 5),
    (0x1FFF, 0x1000, 0x1000),
    (2**63 + 63, 64, 2**63),
  ]
  for x, a, want in vectors:
    eq(MOD.align_down(x, a), want, f'align_down({x:#x}, {a:#x})')


@case
def is_aligned_canonical():
  for x, a in [(0, 4), (4, 4), (8, 4), (6, 2), (0, 1), (7, 1), (4096, 4096)]:
    eq(MOD.is_aligned(x, a), True, f'is_aligned({x}, {a})')
  for x, a in [(1, 4), (2, 4), (3, 4), (6, 4), (4095, 4096), (2**32 + 1, 2)]:
    eq(MOD.is_aligned(x, a), False, f'is_aligned({x}, {a})')


@case
def align_validation():
  for a in (0, 3, 6, 12, -8):
    expect_raises(ValueError, MOD.align_up, 5, a)
    expect_raises(ValueError, MOD.align_down, 5, a)
    expect_raises(ValueError, MOD.is_aligned, 5, a)
  expect_raises(ValueError, MOD.align_up, -1, 8)
  expect_raises(ValueError, MOD.align_down, -1, 8)
  expect_raises(ValueError, MOD.is_aligned, -1, 8)


@case
def align_up_properties():
  for a in (1, 2, 4, 8, 16, 64, 4096):
    prev = 0
    for x in range(513):
      y = MOD.align_up(x, a)
      if not 0 <= y - x < a:
        raise AssertionError(f'0 <= align_up({x},{a})-x < a violated: {y}')
      if y % a:
        raise AssertionError(f'align_up({x},{a}) == {y}, not a multiple')
      eq(MOD.align_up(y, a), y, f'idempotence at ({x},{a})')
      if y < prev:
        raise AssertionError(f'monotonicity broken at ({x},{a})')
      prev = y


@case
def align_down_properties_and_duality():
  for a in (1, 2, 4, 8, 16, 64):
    for x in range(257):
      d = MOD.align_down(x, a)
      if not 0 <= x - d < a or d % a:
        raise AssertionError(f'align_down({x},{a}) == {d}')
      eq(MOD.align_down(d, a), d, f'idempotence at ({x},{a})')
      eq(
        MOD.align_up(x, a), MOD.align_down(x + a - 1, a), f'duality ({x},{a})'
      )
      eq(MOD.is_aligned(x, a), x % a == 0, f'is_aligned({x},{a})')


@case
def add32_canonical():
  vectors = [
    (0xFFFFFFFF, 1, 0),
    (0x80000000, 0x80000000, 0),
    (2, 3, 5),
    (0xFFFFFFFE, 5, 3),
    (0, 0, 0),
    (0x7FFFFFFF, 1, 0x80000000),
  ]
  for a, b, want in vectors:
    eq(MOD.add32(a, b), want, f'add32({a:#x}, {b:#x})')


@case
def sub32_canonical():
  vectors = [
    (0, 1, 0xFFFFFFFF),
    (5, 7, 0xFFFFFFFE),
    (7, 5, 2),
    (0, 0, 0),
    (0, 0x80000000, 0x80000000),
  ]
  for a, b, want in vectors:
    eq(MOD.sub32(a, b), want, f'sub32({a:#x}, {b:#x})')


@case
def mul32_canonical():
  vectors = [
    (0x10000, 0x10000, 0),
    (0xFFFFFFFF, 0xFFFFFFFF, 1),
    (0xDEADBEEF, 1, 0xDEADBEEF),
    (0x80000000, 2, 0),
    (65535, 65535, 0xFFFE0001),
  ]
  for a, b, want in vectors:
    eq(MOD.mul32(a, b), want, f'mul32({a:#x}, {b:#x})')


@case
def u32_negative_inputs_reduce_like_c_casts():
  eq(MOD.add32(-1, 0), 0xFFFFFFFF, 'add32(-1, 0)')
  eq(MOD.add32(-2, 1), 0xFFFFFFFF, 'add32(-2, 1)')
  eq(MOD.sub32(-1, -1), 0, 'sub32(-1, -1)')
  eq(MOD.mul32(-1, -1), 1, 'mul32(-1, -1)')
  eq(MOD.add32(2**64, 5), 5, 'add32(2**64, 5)')


@case
def shl32_canonical():
  vectors = [
    (1, 0, 1),
    (1, 31, 0x80000000),
    (0x80000000, 1, 0),
    (0xFFFFFFFF, 4, 0xFFFFFFF0),
    (0xDEADBEEF, 8, 0xADBEEF00),
    (3, 30, 0xC0000000),
  ]
  for x, n, want in vectors:
    eq(MOD.shl32(x, n), want, f'shl32({x:#x}, {n})')


@case
def shr32_canonical():
  vectors = [
    (0x80000000, 31, 1),
    (0x80000000, 1, 0x40000000),
    (0xFF, 4, 0xF),
    (1, 0, 1),
    (0xDEADBEEF, 16, 0xDEAD),
  ]
  for x, n, want in vectors:
    eq(MOD.shr32(x, n), want, f'shr32({x:#x}, {n})')


@case
def shr32_is_logical_not_arithmetic():
  eq(MOD.shr32(0xFFFFFFF0, 4), 0x0FFFFFFF, 'shr32(0xFFFFFFF0, 4)')
  eq(MOD.shr32(-16, 4), 0x0FFFFFFF, 'shr32(-16, 4)')
  eq(MOD.shr32(-1, 0), 0xFFFFFFFF, 'shr32(-1, 0)')


@case
def shift_count_validation():
  for bad in (32, 33, 64, -1):
    expect_raises(ValueError, MOD.shl32, 1, bad)
    expect_raises(ValueError, MOD.shr32, 1, bad)


@case
def to_signed_canonical():
  vectors = [
    (0xFF, 8, -1),
    (0x80, 8, -128),
    (0x7F, 8, 127),
    (0, 8, 0),
    (0xFE, 8, -2),
    (0xFFFF, 16, -1),
    (0x8000, 16, -32768),
    (0x7FFF, 16, 32767),
    (0xFFFFFFFF, 32, -1),
    (0x80000000, 32, -(2**31)),
    (0x7FFFFFFF, 32, 2**31 - 1),
    (2**64 - 1, 64, -1),
    (2**63, 64, -(2**63)),
    (1, 1, -1),
    (0, 1, 0),
  ]
  for x, bits, want in vectors:
    eq(MOD.to_signed(x, bits), want, f'to_signed({x:#x}, {bits})')


@case
def to_unsigned_canonical():
  vectors = [
    (-1, 8, 0xFF),
    (-128, 8, 0x80),
    (127, 8, 0x7F),
    (-2, 8, 0xFE),
    (256, 8, 0),
    (300, 8, 44),
    (-1, 32, 0xFFFFFFFF),
    (-1, 64, 2**64 - 1),
    (-(2**63), 64, 2**63),
  ]
  for x, bits, want in vectors:
    eq(MOD.to_unsigned(x, bits), want, f'to_unsigned({x}, {bits})')


@case
def to_signed_masks_high_bits():
  eq(MOD.to_signed(0x1FF, 8), -1, 'to_signed(0x1FF, 8)')
  eq(MOD.to_signed(0x100, 8), 0, 'to_signed(0x100, 8)')
  eq(MOD.to_signed(-1, 8), -1, 'to_signed(-1, 8)')


@case
def sign_roundtrip_property():
  for bits in (1, 7, 8, 16):
    lo, hi = -(1 << (bits - 1)), 1 << (bits - 1)
    step = 1 if bits <= 8 else 257
    for v in range(lo, hi, step):
      eq(
        MOD.to_signed(MOD.to_unsigned(v, bits), bits), v, f's(u({v})) @{bits}'
      )
    for u in range(0, 1 << bits, step):
      eq(
        MOD.to_unsigned(MOD.to_signed(u, bits), bits), u, f'u(s({u})) @{bits}'
      )
  for bits in (32, 64):
    for v in (
      -(1 << (bits - 1)),
      -(1 << (bits - 1)) + 1,
      -1,
      0,
      1,
      (1 << (bits - 1)) - 1,
    ):
      eq(
        MOD.to_signed(MOD.to_unsigned(v, bits), bits), v, f's(u({v})) @{bits}'
      )


@case
def to_signed_matches_from_bytes():
  for u in list(range(0, 1 << 16, 257)) + [0xFFFF, 0x8000, 0x7FFF]:
    want = int.from_bytes(u.to_bytes(2, 'little'), 'little', signed=True)
    eq(MOD.to_signed(u, 16), want, f'from_bytes agreement at {u:#x}')


@case
def bits_validation():
  for bad in (0, -1, -8):
    expect_raises(ValueError, MOD.to_signed, 0, bad)
    expect_raises(ValueError, MOD.to_unsigned, 0, bad)


@case
def layout_char_int_short():
  eq(
    MOD.struct_layout([('c', 1, 1), ('i', 4, 4), ('s', 2, 2)]),
    ({'c': 0, 'i': 4, 's': 8}, 12, 4),
  )


@case
def layout_char_double_char():
  eq(
    MOD.struct_layout([('a', 1, 1), ('b', 8, 8), ('c', 1, 1)]),
    ({'a': 0, 'b': 8, 'c': 16}, 24, 8),
  )


@case
def layout_double_first_shrinks():
  eq(
    MOD.struct_layout([('b', 8, 8), ('a', 1, 1), ('c', 1, 1)]),
    ({'b': 0, 'a': 8, 'c': 9}, 16, 8),
  )


@case
def layout_empty_and_single():
  eq(MOD.struct_layout([]), ({}, 0, 1))
  eq(MOD.struct_layout([('x', 4, 4)]), ({'x': 0}, 4, 4))
  eq(MOD.struct_layout([('c', 1, 1)]), ({'c': 0}, 1, 1))


@case
def layout_tail_padding():
  eq(MOD.struct_layout([('s', 2, 2), ('c', 1, 1)]), ({'s': 0, 'c': 2}, 4, 2))
  eq(
    MOD.struct_layout([('hdr', 12, 4), ('b', 1, 1)]),
    ({'hdr': 0, 'b': 12}, 16, 4),
  )


@case
def layout_nested_member():
  eq(
    MOD.struct_layout([('c', 1, 1), ('sub', 16, 8), ('s', 2, 2)]),
    ({'c': 0, 'sub': 8, 's': 24}, 32, 8),
  )


@case
def layout_validation():
  expect_raises(ValueError, MOD.struct_layout, [('a', 1, 1), ('a', 4, 4)])
  expect_raises(ValueError, MOD.struct_layout, [('a', 4, 3)])
  expect_raises(ValueError, MOD.struct_layout, [('a', 4, 0)])
  expect_raises(ValueError, MOD.struct_layout, [('a', -1, 1)])


@case
def layout_matches_reference_on_generated_inputs():
  rng = random.Random(0xA11)
  for trial in range(60):
    fields = []
    for i in range(rng.randint(0, 6)):
      align = 1 << rng.randint(0, 4)
      size = rng.randint(0, 24)
      fields.append((f'f{i}', size, align))
    eq(
      MOD.struct_layout(fields),
      _ref_layout(fields),
      f'trial {trial}: {fields}',
    )


@case
def layout_offset_invariants():
  fields = [('a', 2, 2), ('b', 8, 8), ('c', 1, 1), ('d', 4, 4), ('e', 2, 2)]
  offsets, size, align = MOD.struct_layout(fields)
  end = 0
  for name, fsize, falign in fields:
    off = offsets[name]
    if off % falign:
      raise AssertionError(f'{name} misaligned at offset {off}')
    if off < end:
      raise AssertionError(f'{name} overlaps previous member')
    end = off + fsize
  eq(align, 8, 'struct align')
  if size % align or size < end:
    raise AssertionError(f'bad total size {size} (end {end}, align {align})')


@case
def promote_trap_canonical():
  vectors = [
    (0, 0),
    (1, 1),
    (0x7F, 127),
    (0x80, -128),
    (0xFF, -1),
    (200, -56),
    (0xC3, -61),
  ]
  for x, want in vectors:
    eq(MOD.c_promote_trap(x), want, f'c_promote_trap({x:#x})')


@case
def promote_trap_applies_uint8_cast():
  for x, want in [(0x123, 0x23), (256, 0), (-1, -1), (384, -128), (511, -1)]:
    eq(MOD.c_promote_trap(x), want, f'c_promote_trap({x})')


@case
def promote_trap_full_byte_range():
  for v in range(256):
    want = v - 256 if v >= 128 else v
    eq(MOD.c_promote_trap(v), want, f'v={v}')


@case
def reorder_canonical():
  original = [('c', 1, 1), ('d', 8, 8), ('s', 2, 2)]
  got = MOD.reorder_fields(original)
  eq(got, [('d', 8, 8), ('s', 2, 2), ('c', 1, 1)], 'reorder order')
  eq(_ref_layout(got)[1], 16, 'reordered size')
  eq(_ref_layout(original)[1], 24, 'original size')


@case
def reorder_stable_on_ties():
  got = MOD.reorder_fields([
    ('a', 4, 4),
    ('b', 4, 4),
    ('c', 8, 8),
    ('d', 4, 4),
  ])
  eq(got, [('c', 8, 8), ('a', 4, 4), ('b', 4, 4), ('d', 4, 4)])


@case
def reorder_is_permutation():
  fields = [('a', 2, 2), ('b', 8, 8), ('c', 1, 1), ('d', 4, 4), ('e', 2, 2)]
  got = MOD.reorder_fields(fields)
  eq(sorted(got), sorted(fields), 'permutation of input')
  eq(MOD.reorder_fields([]), [], 'empty input')


@case
def reorder_optimal_vs_brute_force():
  suites = [
    [('a', 1, 1), ('b', 2, 2), ('c', 4, 4), ('d', 8, 8), ('e', 1, 1)],
    [('a', 8, 8), ('b', 1, 1), ('c', 2, 2)],
    [('a', 4, 4), ('b', 4, 4)],
    [('a', 1, 1)],
    [('a', 2, 2), ('b', 8, 8), ('c', 1, 1), ('d', 4, 4), ('e', 2, 2)],
    [('arr', 12, 4), ('b', 8, 8), ('c', 1, 1), ('d', 2, 2)],
    [('a', 16, 16), ('b', 1, 1), ('c', 8, 8), ('d', 2, 2), ('e', 4, 4)],
  ]
  for fields in suites:
    best = min(_ref_layout(list(p))[1] for p in itertools.permutations(fields))
    got = _ref_layout(MOD.reorder_fields(fields))[1]
    eq(got, best, f'brute force on {fields}')


@case
def reorder_validation():
  expect_raises(ValueError, MOD.reorder_fields, [('x', 3, 2)])
  expect_raises(ValueError, MOD.reorder_fields, [('x', 4, 5)])
  expect_raises(ValueError, MOD.reorder_fields, [('x', -4, 4)])


def main():
  passed = failed = 0
  for name, fn in CASES:
    try:
      fn()
    except Exception as exc:
      failed += 1
      print(f'FAIL {name}: {type(exc).__name__}: {exc}')
    else:
      passed += 1
      print(f'PASS {name}')
  print(f'{passed}/{passed + failed} passed')
  sys.exit(1 if failed else 0)


if __name__ == '__main__':
  main()
