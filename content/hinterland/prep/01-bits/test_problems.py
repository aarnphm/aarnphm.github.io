"""Self-contained harness. PRACTICE_MODULE=solutions python3 test_problems.py"""

import importlib
import os
import sys

MOD = importlib.import_module(os.environ.get('PRACTICE_MODULE', 'problems'))

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF

CASES = []


def case(name):
  def register(fn):
    CASES.append((name, fn))
    return fn

  return register


def eq(got, want, label='value'):
  if got != want:
    raise AssertionError(f'{label}: got {got!r}, want {want!r}')


def rejects(fn, label):
  try:
    fn()
  except ValueError:
    return
  except Exception as e:
    raise AssertionError(
      f'{label}: expected ValueError, got {type(e).__name__}: {e}'
    )
  raise AssertionError(f'{label}: expected ValueError, got a result')


def lcg(seed):
  v = seed & M64
  while True:
    v = (v * 6364136223846793005 + 1442695040888963407) & M64
    yield v


# ---------------- pack_rgba / unpack_rgba ----------------


@case('pack_rgba_canonical')
def _():
  eq(MOD.pack_rgba(0x12, 0x34, 0x56, 0x78), 0x12345678)


@case('pack_rgba_zero')
def _():
  eq(MOD.pack_rgba(0, 0, 0, 0), 0)


@case('pack_rgba_all_ones')
def _():
  eq(MOD.pack_rgba(255, 255, 255, 255), 0xFFFFFFFF)


@case('pack_rgba_red_alpha')
def _():
  eq(MOD.pack_rgba(255, 0, 0, 255), 0xFF0000FF)


@case('pack_rgba_rejects_out_of_range')
def _():
  rejects(lambda: MOD.pack_rgba(256, 0, 0, 0), 'r=256')
  rejects(lambda: MOD.pack_rgba(0, -1, 0, 0), 'g=-1')
  rejects(lambda: MOD.pack_rgba(0, 0, 300, 0), 'b=300')
  rejects(lambda: MOD.pack_rgba(0, 0, 0, 999), 'a=999')


@case('unpack_rgba_canonical')
def _():
  eq(MOD.unpack_rgba(0xDEADBEEF), (0xDE, 0xAD, 0xBE, 0xEF))


@case('unpack_rgba_boundaries')
def _():
  eq(MOD.unpack_rgba(0), (0, 0, 0, 0))
  eq(MOD.unpack_rgba(0xFFFFFFFF), (255, 255, 255, 255))
  eq(MOD.unpack_rgba(0x80000001), (0x80, 0, 0, 1))


@case('unpack_rgba_roundtrip_sweep')
def _():
  g = lcg(2024)
  for _ in range(50):
    px = next(g) & M32
    r, gg, b, a = MOD.unpack_rgba(px)
    eq(MOD.pack_rgba(r, gg, b, a), px, f'px={px:#010x}')


@case('unpack_rgba_rejects_out_of_range')
def _():
  rejects(lambda: MOD.unpack_rgba(1 << 32), 'px=2**32')
  rejects(lambda: MOD.unpack_rgba(-1), 'px=-1')


# ---------------- extract_field ----------------


@case('extract_low_byte')
def _():
  eq(MOD.extract_field(0xDEADBEEF, 8, 8), 0xBE)
  eq(MOD.extract_field(0xDEADBEEFCAFEBABE, 0, 8), 0xBE)


@case('extract_mid_field')
def _():
  eq(MOD.extract_field(0xDEADBEEFCAFEBABE, 8, 16), 0xFEBA)


@case('extract_top_nibble')
def _():
  eq(MOD.extract_field(0xDEADBEEFCAFEBABE, 60, 4), 0xD)


@case('extract_bit63')
def _():
  eq(MOD.extract_field(1 << 63, 63, 1), 1)
  eq(MOD.extract_field((1 << 63) - 1, 63, 1), 0)


@case('extract_full_width')
def _():
  eq(MOD.extract_field(0xDEADBEEFCAFEBABE, 0, 64), 0xDEADBEEFCAFEBABE)
  eq(MOD.extract_field(M64, 0, 64), M64)


@case('extract_zero_word')
def _():
  eq(MOD.extract_field(0, 17, 13), 0)


@case('extract_rejects_bad_bounds')
def _():
  rejects(lambda: MOD.extract_field(1 << 64, 0, 8), 'word=2**64')
  rejects(lambda: MOD.extract_field(0, -1, 8), 'offset=-1')
  rejects(lambda: MOD.extract_field(0, 0, 0), 'width=0')
  rejects(lambda: MOD.extract_field(0, 60, 5), 'offset+width=65')
  rejects(lambda: MOD.extract_field(0, 64, 1), 'offset=64')


# ---------------- insert_field ----------------


@case('insert_canonical')
def _():
  eq(MOD.insert_field(0xDEADBEEF, 8, 8, 0x42), 0xDEAD42EF)


@case('insert_top_byte_into_zero')
def _():
  eq(MOD.insert_field(0, 56, 8, 0x12), 0x1200000000000000)


@case('insert_clears_inside_all_ones')
def _():
  eq(MOD.insert_field(M64, 8, 8, 0), 0xFFFFFFFFFFFF00FF)


@case('insert_bit63')
def _():
  eq(MOD.insert_field(0, 63, 1, 1), 1 << 63)
  eq(MOD.insert_field(M64, 63, 1, 0), (1 << 63) - 1)


@case('insert_full_width')
def _():
  eq(MOD.insert_field(M64, 0, 64, 0x0123456789ABCDEF), 0x0123456789ABCDEF)


@case('insert_roundtrip_preserves_outside')
def _():
  g = lcg(999)
  for _ in range(100):
    word = next(g)
    sel = next(g)
    offset = sel % 64
    width = 1 + (sel >> 8) % (64 - offset)
    value = next(g) & ((1 << width) - 1)
    out = MOD.insert_field(word, offset, width, value)
    label = f'word={word:#x} off={offset} w={width}'
    eq(MOD.extract_field(out, offset, width), value, label + ' field')
    outside = M64 ^ (((1 << width) - 1) << offset)
    eq(out & outside, word & outside, label + ' outside bits')


@case('insert_rejects_bad_args')
def _():
  rejects(lambda: MOD.insert_field(0, 0, 4, 16), 'value=16 in width 4')
  rejects(lambda: MOD.insert_field(0, 0, 4, -1), 'value=-1')
  rejects(lambda: MOD.insert_field(0, 60, 5, 0), 'offset+width=65')
  rejects(lambda: MOD.insert_field(1 << 64, 0, 8, 0), 'word=2**64')


# ---------------- next_pow2 ----------------


@case('next_pow2_small_vectors')
def _():
  for x, want in (
    (0, 1),
    (1, 1),
    (2, 2),
    (3, 4),
    (4, 4),
    (5, 8),
    (7, 8),
    (9, 16),
  ):
    eq(MOD.next_pow2(x), want, f'x={x}')


@case('next_pow2_page_sizes')
def _():
  eq(MOD.next_pow2(0x1000), 0x1000)
  eq(MOD.next_pow2(0x1001), 0x2000)


@case('next_pow2_crosses_bit31')
def _():
  eq(MOD.next_pow2((1 << 31) - 1), 1 << 31)
  eq(MOD.next_pow2(1 << 31), 1 << 31)
  eq(MOD.next_pow2((1 << 31) + 1), 1 << 32)


@case('next_pow2_bit63_boundary')
def _():
  eq(MOD.next_pow2((1 << 63) - 1), 1 << 63)
  eq(MOD.next_pow2(1 << 63), 1 << 63)


@case('next_pow2_sweep_oracle')
def _():
  for x in range(1, 4097):
    eq(MOD.next_pow2(x), 1 << (x - 1).bit_length(), f'x={x}')


@case('next_pow2_rejects_out_of_range')
def _():
  rejects(lambda: MOD.next_pow2(-1), 'x=-1')
  rejects(lambda: MOD.next_pow2((1 << 63) + 1), 'x=2**63+1')


# ---------------- reverse_bits32 ----------------


@case('reverse_zero')
def _():
  eq(MOD.reverse_bits32(0), 0)


@case('reverse_bit0_to_bit31')
def _():
  eq(MOD.reverse_bits32(1), 0x80000000)


@case('reverse_bit31_to_bit0')
def _():
  eq(MOD.reverse_bits32(0x80000000), 1)


@case('reverse_all_ones')
def _():
  eq(MOD.reverse_bits32(M32), M32)


@case('reverse_canonical')
def _():
  eq(MOD.reverse_bits32(0x12345678), 0x1E6A2C48)


@case('reverse_leetcode_vector')
def _():
  eq(MOD.reverse_bits32(43261596), 964176192)


@case('reverse_involution_sweep')
def _():
  g = lcg(7)
  for _ in range(100):
    x = next(g) & M32
    want = int(format(x, '032b')[::-1], 2)
    r = MOD.reverse_bits32(x)
    eq(r, want, f'x={x:#010x}')
    eq(MOD.reverse_bits32(r), x, f'involution x={x:#010x}')


@case('reverse_rejects_out_of_range')
def _():
  rejects(lambda: MOD.reverse_bits32(1 << 32), 'x=2**32')
  rejects(lambda: MOD.reverse_bits32(-1), 'x=-1')


# ---------------- sar32 ----------------


@case('sar32_negative_by_1')
def _():
  eq(MOD.sar32(0x80000000, 1), 0xC0000000)


@case('sar32_negative_by_31')
def _():
  eq(MOD.sar32(0x80000000, 31), 0xFFFFFFFF)


@case('sar32_all_ones_stays')
def _():
  eq(MOD.sar32(0xFFFFFFFF, 16), 0xFFFFFFFF)
  eq(MOD.sar32(0xFFFFFFFF, 31), 0xFFFFFFFF)


@case('sar32_positive_is_logical')
def _():
  eq(MOD.sar32(0x7FFFFFFF, 1), 0x3FFFFFFF)
  eq(MOD.sar32(0x7FFFFFFF, 31), 0)


@case('sar32_shift_zero_identity')
def _():
  eq(MOD.sar32(0xDEADBEEF, 0), 0xDEADBEEF)
  eq(MOD.sar32(0x00000001, 0), 1)


@case('sar32_bit0')
def _():
  eq(MOD.sar32(1, 1), 0)


@case('sar32_canonical_deadbeef')
def _():
  eq(MOD.sar32(0xDEADBEEF, 4), 0xFDEADBEE)


@case('sar32_oracle_sweep')
def _():
  g = lcg(31337)
  for _ in range(50):
    x = next(g) & M32
    signed = x - (1 << 32) if x >> 31 else x
    for n in (0, 1, 7, 15, 30, 31):
      eq(MOD.sar32(x, n), (signed >> n) & M32, f'x={x:#010x} n={n}')


@case('sar32_rejects_bad_args')
def _():
  rejects(lambda: MOD.sar32(0, 32), 'n=32 (C UB)')
  rejects(lambda: MOD.sar32(0, -1), 'n=-1')
  rejects(lambda: MOD.sar32(1 << 32, 0), 'x=2**32')
  rejects(lambda: MOD.sar32(-1, 0), 'x=-1')


# ---------------- popcount_swar ----------------


@case('popcount_zero')
def _():
  eq(MOD.popcount_swar(0), 0)


@case('popcount_bit0')
def _():
  eq(MOD.popcount_swar(1), 1)


@case('popcount_bit63')
def _():
  eq(MOD.popcount_swar(1 << 63), 1)


@case('popcount_all_ones')
def _():
  eq(MOD.popcount_swar(M64), 64)


@case('popcount_alternating')
def _():
  eq(MOD.popcount_swar(0x5555555555555555), 32)
  eq(MOD.popcount_swar(0xAAAAAAAAAAAAAAAA), 32)
  eq(MOD.popcount_swar(0x3333333333333333), 32)


@case('popcount_nibble_ladder')
def _():
  eq(MOD.popcount_swar(0x0123456789ABCDEF), 32)


@case('popcount_deadbeef')
def _():
  eq(MOD.popcount_swar(0xDEADBEEFDEADBEEF), 48)


@case('popcount_sweep_oracle')
def _():
  g = lcg(0x9E3779B97F4A7C15)
  for _ in range(200):
    v = next(g)
    eq(MOD.popcount_swar(v), bin(v).count('1'), f'v={v:#018x}')


@case('popcount_rejects_out_of_range')
def _():
  rejects(lambda: MOD.popcount_swar(-1), 'x=-1')
  rejects(lambda: MOD.popcount_swar(1 << 64), 'x=2**64')


# ---------------- submasks ----------------


@case('submasks_zero')
def _():
  eq(MOD.submasks(0), [0])


@case('submasks_canonical_101')
def _():
  eq(MOD.submasks(0b101), [0b101, 0b100, 0b001, 0b000])


@case('submasks_full_byte_is_countdown')
def _():
  eq(MOD.submasks(0xFF), list(range(255, -1, -1)))


@case('submasks_bit63')
def _():
  eq(MOD.submasks(1 << 63), [1 << 63, 0])


@case('submasks_properties_11010')
def _():
  m = 0b11010
  out = MOD.submasks(m)
  eq(len(out), 8, 'count')
  eq(out[0], m, 'first')
  eq(out[-1], 0, 'last')
  eq(len(set(out)), 8, 'distinct')
  if not all(a > b for a, b in zip(out, out[1:])):
    raise AssertionError(f'not strictly decreasing: {out}')
  if not all((s | m) == m for s in out):
    raise AssertionError(f'non-submask present: {out}')


@case('submasks_sparse_high_and_low')
def _():
  m = (1 << 63) | (1 << 32) | 1
  out = MOD.submasks(m)
  eq(len(out), 8, 'count')
  eq(out[0], m, 'first')
  eq(out[-1], 0, 'last')
  expected = {
    a | b | c for a in (0, 1) for b in (0, 1 << 32) for c in (0, 1 << 63)
  }
  eq(set(out), expected, 'members')


@case('submasks_rejects_out_of_range')
def _():
  rejects(lambda: MOD.submasks(-1), 'm=-1')
  rejects(lambda: MOD.submasks(1 << 64), 'm=2**64')


def main():
  passed = 0
  for name, fn in CASES:
    try:
      fn()
    except Exception as e:
      print(f'FAIL {name}: {type(e).__name__}: {e}')
    else:
      print(f'PASS {name}')
      passed += 1
  total = len(CASES)
  print(f'{passed}/{total} passed')
  sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
  main()
