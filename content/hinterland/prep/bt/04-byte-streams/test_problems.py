"""Self-contained test harness for 04-byte-streams. Stdlib only.

python3 test_problems.py                       # tests problems.py
PRACTICE_MODULE=solutions python3 test_problems.py
"""

import importlib
import math
import os
import random
import struct
import sys

MOD = importlib.import_module(os.environ.get('PRACTICE_MODULE', 'problems'))

PASSED = 0
FAILED = 0


def check(name, fn):
  global PASSED, FAILED
  try:
    fn()
    PASSED += 1
    print(f'PASS {name}')
  except Exception as e:
    FAILED += 1
    print(f'FAIL {name}: {type(e).__name__}: {e}')


def expect_raises(exc, fn, *args, **kwargs):
  try:
    fn(*args, **kwargs)
  except exc:
    return
  except Exception as e:
    raise AssertionError(
      f'expected {exc.__name__}, got {type(e).__name__}: {e}'
    )
  raise AssertionError(f'expected {exc.__name__}, nothing raised')


def eq(got, want, label=''):
  assert got == want, f'{label}: got {got!r}, want {want!r}'


# ---------------------------------------------------------------- read_u32_*


def t_u32le_golden():
  eq(MOD.read_u32_le(b'\x44\x33\x22\x11', 0), 0x11223344)
  eq(MOD.read_u32_le(b'\x01\x00\x00\x00', 0), 1)
  eq(MOD.read_u32_le(b'\xff\xff\xff\xff', 0), 0xFFFFFFFF)


def t_u32be_golden():
  eq(MOD.read_u32_be(b'\x11\x22\x33\x44', 0), 0x11223344)
  eq(MOD.read_u32_be(b'\x00\x00\x00\x01', 0), 1)
  eq(MOD.read_u32_be(b'\xff\xff\xff\xff', 0), 0xFFFFFFFF)


def t_u32_oracle_sweep():
  rng = random.Random(0xBEEF)
  buf = bytes(rng.randrange(256) for _ in range(64))
  for off in (0, 1, 7, 30, 60):
    eq(
      MOD.read_u32_le(buf, off),
      int.from_bytes(buf[off : off + 4], 'little'),
      f'le off={off}',
    )
    eq(
      MOD.read_u32_be(buf, off),
      struct.unpack_from('>I', buf, off)[0],
      f'be off={off}',
    )


def t_u32_offset_middle():
  buf = b'\xaa\xbb' + b'\x44\x33\x22\x11' + b'\xcc'
  eq(MOD.read_u32_le(buf, 2), 0x11223344)
  eq(MOD.read_u32_be(buf, 2), 0x44332211)


def t_u32_bytearray():
  eq(MOD.read_u32_le(bytearray(b'\x78\x56\x34\x12'), 0), 0x12345678)


def t_u32_last_valid_offset():
  buf = bytes(range(8))
  eq(MOD.read_u32_le(buf, 4), int.from_bytes(buf[4:8], 'little'))


def t_u32_bounds_errors():
  buf = b'\x00\x01\x02\x03\x04'
  for f in (MOD.read_u32_le, MOD.read_u32_be):
    expect_raises(ValueError, f, buf, -1)
    expect_raises(ValueError, f, buf, 2)
    expect_raises(ValueError, f, buf, 5)
    expect_raises(ValueError, f, buf, 100)
    expect_raises(ValueError, f, b'', 0)
    expect_raises(ValueError, f, b'\x00\x01\x02', 0)


# --------------------------------------------------------------- write_i64_le


def t_i64_golden():
  eq(MOD.write_i64_le(0), b'\x00' * 8)
  eq(MOD.write_i64_le(-1), b'\xff' * 8)
  eq(MOD.write_i64_le(-2), b'\xfe' + b'\xff' * 7)
  eq(MOD.write_i64_le(1), b'\x01' + b'\x00' * 7)


def t_i64_width_boundaries():
  eq(MOD.write_i64_le(-(2**63)), b'\x00' * 7 + b'\x80')
  eq(MOD.write_i64_le(2**63 - 1), b'\xff' * 7 + b'\x7f')
  eq(MOD.write_i64_le(2**32), b'\x00\x00\x00\x00\x01\x00\x00\x00')


def t_i64_oracle_sweep():
  vals = [
    0,
    1,
    -1,
    127,
    128,
    -128,
    -129,
    255,
    256,
    0x0123456789ABCDEF,
    -0x0123456789ABCDEF,
    2**63 - 1,
    -(2**63),
    -305419896,
  ]
  for n in vals:
    eq(MOD.write_i64_le(n), struct.pack('<q', n), f'n={n}')


def t_i64_range_errors():
  expect_raises(ValueError, MOD.write_i64_le, 2**63)
  expect_raises(ValueError, MOD.write_i64_le, -(2**63) - 1)
  expect_raises(ValueError, MOD.write_i64_le, 2**64)


# -------------------------------------------------------------- float32_parts


def t_f32_normal():
  eq(MOD.float32_parts(0x3F800000), (0, 127, 0, 'normal'), '1.0')
  eq(MOD.float32_parts(0xC0200000), (1, 128, 0x200000, 'normal'), '-2.5')
  bits = struct.unpack('<I', struct.pack('<f', -2.5))[0]
  eq(bits, 0xC0200000, 'struct oracle for -2.5')


def t_f32_zeroes():
  eq(MOD.float32_parts(0x00000000), (0, 0, 0, 'zero'), '+0.0')
  eq(MOD.float32_parts(0x80000000), (1, 0, 0, 'zero'), '-0.0')


def t_f32_inf():
  eq(MOD.float32_parts(0x7F800000), (0, 255, 0, 'inf'), '+inf')
  eq(MOD.float32_parts(0xFF800000), (1, 255, 0, 'inf'), '-inf')
  inf_bits = struct.unpack('<I', struct.pack('<f', math.inf))[0]
  eq(inf_bits, 0x7F800000, 'struct oracle for inf')


def t_f32_nan():
  eq(MOD.float32_parts(0x7FC00000), (0, 255, 0x400000, 'nan'), 'quiet nan')
  eq(MOD.float32_parts(0x7F800001), (0, 255, 1, 'nan'), 'payload nan')
  assert math.isnan(struct.unpack('<f', struct.pack('<I', 0x7FC00000))[0])


def t_f32_smallest_subnormal():
  eq(MOD.float32_parts(0x00000001), (0, 0, 1, 'subnormal'))
  val = struct.unpack('<f', struct.pack('<I', 0x00000001))[0]
  eq(val, 2**-149, 'smallest subnormal is 2**-149')
  eq(
    MOD.float32_parts(0x007FFFFF),
    (0, 0, (1 << 23) - 1, 'subnormal'),
    'largest subnormal',
  )


def t_f32_range_errors():
  expect_raises(ValueError, MOD.float32_parts, -1)
  expect_raises(ValueError, MOD.float32_parts, 2**32)


# ------------------------------------------------------ float64 bits bridge


def t_f64_golden_bits():
  eq(MOD.float_to_bits(1.0), 0x3FF0000000000000)
  eq(MOD.float_to_bits(-0.0), 0x8000000000000000)
  eq(MOD.float_to_bits(math.inf), 0x7FF0000000000000)
  eq(MOD.float_to_bits(0.1), 0x3FB999999999999A)


def t_f64_bits_to_float_golden():
  eq(MOD.bits_to_float(0x3FF0000000000000), 1.0)
  eq(MOD.bits_to_float(0x7FF0000000000000), math.inf)
  eq(MOD.bits_to_float(0xFFF0000000000000), -math.inf)
  eq(MOD.bits_to_float(1), 2**-1074, 'smallest f64 subnormal')


def t_f64_negative_zero():
  z = MOD.bits_to_float(0x8000000000000000)
  eq(z, 0.0, '-0.0 == 0.0')
  assert math.copysign(1.0, z) < 0, 'sign bit lost on -0.0'


def t_f64_nan_roundtrip():
  f = MOD.bits_to_float(0x7FF8000000000000)
  assert math.isnan(f), 'canonical quiet NaN should be nan'
  eq(MOD.float_to_bits(f), 0x7FF8000000000000, 'nan bit pattern')


def t_f64_roundtrip_property():
  vals = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.1,
    2.0**-1074,
    1.7976931348623157e308,
    math.pi,
    math.inf,
    -math.inf,
    3.5e-320,
  ]
  for v in vals:
    bits = MOD.float_to_bits(v)
    oracle = int.from_bytes(struct.pack('<d', v), 'little')
    eq(bits, oracle, f'oracle v={v!r}')
    eq(MOD.float_to_bits(MOD.bits_to_float(bits)), bits, f'roundtrip v={v!r}')


def t_f64_bits_range_errors():
  expect_raises(ValueError, MOD.bits_to_float, -1)
  expect_raises(ValueError, MOD.bits_to_float, 2**64)


# --------------------------------------------------------------- BinaryReader


def _stream():
  return (
    bytes([0x2A])
    + (0x1234).to_bytes(2, 'little')
    + (0xDEADBEEF).to_bytes(4, 'little')
    + b'\xac\x02'
    + b'hi\x00'
    + b'tail'
  )


def t_reader_composite_walk():
  r = MOD.BinaryReader(_stream())
  eq(r.tell(), 0)
  eq(r.read_u8(), 0x2A)
  eq(r.read_u16_le(), 0x1234)
  eq(r.read_u32_le(), 0xDEADBEEF)
  eq(r.tell(), 7)
  eq(r.read_uvarint(), 300)
  eq(r.read_cstring(), b'hi')
  eq(r.tell(), 12)
  eq(r.read_bytes(4), b'tail')
  eq(r.tell(), 16)


def t_reader_seek():
  r = MOD.BinaryReader(_stream())
  r.seek(3)
  eq(r.read_u32_le(), 0xDEADBEEF)
  r.seek(0)
  eq(r.read_u8(), 0x2A)
  r.seek(len(_stream()))
  eq(r.tell(), len(_stream()), 'seek to end is legal')
  expect_raises(ValueError, r.seek, -1)
  expect_raises(ValueError, r.seek, len(_stream()) + 1)


def t_reader_eof_no_advance():
  r = MOD.BinaryReader(b'\x01\x02\x03')
  expect_raises(ValueError, r.read_u32_le)
  eq(r.tell(), 0, 'cursor after failed u32')
  eq(r.read_u16_le(), 0x0201)
  expect_raises(ValueError, r.read_u16_le)
  eq(r.tell(), 2, 'cursor after failed u16')
  eq(r.read_u8(), 3)
  expect_raises(ValueError, r.read_u8)
  eq(r.tell(), 3, 'cursor after failed u8 at end')


def t_reader_uvarint_basics():
  eq(MOD.BinaryReader(b'\x00').read_uvarint(), 0)
  eq(MOD.BinaryReader(b'\x7f').read_uvarint(), 127)
  eq(MOD.BinaryReader(b'\x80\x01').read_uvarint(), 128)
  r = MOD.BinaryReader(b'\xac\x02\xff')
  eq(r.read_uvarint(), 300)
  eq(r.tell(), 2)


def t_reader_uvarint_overlong_accepted():
  r = MOD.BinaryReader(b'\x80\x00')
  eq(r.read_uvarint(), 0, 'overlong zero decodes')
  eq(r.tell(), 2)


def t_reader_uvarint_u64_max():
  r = MOD.BinaryReader(b'\xff' * 9 + b'\x01')
  eq(r.read_uvarint(), 2**64 - 1)
  eq(r.tell(), 10)


def t_reader_uvarint_truncated():
  r = MOD.BinaryReader(b'\x80')
  expect_raises(ValueError, r.read_uvarint)
  eq(r.tell(), 0, 'cursor after truncated varint')
  expect_raises(ValueError, MOD.BinaryReader(b'').read_uvarint)


def t_reader_uvarint_too_long():
  r = MOD.BinaryReader(b'\x80' * 10 + b'\x01')
  expect_raises(ValueError, r.read_uvarint)
  eq(r.tell(), 0, 'cursor after over-length varint')


def t_reader_uvarint_overflow_64():
  r = MOD.BinaryReader(b'\xff' * 9 + b'\x02')
  expect_raises(ValueError, r.read_uvarint)
  eq(r.tell(), 0, 'cursor after overflowing varint')


def t_reader_read_bytes_edges():
  r = MOD.BinaryReader(b'abc')
  eq(r.read_bytes(0), b'')
  expect_raises(ValueError, r.read_bytes, -1)
  expect_raises(ValueError, r.read_bytes, 4)
  eq(r.tell(), 0, 'cursor after failed read_bytes')
  eq(r.read_bytes(3), b'abc')


def t_reader_cstring_edges():
  eq(MOD.BinaryReader(b'\x00').read_cstring(), b'')
  r = MOD.BinaryReader(b'abc')
  expect_raises(ValueError, r.read_cstring)
  eq(r.tell(), 0, 'cursor after unterminated cstring')
  r2 = MOD.BinaryReader(b'a\x00b\x00')
  eq(r2.read_cstring(), b'a')
  eq(r2.read_cstring(), b'b')
  eq(r2.tell(), 4)


def t_reader_bytearray_and_oracle():
  rng = random.Random(7)
  raw = bytearray(rng.randrange(256) for _ in range(12))
  r = MOD.BinaryReader(raw)
  eq(r.read_u32_le(), struct.unpack_from('<I', bytes(raw), 0)[0])
  eq(r.read_u16_le(), struct.unpack_from('<H', bytes(raw), 4)[0])
  eq(r.read_u8(), raw[6])


# -------------------------------------------------------------------- bswap32


def t_bswap_golden():
  eq(MOD.bswap32(0x11223344), 0x44332211)
  eq(MOD.bswap32(0xDEADBEEF), 0xEFBEADDE)
  eq(MOD.bswap32(0), 0)
  eq(MOD.bswap32(0xFF), 0xFF000000)
  eq(MOD.bswap32(0xFFFFFFFF), 0xFFFFFFFF)


def t_bswap_oracle_involution():
  rng = random.Random(0x1234)
  for _ in range(50):
    x = rng.randrange(2**32)
    want = int.from_bytes(x.to_bytes(4, 'big'), 'little')
    eq(MOD.bswap32(x), want, f'x={x:#010x}')
    eq(MOD.bswap32(MOD.bswap32(x)), x, f'involution x={x:#010x}')


def t_bswap_range_errors():
  expect_raises(ValueError, MOD.bswap32, -1)
  expect_raises(ValueError, MOD.bswap32, 2**32)


# -------------------------------------------------------------------- hexdump


def t_hexdump_empty():
  eq(MOD.hexdump(b''), '')


def t_hexdump_partial_row():
  want = '00000000  00 01 02' + ' ' * 42 + '|...|'
  eq(MOD.hexdump(b'\x00\x01\x02'), want)


def t_hexdump_full_row():
  want = (
    '00000000  61 62 63 64 65 66 67 68  '
    '69 6a 6b 6c 6d 6e 6f 70  |abcdefghijklmnop|'
  )
  eq(MOD.hexdump(bytes(range(0x61, 0x71))), want)


def t_hexdump_two_rows():
  buf = bytes(range(0x61, 0x71)) + b'qr'
  want = (
    '00000000  61 62 63 64 65 66 67 68  '
    '69 6a 6b 6c 6d 6e 6f 70  |abcdefghijklmnop|\n'
    '00000010  71 72' + ' ' * 45 + '|qr|'
  )
  eq(MOD.hexdump(buf), want)


def t_hexdump_printable_boundary():
  want = '00000000  1f 20 7e 7f' + ' ' * 39 + '|. ~.|'
  eq(MOD.hexdump(b'\x1f\x20\x7e\x7f'), want)


def t_hexdump_png_header():
  buf = b'\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR'
  want = (
    '00000000  89 50 4e 47 0d 0a 1a 0a  '
    '00 00 00 0d 49 48 44 52  |.PNG........IHDR|'
  )
  eq(MOD.hexdump(buf), want)


def t_hexdump_no_trailing_newline():
  out = MOD.hexdump(bytes(32))
  assert isinstance(out, str), 'must return str'
  assert not out.endswith('\n'), 'no trailing newline'
  eq(out.count('\n'), 1, '32 bytes -> 2 rows, 1 separator')
  assert out.split('\n')[1].startswith('00000010  '), 'second-row offset'


CASES = [
  ('read_u32_le golden vectors', t_u32le_golden),
  ('read_u32_be golden vectors', t_u32be_golden),
  ('read_u32_* oracle sweep vs from_bytes/struct', t_u32_oracle_sweep),
  ('read_u32_* mid-buffer offset', t_u32_offset_middle),
  ('read_u32_le bytearray input', t_u32_bytearray),
  ('read_u32_le last valid offset', t_u32_last_valid_offset),
  ('read_u32_* bounds ValueError', t_u32_bounds_errors),
  ('write_i64_le golden vectors', t_i64_golden),
  ('write_i64_le width boundaries', t_i64_width_boundaries),
  ("write_i64_le oracle sweep vs struct '<q'", t_i64_oracle_sweep),
  ('write_i64_le range ValueError', t_i64_range_errors),
  ('float32_parts normals', t_f32_normal),
  ('float32_parts signed zeroes', t_f32_zeroes),
  ('float32_parts infinities', t_f32_inf),
  ('float32_parts NaNs', t_f32_nan),
  ('float32_parts smallest subnormal 2**-149', t_f32_smallest_subnormal),
  ('float32_parts range ValueError', t_f32_range_errors),
  ('float_to_bits golden vectors', t_f64_golden_bits),
  ('bits_to_float golden vectors', t_f64_bits_to_float_golden),
  ('bits_to_float negative zero sign', t_f64_negative_zero),
  ('float64 NaN bit round-trip', t_f64_nan_roundtrip),
  ('float64 round-trip property + oracle', t_f64_roundtrip_property),
  ('bits_to_float range ValueError', t_f64_bits_range_errors),
  ('BinaryReader composite walk', t_reader_composite_walk),
  ('BinaryReader tell/seek', t_reader_seek),
  ('BinaryReader EOF leaves cursor', t_reader_eof_no_advance),
  ('BinaryReader uvarint basics', t_reader_uvarint_basics),
  (
    'BinaryReader uvarint overlong accepted',
    t_reader_uvarint_overlong_accepted,
  ),
  ('BinaryReader uvarint u64 max (10 bytes)', t_reader_uvarint_u64_max),
  ('BinaryReader uvarint truncated', t_reader_uvarint_truncated),
  ('BinaryReader uvarint >10 bytes rejected', t_reader_uvarint_too_long),
  (
    'BinaryReader uvarint 64-bit overflow rejected',
    t_reader_uvarint_overflow_64,
  ),
  ('BinaryReader read_bytes edges', t_reader_read_bytes_edges),
  ('BinaryReader read_cstring edges', t_reader_cstring_edges),
  ('BinaryReader bytearray + struct oracle', t_reader_bytearray_and_oracle),
  ('bswap32 golden vectors', t_bswap_golden),
  ('bswap32 oracle + involution', t_bswap_oracle_involution),
  ('bswap32 range ValueError', t_bswap_range_errors),
  ('hexdump empty input', t_hexdump_empty),
  ('hexdump partial row padding', t_hexdump_partial_row),
  ('hexdump full 16-byte row', t_hexdump_full_row),
  ('hexdump two rows + offset column', t_hexdump_two_rows),
  ('hexdump printable boundary 0x20/0x7e', t_hexdump_printable_boundary),
  ('hexdump PNG header realistic vector', t_hexdump_png_header),
  ('hexdump str type, no trailing newline', t_hexdump_no_trailing_newline),
]


def main():
  for name, fn in CASES:
    check(name, fn)
  total = PASSED + FAILED
  print(f'{PASSED}/{total} passed')
  if FAILED:
    sys.exit(1)


if __name__ == '__main__':
  main()
