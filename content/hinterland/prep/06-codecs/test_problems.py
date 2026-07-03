"""Self-contained test harness for the classic-codecs module. Stdlib only.

python3 test_problems.py                              # test problems.py
PRACTICE_MODULE=solutions python3 test_problems.py    # test solutions.py
"""

import base64
import importlib
import os
import random
import sys

MOD = importlib.import_module(os.environ.get('PRACTICE_MODULE', 'problems'))

CASES = []


def case(fn):
  CASES.append((fn.__name__, fn))
  return fn


def eq(got, want, what='value'):
  if got != want:
    raise AssertionError(f'{what}: got {got!r}, want {want!r}')


def expect_error(thunk, token, what='call'):
  try:
    thunk()
  except ValueError as e:
    if token.lower() not in str(e).lower():
      raise AssertionError(
        f'{what}: expected {token!r} in error, got {str(e)!r}'
      )
    return
  raise AssertionError(
    f'{what}: expected ValueError containing {token!r}, nothing raised'
  )


UTF8_VECTORS = [
  (0x00, b'\x00'),
  (0x41, b'A'),
  (0x7F, b'\x7f'),
  (0x80, b'\xc2\x80'),
  (0xE9, b'\xc3\xa9'),
  (0x7FF, b'\xdf\xbf'),
  (0x800, b'\xe0\xa0\x80'),
  (0x20AC, b'\xe2\x82\xac'),
  (0xFFFF, b'\xef\xbf\xbf'),
  (0x10000, b'\xf0\x90\x80\x80'),
  (0x1F600, b'\xf0\x9f\x98\x80'),
  (0x10FFFF, b'\xf4\x8f\xbf\xbf'),
]


def rand_codepoint(rng):
  while True:
    cp = rng.randrange(0x110000)
    if not 0xD800 <= cp <= 0xDFFF:
      return cp


@case
def utf8_encode_boundary_vectors():
  for cp, want in UTF8_VECTORS:
    eq(MOD.utf8_encode(cp), want, f'utf8_encode(U+{cp:04X})')


@case
def utf8_encode_random_vs_stdlib():
  rng = random.Random(1337)
  for _ in range(2000):
    cp = rand_codepoint(rng)
    eq(
      MOD.utf8_encode(cp), chr(cp).encode('utf-8'), f'utf8_encode(U+{cp:04X})'
    )


@case
def utf8_encode_rejects_out_of_range():
  for cp in (-1, -100, 0x110000, 0x7FFFFFFF):
    expect_error(
      lambda cp=cp: MOD.utf8_encode(cp), 'out of range', f'utf8_encode({cp})'
    )


@case
def utf8_encode_rejects_surrogates():
  for cp in (0xD800, 0xDBFF, 0xDC00, 0xDFFF):
    expect_error(
      lambda cp=cp: MOD.utf8_encode(cp),
      'surrogate',
      f'utf8_encode(U+{cp:04X})',
    )


@case
def utf8_decode_single_vectors():
  eq(MOD.utf8_decode(b''), [], "utf8_decode(b'')")
  eq(
    MOD.utf8_decode(b'hello'),
    [104, 101, 108, 108, 111],
    "utf8_decode(b'hello')",
  )
  for cp, enc in UTF8_VECTORS:
    eq(MOD.utf8_decode(enc), [cp], f'utf8_decode({enc.hex()})')


@case
def utf8_decode_mixed_stream():
  stream = b''.join(enc for _, enc in UTF8_VECTORS)
  eq(MOD.utf8_decode(stream), [cp for cp, _ in UTF8_VECTORS], 'mixed stream')


@case
def utf8_decode_rejects_lone_continuation():
  for buf in (b'\x80', b'\xbf', b'A\x80B'):
    expect_error(
      lambda buf=buf: MOD.utf8_decode(buf), 'invalid start byte', buf.hex()
    )


@case
def utf8_decode_rejects_invalid_lead():
  for buf in (b'\xf8\x80\x80\x80\x80', b'\xfe', b'\xff'):
    expect_error(
      lambda buf=buf: MOD.utf8_decode(buf), 'invalid start byte', buf.hex()
    )


@case
def utf8_decode_rejects_overlong_2byte():
  for buf in (b'\xc0\x80', b'\xc0\xaf', b'\xc1\xbf'):
    expect_error(lambda buf=buf: MOD.utf8_decode(buf), 'overlong', buf.hex())


@case
def utf8_decode_rejects_overlong_3byte():
  for buf in (b'\xe0\x80\x80', b'\xe0\x9f\xbf'):
    expect_error(lambda buf=buf: MOD.utf8_decode(buf), 'overlong', buf.hex())


@case
def utf8_decode_rejects_overlong_4byte():
  for buf in (b'\xf0\x80\x80\x80', b'\xf0\x8f\xbf\xbf'):
    expect_error(lambda buf=buf: MOD.utf8_decode(buf), 'overlong', buf.hex())


@case
def utf8_decode_rejects_surrogates():
  for buf in (b'\xed\xa0\x80', b'\xed\xbf\xbf'):
    expect_error(lambda buf=buf: MOD.utf8_decode(buf), 'surrogate', buf.hex())


@case
def utf8_decode_rejects_truncated():
  for buf in (b'\xc3', b'\xe2\x82', b'\xf0\x9f\x98', b'hello\xe2\x82'):
    expect_error(lambda buf=buf: MOD.utf8_decode(buf), 'truncated', buf.hex())


@case
def utf8_decode_rejects_bad_continuation():
  for buf in (
    b'\xc3\x28',
    b'\xe2\x28\xac',
    b'\xe2\x82\xc3',
    b'\xf0\x9f\x28\x80',
  ):
    expect_error(
      lambda buf=buf: MOD.utf8_decode(buf), 'invalid continuation', buf.hex()
    )


@case
def utf8_decode_rejects_beyond_max():
  for buf in (b'\xf4\x90\x80\x80', b'\xf5\x80\x80\x80', b'\xf7\xbf\xbf\xbf'):
    expect_error(
      lambda buf=buf: MOD.utf8_decode(buf), 'out of range', buf.hex()
    )


@case
def utf8_decode_random_vs_stdlib():
  rng = random.Random(1337)
  for _ in range(300):
    s = ''.join(chr(rand_codepoint(rng)) for _ in range(rng.randrange(0, 40)))
    buf = s.encode('utf-8')
    eq(MOD.utf8_decode(buf), [ord(c) for c in s], f'decode {buf.hex()[:40]}')


@case
def utf8_fuzz_agrees_with_stdlib():
  rng = random.Random(1337)
  accepted = rejected = 0
  for _ in range(600):
    blob = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 13)))
    try:
      want = [ord(c) for c in blob.decode('utf-8')]
      stdlib_ok = True
    except UnicodeDecodeError:
      stdlib_ok = False
    if stdlib_ok:
      accepted += 1
      eq(MOD.utf8_decode(blob), want, f'fuzz accept {blob.hex()}')
    else:
      rejected += 1
      expect_error(
        lambda blob=blob: MOD.utf8_decode(blob),
        '',
        f'fuzz reject {blob.hex()}',
      )
  if accepted == 0 or rejected == 0:
    raise AssertionError(
      f'fuzz did not exercise both paths: {accepted}/{rejected}'
    )


B64_RFC_VECTORS = [
  (b'', ''),
  (b'f', 'Zg=='),
  (b'fo', 'Zm8='),
  (b'foo', 'Zm9v'),
  (b'foob', 'Zm9vYg=='),
  (b'fooba', 'Zm9vYmE='),
  (b'foobar', 'Zm9vYmFy'),
]


@case
def b64_encode_rfc4648_vectors():
  for data, want in B64_RFC_VECTORS:
    eq(MOD.b64_encode(data), want, f'b64_encode({data!r})')


@case
def b64_encode_binary_vectors():
  for data, want in [
    (b'Man', 'TWFu'),
    (b'\x00', 'AA=='),
    (b'\x00\x00', 'AAA='),
    (b'\x00\x00\x00', 'AAAA'),
    (b'\xff\xff\xff', '////'),
    (b'\xfb\xef\xbe', '++++'),
  ]:
    eq(MOD.b64_encode(data), want, f'b64_encode({data!r})')


@case
def b64_encode_random_vs_stdlib():
  rng = random.Random(1337)
  for _ in range(400):
    data = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 80)))
    eq(
      MOD.b64_encode(data),
      base64.b64encode(data).decode('ascii'),
      f'encode {data.hex()[:32]}',
    )


@case
def b64_decode_rfc4648_vectors():
  for data, enc in B64_RFC_VECTORS:
    eq(MOD.b64_decode(enc), data, f'b64_decode({enc!r})')


@case
def b64_roundtrip_random():
  rng = random.Random(1337)
  for _ in range(400):
    data = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 80)))
    eq(MOD.b64_decode(MOD.b64_encode(data)), data, 'own round trip')
    eq(
      MOD.b64_decode(base64.b64encode(data).decode('ascii')),
      data,
      'decode stdlib output',
    )


@case
def b64_decode_rejects_bad_length():
  for s in ('A', 'AB', 'ABC', 'TQ=', 'AAAAA', '='):
    expect_error(lambda s=s: MOD.b64_decode(s), 'length', repr(s))


@case
def b64_decode_rejects_bad_padding():
  for s in ('T===', 'TQ=A', '=AAA', 'AB=C', '====', 'TWFu=AAA'):
    expect_error(lambda s=s: MOD.b64_decode(s), 'padding', repr(s))


@case
def b64_decode_rejects_invalid_chars():
  for s in ('TW!u', 'TW u', 'TW\nu', 'TWF.', 'TWF\x00', 'TWéu'):
    expect_error(lambda s=s: MOD.b64_decode(s), 'invalid character', repr(s))


@case
def b64_decode_noncanonical_trailing_bits_match_stdlib():
  for s in ('TR==', 'TWE=', 'TWF='):
    eq(MOD.b64_decode(s), base64.b64decode(s), f'noncanonical {s!r}')


RLE_VECTORS = [
  (b'', b''),
  (b'A', b'\x01A'),
  (b'AAAABBBCCD', b'\x04A\x03B\x02C\x01D'),
  (b'ABAB', b'\x01A\x01B\x01A\x01B'),
]


@case
def rle_encode_vectors():
  for data, want in RLE_VECTORS:
    eq(MOD.rle_encode(data), want, f'rle_encode({data!r})')


@case
def rle_decode_vectors():
  for data, enc in RLE_VECTORS:
    eq(MOD.rle_decode(enc), data, f'rle_decode({enc!r})')


@case
def rle_run_split_at_255():
  eq(MOD.rle_encode(b'A' * 255), b'\xffA', '255 run')
  eq(MOD.rle_encode(b'A' * 256), b'\xffA\x01A', '256 run')
  eq(MOD.rle_encode(b'A' * 300), b'\xffA\x2dA', '300 run')
  for n in (255, 256, 300, 511, 1000):
    eq(
      MOD.rle_decode(MOD.rle_encode(b'z' * n)), b'z' * n, f'run {n} round trip'
    )


@case
def rle_roundtrip_random():
  rng = random.Random(1337)
  for _ in range(120):
    data = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 200)))
    eq(MOD.rle_decode(MOD.rle_encode(data)), data, 'uniform round trip')
  for _ in range(60):
    runs = bytearray()
    for _ in range(rng.randrange(1, 10)):
      runs += bytes([rng.randrange(256)]) * rng.randrange(1, 600)
    eq(
      MOD.rle_decode(MOD.rle_encode(bytes(runs))),
      bytes(runs),
      'runny round trip',
    )


@case
def rle_decode_rejects_odd_length():
  for buf in (b'\x01', b'\x02A\x03'):
    expect_error(lambda buf=buf: MOD.rle_decode(buf), 'odd', buf.hex())


@case
def rle_decode_rejects_zero_count():
  for buf in (b'\x00A', b'\x01A\x00B'):
    expect_error(lambda buf=buf: MOD.rle_decode(buf), 'zero count', buf.hex())


@case
def bitpack_worked_example():
  eq(MOD.bitpack([1, 2, 3, 4, 5], 3), bytes([0xD1, 0x58]), 'bitpack k=3')
  eq(
    MOD.bitunpack(bytes([0xD1, 0x58]), 3, 5), [1, 2, 3, 4, 5], 'bitunpack k=3'
  )


@case
def bitpack_k1_vector():
  eq(MOD.bitpack([1, 0, 1, 1, 0, 0, 1, 0, 1], 1), b'\x4d\x01', 'k=1 nine bits')
  eq(MOD.bitpack([1], 1), b'\x01', 'k=1 single')
  eq(
    MOD.bitunpack(b'\x4d\x01', 1, 9), [1, 0, 1, 1, 0, 0, 1, 0, 1], 'k=1 unpack'
  )


@case
def bitpack_k8_identity():
  eq(MOD.bitpack(list(b'hello world'), 8), b'hello world', 'k=8 pack identity')
  eq(
    MOD.bitunpack(b'hello world', 8, 11),
    list(b'hello world'),
    'k=8 unpack identity',
  )


@case
def bitpack_empty_and_zero_n():
  eq(MOD.bitpack([], 5), b'', 'empty pack')
  eq(MOD.bitunpack(b'', 5, 0), [], 'empty unpack')


@case
def bitpack_k64_extremes():
  values = [(1 << 64) - 1, 0, 123456789]
  packed = MOD.bitpack(values, 64)
  eq(len(packed), 24, 'k=64 length')
  eq(MOD.bitunpack(packed, 64, 3), values, 'k=64 round trip')


@case
def bitpack_roundtrip_random():
  rng = random.Random(1337)
  for k in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 24, 31, 32, 33, 63, 64):
    n = rng.randrange(0, 40)
    values = [rng.randrange(1 << k) for _ in range(n)]
    packed = MOD.bitpack(values, k)
    eq(len(packed), (n * k + 7) // 8, f'packed length k={k}')
    eq(MOD.bitunpack(packed, k, n), values, f'round trip k={k}')


@case
def bitpack_rejects_bad_k():
  for k in (0, -1, 65):
    expect_error(
      lambda k=k: MOD.bitpack([0], k), 'k out of range', f'bitpack k={k}'
    )
    expect_error(
      lambda k=k: MOD.bitunpack(b'\x00', k, 1),
      'k out of range',
      f'bitunpack k={k}',
    )


@case
def bitpack_rejects_out_of_range_values():
  expect_error(lambda: MOD.bitpack([8], 3), 'out of range for', 'value 8, k=3')
  expect_error(lambda: MOD.bitpack([-1], 3), 'out of range for', 'value -1')
  expect_error(
    lambda: MOD.bitpack([1 << 64], 64), 'out of range for', 'value 2**64'
  )


@case
def bitunpack_rejects_wrong_length():
  expect_error(
    lambda: MOD.bitunpack(b'\xd1', 3, 5), 'length', 'truncated buffer'
  )
  expect_error(
    lambda: MOD.bitunpack(b'\xd1\x58\x00', 3, 5), 'length', 'trailing bytes'
  )
  expect_error(lambda: MOD.bitunpack(b'', 3, -1), 'length', 'negative n')


@case
def dv_empty():
  eq(MOD.delta_varint_encode([]), b'', 'encode []')
  eq(MOD.delta_varint_decode(b''), [], "decode b''")


@case
def dv_worked_example():
  enc = MOD.delta_varint_encode([1000, 1005, 1004, 1010])
  eq(enc, bytes.fromhex('d00f0a010c'), 'capstone bytes')
  eq(
    MOD.delta_varint_decode(enc),
    [1000, 1005, 1004, 1010],
    'capstone round trip',
  )


@case
def dv_single_value_vectors():
  for value, want in [
    (0, b'\x00'),
    (-1, b'\x01'),
    (1, b'\x02'),
    (64, b'\x80\x01'),
    (-64, b'\x7f'),
  ]:
    eq(MOD.delta_varint_encode([value]), want, f'encode [{value}]')
    eq(MOD.delta_varint_decode(want), [value], f'decode {want.hex()}')


@case
def dv_sorted_timestamps_compress():
  rng = random.Random(1337)
  ts = []
  t = 1_700_000_000
  for _ in range(200):
    t += rng.randrange(0, 64)
    ts.append(t)
  enc = MOD.delta_varint_encode(ts)
  eq(MOD.delta_varint_decode(enc), ts, 'timestamp round trip')
  if len(enc) > 210:
    raise AssertionError(
      f'expected < 210 bytes for 200 small-delta timestamps, got {len(enc)}'
    )


@case
def dv_unsorted_negative_roundtrip():
  seqs = [
    [5, 3, 8, 8, -2, 100],
    [-1000, -999, -2000, 0],
    [0, 0, 0],
    list(range(100, 0, -1)),
  ]
  for seq in seqs:
    eq(
      MOD.delta_varint_decode(MOD.delta_varint_encode(seq)),
      seq,
      f'seq {seq[:4]}...',
    )


@case
def dv_int64_extreme_wrap():
  lo, hi = -(1 << 63), (1 << 63) - 1
  for seq in ([lo, hi], [hi, lo], [lo, hi, lo], [0, hi, lo, 0]):
    enc = MOD.delta_varint_encode(seq)
    eq(MOD.delta_varint_decode(enc), seq, f'wrap seq {seq}')
  eq(len(MOD.delta_varint_encode([lo, hi])), 11, 'wrapped delta is 1 byte')


@case
def dv_uvarint_boundary_max_u64():
  eq(
    MOD.delta_varint_decode(b'\xff' * 9 + b'\x01'),
    [-(1 << 63)],
    'max u64 zigzag',
  )


@case
def dv_encode_rejects_non_int64():
  for value in (1 << 63, -(1 << 63) - 1, 1 << 100):
    expect_error(
      lambda value=value: MOD.delta_varint_encode([value]),
      'int64',
      f'encode [{value}]',
    )


@case
def dv_decode_rejects_truncated():
  for buf in (b'\x80', b'\xd0', b'\x0a\xd0\x0f\x80'):
    expect_error(
      lambda buf=buf: MOD.delta_varint_decode(buf), 'truncated', buf.hex()
    )


@case
def dv_decode_rejects_too_long():
  expect_error(
    lambda: MOD.delta_varint_decode(b'\xff' * 10),
    'too long',
    '10 continuation bytes',
  )
  expect_error(
    lambda: MOD.delta_varint_decode(b'\x80' * 10 + b'\x00'),
    'too long',
    '11-byte varint',
  )


@case
def dv_decode_rejects_overflow():
  for buf in (b'\xff' * 9 + b'\x7f', b'\x80' * 9 + b'\x02'):
    expect_error(
      lambda buf=buf: MOD.delta_varint_decode(buf), 'overflow', buf.hex()
    )


@case
def dv_decode_accepts_nonminimal():
  eq(MOD.delta_varint_decode(b'\x80\x00'), [0], 'nonminimal 80 00')
  eq(MOD.delta_varint_decode(b'\x82\x00'), [1], 'nonminimal 82 00')


@case
def dv_roundtrip_random():
  rng = random.Random(1337)
  for _ in range(80):
    m = rng.randrange(0, 60)
    seq = [rng.randrange(-(1 << 63), 1 << 63) for _ in range(m)]
    eq(
      MOD.delta_varint_decode(MOD.delta_varint_encode(seq)),
      seq,
      'random int64 seq',
    )
  for _ in range(40):
    t = rng.randrange(1 << 40)
    seq = []
    for _ in range(rng.randrange(0, 60)):
      t += rng.randrange(-5, 1000)
      seq.append(t)
    eq(
      MOD.delta_varint_decode(MOD.delta_varint_encode(seq)),
      seq,
      'random timestamps',
    )


@case
def fletcher16_vectors():
  for data, want in [
    (b'abcde', 0xC8F0),
    (b'abcdef', 0x2057),
    (b'abcdefgh', 0x0627),
    (b'', 0x0000),
    (b'\x01\x02', 0x0403),
  ]:
    eq(MOD.fletcher16(data), want, f'fletcher16({data!r})')


@case
def fletcher16_order_sensitivity_and_00ff_blindspot():
  if MOD.fletcher16(b'AB') == MOD.fletcher16(b'BA'):
    raise AssertionError('fletcher16 must detect byte reordering')
  eq(MOD.fletcher16(b'\x00'), 0x0000, 'fletcher16(00)')
  eq(MOD.fletcher16(b'\xff'), 0x0000, 'fletcher16(ff) mod-255 blind spot')
  eq(
    MOD.fletcher16(b'\x00\xab'),
    MOD.fletcher16(b'\xff\xab'),
    '00/ff congruence',
  )


def main():
  random.seed(1337)
  passed = 0
  for name, fn in CASES:
    try:
      fn()
    except Exception as e:
      print(f'FAIL {name}: {e!r}')
    else:
      print(f'PASS {name}')
      passed += 1
  total = len(CASES)
  print(f'{passed}/{total} passed')
  sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
  main()
