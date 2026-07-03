"""Self-contained test harness for the varint module. stdlib only.

Usage:
    python3 test_problems.py                            # tests problems.py
    PRACTICE_MODULE=solutions python3 test_problems.py  # tests solutions.py
"""

import importlib
import os
import sys

MOD = importlib.import_module(os.environ.get('PRACTICE_MODULE', 'problems'))

PASSED = 0
FAILED = 0


def case(name, fn):
  global PASSED, FAILED
  try:
    fn()
  except Exception as exc:
    FAILED += 1
    print(f'FAIL {name}: {type(exc).__name__}: {exc}')
  else:
    PASSED += 1
    print(f'PASS {name}')


def eq(got, want):
  if got != want:
    raise AssertionError(f'got {got!r}, want {want!r}')


def raises(fn, substr):
  try:
    fn()
  except ValueError as exc:
    msg = str(exc).lower()
    if substr not in msg:
      raise AssertionError(
        f'wrong error class: got {msg!r}, want substring {substr!r}'
      )
  else:
    raise AssertionError(
      f'expected ValueError containing {substr!r}, got none'
    )


# ---------------------------------------------------------------- uvarint

UVARINT_VECTORS = [
  (0, '00'),
  (1, '01'),
  (127, '7f'),
  (128, '8001'),
  (300, 'ac02'),
  (16383, 'ff7f'),
  (16384, '808001'),
  (2**32 - 1, 'ffffffff0f'),
  (2**63, '80' * 9 + '01'),
  (2**64 - 1, 'ff' * 9 + '01'),
]

for n, h in UVARINT_VECTORS:
  case(
    f'encode_uvarint({n}) == {h}',
    lambda n=n, h=h: eq(MOD.encode_uvarint(n), bytes.fromhex(h)),
  )
  case(
    f'decode_uvarint({h}) == ({n}, {len(h) // 2})',
    lambda n=n, h=h: eq(
      MOD.decode_uvarint(bytes.fromhex(h)), (n, len(h) // 2)
    ),
  )


def uvarint_boundaries():
  for k in range(1, 10):
    lo, hi = 2 ** (7 * k) - 1, 2 ** (7 * k)
    eq(len(MOD.encode_uvarint(lo)), k)
    eq(MOD.decode_uvarint(MOD.encode_uvarint(lo)), (lo, k))
    if hi <= 2**64 - 1:
      eq(len(MOD.encode_uvarint(hi)), k + 1)
      eq(MOD.decode_uvarint(MOD.encode_uvarint(hi)), (hi, k + 1))


case('uvarint 7-bit width boundaries roundtrip', uvarint_boundaries)
case(
  'decode_uvarint honors offset, returns consumed not new offset',
  lambda: eq(MOD.decode_uvarint(b'\xff\xac\x02\x99', 1), (300, 2)),
)
case(
  'decode_uvarint at exact end offset raises truncated',
  lambda: raises(lambda: MOD.decode_uvarint(b'\x01', 1), 'truncated'),
)
case(
  'uvarint empty buffer -> truncated',
  lambda: raises(lambda: MOD.decode_uvarint(b''), 'truncated'),
)
case(
  'uvarint lone continuation byte 80 -> truncated',
  lambda: raises(lambda: MOD.decode_uvarint(b'\x80'), 'truncated'),
)
case(
  'uvarint 80 80 (mid-value EOF) -> truncated',
  lambda: raises(lambda: MOD.decode_uvarint(b'\x80\x80'), 'truncated'),
)
case(
  'uvarint 80*10 + 01 -> too long',
  lambda: raises(
    lambda: MOD.decode_uvarint(b'\x80' * 10 + b'\x01'), 'too long'
  ),
)
case(
  'uvarint ff*10 at EOF -> too long (not truncated)',
  lambda: raises(lambda: MOD.decode_uvarint(b'\xff' * 10), 'too long'),
)
case(
  'uvarint ff*9 + 02 (10th group > 1) -> too long',
  lambda: raises(
    lambda: MOD.decode_uvarint(b'\xff' * 9 + b'\x02'), 'too long'
  ),
)
case(
  'uvarint ff*9 + 7f -> too long',
  lambda: raises(
    lambda: MOD.decode_uvarint(b'\xff' * 9 + b'\x7f'), 'too long'
  ),
)
case(
  'uvarint 80 00 -> overlong',
  lambda: raises(lambda: MOD.decode_uvarint(b'\x80\x00'), 'overlong'),
)
case(
  'uvarint ff 80 00 -> overlong',
  lambda: raises(lambda: MOD.decode_uvarint(b'\xff\x80\x00'), 'overlong'),
)
case(
  'uvarint ff*9 + 00 -> overlong',
  lambda: raises(
    lambda: MOD.decode_uvarint(b'\xff' * 9 + b'\x00'), 'overlong'
  ),
)
case(
  'uvarint 00 is canonical (single zero byte fine)',
  lambda: eq(MOD.decode_uvarint(b'\x00'), (0, 1)),
)
case(
  'encode_uvarint(-1) -> range error',
  lambda: raises(lambda: MOD.encode_uvarint(-1), 'range'),
)
case(
  'encode_uvarint(2**64) -> range error',
  lambda: raises(lambda: MOD.encode_uvarint(2**64), 'range'),
)

# ---------------------------------------------------------------- zigzag

ZIGZAG_VECTORS = [
  (0, 0),
  (-1, 1),
  (1, 2),
  (-2, 3),
  (2, 4),
  (2**63 - 1, 2**64 - 2),
  (-(2**63), 2**64 - 1),
]

for s, u in ZIGZAG_VECTORS:
  case(
    f'zigzag_encode({s}) == {u}', lambda s=s, u=u: eq(MOD.zigzag_encode(s), u)
  )
  case(
    f'zigzag_decode({u}) == {s}', lambda s=s, u=u: eq(MOD.zigzag_decode(u), s)
  )


def zigzag_sweep():
  for s in (
    0,
    1,
    -1,
    2,
    -2,
    63,
    -63,
    64,
    -64,
    127,
    -128,
    300,
    -300,
    2**31,
    -(2**31),
    2**62,
    -(2**62),
    2**63 - 1,
    -(2**63),
  ):
    eq(MOD.zigzag_decode(MOD.zigzag_encode(s)), s)


case('zigzag roundtrip sweep', zigzag_sweep)
case(
  'zigzag_encode(2**63) -> range error',
  lambda: raises(lambda: MOD.zigzag_encode(2**63), 'range'),
)
case(
  'zigzag_encode(-2**63 - 1) -> range error',
  lambda: raises(lambda: MOD.zigzag_encode(-(2**63) - 1), 'range'),
)
case(
  'zigzag_decode(-1) -> range error',
  lambda: raises(lambda: MOD.zigzag_decode(-1), 'range'),
)
case(
  'zigzag_decode(2**64) -> range error',
  lambda: raises(lambda: MOD.zigzag_decode(2**64), 'range'),
)

# ---------------------------------------------------------------- svarint

SVARINT_VECTORS = [
  (0, '00'),
  (-1, '01'),
  (1, '02'),
  (-2, '03'),
  (2, '04'),
  (63, '7e'),
  (-64, '7f'),
  (64, '8001'),
  (-65, '8101'),
  (2**63 - 1, 'fe' + 'ff' * 8 + '01'),
  (-(2**63), 'ff' * 9 + '01'),
]

for s, h in SVARINT_VECTORS:
  case(
    f'encode_svarint({s}) == {h}',
    lambda s=s, h=h: eq(MOD.encode_svarint(s), bytes.fromhex(h)),
  )
  case(
    f'decode_svarint({h}) == ({s}, {len(h) // 2})',
    lambda s=s, h=h: eq(
      MOD.decode_svarint(bytes.fromhex(h)), (s, len(h) // 2)
    ),
  )

case(
  'decode_svarint honors offset',
  lambda: eq(MOD.decode_svarint(b'\x00\xac\x02', 1), (150, 2)),
)
case(
  'decode_svarint odd u -> negative',
  lambda: eq(MOD.decode_svarint(b'\xab\x02'), (-150, 2)),
)
case(
  'decode_svarint propagates overlong',
  lambda: raises(lambda: MOD.decode_svarint(b'\x80\x00'), 'overlong'),
)
case(
  'decode_svarint propagates truncated',
  lambda: raises(lambda: MOD.decode_svarint(b'\x80'), 'truncated'),
)
case(
  'encode_svarint(2**63) -> range error',
  lambda: raises(lambda: MOD.encode_svarint(2**63), 'range'),
)

# ---------------------------------------------------------------- vlq

VLQ_VECTORS = [
  (0, '00'),
  (1, '01'),
  (127, '7f'),
  (128, '8100'),
  (300, '822c'),
  (16383, 'ff7f'),
  (16384, '818000'),
  (0x0FFFFFFF, 'ffffff7f'),
  (2**32 - 1, '8fffffff7f'),
  (2**64 - 1, '81' + 'ff' * 8 + '7f'),
]

for n, h in VLQ_VECTORS:
  case(
    f'encode_vlq({n}) == {h}',
    lambda n=n, h=h: eq(MOD.encode_vlq(n), bytes.fromhex(h)),
  )
  case(
    f'decode_vlq({h}) == ({n}, {len(h) // 2})',
    lambda n=n, h=h: eq(MOD.decode_vlq(bytes.fromhex(h)), (n, len(h) // 2)),
  )


def vlq_boundaries():
  for k in range(1, 10):
    lo, hi = 2 ** (7 * k) - 1, 2 ** (7 * k)
    eq(len(MOD.encode_vlq(lo)), k)
    eq(MOD.decode_vlq(MOD.encode_vlq(lo)), (lo, k))
    if hi <= 2**64 - 1:
      eq(len(MOD.encode_vlq(hi)), k + 1)
      eq(MOD.decode_vlq(MOD.encode_vlq(hi)), (hi, k + 1))


case('vlq 7-bit width boundaries roundtrip', vlq_boundaries)
case(
  'decode_vlq honors offset',
  lambda: eq(MOD.decode_vlq(b'\x00\x82\x2c', 1), (300, 2)),
)
case(
  'vlq empty buffer -> truncated',
  lambda: raises(lambda: MOD.decode_vlq(b''), 'truncated'),
)
case(
  'vlq lone 81 -> truncated',
  lambda: raises(lambda: MOD.decode_vlq(b'\x81'), 'truncated'),
)
case(
  'vlq leading 80 -> overlong (even at EOF)',
  lambda: raises(lambda: MOD.decode_vlq(b'\x80'), 'overlong'),
)
case(
  'vlq 80 00 -> overlong',
  lambda: raises(lambda: MOD.decode_vlq(b'\x80\x00'), 'overlong'),
)
case(
  'vlq 11 groups -> too long',
  lambda: raises(
    lambda: MOD.decode_vlq(b'\x81' + b'\x80' * 9 + b'\x00'), 'too long'
  ),
)
case(
  'vlq 10 groups leading 2 (overflow) -> too long',
  lambda: raises(
    lambda: MOD.decode_vlq(b'\x82' + b'\xff' * 8 + b'\x7f'), 'too long'
  ),
)
case(
  'encode_vlq(-1) -> range error',
  lambda: raises(lambda: MOD.encode_vlq(-1), 'range'),
)
case(
  'encode_vlq(2**64) -> range error',
  lambda: raises(lambda: MOD.encode_vlq(2**64), 'range'),
)

# ---------------------------------------------------------------- sequence

case('seq empty -> []', lambda: eq(MOD.decode_uvarint_seq(b''), []))
case('seq single', lambda: eq(MOD.decode_uvarint_seq(b'\x2c'), [44]))
case(
  'seq zeros', lambda: eq(MOD.decode_uvarint_seq(b'\x00\x00\x00'), [0, 0, 0])
)
case(
  'seq mixed canonical stream',
  lambda: eq(
    MOD.decode_uvarint_seq(
      bytes.fromhex('00017f8001ac02808001' + 'ff' * 9 + '01')
    ),
    [0, 1, 127, 128, 300, 16384, 2**64 - 1],
  ),
)
case(
  'seq trailing dangling 80 -> truncated',
  lambda: raises(lambda: MOD.decode_uvarint_seq(b'\x00\x80'), 'truncated'),
)
case(
  'seq embedded overlong -> overlong',
  lambda: raises(
    lambda: MOD.decode_uvarint_seq(b'\x00\x80\x00\x01'), 'overlong'
  ),
)
case(
  'seq embedded too long -> too long',
  lambda: raises(
    lambda: MOD.decode_uvarint_seq(b'\x01' + b'\x80' * 10 + b'\x01'),
    'too long',
  ),
)

# ---------------------------------------------------------------- prefixvarint

PV_VECTORS = [
  (0, '00'),
  (1, '01'),
  (127, '7f'),
  (128, '8080'),
  (300, '812c'),
  (16383, 'bfff'),
  (16384, 'c04000'),
  (2**21 - 1, 'dfffff'),
  (2**32 - 1, 'f0ffffffff'),
  (2**56 - 1, 'fe' + 'ff' * 7),
  (2**56, 'ff01' + '00' * 7),
  (2**63, 'ff80' + '00' * 7),
  (2**64 - 1, 'ff' * 9),
]

for n, h in PV_VECTORS:
  case(
    f'pv_encode({n}) == {h}',
    lambda n=n, h=h: eq(MOD.pv_encode(n), bytes.fromhex(h)),
  )
  case(
    f'pv_decode({h}) == ({n}, {len(h) // 2})',
    lambda n=n, h=h: eq(MOD.pv_decode(bytes.fromhex(h)), (n, len(h) // 2)),
  )


def pv_boundaries():
  for k in range(1, 9):
    lo, hi = 2 ** (7 * k) - 1, 2 ** (7 * k)
    eq(len(MOD.pv_encode(lo)), k)
    eq(MOD.pv_decode(MOD.pv_encode(lo)), (lo, k))
    eq(len(MOD.pv_encode(hi)), k + 1)
    eq(MOD.pv_decode(MOD.pv_encode(hi)), (hi, k + 1))
  eq(MOD.pv_decode(MOD.pv_encode(2**64 - 1)), (2**64 - 1, 9))


case('pv width boundaries roundtrip (L=1..9)', pv_boundaries)
case(
  'pv_decode honors offset',
  lambda: eq(MOD.pv_decode(b'\xff\x81\x2c', 1), (300, 2)),
)
case(
  'pv empty buffer -> truncated',
  lambda: raises(lambda: MOD.pv_decode(b''), 'truncated'),
)
case(
  'pv lone c0 (needs 3 bytes) -> truncated',
  lambda: raises(lambda: MOD.pv_decode(b'\xc0'), 'truncated'),
)
case(
  'pv c0 40 (needs 3 bytes, has 2) -> truncated',
  lambda: raises(lambda: MOD.pv_decode(b'\xc0\x40'), 'truncated'),
)
case(
  'pv ff + 7 bytes (needs 9, has 8) -> truncated',
  lambda: raises(lambda: MOD.pv_decode(b'\xff' + b'\x00' * 7), 'truncated'),
)
case(
  'pv 80 00 (2-byte zero) -> overlong',
  lambda: raises(lambda: MOD.pv_decode(b'\x80\x00'), 'overlong'),
)
case(
  'pv 80 7f (2-byte 127) -> overlong',
  lambda: raises(lambda: MOD.pv_decode(b'\x80\x7f'), 'overlong'),
)
case(
  'pv ff + 2**56-1 (9-byte value that fits 8) -> overlong',
  lambda: raises(
    lambda: MOD.pv_decode(b'\xff' + (2**56 - 1).to_bytes(8, 'big')), 'overlong'
  ),
)
case(
  'pv_encode(-1) -> range error',
  lambda: raises(lambda: MOD.pv_encode(-1), 'range'),
)
case(
  'pv_encode(2**64) -> range error',
  lambda: raises(lambda: MOD.pv_encode(2**64), 'range'),
)

# ---------------------------------------------------------------- summary

total = PASSED + FAILED
print(f'{PASSED}/{total} passed')
sys.exit(1 if FAILED else 0)
