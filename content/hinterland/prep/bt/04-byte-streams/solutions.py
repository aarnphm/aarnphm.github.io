"""Reference solutions: representing numbers as byte streams."""

import struct

U32_MAX = (1 << 32) - 1
U64_MAX = (1 << 64) - 1
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1


def _check_u32_bounds(buf, offset):
  if offset < 0 or offset + 4 > len(buf):
    raise ValueError(f'need 4 bytes at offset {offset}, have {len(buf)}')


def read_u32_le(buf, offset):
  _check_u32_bounds(buf, offset)
  v = 0
  for i in range(4):
    v |= buf[offset + i] << (8 * i)
  return v


def read_u32_be(buf, offset):
  _check_u32_bounds(buf, offset)
  v = 0
  for i in range(4):
    v = (v << 8) | buf[offset + i]
  return v


def write_i64_le(n):
  if not I64_MIN <= n <= I64_MAX:
    raise ValueError(f'{n} outside signed 64-bit range')
  # masking IS the two's-complement encode: Python & operates on the
  # infinite sign-extended form, so n & U64_MAX == n mod 2**64
  u = n & U64_MAX
  return bytes((u >> (8 * i)) & 0xFF for i in range(8))


def float32_parts(bits):
  if not 0 <= bits <= U32_MAX:
    raise ValueError(f'{bits} outside u32 range')
  sign = bits >> 31
  exponent = (bits >> 23) & 0xFF
  mantissa = bits & ((1 << 23) - 1)
  if exponent == 0:
    category = 'zero' if mantissa == 0 else 'subnormal'
  elif exponent == 0xFF:
    category = 'inf' if mantissa == 0 else 'nan'
  else:
    category = 'normal'
  return (sign, exponent, mantissa, category)


def float_to_bits(f):
  return struct.unpack('<Q', struct.pack('<d', f))[0]


def bits_to_float(b):
  if not 0 <= b <= U64_MAX:
    raise ValueError(f'{b} outside u64 range')
  return struct.unpack('<d', struct.pack('<Q', b))[0]


class BinaryReader:
  def __init__(self, data):
    self._data = data
    self._pos = 0

  def tell(self):
    return self._pos

  def seek(self, pos):
    if not 0 <= pos <= len(self._data):
      raise ValueError(f'seek to {pos} outside [0, {len(self._data)}]')
    self._pos = pos

  def _take(self, n):
    if self._pos + n > len(self._data):
      raise ValueError(
        f'need {n} bytes at {self._pos}, have {len(self._data) - self._pos}'
      )
    chunk = bytes(self._data[self._pos : self._pos + n])
    self._pos += n
    return chunk

  def read_u8(self):
    return self._take(1)[0]

  def read_u16_le(self):
    b = self._take(2)
    return b[0] | b[1] << 8

  def read_u32_le(self):
    b = self._take(4)
    return b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24

  def read_uvarint(self):
    value = 0
    pos = self._pos
    for i in range(10):
      if pos + i >= len(self._data):
        raise ValueError('truncated varint')
      byte = self._data[pos + i]
      value |= (byte & 0x7F) << (7 * i)
      if not byte & 0x80:
        if value > U64_MAX:
          raise ValueError('varint exceeds 64 bits')
        self._pos = pos + i + 1
        return value
    raise ValueError('varint longer than 10 bytes')

  def read_bytes(self, n):
    if n < 0:
      raise ValueError(f'negative length {n}')
    return self._take(n)

  def read_cstring(self):
    idx = self._data.find(b'\x00', self._pos)
    if idx < 0:
      raise ValueError('unterminated cstring')
    s = bytes(self._data[self._pos : idx])
    self._pos = idx + 1
    return s


def bswap32(x):
  if not 0 <= x <= U32_MAX:
    raise ValueError(f'{x} outside u32 range')
  return (
    (x & 0xFF) << 24 | (x & 0xFF00) << 8 | (x >> 8) & 0xFF00 | (x >> 24) & 0xFF
  )


def hexdump(buf):
  rows = []
  for off in range(0, len(buf), 16):
    chunk = buf[off : off + 16]
    cells = [f'{b:02x}' for b in chunk] + ['  '] * (16 - len(chunk))
    left = ' '.join(cells[:8])
    right = ' '.join(cells[8:])
    ascii_panel = ''.join(chr(b) if 0x20 <= b <= 0x7E else '.' for b in chunk)
    rows.append(f'{off:08x}  {left}  {right}  |{ascii_panel}|')
  return '\n'.join(rows)
