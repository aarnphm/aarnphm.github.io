"""Reference implementations for the varint problems. stdlib only."""

MASK64 = (1 << 64) - 1


def encode_uvarint(n: int) -> bytes:
  if n < 0 or n > MASK64:
    raise ValueError('out of range: need 0 <= n < 2**64')
  out = bytearray()
  while True:
    group = n & 0x7F
    n >>= 7
    if n:
      out.append(group | 0x80)
    else:
      out.append(group)
      return bytes(out)


def decode_uvarint(buf: bytes, offset: int = 0) -> tuple[int, int]:
  value = 0
  shift = 0
  pos = offset
  while True:
    if pos - offset == 10:
      raise ValueError('too long: more than 10 groups')
    if pos >= len(buf):
      raise ValueError('truncated: buffer ends mid-varint')
    b = buf[pos]
    pos += 1
    value |= (b & 0x7F) << shift
    shift += 7
    if not b & 0x80:
      break
  if value > MASK64:
    raise ValueError('too long: value overflows u64')
  if pos - offset > 1 and b == 0:
    raise ValueError('overlong: trailing zero group is non-canonical')
  return value, pos - offset


def zigzag_encode(n: int) -> int:
  if n < -(1 << 63) or n >= 1 << 63:
    raise ValueError('out of range: need -2**63 <= n < 2**63')
  # Python >> on negatives sign-extends over infinite bits; mask truncates.
  return ((n << 1) ^ (n >> 63)) & MASK64


def zigzag_decode(u: int) -> int:
  if u < 0 or u > MASK64:
    raise ValueError('out of range: need 0 <= u < 2**64')
  return (u >> 1) ^ -(u & 1)


def encode_svarint(n: int) -> bytes:
  return encode_uvarint(zigzag_encode(n))


def decode_svarint(buf: bytes, offset: int = 0) -> tuple[int, int]:
  u, consumed = decode_uvarint(buf, offset)
  return zigzag_decode(u), consumed


def encode_vlq(n: int) -> bytes:
  if n < 0 or n > MASK64:
    raise ValueError('out of range: need 0 <= n < 2**64')
  groups = [n & 0x7F]
  n >>= 7
  while n:
    groups.append(n & 0x7F)
    n >>= 7
  groups.reverse()
  out = bytearray(g | 0x80 for g in groups[:-1])
  out.append(groups[-1])
  return bytes(out)


def decode_vlq(buf: bytes, offset: int = 0) -> tuple[int, int]:
  value = 0
  pos = offset
  while True:
    if pos - offset == 10:
      raise ValueError('too long: more than 10 groups')
    if pos >= len(buf):
      raise ValueError('truncated: buffer ends mid-vlq')
    b = buf[pos]
    pos += 1
    if pos - offset == 1 and b == 0x80:
      raise ValueError('overlong: leading zero group is non-canonical')
    value = (value << 7) | (b & 0x7F)
    if not b & 0x80:
      break
  if value > MASK64:
    raise ValueError('too long: value overflows u64')
  return value, pos - offset


def decode_uvarint_seq(buf: bytes) -> list[int]:
  values = []
  offset = 0
  while offset < len(buf):
    value, consumed = decode_uvarint(buf, offset)
    values.append(value)
    offset += consumed
  return values


def pv_encode(n: int) -> bytes:
  if n < 0 or n > MASK64:
    raise ValueError('out of range: need 0 <= n < 2**64')
  bits = n.bit_length() or 1
  if bits > 56:
    return b'\xff' + n.to_bytes(8, 'big')
  length = -(-bits // 7)
  # First byte's top `length` bits: (length-1) ones then a zero.
  prefix = ((1 << (length - 1)) - 1) << 1
  return ((prefix << (7 * length)) | n).to_bytes(length, 'big')


def pv_decode(buf: bytes, offset: int = 0) -> tuple[int, int]:
  if offset >= len(buf):
    raise ValueError('truncated: empty buffer')
  b0 = buf[offset]
  ones = 8 - (b0 ^ 0xFF).bit_length()
  length = 9 if ones == 8 else ones + 1
  if len(buf) - offset < length:
    raise ValueError('truncated: buffer ends mid-prefixvarint')
  if length == 9:
    value = int.from_bytes(buf[offset + 1 : offset + 9], 'big')
    if value < 1 << 56:
      raise ValueError('overlong: fits in a shorter encoding')
    return value, 9
  value = b0 & ((1 << (8 - length)) - 1)
  for i in range(offset + 1, offset + length):
    value = (value << 8) | buf[i]
  if length > 1 and value < 1 << (7 * (length - 1)):
    raise ValueError('overlong: fits in a shorter encoding')
  return value, length
