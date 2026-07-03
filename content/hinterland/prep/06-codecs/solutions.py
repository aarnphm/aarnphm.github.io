"""Reference solutions for the classic-codecs module. Stdlib only."""

_MASK64 = (1 << 64) - 1
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def utf8_encode(cp: int) -> bytes:
  if cp < 0 or cp > 0x10FFFF:
    raise ValueError(f'codepoint out of range: {cp}')
  if 0xD800 <= cp <= 0xDFFF:
    raise ValueError(f'surrogate U+{cp:04X} is not encodable')
  if cp < 0x80:
    return bytes([cp])
  if cp < 0x800:
    return bytes([0xC0 | cp >> 6, 0x80 | cp & 0x3F])
  if cp < 0x10000:
    return bytes([0xE0 | cp >> 12, 0x80 | cp >> 6 & 0x3F, 0x80 | cp & 0x3F])
  return bytes([
    0xF0 | cp >> 18,
    0x80 | cp >> 12 & 0x3F,
    0x80 | cp >> 6 & 0x3F,
    0x80 | cp & 0x3F,
  ])


_UTF8_MIN = (0, 0, 0x80, 0x800, 0x10000)


def utf8_decode(buf: bytes) -> list[int]:
  out: list[int] = []
  i, n = 0, len(buf)
  while i < n:
    b = buf[i]
    if b < 0x80:
      out.append(b)
      i += 1
      continue
    if b < 0xC0 or b > 0xF7:
      raise ValueError(f'invalid start byte 0x{b:02X} at offset {i}')
    if b < 0xE0:
      need, cp = 1, b & 0x1F
    elif b < 0xF0:
      need, cp = 2, b & 0x0F
    else:
      need, cp = 3, b & 0x07
    if i + need >= n:
      raise ValueError(f'truncated sequence at offset {i}')
    for j in range(1, need + 1):
      c = buf[i + j]
      if c & 0xC0 != 0x80:
        raise ValueError(
          f'invalid continuation byte 0x{c:02X} at offset {i + j}'
        )
      cp = cp << 6 | c & 0x3F
    length = need + 1
    if cp < _UTF8_MIN[length]:
      raise ValueError(f'overlong {length}-byte encoding at offset {i}')
    if 0xD800 <= cp <= 0xDFFF:
      raise ValueError(f'surrogate U+{cp:04X} at offset {i}')
    if cp > 0x10FFFF:
      raise ValueError(f'codepoint out of range at offset {i}')
    out.append(cp)
    i += length
  return out


_B64_ALPHABET = (
  'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
)
_B64_REV = {c: v for v, c in enumerate(_B64_ALPHABET)}


def b64_encode(data: bytes) -> str:
  out: list[str] = []
  for i in range(0, len(data), 3):
    chunk = data[i : i + 3]
    word = int.from_bytes(chunk, 'big') << 8 * (3 - len(chunk))
    keep = len(chunk) + 1
    out.append(
      ''.join(_B64_ALPHABET[word >> s & 0x3F] for s in (18, 12, 6, 0)[:keep])
    )
    out.append('=' * (4 - keep))
  return ''.join(out)


def b64_decode(s: str) -> bytes:
  if len(s) % 4:
    raise ValueError(f'length {len(s)} is not a multiple of 4')
  pad = 2 if s.endswith('==') else 1 if s.endswith('=') else 0
  body = s[: len(s) - pad] if pad else s
  if '=' in body:
    raise ValueError("misplaced padding character '='")
  acc = nbits = 0
  out = bytearray()
  for ch in body:
    v = _B64_REV.get(ch)
    if v is None:
      raise ValueError(f'invalid character {ch!r}')
    acc = acc << 6 | v
    nbits += 6
    if nbits >= 8:
      nbits -= 8
      out.append(acc >> nbits & 0xFF)
  return bytes(out)


def rle_encode(data: bytes) -> bytes:
  out = bytearray()
  i, n = 0, len(data)
  while i < n:
    b = data[i]
    j = i
    while j < n and data[j] == b and j - i < 255:
      j += 1
    out.append(j - i)
    out.append(b)
    i = j
  return bytes(out)


def rle_decode(buf: bytes) -> bytes:
  if len(buf) % 2:
    raise ValueError(f'odd-length input: {len(buf)} bytes')
  out = bytearray()
  for i in range(0, len(buf), 2):
    count = buf[i]
    if count == 0:
      raise ValueError(f'zero count at offset {i}')
    out += bytes([buf[i + 1]]) * count
  return bytes(out)


def bitpack(values: list[int], k: int) -> bytes:
  if not 1 <= k <= 64:
    raise ValueError(f'k out of range: {k}')
  limit = 1 << k
  acc = nbits = 0
  out = bytearray()
  for v in values:
    if v < 0 or v >= limit:
      raise ValueError(f'value {v} out of range for {k} bits')
    acc |= v << nbits
    nbits += k
    while nbits >= 8:
      out.append(acc & 0xFF)
      acc >>= 8
      nbits -= 8
  if nbits:
    out.append(acc & 0xFF)
  return bytes(out)


def bitunpack(buf: bytes, k: int, n: int) -> list[int]:
  if not 1 <= k <= 64:
    raise ValueError(f'k out of range: {k}')
  expected = (n * k + 7) // 8 if n >= 0 else -1
  if n < 0 or len(buf) != expected:
    raise ValueError(
      f'buffer length {len(buf)} != expected {expected} for n={n}, k={k}'
    )
  mask = (1 << k) - 1
  out: list[int] = []
  acc = nbits = 0
  i = 0
  for _ in range(n):
    while nbits < k:
      acc |= buf[i] << nbits
      i += 1
      nbits += 8
    out.append(acc & mask)
    acc >>= k
    nbits -= k
  return out


def _zigzag64(d: int) -> int:
  # wrap the exact delta into int64 two's complement first (Parquet semantics)
  d &= _MASK64
  if d > _INT64_MAX:
    d -= 1 << 64
  return ((d << 1) ^ (d >> 63)) & _MASK64


def _unzigzag64(z: int) -> int:
  return (z >> 1) ^ -(z & 1)


def _uvarint_append(u: int, out: bytearray) -> None:
  while u >= 0x80:
    out.append(u & 0x7F | 0x80)
    u >>= 7
  out.append(u)


def _uvarint_read(buf: bytes, i: int) -> tuple[int, int]:
  u = shift = 0
  for j in range(10):
    if i + j >= len(buf):
      raise ValueError(f'truncated varint at offset {i}')
    b = buf[i + j]
    u |= (b & 0x7F) << shift
    if not b & 0x80:
      if u > _MASK64:
        raise ValueError(f'varint overflows 64 bits at offset {i}')
      return u, i + j + 1
    shift += 7
  raise ValueError(f'varint too long at offset {i}')


def delta_varint_encode(ints: list[int]) -> bytes:
  out = bytearray()
  prev = 0
  for x in ints:
    if x < _INT64_MIN or x > _INT64_MAX:
      raise ValueError(f'value {x} does not fit in int64')
    _uvarint_append(_zigzag64(x - prev), out)
    prev = x
  return bytes(out)


def delta_varint_decode(buf: bytes) -> list[int]:
  out: list[int] = []
  prev = 0
  i = 0
  while i < len(buf):
    z, i = _uvarint_read(buf, i)
    # running sum wraps mod 2**64 back into int64 range, undoing encode-side wrap
    prev = ((prev + _unzigzag64(z) + (1 << 63)) & _MASK64) - (1 << 63)
    out.append(prev)
  return out


def fletcher16(buf: bytes) -> int:
  s1 = s2 = 0
  for b in buf:
    s1 = (s1 + b) % 255
    s2 = (s2 + s1) % 255
  return s2 << 8 | s1
