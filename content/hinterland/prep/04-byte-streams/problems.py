"""Practice stubs: representing numbers as byte streams.

Implement each function/class per its docstring. Run test_problems.py to check:
    python3 test_problems.py                       # against this file
    PRACTICE_MODULE=solutions python3 test_problems.py
"""


def read_u32_le(buf, offset):
  """Decode an unsigned 32-bit little-endian integer using shifts only.

  [warmup | screen-core]

  Format: 4 bytes at buf[offset:offset+4], least significant byte first.
  Example: buf = b'\\x44\\x33\\x22\\x11', offset = 0 -> 0x11223344.

  Args:
      buf: bytes or bytearray.
      offset: int, start of the 4-byte field.
  Returns:
      int in [0, 2**32 - 1].
  Raises:
      ValueError: if offset < 0 or offset + 4 > len(buf).

  Constraint: build the value with indexing, shifts, and bitwise ors only —
  no struct, no int.from_bytes.
  """
  raise NotImplementedError


def read_u32_be(buf, offset):
  """Decode an unsigned 32-bit big-endian integer using shifts only.

  [warmup | screen-core]

  Format: 4 bytes at buf[offset:offset+4], most significant byte first.
  Example: buf = b'\\x11\\x22\\x33\\x44', offset = 0 -> 0x11223344.

  Args / Returns / Raises: identical to read_u32_le.
  Same constraint: shifts only, no struct, no int.from_bytes.
  """
  raise NotImplementedError


def write_i64_le(n):
  """Encode a signed 64-bit integer as 8 little-endian bytes, manually.

  [core | screen-core]

  Format: two's complement, width 64, least significant byte first.
  Examples: 0 -> b'\\x00' * 8; -1 -> b'\\xff' * 8;
            -2 -> b'\\xfe' + b'\\xff' * 7.

  Args:
      n: int with -2**63 <= n <= 2**63 - 1.
  Returns:
      bytes of length 8.
  Raises:
      ValueError: if n is outside the signed 64-bit range.

  Constraint: perform the two's-complement conversion and byte extraction
  yourself (mask to unsigned, then shift out bytes) — no struct, no
  int.to_bytes.
  """
  raise NotImplementedError


def float32_parts(bits):
  """Split raw IEEE 754 binary32 bits into fields and classify the value.

  [core | depth]

  Layout (bit 31 down to bit 0): 1 sign | 8 exponent (bias 127) | 23 mantissa.

  Args:
      bits: int in [0, 2**32 - 1], the raw bit pattern.
  Returns:
      (sign, exponent, mantissa, category) where sign is 0 or 1, exponent is
      the raw biased field in [0, 255], mantissa is the raw 23-bit field, and
      category is one of "zero", "subnormal", "normal", "inf", "nan":
          exponent == 0,   mantissa == 0 -> "zero"
          exponent == 0,   mantissa != 0 -> "subnormal"
          exponent == 255, mantissa == 0 -> "inf"
          exponent == 255, mantissa != 0 -> "nan"
          otherwise                      -> "normal"
  Raises:
      ValueError: if bits is outside [0, 2**32 - 1].

  Example: 0x3F800000 (1.0) -> (0, 127, 0, "normal").
  """
  raise NotImplementedError


def float_to_bits(f):
  """Bit-cast a Python float (IEEE 754 binary64) to its raw u64 pattern.

  [warmup | depth]

  Args:
      f: float (any value, including inf, nan, -0.0).
  Returns:
      int in [0, 2**64 - 1]: the exact bit pattern.
      Examples: 1.0 -> 0x3FF0000000000000; -0.0 -> 0x8000000000000000.

  Use struct for the reinterpretation; this is the idiom under test.
  """
  raise NotImplementedError


def bits_to_float(b):
  """Bit-cast a raw u64 pattern to the Python float it represents.

  [warmup | depth]

  Inverse of float_to_bits: bits_to_float(float_to_bits(f)) reproduces f
  bit-for-bit (including -0.0 and NaN patterns).

  Args:
      b: int in [0, 2**64 - 1].
  Returns:
      float.
  Raises:
      ValueError: if b is outside [0, 2**64 - 1].
  """
  raise NotImplementedError


class BinaryReader:
  """Sequential decoder over an in-memory byte buffer — the composite that
  covers most live screens in this topic.

  [core | screen-core]

  Wraps a bytes/bytearray value and maintains a cursor. Every read consumes
  from the cursor and advances it; on any failure the reader raises
  ValueError and leaves the cursor where it was (advance only on success).

  Methods:
      __init__(data): data is bytes or bytearray; cursor starts at 0.
      tell() -> int: current cursor.
      seek(pos): set cursor to absolute pos; ValueError unless
          0 <= pos <= len(data).
      read_u8() -> int: one byte.
      read_u16_le() -> int: 2 bytes, little-endian unsigned.
      read_u32_le() -> int: 4 bytes, little-endian unsigned.
      read_uvarint() -> int: LEB128 — 7 value bits per byte, low group
          first, high bit set means "more bytes follow".
          Example: 300 = b'\\xac\\x02'. Errors (all ValueError, cursor
          unmoved): truncated (continuation bit set at EOF); longer than 10
          bytes; decoded value > 2**64 - 1. Overlong-but-terminating
          encodings within 10 bytes (e.g. b'\\x80\\x00' for 0) are accepted.
      read_bytes(n) -> bytes: exactly n bytes; ValueError if n < 0 or
          fewer than n remain.
      read_cstring() -> bytes: bytes up to the next NUL, consuming the NUL,
          returned without it; ValueError (cursor unmoved) if no NUL occurs
          before end of buffer.

  All multi-byte fixed-width reads raise ValueError when fewer bytes remain
  than needed.
  """

  def __init__(self, data):
    raise NotImplementedError

  def tell(self):
    raise NotImplementedError

  def seek(self, pos):
    raise NotImplementedError

  def read_u8(self):
    raise NotImplementedError

  def read_u16_le(self):
    raise NotImplementedError

  def read_u32_le(self):
    raise NotImplementedError

  def read_uvarint(self):
    raise NotImplementedError

  def read_bytes(self, n):
    raise NotImplementedError

  def read_cstring(self):
    raise NotImplementedError


def bswap32(x):
  """Reverse the byte order of a 32-bit unsigned value.

  [warmup | screen-core]

  Example: 0x11223344 -> 0x44332211. bswap32 is its own inverse.

  Args:
      x: int in [0, 2**32 - 1].
  Returns:
      int in [0, 2**32 - 1].
  Raises:
      ValueError: if x is outside [0, 2**32 - 1].

  Constraint: shifts and masks only (this is the htonl/ntohl body).
  """
  raise NotImplementedError


def hexdump(buf):
  """Render bytes as canonical hexdump rows (offset | hex panel | ASCII).

  [hard | screen-core]

  Row format, exactly:
      {offset:08x}  {hex8}  {hex8}  |{ascii}|
  where:
    - offset is the row's starting index, 8 lowercase hex digits;
    - each hex8 is 8 columns of two lowercase hex digits joined by single
      spaces (23 chars); columns past the end of buf are two spaces, so the
      panel stays fixed width on the last row;
    - two spaces separate offset/panel/panel/ascii sections;
    - ascii holds one char per real byte only: the byte itself if
      0x20 <= b <= 0x7e, else '.'.
  Rows cover 16 bytes each and are joined with '\\n'; no trailing newline.

  Example (16 bytes 0x61..0x70):
  00000000  61 62 63 64 65 66 67 68  69 6a 6b 6c 6d 6e 6f 70  |abcdefghijklmnop|

  Args:
      buf: bytes or bytearray.
  Returns:
      str; empty string for empty input.
  """
  raise NotImplementedError
