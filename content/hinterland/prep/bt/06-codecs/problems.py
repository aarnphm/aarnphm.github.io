"""Practice stubs: classic codecs (UTF-8, base64, RLE, bitpacking, delta+zigzag+uvarint, Fletcher-16).

Each docstring IS the format spec; the tests enforce it byte-for-byte,
including the quoted error-message tokens. Run:

    python3 test_problems.py                       # against this module
    PRACTICE_MODULE=solutions python3 test_problems.py   # against the reference

Stdlib only; b64_* may not import base64/binascii.
"""


def utf8_encode(cp: int) -> bytes:
  """Encode one Unicode codepoint as UTF-8.

  [screen-core | difficulty: core]

  Templates (x = payload bits, most significant bits in the lead byte):
      U+0000..U+007F     -> 0xxxxxxx
      U+0080..U+07FF     -> 110xxxxx 10xxxxxx
      U+0800..U+FFFF     -> 1110xxxx 10xxxxxx 10xxxxxx
      U+10000..U+10FFFF  -> 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
  Always the shortest form that fits.

  Args:
      cp: codepoint as an int.
  Returns:
      1-4 bytes.
  Errors:
      ValueError containing 'out of range' if cp < 0 or cp > 0x10FFFF.
      ValueError containing 'surrogate' if 0xD800 <= cp <= 0xDFFF.
  """
  raise NotImplementedError


def utf8_decode(buf: bytes) -> list[int]:
  """Strictly decode UTF-8 (RFC 3629) into a list of codepoints.

  [screen-core | difficulty: hard]

  Must match bytes.decode('utf-8') accept/reject behavior exactly.
  Reject with ValueError whose message contains the token:
      'invalid start byte'   lead byte 10xxxxxx (lone continuation) or 0xF8..0xFF
      'truncated'            sequence runs past the end of buf
      'invalid continuation' a following byte is not 10xxxxxx
      'overlong'             assembled cp fits a shorter form (e.g. C0 80, E0 80 80)
      'surrogate'            cp in U+D800..U+DFFF (e.g. ED A0 80)
      'out of range'         cp > 0x10FFFF (e.g. F4 90 80 80, F5 80 80 80)
  Check order per sequence: start-byte class, truncation, continuation
  bytes, then value checks (overlong, surrogate, range) on the assembled cp.

  Args:
      buf: byte string.
  Returns:
      list of int codepoints. b'' -> [].
  """
  raise NotImplementedError


def b64_encode(data: bytes) -> str:
  """Standard base64 (RFC 4648 section 4), no library.

  [screen-core | difficulty: core]

  Alphabet A-Za-z0-9+/ with '=' padding. 3 input bytes -> 4 chars;
  tail of 1 byte -> 2 chars + '=='; tail of 2 bytes -> 3 chars + '='.
  Dangling low bits of the final symbol are zero. b'' -> ''.

  Args:
      data: arbitrary bytes.
  Returns:
      padded base64 str, length 4*ceil(len(data)/3).
  """
  raise NotImplementedError


def b64_decode(s: str) -> bytes:
  """Strict decode of standard base64, no library.

  [screen-core | difficulty: core]

  Validation, in order; ValueError containing the token:
      'length'             len(s) % 4 != 0
      'padding'            '=' anywhere except as the final one or two
                           chars ('===' is never legal)
      'invalid character'  any non-alphabet char, including whitespace
  Non-canonical dangling bits are accepted (e.g. 'TR==' -> b'M', matching
  base64.b64decode); know why strict systems reject them.

  Args:
      s: base64 text.
  Returns:
      decoded bytes. '' -> b''.
  """
  raise NotImplementedError


def rle_encode(data: bytes) -> bytes:
  """Byte-oriented run-length encode.

  [screen-core | difficulty: warmup]

  Output is a concatenation of 2-byte pairs [count][value], count in
  1..255. Maximal runs are split greedily: 300 x 0x41 -> FF 41 2D 41.
  Round-trips arbitrary bytes (worst case 2x expansion). b'' -> b''.

  Args:
      data: arbitrary bytes.
  Returns:
      encoded pairs.
  """
  raise NotImplementedError


def rle_decode(buf: bytes) -> bytes:
  """Inverse of rle_encode.

  [screen-core | difficulty: warmup]

  Errors (ValueError containing the token):
      'odd'         len(buf) is odd (dangling half-pair)
      'zero count'  any count byte is 0 (encoder never emits it)

  Args:
      buf: encoded pairs.
  Returns:
      original bytes.
  """
  raise NotImplementedError


def bitpack(values: list[int], k: int) -> bytes:
  """Pack unsigned k-bit values into ceil(n*k/8) bytes, LSB-first.

  [depth | difficulty: hard]

  Bit order (Parquet RLE/bit-packed hybrid convention): value i occupies
  stream bits [i*k, (i+1)*k); stream bit j is bit (j % 8) of byte
  (j // 8), bit 0 = least significant. Unused high bits of the final
  byte are 0.
  Example: bitpack([1, 2, 3, 4, 5], 3) == bytes([0xD1, 0x58]).

  Args:
      values: ints, each in [0, 2**k).
      k: bit width, 1..64.
  Errors:
      ValueError containing 'k out of range' if not 1 <= k <= 64.
      ValueError containing 'out of range for' if any value is negative
      or >= 2**k.
  Returns:
      packed bytes; [] -> b''.
  """
  raise NotImplementedError


def bitunpack(buf: bytes, k: int, n: int) -> list[int]:
  """Inverse of bitpack: extract n k-bit values.

  [depth | difficulty: hard]

  Args:
      buf: packed bytes.
      k: bit width, 1..64.
      n: number of values, >= 0.
  Errors:
      ValueError containing 'k out of range' if not 1 <= k <= 64.
      ValueError containing 'length' if n < 0 or
      len(buf) != ceil(n*k/8) (truncated or trailing bytes).
  Returns:
      list of n ints.
  """
  raise NotImplementedError


def delta_varint_encode(ints: list[int]) -> bytes:
  """Capstone: delta -> zigzag -> uvarint (the timestamp stack).

  [screen-core | difficulty: core]

  For each i: d_i = ints[i] - ints[i-1] with ints[-1] taken as 0,
  wrapped into int64 two's complement (mod 2**64, Parquet semantics);
  z_i = zigzag64(d_i) = ((d << 1) ^ (d >> 63)) as uint64; z_i emitted
  as LEB128 uvarint: 7 bits per byte, least-significant group first,
  MSB set = continue; at most 10 bytes.
  Example: [1000, 1005, 1004, 1010] -> D0 0F 0A 01 0C.

  Args:
      ints: list of ints, each in [-2**63, 2**63 - 1].
  Errors:
      ValueError containing 'int64' if any value is outside int64.
  Returns:
      encoded bytes; [] -> b''.
  """
  raise NotImplementedError


def delta_varint_decode(buf: bytes) -> list[int]:
  """Inverse of delta_varint_encode; reads uvarints until buf is exhausted.

  [screen-core | difficulty: hard]

  Running sum wraps back into int64 range mod 2**64, so every encodable
  sequence round-trips exactly.
  Errors (ValueError containing the token):
      'truncated'  buf ends mid-varint (continuation bit set at end)
      'too long'   a varint still has its continuation bit set after
                   10 bytes
      'overflow'   a 10-byte varint carries bits at or above 2**64
  Non-minimal encodings (e.g. 80 00 for 0) are accepted, protobuf-style.

  Args:
      buf: encoded bytes.
  Returns:
      list of ints. b'' -> [].
  """
  raise NotImplementedError


def fletcher16(buf: bytes) -> int:
  """Fletcher-16 checksum.

  [depth | difficulty: warmup]

  s1 = s2 = 0; for each byte b: s1 = (s1 + b) % 255; s2 = (s2 + s1) % 255.
  Return (s2 << 8) | s1.
  Vectors: b'abcde' -> 0xC8F0, b'abcdef' -> 0x2057, b'abcdefgh' -> 0x0627,
  b'' -> 0. Note the mod-255 blind spot: fletcher16(b'\\x00') ==
  fletcher16(b'\\xff') == 0.

  Args:
      buf: arbitrary bytes.
  Returns:
      int in [0, 0xFFFF].
  """
  raise NotImplementedError
