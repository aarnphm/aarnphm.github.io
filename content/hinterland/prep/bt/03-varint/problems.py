"""Varint practice problems.

Implement every function below, then run:

    python3 test_problems.py                      # tests your implementations
    PRACTICE_MODULE=solutions python3 test_problems.py   # tests the reference

All byte examples in docstrings are hex. "screen-core" marks problems likely
to be asked live in a 60-minute screen; "depth" marks problems whose value is
the follow-up discussion they prepare you for.
"""


def encode_uvarint(n: int) -> bytes:
  """Encode an unsigned integer as a protobuf-style LEB128 varint.

  Format: split n into 7-bit groups starting from the least-significant
  bits. Emit one byte per group, least-significant group first. Set the
  high bit (0x80) on every byte except the last. The encoding must be
  minimal (never emit a trailing zero group).

  Vectors: 0 -> 00; 1 -> 01; 127 -> 7f; 128 -> 80 01; 300 -> ac 02;
  16383 -> ff 7f; 16384 -> 80 80 01; 2**32-1 -> ff ff ff ff 0f;
  2**64-1 -> ff ff ff ff ff ff ff ff ff 01 (10 bytes).

  Args:
      n: integer in [0, 2**64 - 1].

  Returns:
      1 to 10 bytes.

  Raises:
      ValueError: if n is negative or >= 2**64 (message mentions "range").

  Difficulty: warmup. screen-core -- this function opens most varint
  interviews; get the n == 0 single-byte case right.
  """
  raise NotImplementedError


def decode_uvarint(buf: bytes, offset: int = 0) -> tuple[int, int]:
  """Decode one LEB128 uvarint from buf starting at offset. Strict decoder.

  Read bytes until one has the high bit clear. Byte i (0-based, counting
  from offset) contributes its low 7 bits at shift 7*i (first byte is
  least significant).

  Reject all three malformed classes with ValueError; the message must
  start with the class name so callers can distinguish them:

    "truncated" -- buffer ends while the last byte read had the
        continuation bit set (an empty region counts). A streaming caller
        may treat this as "need more bytes".
    "too long"  -- unrecoverable regardless of any suffix: either ten
        bytes were read all with the continuation bit set (an 11th group
        can never be valid for u64), or exactly ten groups terminate but
        the tenth group exceeds 1 so the value would be >= 2**64. Ten
        continuation bytes at end-of-buffer are "too long", not
        "truncated".
    "overlong"  -- terminates and fits u64 but is not minimal: more than
        one byte consumed and the final group is 0. Example: 80 00 is a
        2-byte encoding of 0; canonical is 00.

  Args:
      buf: bytes-like object.
      offset: start index, 0 <= offset <= len(buf).

  Returns:
      (value, bytes_consumed). bytes_consumed counts from offset; it is
      NOT the new offset.

  Difficulty: core. screen-core -- the single most likely live question;
  the error taxonomy is where candidates get separated.
  """
  raise NotImplementedError


def zigzag_encode(n: int) -> int:
  """Map a signed 64-bit integer to an unsigned one, small magnitudes first.

  zigzag(n) = (n << 1) ^ (n >> 63) evaluated in 64-bit two's-complement
  semantics (the >> is an arithmetic shift that broadcasts the sign bit).
  Mapping: 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...,
  2**63-1 -> 2**64-2, -2**63 -> 2**64-1.

  Python note: ints are unbounded and >> on a negative value sign-extends
  over infinitely many bits, so the result must be masked to 64 bits
  explicitly.

  Args:
      n: integer in [-2**63, 2**63 - 1].

  Returns:
      integer in [0, 2**64 - 1].

  Raises:
      ValueError: n outside the signed 64-bit domain (message mentions
      "range").

  Difficulty: core. screen-core -- the standard follow-up to uvarint
  ("now handle negatives without paying 10 bytes").
  """
  raise NotImplementedError


def zigzag_decode(u: int) -> int:
  """Inverse of zigzag_encode: (u >> 1) ^ -(u & 1).

  -(u & 1) is 0 for even u and all-ones for odd u, i.e. a conditional
  complement. In Python no masking is needed: for u in [0, 2**64) the
  result lands exactly in [-2**63, 2**63).

  Args:
      u: integer in [0, 2**64 - 1].

  Returns:
      integer in [-2**63, 2**63 - 1].

  Raises:
      ValueError: u outside the unsigned 64-bit domain (message mentions
      "range").

  Difficulty: core. screen-core.
  """
  raise NotImplementedError


def encode_svarint(n: int) -> bytes:
  """Encode a signed 64-bit integer as zigzag-then-uvarint (protobuf sint64).

  Vectors: 0 -> 00; -1 -> 01; 1 -> 02; -2 -> 03; -64 -> 7f; 64 -> 80 01;
  -2**63 -> ff ff ff ff ff ff ff ff ff 01.

  Args:
      n: integer in [-2**63, 2**63 - 1].

  Returns:
      1 to 10 bytes.

  Raises:
      ValueError: n outside the signed 64-bit domain.

  Difficulty: warmup (composition of two solved parts). screen-core --
  this is the sint64 wire format verbatim.
  """
  raise NotImplementedError


def decode_svarint(buf: bytes, offset: int = 0) -> tuple[int, int]:
  """Decode one zigzag-varint (sint64) from buf starting at offset.

  Composition: decode_uvarint then zigzag_decode. Propagates
  decode_uvarint's three error classes unchanged (truncated / too long /
  overlong).

  Args:
      buf: bytes-like object.
      offset: start index.

  Returns:
      (signed_value, bytes_consumed).

  Difficulty: warmup. screen-core.
  """
  raise NotImplementedError


def encode_vlq(n: int) -> bytes:
  """Encode an unsigned integer as a MIDI-style VLQ.

  Same 7-bit groups and same continuation convention as LEB128 (0x80 set
  on every byte except the last), but the MOST-significant group comes
  FIRST. Minimal encoding only (no leading zero group).

  Vectors: 0 -> 00; 127 -> 7f; 128 -> 81 00; 300 -> 82 2c;
  16384 -> 81 80 00; 0x0FFFFFFF -> ff ff ff 7f;
  2**64-1 -> 81 ff ff ff ff ff ff ff ff 7f (10 bytes).
  Contrast with LEB128: 300 -> ac 02, 128 -> 80 01.

  Args:
      n: integer in [0, 2**64 - 1].

  Returns:
      1 to 10 bytes.

  Raises:
      ValueError: if n is negative or >= 2**64.

  Difficulty: core. depth -- interviewers use it to test whether you
  understood group order rather than memorized LEB, and the encode side
  forces you to notice you need the group count up front.
  """
  raise NotImplementedError


def decode_vlq(buf: bytes, offset: int = 0) -> tuple[int, int]:
  """Decode one MSB-group-first VLQ from buf starting at offset. Strict.

  Accumulate value = (value << 7) | (b & 0x7F) per byte; stop at the
  first byte with the high bit clear. ValueError classes, message must
  start with the class name:

    "truncated" -- buffer ends while the continuation bit was set.
    "too long"  -- an 11th group would be needed, or exactly ten groups
        terminate but the value exceeds 2**64 - 1 (leading group > 1).
    "overlong"  -- the FIRST byte is exactly 0x80 (a leading zero group).
        Rejected immediately, even if the buffer ends right there: no
        canonical VLQ begins with 0x80. Mirror image of LEB128, where the
        redundant group is the TRAILING 00 byte.

  Args:
      buf: bytes-like object.
      offset: start index.

  Returns:
      (value, bytes_consumed).

  Difficulty: core. depth.
  """
  raise NotImplementedError


def decode_uvarint_seq(buf: bytes) -> list[int]:
  """Decode the entire buffer as back-to-back LEB128 uvarints.

  The buffer must be exactly a concatenation of zero or more valid
  canonical uvarints; every byte must belong to one. Decode left to
  right, propagating decode_uvarint's ValueError classes unchanged.
  Trailing garbage necessarily fails one of them (e.g. a dangling 80 is
  "truncated"). Empty buffer -> [].

  Vector: 00 01 7f 80 01 ac 02 80 80 01 ff*9 01
          -> [0, 1, 127, 128, 300, 16384, 2**64 - 1].

  Args:
      buf: bytes-like object.

  Returns:
      list of decoded values in order.

  Difficulty: core. screen-core -- the standard second act ("now parse a
  stream of them"); graded on offset discipline.
  """
  raise NotImplementedError


def pv_encode(n: int) -> bytes:
  """Encode u64 as a PrefixVarint with a UTF-8-style leading-ones tag.

  All tag bits live in the first byte. Total length L is 1..9:

    L in 1..8: the first byte's top L bits are (L-1) one-bits followed by
        one zero-bit; its remaining 8-L bits are the HIGH bits of the
        payload. The following L-1 bytes carry the remaining payload
        bits, big-endian, 8 bits per byte. Capacity: 7*L payload bits.
          L=1: 0xxxxxxx                      n < 2**7
          L=2: 10xxxxxx xxxxxxxx             n < 2**14
          L=3: 110xxxxx xxxxxxxx xxxxxxxx    n < 2**21
          ...
          L=8: 11111110 then 7 payload bytes n < 2**56
    L = 9: first byte 0xFF, then the value as exactly 8 raw big-endian
        bytes (full 64 bits), used for n >= 2**56.

  Always emit the minimal L for n.

  Vectors: 0 -> 00; 127 -> 7f; 128 -> 80 80; 300 -> 81 2c; 16383 -> bf ff;
  16384 -> c0 40 00; 2**32-1 -> f0 ff ff ff ff;
  2**56-1 -> fe ff ff ff ff ff ff ff; 2**56 -> ff 01 00 00 00 00 00 00 00;
  2**64-1 -> ff ff ff ff ff ff ff ff ff (9 bytes; max is 9, not 10).

  Args:
      n: integer in [0, 2**64 - 1].

  Returns:
      1 to 9 bytes.

  Raises:
      ValueError: if n is negative or >= 2**64.

  Difficulty: hard. depth -- the design that turns LEB's branch-per-byte
  into one length dispatch; expect to discuss clz and overread.
  """
  raise NotImplementedError


def pv_decode(buf: bytes, offset: int = 0) -> tuple[int, int]:
  """Decode one PrefixVarint (format of pv_encode) from buf at offset.

  Length comes from the first byte alone: L = (count of leading one
  bits) + 1, except 8 leading ones (0xFF) means L = 9. Then read the
  payload: the low 8-L bits of byte one are the payload's high bits,
  followed by L-1 raw big-endian bytes (for L = 9: byte one carries no
  payload, the next 8 bytes are the value).

  ValueError classes, message must start with the class name:

    "truncated" -- fewer than L bytes available from offset (empty
        region included).
    "overlong"  -- the decoded value would fit in a smaller L: for
        L in 2..8, value < 2**(7*(L-1)); for L = 9, value < 2**56.

  There is no "too long" class: the tag caps L at 9 by construction.

  Args:
      buf: bytes-like object.
      offset: start index.

  Returns:
      (value, bytes_consumed).

  Difficulty: hard. depth.
  """
  raise NotImplementedError
