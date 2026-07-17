"""Bit and byte manipulation drills: fixed-width idioms emulated in Python.

Conventions for the whole module:
- "u32"/"u64" means a Python int constrained to [0, 2**32) / [0, 2**64): the
  nonnegative bit pattern, never a Python negative.
- Bit k is the k-th least significant bit (bit 0 = LSB).
- Every domain violation raises ValueError. Nothing else is raised.
"""


def pack_rgba(r: int, g: int, b: int, a: int) -> int:
  """Pack four byte channels into one u32 laid out 0xRRGGBBAA.

  Layout (bit positions within the integer, not memory bytes):
      bits 31..24 = r, bits 23..16 = g, bits 15..8 = b, bits 7..0 = a
  so pack_rgba(0x12, 0x34, 0x56, 0x78) == 0x12345678.

  Args:
      r, g, b, a: ints, each in [0, 255].
  Returns:
      u32 with the channels packed as above.
  Errors:
      ValueError if any channel is outside [0, 255].
  Difficulty: warmup. Screen-core: packing header fields into a word is the
  opening move of most encoding screens.
  """
  raise NotImplementedError


def unpack_rgba(px: int) -> tuple[int, int, int, int]:
  """Inverse of pack_rgba: split a u32 0xRRGGBBAA into its four channels.

  unpack_rgba(0xDEADBEEF) == (0xDE, 0xAD, 0xBE, 0xEF), and
  pack_rgba(*unpack_rgba(px)) == px for every valid px.

  Args:
      px: u32 in [0, 2**32).
  Returns:
      (r, g, b, a) tuple of ints, each in [0, 255].
  Errors:
      ValueError if px is outside [0, 2**32).
  Difficulty: warmup. Screen-core.
  """
  raise NotImplementedError


def extract_field(word: int, offset: int, width: int) -> int:
  """Read the width-bit unsigned field of word starting at bit offset.

  The field occupies bits offset .. offset+width-1 and is returned
  right-aligned:
      extract_field(0xDEADBEEF, 8, 8) == 0xBE
      extract_field(w, 0, 64) == w    (full-width field is legal here; note
                                       that (1 << 64) - 1 would be UB in C)

  Args:
      word: u64 in [0, 2**64).
      offset: int >= 0.
      width: int >= 1, with offset + width <= 64.
  Returns:
      int in [0, 2**width).
  Errors:
      ValueError if word is outside u64 range, offset < 0, width < 1, or
      offset + width > 64.
  Difficulty: warmup. Screen-core: the header-decoding primitive.
  """
  raise NotImplementedError


def insert_field(word: int, offset: int, width: int, value: int) -> int:
  """Return word with its width-bit field at bit offset replaced by value.

  All bits outside offset .. offset+width-1 are preserved:
      insert_field(0xDEADBEEF, 8, 8, 0x42) == 0xDEAD42EF
      extract_field(insert_field(w, o, k, v), o, k) == v

  Args:
      word: u64 in [0, 2**64).
      offset: int >= 0.
      width: int >= 1, with offset + width <= 64.
      value: int in [0, 2**width); it must fit the field exactly.
  Returns:
      u64 with the field overwritten.
  Errors:
      ValueError if word/offset/width violate the extract_field bounds, or
      if value is outside [0, 2**width).
  Difficulty: core. Screen-core: clear-then-or, and the mask discipline
  around Python's negative ~.
  """
  raise NotImplementedError


def next_pow2(x: int) -> int:
  """Smallest power of two >= x, computed u64-style (bit smear, then +1).

  next_pow2(0) == 1, next_pow2(1) == 1, next_pow2(3) == 4,
  next_pow2(2**63) == 2**63.

  Args:
      x: int in [0, 2**63]. The bound exists because the next power of two
         above 2**63 does not fit a u64 (the C version wraps to 0).
  Returns:
      power of two p with p >= x and p >= 1.
  Errors:
      ValueError if x < 0 or x > 2**63.
  Difficulty: warmup. Screen-core (hash-table sizing, allocator rounding);
  the x == 0 edge and the overflow bound are what gets probed.
  """
  raise NotImplementedError


def reverse_bits32(x: int) -> int:
  """Reverse the 32 bits of a u32 (bit 0 swaps with bit 31, etc.).

  reverse_bits32(0x00000001) == 0x80000000
  reverse_bits32(0x12345678) == 0x1E6A2C48
  reverse_bits32(reverse_bits32(x)) == x

  Required approach: the divide-and-conquer mask ladder (swap adjacent bits,
  then pairs, nibbles, bytes, halfwords), not a 32-iteration bit loop.

  Args:
      x: u32 in [0, 2**32).
  Returns:
      u32 with the bit order reversed.
  Errors:
      ValueError if x is outside [0, 2**32).
  Difficulty: core. Screen-core (LeetCode 190 shape); the follow-up is
  always "without the per-bit loop".
  """
  raise NotImplementedError


def sar32(x: int, n: int) -> int:
  """Arithmetic shift right of a u32 bit pattern, returning a u32 pattern.

  Interpret x as a 32-bit two's complement value, shift right n filling
  with the sign bit, reinterpret the result as u32:
      sar32(0x80000000, 1)  == 0xC0000000
      sar32(0x7FFFFFFF, 1)  == 0x3FFFFFFF
      sar32(0xFFFFFFFF, 16) == 0xFFFFFFFF

  Args:
      x: u32 in [0, 2**32).
      n: shift count in [0, 31]. (n >= 32 is UB in C; rejected here.)
  Returns:
      u32 bit pattern of the arithmetic shift.
  Errors:
      ValueError if x is outside u32 range or n outside [0, 31].
  Difficulty: core. Depth: exercises the hop between "Python int" and
  "hardware register" views; the sign-extension idiom inside is the same
  one used to decode signed bitfields.
  """
  raise NotImplementedError


def popcount_swar(x: int) -> int:
  """Population count of a u64 with no per-bit loop and no bit_count().

  Required approach: SWAR reduction (pairs -> nibbles -> bytes using the
  0x5555…, 0x3333…, 0x0F0F… masks, finished by the 0x0101…01 multiply or an
  equivalent fixed sequence of word ops). A Kernighan `while x:` loop is
  good practice elsewhere; this stub asks for the loop-free ladder, and an
  interviewer checks the approach even though tests only check values.

  Args:
      x: u64 in [0, 2**64).
  Returns:
      number of set bits, in [0, 64].
  Errors:
      ValueError if x is outside u64 range.
  Difficulty: hard (to derive live; memorize the ladder). Depth: the stock
  escalation after you write Kernighan is "now without the loop", and the
  Python-specific trap is masking the multiply before the final shift.
  """
  raise NotImplementedError


def submasks(m: int) -> list[int]:
  """Every submask of m, in decreasing order, via s = (s - 1) & m.

  A submask s satisfies s | m == m (equivalently s & ~m == 0). The output
  starts at m, strictly decreases, ends at 0, and contains exactly
  2**popcount(m) elements:
      submasks(0b101) == [0b101, 0b100, 0b001, 0b000]
      submasks(0) == [0]

  Args:
      m: u64 in [0, 2**64). Callers keep popcount(m) small; tests do.
  Returns:
      list[int] of all submasks, m first, 0 last.
  Errors:
      ValueError if m is outside u64 range.
  Difficulty: hard if unseen, mechanical once drilled. Depth: bitmask-DP
  staple; be ready to argue termination (the 0 exit, since (0-1) & m == m
  would cycle) and the sum-over-all-m 3^n bound.
  """
  raise NotImplementedError
