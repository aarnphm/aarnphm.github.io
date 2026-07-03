"""Module 02: unsigned semantics + data alignment. Practice stubs.

Implement each function per its docstring, then run:

    python3 test_problems.py                              # your implementations
    PRACTICE_MODULE=solutions python3 test_problems.py    # reference sanity check

Tags: difficulty [warmup | core | hard]; screen-core = likely asked live,
depth = pays off in follow-up discussion.
"""

MASK32 = (1 << 32) - 1


def align_up(x: int, a: int) -> int:
  """Round x up to the nearest multiple of the alignment a.

  [warmup | screen-core]

  The interview one-liner: (x + a - 1) & ~(a - 1), valid only because a
  is a power of two (a - 1 is then a contiguous low-bit mask). The
  spelling x & -a is equivalent: Python ints behave as infinite two's
  complement, so -a == ~(a - 1).

  Args:
      x: byte offset / address / size; must be >= 0.
      a: alignment; must be a power of two >= 1.

  Returns:
      Smallest multiple of a that is >= x. Contract: align_up(0, a) == 0;
      align_up(x, a) == x when x is already a multiple of a (idempotent);
      0 <= align_up(x, a) - x < a.

  Raises:
      ValueError: if a is not a power of two >= 1, or x < 0.
  """
  raise NotImplementedError


def align_down(x: int, a: int) -> int:
  """Round x down to the nearest multiple of the alignment a.

  [warmup | screen-core]

  x & ~(a - 1), i.e. clear the low log2(a) bits. Same duality worth
  stating live: align_up(x, a) == align_down(x + a - 1, a).

  Args:
      x: must be >= 0.
      a: alignment; must be a power of two >= 1.

  Returns:
      Largest multiple of a that is <= x. align_down(0, a) == 0;
      already-aligned x comes back unchanged; 0 <= x - align_down(x, a) < a.

  Raises:
      ValueError: if a is not a power of two >= 1, or x < 0.
  """
  raise NotImplementedError


def is_aligned(x: int, a: int) -> bool:
  """Report whether x is a multiple of the alignment a.

  [warmup | screen-core]

  (x & (a - 1)) == 0. Note is_aligned(0, a) is True for every a, and
  everything is aligned to a == 1.

  Args:
      x: must be >= 0.
      a: alignment; must be a power of two >= 1.

  Returns:
      True iff x is a multiple of a.

  Raises:
      ValueError: if a is not a power of two >= 1, or x < 0.
  """
  raise NotImplementedError


def add32(a: int, b: int) -> int:
  """Wrapping u32 add: (a + b) mod 2^32.

  [warmup | screen-core]

  Models C uint32_t addition, where overflow is defined as reduction
  mod 2^32. Inputs may be ANY Python ints (negative included): they are
  taken mod 2^32, exactly C's conversion-to-unsigned rule, so
  add32(-1, 0) == 0xFFFFFFFF. In Python the entire job is one mask.

  Args:
      a: arbitrary Python int.
      b: arbitrary Python int.

  Returns:
      (a + b) mod 2^32, an int in [0, 2^32). add32(0xFFFFFFFF, 1) == 0.
  """
  raise NotImplementedError


def sub32(a: int, b: int) -> int:
  """Wrapping u32 subtract: (a - b) mod 2^32.

  [warmup | screen-core]

  sub32(0, 1) == 0xFFFFFFFF: unsigned underflow is the same mod-2^32
  reduction. Python's negative intermediate is erased by the mask.

  Args:
      a: arbitrary Python int.
      b: arbitrary Python int.

  Returns:
      (a - b) mod 2^32, an int in [0, 2^32).
  """
  raise NotImplementedError


def mul32(a: int, b: int) -> int:
  """Wrapping u32 multiply: (a * b) mod 2^32.

  [warmup | screen-core]

  mul32(0x10000, 0x10000) == 0 and mul32(0xFFFFFFFF, 0xFFFFFFFF) == 1
  (since (2^32 - 1)^2 = 2^64 - 2^33 + 1). C contrast worth knowing:
  uint32_t * uint32_t is defined (wraps) but uint16_t * uint16_t is
  signed-overflow UB because both operands promote to int.

  Args:
      a: arbitrary Python int.
      b: arbitrary Python int.

  Returns:
      (a * b) mod 2^32, an int in [0, 2^32).
  """
  raise NotImplementedError


def shl32(x: int, n: int) -> int:
  """Wrapping u32 left shift: (x * 2^n) mod 2^32.

  [warmup | screen-core]

  Bits shifted past bit 31 are discarded: shl32(0x80000000, 1) == 0.
  C contrast: shifting by n >= 32 or n < 0 is UB (x86 masks the count
  mod 32 so x << 32 == x at runtime; ARM32 register shifts give 0;
  the compiler may fold to anything). This function makes the shift
  count policy explicit instead of undefined.

  Args:
      x: arbitrary Python int (taken mod 2^32).
      n: shift count; must satisfy 0 <= n < 32.

  Returns:
      (x << n) mod 2^32, an int in [0, 2^32).

  Raises:
      ValueError: if n < 0 or n >= 32.
  """
  raise NotImplementedError


def shr32(x: int, n: int) -> int:
  """LOGICAL right shift of the u32 value of x: zeros shift in.

  [warmup | screen-core]

  Mask BEFORE shifting: (x & 0xFFFFFFFF) >> n. Python's >> is
  arithmetic (sign-propagating) on negative ints, so shr32(-16, 4)
  must be 0x0FFFFFFF, never -1. Mask-then-shift is THE idiom for
  emulating unsigned shifts in Python.

  Args:
      x: arbitrary Python int (taken mod 2^32 first).
      n: shift count; must satisfy 0 <= n < 32.

  Returns:
      (x mod 2^32) >> n with zero fill, an int in [0, 2^32).

  Raises:
      ValueError: if n < 0 or n >= 32.
  """
  raise NotImplementedError


def to_unsigned(x: int, bits: int) -> int:
  """Two's-complement bit pattern of x in `bits` bits: x mod 2^bits.

  [core | screen-core]

  Exactly C's conversion to an unsigned type, defined for ALL inputs
  (C11 6.3.1.3): to_unsigned(-1, 8) == 0xFF, to_unsigned(-128, 8) == 0x80,
  to_unsigned(300, 8) == 44. Round-trip law:
  to_signed(to_unsigned(v, b), b) == v for v in [-2^(b-1), 2^(b-1)).

  Args:
      x: arbitrary Python int.
      bits: width; must be >= 1.

  Returns:
      x mod 2^bits, an int in [0, 2^bits).

  Raises:
      ValueError: if bits < 1.
  """
  raise NotImplementedError


def to_signed(x: int, bits: int) -> int:
  """Reinterpret the low `bits` bits of x as a two's-complement value.

  [core | screen-core]

  The xor-and-subtract trick, with s = 2^(bits-1):
      ((x & (2^bits - 1)) ^ s) - s
  Xor with the sign bit swaps the halves of [0, 2^bits); subtracting s
  lands in [-s, s), the unique value congruent to x mod 2^bits.
  Byte-width equivalent that round-trips:
      int.from_bytes(u.to_bytes(bits // 8, "little"), "little", signed=True)

  Args:
      x: arbitrary Python int; only its low `bits` bits matter
         (to_signed(0x1FF, 8) == -1).
      bits: width; must be >= 1.

  Returns:
      int v in [-2^(bits-1), 2^(bits-1)) with v congruent to x mod 2^bits.
      to_signed(0xFF, 8) == -1; to_signed(0x7F, 8) == 127;
      to_signed(0x80, 8) == -128.

  Raises:
      ValueError: if bits < 1.
  """
  raise NotImplementedError


def struct_layout(
  fields: 'list[tuple[str, int, int]]',
) -> 'tuple[dict[str, int], int, int]':
  """Compute C struct member offsets, total size, and alignment.

  [core | screen-core] A real interview question, usually posed as
  "implement sizeof".

  Layout algorithm (what gcc/clang do without pragmas):
    1. cursor = 0. For each (name, size, align) in declaration order,
       place the member at offset = align_up(cursor, align), then
       cursor = offset + size. C never reorders members.
    2. Struct alignment A = max member alignment, or 1 if no fields.
    3. Total size = align_up(final cursor, A): tail padding, so every
       element of an array of this struct keeps all members aligned.

  Example:
      struct_layout([("c", 1, 1), ("i", 4, 4), ("s", 2, 2)])
      == ({"c": 0, "i": 4, "s": 8}, 12, 4)

  Args:
      fields: [(name, size, align), ...] in declaration order; size >= 0,
          align a power of two >= 1, names unique. (Real C members always
          have size % align == 0; this function does not require it.)

  Returns:
      (offsets, size, align): offsets maps each name to its byte offset;
      size is the tail-padded sizeof; align is the struct alignment.
      Empty input returns ({}, 0, 1).

  Raises:
      ValueError: on a duplicate name, size < 0, or align not a power
      of two >= 1.
  """
  raise NotImplementedError


def c_promote_trap(x: int) -> int:
  """Simulate what C computes for ((uint8_t)x) << 24 >> 24 with 32-bit int.

  [core | depth] Documents the integer-promotion sign-extension trap.

  Step by step on a two's-complement machine with 32-bit int (gcc/clang
  on x86-64 and arm64):
    1. (uint8_t)x   -> v = x mod 256.
    2. Promotion: uint8_t has rank < int and int holds all of 0..255,
       so v becomes a SIGNED int before any shift happens.
    3. v << 24      -> for v >= 128 this sets bit 31 (formally UB for
       a signed left shift that overflows, C11 6.5.7p4; every mainstream
       compiler produces the two's-complement pattern; C++20 defines it).
    4. >> 24 of a signed int is arithmetic (implementation-defined in C,
       sign-extending on every real compiler): the byte comes back
       sign-EXTENDED.

  Net effect: the author expected 0..255 back; C returns v - 256 for
  v >= 128. The C fix is ((uint32_t)x << 24) >> 24.

  Args:
      x: arbitrary Python int; only x mod 256 matters (the uint8_t cast).

  Returns:
      int in [-128, 127], the value the C expression evaluates to:
      c_promote_trap(0xFF) == -1, c_promote_trap(0x7F) == 127,
      c_promote_trap(200) == -56.
  """
  raise NotImplementedError


def reorder_fields(
  fields: 'list[tuple[str, int, int]]',
) -> 'list[tuple[str, int, int]]':
  """Reorder struct fields to minimize sizeof under C layout rules.

  [hard | depth] The follow-up after struct_layout: "now shrink it".

  Required order: alignment DESCENDING, stable (fields with equal
  alignment keep their declaration order; determinism matters).
  Why this is optimal rather than a heuristic: alignments are powers of
  two and each size is a multiple of its alignment, so after sorting,
  every prefix sum is a multiple of every remaining (smaller or equal)
  alignment; no internal padding is ever inserted, giving
  sizeof = align_up(sum(sizes), max(align)), and no ordering beats that
  (sizeof >= sum of sizes and must be a multiple of the max alignment).
  The tests verify optimality against brute force over all permutations.

  Args:
      fields: [(name, size, align), ...]; align a power of two >= 1,
          size a non-negative multiple of align (true of every real C
          type: scalars have size == align, aggregates are tail-padded).

  Returns:
      A new list, a permutation of `fields`, whose struct_layout size
      is minimal. [] for [].

  Raises:
      ValueError: if some align is not a power of two >= 1, or some
      size is negative or not a multiple of its align.
  """
  raise NotImplementedError
