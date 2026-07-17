"""Reference solutions for module 02: unsigned semantics + data alignment."""

MASK32 = (1 << 32) - 1


def _require_pow2(a: int) -> None:
  if a < 1 or a & (a - 1):
    raise ValueError(f'alignment must be a power of two >= 1, got {a}')


def _require_nonneg(x: int) -> None:
  if x < 0:
    raise ValueError(f'x must be non-negative, got {x}')


def align_up(x: int, a: int) -> int:
  """Smallest multiple of a that is >= x."""
  _require_pow2(a)
  _require_nonneg(x)
  return (x + a - 1) & -a


def align_down(x: int, a: int) -> int:
  """Largest multiple of a that is <= x."""
  _require_pow2(a)
  _require_nonneg(x)
  return x & -a


def is_aligned(x: int, a: int) -> bool:
  """True iff x is a multiple of a."""
  _require_pow2(a)
  _require_nonneg(x)
  return x & (a - 1) == 0


def add32(a: int, b: int) -> int:
  """(a + b) mod 2^32."""
  return (a + b) & MASK32


def sub32(a: int, b: int) -> int:
  """(a - b) mod 2^32."""
  return (a - b) & MASK32


def mul32(a: int, b: int) -> int:
  """(a * b) mod 2^32."""
  return (a * b) & MASK32


def _require_shift(n: int) -> None:
  if not 0 <= n < 32:
    raise ValueError(f'shift count must be in [0, 32), got {n}')


def shl32(x: int, n: int) -> int:
  """(x << n) mod 2^32; n in [0, 32)."""
  _require_shift(n)
  return (x << n) & MASK32


def shr32(x: int, n: int) -> int:
  """Logical right shift of the u32 value of x; n in [0, 32)."""
  _require_shift(n)
  return (x & MASK32) >> n


def to_unsigned(x: int, bits: int) -> int:
  """x mod 2^bits: C's conversion of x to an unsigned bits-wide type."""
  if bits < 1:
    raise ValueError(f'bits must be >= 1, got {bits}')
  return x & ((1 << bits) - 1)


def to_signed(x: int, bits: int) -> int:
  """Two's-complement value of the low `bits` bits of x."""
  if bits < 1:
    raise ValueError(f'bits must be >= 1, got {bits}')
  sign = 1 << (bits - 1)
  # xor biases by 2^(bits-1) mod 2^bits (swaps halves), subtract re-centers
  return ((x & ((1 << bits) - 1)) ^ sign) - sign


def struct_layout(
  fields: 'list[tuple[str, int, int]]',
) -> 'tuple[dict[str, int], int, int]':
  """C layout: (name -> offset, sizeof, alignof)."""
  offsets: 'dict[str, int]' = {}
  cursor = 0
  align = 1
  for name, size, falign in fields:
    _require_pow2(falign)
    if size < 0:
      raise ValueError(f'size must be >= 0, got {size} for {name!r}')
    if name in offsets:
      raise ValueError(f'duplicate field name {name!r}')
    cursor = (cursor + falign - 1) & -falign
    offsets[name] = cursor
    cursor += size
    align = max(align, falign)
  return offsets, (cursor + align - 1) & -align, align


def c_promote_trap(x: int) -> int:
  """What C computes for ((uint8_t)x) << 24 >> 24 with 32-bit int."""
  pattern = ((x & 0xFF) << 24) & MASK32
  promoted = to_signed(pattern, 32)
  # Python >> on negatives is arithmetic, matching every real C compiler
  return promoted >> 24


def reorder_fields(
  fields: 'list[tuple[str, int, int]]',
) -> 'list[tuple[str, int, int]]':
  """Size-minimizing order: alignment descending, ties keep source order."""
  for name, size, falign in fields:
    _require_pow2(falign)
    if size < 0 or size % falign:
      raise ValueError(
        f'size must be a non-negative multiple of align for {name!r}'
      )
  return sorted(fields, key=lambda f: -f[2])
