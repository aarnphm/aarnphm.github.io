"""Reference solutions for the bits module. Stdlib only."""

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF


def _require(cond: bool, msg: str) -> None:
  if not cond:
    raise ValueError(msg)


def _check_field_bounds(word: int, offset: int, width: int) -> None:
  _require(0 <= word <= M64, f'word={word:#x} outside u64')
  _require(offset >= 0, f'offset={offset} negative')
  _require(width >= 1, f'width={width} < 1')
  _require(offset + width <= 64, f'offset+width={offset + width} > 64')


def pack_rgba(r: int, g: int, b: int, a: int) -> int:
  for name, c in (('r', r), ('g', g), ('b', b), ('a', a)):
    _require(0 <= c <= 0xFF, f'{name}={c} outside [0, 255]')
  return (r << 24) | (g << 16) | (b << 8) | a


def unpack_rgba(px: int) -> tuple[int, int, int, int]:
  _require(0 <= px <= M32, f'px={px} outside u32')
  return ((px >> 24) & 0xFF, (px >> 16) & 0xFF, (px >> 8) & 0xFF, px & 0xFF)


def extract_field(word: int, offset: int, width: int) -> int:
  _check_field_bounds(word, offset, width)
  return (word >> offset) & ((1 << width) - 1)


def insert_field(word: int, offset: int, width: int, value: int) -> int:
  _check_field_bounds(word, offset, width)
  _require(
    0 <= value < (1 << width), f'value={value} does not fit {width} bits'
  )
  mask = ((1 << width) - 1) << offset
  return (word & ~mask & M64) | (value << offset)


def next_pow2(x: int) -> int:
  _require(0 <= x <= 1 << 63, f'x={x} outside [0, 2**63]')
  if x <= 1:
    return 1
  x -= 1
  x |= x >> 1
  x |= x >> 2
  x |= x >> 4
  x |= x >> 8
  x |= x >> 16
  x |= x >> 32
  return x + 1


def reverse_bits32(x: int) -> int:
  _require(0 <= x <= M32, f'x={x} outside u32')
  x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1)
  x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2)
  x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4)
  x = ((x >> 8) & 0x00FF00FF) | ((x & 0x00FF00FF) << 8)
  return ((x >> 16) | (x << 16)) & M32


def sar32(x: int, n: int) -> int:
  _require(0 <= x <= M32, f'x={x} outside u32')
  _require(0 <= n <= 31, f'n={n} outside [0, 31]')
  # (x ^ 2**31) - 2**31 reinterprets the u32 pattern as a signed value;
  # Python's >> on a negative is arithmetic, then the mask returns to u32.
  signed = (x ^ 0x80000000) - 0x80000000
  return (signed >> n) & M32


def popcount_swar(x: int) -> int:
  _require(0 <= x <= M64, f'x={x} outside u64')
  x -= (x >> 1) & 0x5555555555555555
  x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
  x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
  # The 0x0101..01 multiply sums all eight bytes into the top byte; Python
  # ints do not wrap, so mask to u64 before extracting it.
  return ((x * 0x0101010101010101) & M64) >> 56


def submasks(m: int) -> list[int]:
  _require(0 <= m <= M64, f'm={m} outside u64')
  out = []
  s = m
  while True:
    out.append(s)
    if s == 0:
      break
    s = (s - 1) & m
  return out
