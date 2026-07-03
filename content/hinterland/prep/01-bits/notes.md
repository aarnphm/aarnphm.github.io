# Bits, bytes, shifts

The register model, the mask idioms, and the exact places Python and C disagree. Everything downstream in this kit (varints, headers, streaming decoders) compiles to this layer.

## 1. Core mental model

A `u32`/`u64` is a bit vector with arithmetic mod $2^{32}$/$2^{64}$. Two's complement is the same bits reread with the top bit weighted $-2^{N-1}$: when bit $N-1$ is set, signed = unsigned $-\,2^N$. One adder circuit serves both readings; only right shift, comparison, widening, and overflow rules care which reading you meant.

Negation is flip-and-increment, $-x = \lnot x + 1$:

```
 6 = 0000_0110
~6 = 1111_1001
+1 = 1111_1010 = 0xFA = 250 = 256 - 6   (-6 as a u8 pattern)
```

Python ints are two's complement at infinite width. `-1` is an unbounded run of 1s, `~x == -x - 1` holds exactly, and `& | ^ >>` on negatives behave as if sign-extended forever. `bin(-5)` prints `'-0b101'` (sign-magnitude display, not the conceptual pattern); to see a pattern, mask first: `format(-5 & 0xFF, '08b') == '11111011'`.

**Emulating fixed width in Python**, the discipline used everywhere in this kit: keep every value in $[0, 2^N)$ and re-mask after each op that can escape the range, which is `+ - * <<`, `~`, and unary `-`. `>>` and `& mask` cannot escape.

```python
M32 = 0xFFFFFFFF
add32 = (a + b) & M32
not32 = ~x & M32  # ~x alone is negative in Python
neg32 = -x & M32  # == (~x + 1) & M32
```

Logical shift right of a value whose bit 31 is set: if you kept it masked it is an ordinary nonnegative int, and `>>` already zero-fills: `0x80000000 >> 4 == 0x08000000`. The same bit pattern held in a C signed `int` shifts arithmetically: `(int)0x80000000 >> 4` yields the `0xF8000000` pattern. Python only shifts arithmetically when an actual negative leaks in; `(x & M32) >> n` restores the logical shift. Rule: mask before `>>` when provenance is unclear, mask after everything that grows.

## 2. Logical vs arithmetic shift, and the language deltas

Left shift is one concept. Right shift is two: logical (zero fill) and arithmetic (sign fill; floor division by $2^n$).

```
-7 = ...1111_1001
-7 >> 1 = ...1111_1100 = -4        arithmetic: floors toward -inf
C:      -7 / 2 = -3                division truncates toward 0 (mandated since C99)
Python: -7 // 2 == -7 >> 1 == -4   floor everywhere, internally consistent
```

So ASR is signed division only up to rounding; compilers that lower `x / 2` for signed `x` to a shift must add a bias correction, and this rounding gap is exactly why.

| operation                           | Python                                                | C                                                                                                        | Rust                                                                     | Go                                                               |
| ----------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| `x >> n`, x negative signed         | arithmetic over infinite sign bits: `-1 >> 100 == -1` | implementation-defined, even in C23 (arithmetic on every mainstream compiler); C++20 mandates arithmetic | arithmetic on `iN`, logical on `uN`                                      | arithmetic                                                       |
| `x << n`, x negative                | exact math                                            | **UB**                                                                                                   | defined (two's complement bits shift)                                    | defined                                                          |
| shift count $\ge$ width or negative | exact math: `1 << 200` is a big int                   | **UB**                                                                                                   | panic in debug, count masked in release; use `checked_shl`/`rotate_left` | defined: 0, or all sign bits for signed `>>`                     |
| `1 << 31` into 32-bit signed        | fine                                                  | **UB** (signed overflow); write `1u << 31`, `1ull << 63`                                                 | `1i32 << 31 == i32::MIN`, defined                                        | constant overflow is a compile error; runtime var wraps, defined |

Why C leaves count $\ge$ width undefined: hardware disagrees. x86 masks the count to 5 bits for 32-bit operands (6 for 64-bit), so `x << 32` executes as `x << 0`; ARM32 uses the low byte of the count register, so the same shift gives 0. C refuses to pick, so you guard variable counts.

## 3. Single bits and bit fields

Bit `k`, counting from 0 at the LSB:

```python
(x >> k) & 1  # test
x | (1 << k)  # set
x & ~(
  1 << k
)  # clear   (Python: already correct; ~ makes a negative, & with nonneg x is fine)
x ^ (1 << k)  # toggle
(x & ~(1 << k)) | (v << k)  # update to v in {0, 1}: clear then or
```

The field primitive, the single most-asked shape in encoding screens:

```python
mask = (1 << width) - 1
get = (word >> offset) & mask
put = (word & ~(mask << offset)) | ((value & mask) << offset)
```

Worked, `word = 0xDEAD_BEEF`, offset 8, width 8:

```
get: 0xDEADBEEF >> 8 = 0x00DEADBE;  & 0xFF -> 0xBE
put value=0x42:
  mask << 8      = 0x0000_FF00
  word cleared   = 0xDEAD_00EF
  | (0x42 << 8)  = 0xDEAD_42EF
```

C trap: building the mask. `(1 << width) - 1` is UB at `width == 32` for a u32 (and already UB at 31 without the `u` suffix, since bare `1` is a signed int). A safe u64 form for $1 \le \text{width} \le 64$: `(((1ull << (width - 1)) - 1) << 1) | 1`. Python needs none of this; when you write `(1 << width) - 1` in an interview, say out loud that width = word size would be UB in C. Cheap, high-signal.

C bit-field structs (`unsigned f : 3;`) have implementation-defined ordering and padding, so wire formats are always decoded with explicit shifts and masks, never struct overlay.

## 4. Two's complement tricks

**Sign-extend a k-bit field** (interpret the output of a field extract as signed):

```python
def sext(v, k):
  return (v ^ (1 << (k - 1))) - (1 << (k - 1))
```

XOR with $2^{k-1}$ toggles the field's sign bit, which adds or subtracts $2^{k-1}$; the subtraction recenters both cases. For 4-bit `v = 0b1010` (10): `10 ^ 8 = 2`, then `2 - 8 = -6`, and `1010` is indeed $-6$ in 4-bit two's complement. Branchy equivalent: `v - (1 << k) if v >> (k - 1) else v`. In C, casting an out-of-range u32 to i32 is implementation-defined (all mainstream compilers wrap); the xor-sub trick is the portable spelling.

**Lowest set bit family**, all consequences of borrow/carry propagation through trailing zeros:

```python
x & -x  # isolate lowest set bit
x & (x - 1)  # strip lowest set bit
x | (x - 1)  # smear: set everything below the lowest set bit
~x & (x + 1)  # isolate lowest clear bit
```

`x & -x` works because $-x = \lnot x + 1$: the +1 carries through the trailing 1s of $\lnot x$ (the trailing 0s of $x$) and stops at $x$'s lowest set bit; below it zeros, at it a 1, above it the complement.

```
x     = 0110_1000
x - 1 = 0110_0111     x & (x-1) = 0110_0000
-x    = 1001_1000     x & -x    = 0000_1000
```

**Power-of-two test**: `x != 0 and (x & (x - 1)) == 0`. The guard matters: `0 & -1 == 0`, so 0 passes the naked test.

**Round up to the next power of two** (bit smear then increment):

```python
def next_pow2_u64(x):  # for 1 <= x <= 2**63
  x -= 1
  x |= x >> 1
  x |= x >> 2
  x |= x >> 4
  x |= x >> 8
  x |= x >> 16
  x |= x >> 32
  return x + 1
```

The initial decrement makes exact powers map to themselves. Each OR doubles the width of the solid run of 1s below the top bit, giving $2^k - 1$, then +1. Worked at byte width: $x = 38 = \texttt{0b100110}$, minus 1 is `100101`, smear gives `111111`, plus 1 is `1000000` = 64. Edges: at $x = 0$ the Python smear of $-1$ stays $-1$ and returns 0, so special-case 0; above $2^{63}$ the u64 version wraps to 0. Python cheat: `1 << (x - 1).bit_length()` for $x \ge 1$. C++20: `std::bit_ceil`.

## 5. Popcount, ctz, clz

**Kernighan**, $O(\text{set bits})$, the expected first answer:

```python
n = 0
while x:
  x &= x - 1  # strip one set bit per iteration
  n += 1
```

`0b0110_1000 -> 0b0110_0000 -> 0b0100_0000 -> 0`, three iterations, three bits.

**SWAR**, $O(\log w)$ with no data-dependent loop; the standard follow-up is "now without looping per bit":

```python
def popcount64(x):
  x -= (x >> 1) & 0x5555555555555555
  x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
  x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
  return ((x * 0x0101010101010101) & 0xFFFFFFFFFFFFFFFF) >> 56
```

The constants alternate at doubling granularity: `0x55…` = `0101…` (1-bit fields), `0x33…` = `0011…` (2-bit), `0x0F…` (4-bit). Line 1 is the two-bit trick: a chunk holding $2a + b$ has popcount $a + b = (2a+b) - a$, i.e. subtract the high bit once. Line 2 adds adjacent 2-bit counts into 4-bit fields (max 4, fits). Line 3 adds adjacent nibbles into bytes; one mask after the add suffices since 8 fits a nibble's headroom. The multiply by `0x0101…01` is a byte-sum: every byte lands once in the top byte's column and the total (max 64) fits a byte; `>> 56` extracts it. In C the multiply wraps for free; in Python you must `& M64` before shifting, and that line is where naive ports break.

Worked at byte width, `b = 0b1101_0110` (five bits):

```
pairs:   11,01,01,10  ->  10,01,01,01     (each ab -> ab - a)
nibbles: 10+01, 01+01 ->  0011, 0010      (3 and 2)
byte:    0011 + 0010  ->  0101 = 5
```

**Byte table**: `TABLE = bytes(bin(i).count("1") for i in range(256))`, then sum `TABLE[(x >> s) & 0xFF]` over byte offsets. Wins in hot loops where the 256-byte table stays in L1.

Production answers: Python `x.bit_count()` (3.10+; before that `bin(x).count('1')`), C `__builtin_popcountll`, C++20 `std::popcount`, Rust `count_ones`, Go `math/bits.OnesCount64`, x86 `POPCNT`, NEON `CNT`+`ADDV`.

**Trailing / leading zeros**:

```python
ctz = (x & -x).bit_length() - 1  # x > 0; returns -1 at x == 0, guard it
clz32 = 32 - x.bit_length()  # correct even at x == 0 (gives 32)
```

`x.bit_length() - 1` is floor log2. C's `__builtin_ctz(0)` and `__builtin_clz(0)` are **UB** (BSF/BSR leave the register undefined at 0; TZCNT/LZCNT later fixed the hardware). C++20 `std::countr_zero(0) == width` is defined, as are Rust `trailing_zeros(0)` and Go `bits.TrailingZeros(0)`. "What does yours do at zero" is a stock probe.

Depth: branch-free ctz without an instruction = De Bruijn multiply, `table[((x & -x) * 0x077CB531) >> 27]` for u32. Name it; nobody derives it live.

## 6. Rearranging bits and bytes

**Reverse the bits of a u32**: divide and conquer with the same mask family as popcount, swapping at doubling granularity. Drill the 8-bit version until automatic:

```
b = 1011_0010
swap adjacent bits:  ((b>>1) & 0x55) | ((b & 0x55) << 1)  ->  0111_0001
swap bit pairs:      ((b>>2) & 0x33) | ((b & 0x33) << 2)  ->  1101_0100
swap nibbles:        ((b>>4) | (b<<4)) & 0xFF             ->  0100_1101   reversed
```

u32 adds two rungs:

```python
def reverse_bits32(x):
  x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1)
  x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2)
  x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4)
  x = ((x >> 8) & 0x00FF00FF) | ((x & 0x00FF00FF) << 8)
  return ((x >> 16) | (x << 16)) & 0xFFFFFFFF
```

In the first four lines both operands are pre-masked, so nothing escapes 32 bits. The last line shifts all of `x` left, so the final `& M32` does the cleanup; forgetting it is the classic Python port bug. AArch64 has `RBIT` in one instruction; x86 has no bit reverse.

**Byte swap** (endianness flip):

```python
def bswap32(x):
  return (
    (x >> 24) | ((x >> 8) & 0xFF00) | ((x << 8) & 0xFF0000) | (x << 24)
  ) & 0xFFFFFFFF
```

`0x12345678 -> 0x12 | 0x3400 | 0x560000 | 0x78000000 = 0x78563412`. u64 by SWAR rungs (pairs of bytes, then halfwords, then words):

```python
def bswap64(x):
  x = ((x >> 8) & 0x00FF00FF00FF00FF) | ((x & 0x00FF00FF00FF00FF) << 8)
  x = ((x >> 16) & 0x0000FFFF0000FFFF) | ((x & 0x0000FFFF0000FFFF) << 16)
  return ((x >> 32) | (x << 32)) & 0xFFFFFFFFFFFFFFFF
```

Production: `__builtin_bswap64`, Rust `swap_bytes`/`to_be`, Go `bits.ReverseBytes64`, Python `int.to_bytes(8, 'little')` / `int.from_bytes(b, 'big')` or `struct.pack('<Q' / '>Q', x)`, hardware `BSWAP`/`REV`/`MOVBE`.

Endianness in one breath: the u32 `0x12345678` in little-endian memory is bytes `78 56 34 12`; big-endian (network order) is `12 34 56 78`. Shifts operate on the integer value and are endianness-free; endianness exists only at the memory/wire boundary. `(word >> 24) & 0xFF` means "most significant byte", which coincides with wire byte 0 only in big-endian formats.

**Rotate at fixed width.** Python and C both lack a rotate operator; emulate and let compilers pattern-match to `ROL`:

```python
def rotl32(x, n):
  n &= 31
  return (
    (x << n) | (x >> (32 - n))
  ) & 0xFFFFFFFF  # n=0: x >> 32 == 0 in Python, harmless


def rotr32(x, n):
  n &= 31
  return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF
```

`rotl8(0b1011_0010, 3) = 0b1001_0101`: the top three bits wrap to the bottom. In C, `x >> (32 - n)` is UB at `n == 0`; the canonical UB-free form is `(x << n) | (x >> (-n & 31))`. Rust: `rotate_left`. Rotates are the workhorse of every ARX primitive (ChaCha, SipHash, SHA), which is where hashing follow-ups go.

## 7. XOR

`x ^ x = 0`, `x ^ 0 = x`, associative and commutative: XOR is carry-less addition in $(\mathbb{Z}/2)^n$. Consequences:

- **Find the odd one out**: XOR an array where everything else appears twice; pairs annihilate: `5 ^ 7 ^ 5 == 7`. Variants: missing number in $0..n$ (XOR both ranges); two singletons (XOR all to get `d`, split the array by `d & -d`, recurse).
- **Parity** = popcount mod 2 by folding halves: `x ^= x >> 16; x ^= x >> 8; x ^= x >> 4; x ^= x >> 2; x ^= x >> 1; parity = x & 1`. Each fold XORs the halves, so the final low bit accumulates every bit's parity.
- **Hamming distance**: `popcount(a ^ b)`; XOR marks exactly the differing bits.
- **XOR swap**: `a ^= b; b ^= a; a ^= b`. Know it, refuse to use it, and say why: it zeroes both when the operands alias the same memory (`swap(&v[i], &v[i])`), and it chains three dependent ops where the compiler's register renaming made the temporary free anyway.

## 8. Branchless idioms

C, i32, with `m = x >> 31` (arithmetic shift gives all-ones for negative, zero otherwise):

```c
abs:      (x ^ m) - m            /* m = -1: reduces to ~x + 1; m = 0: identity. UB at INT_MIN */
min(x,y): y ^ ((x ^ y) & -(x < y))   /* -(x<y) is all-ones when true, selecting x */
max(x,y): x ^ ((x ^ y) & -(x < y))
negate if f in {0,1}:  (x ^ -f) + f
select:   b ^ ((a ^ b) & mask)   /* mask all-ones -> a, zero -> b; constant-time */
```

When it matters: mispredicted branches cost a pipeline flush (15 to 20 cycles) versus 2 to 3 ALU ops, so branchless wins only on unpredictable data, and compilers already emit `cmov` for simple ternaries; measure first. Constant-time selection is mandatory in crypto, where a data-dependent branch is a timing leak. SIMD lanes cannot branch at all, so mask-select is the only control flow. In Python, never for speed (interpreter dispatch dwarfs the ALU op); only when emulating hardware. Interview move: write the readable branch, name the branchless form, switch only if asked.

`INT_MIN` caveat: `abs`, unary minus, and the branchless forms all overflow at $-2^{31}$, which has no positive counterpart in $[-2^{31}, 2^{31})$; signed overflow is UB in C. `abs(INT_MIN)` is a stock find-the-bug.

## 9. Enumerating submasks

All submasks of `m` (all `s` with `s | m == m`), in strictly decreasing order:

```python
s = m
while True:
  visit(s)
  if s == 0:
    break
  s = (s - 1) & m
```

`s - 1` flips the lowest set bit of `s` and turns on everything below it; `& m` keeps only the positions `m` owns. Net effect: decrement in the compressed coordinate system of `m`'s bits, so every submask appears exactly once, descending. `m = 0b101` yields `101, 100, 001, 000`. Count is $2^{\mathrm{popcount}(m)}$.

Two classic bugs: `while s:` as the loop condition silently drops the empty submask; and continuing past 0 loops forever, since `(0 - 1) & m == m` (in fixed width too: `0xFFFF… & m == m`). Hence the do-while shape with an explicit exit at 0.

Depth: summed over all $m$ of an $n$-bit universe, $\sum_m 2^{\mathrm{popcount}(m)} = 3^n$ (each bit is out of `m`, in `m` but not `s`, or in both), the complexity signature of subset-sum DP and SOS DP.

## 10. Gotchas and interviewer follow-ups

- **"What does it do at zero?"** Kernighan popcount: 0, correct. `(x & -x).bit_length() - 1`: returns $-1$, guard it. `__builtin_ctz(0)`/`__builtin_clz(0)`: UB. C++20/Rust/Go count ops: defined, return width. Power-of-two test without `x != 0`: wrongly accepts 0. `next_pow2(0)` via smear: returns 0 (or wraps), special-case to 1.
- **"What if width equals the word size?"** `(1 << 64) - 1` on u64 in C is UB twice (signed overflow of `1`, count = width). Python computes it exactly. Either bound the API or build the mask in two steps.
- **"Why did you mask after `~`?"** Python `~x` is $-x-1$, a negative with infinite sign bits; `~x & M32` is the u32 NOT. Same for unary minus.
- **"Shift by 32?"** C: UB. Python and Go: defined. Rust: debug panic, release masks the count. x86 would mask to `x << 0`, ARM32 would give 0, which is exactly why C refuses to define it.
- **"Is `>>` the same as dividing?"** Arithmetic shift floors, C signed division truncates: `-7 >> 1 == -4` but `-7 / 2 == -3`. Python `//` floors, so `>>` and `//` agree there.
- **"Why `uint8_t*` and not `char*` for byte parsing?"** `char` may be signed: byte `0xFF` promotes to `int` $-1$, so `c >> 4` gives `0xFFFFFFFF...` not `0x0F`. Decode through `uint8_t` or mask with `& 0xFF` at every read.
- **Signed/unsigned comparison**: in C, `-1 < 1u` is false; the $-1$ converts to `0xFFFFFFFF`. The realistic form: `if (len - offset < need)` with unsigned `len` underflows to huge instead of negative. Compare before subtracting, or check `offset > len` first.
- **Overflow detection**: unsigned wrap is defined in C; detect with `sum = a + b; if (sum < a)`. Signed overflow is UB; use `__builtin_add_overflow` or C23 `<stdckdint.h>`. Python never wraps, so emulate with `& M64` and compare, and remember that "loop until it wraps" never terminates on Python ints.
- **Python display traps**: `bin`/`hex` on negatives print sign-magnitude (`hex(-1) == '-0x1'`); always mask before formatting patterns: `format(x & M32, '08x')`.
- **The escalation ladder** interviewers walk for popcount and friends: per-bit loop, Kernighan, SWAR, byte table, hardware intrinsic. Know where each wins and volunteer the next rung before being pushed.
- **"What byte order does your packed u32 have?"** None until it hits memory. Integer layout (shift positions) and memory layout (endianness) are different questions; specify the wire order, then `to_bytes(4, 'big')` or `struct.pack('>I', x)`.

## 11. Rapid-fire drills

1. Isolate the lowest set bit → `x & -x`.
2. Clear the lowest set bit → `x & (x - 1)`.
3. Power-of-two test → `x != 0 and (x & (x - 1)) == 0`; the zero guard is the point.
4. `-1 >> 1` in Python → `-1` (arithmetic over infinite sign bits).
5. `0x80000000 >> 4` in Python → `0x08000000` (nonnegative int, logical fill).
6. Logical shift right of a possibly-negative Python int at u32 → `(x & 0xFFFFFFFF) >> n`.
7. Sign-extend 12-bit `0x8A3` → `(0x8A3 ^ 0x800) - 0x800 = -1885`.
8. `x & (x - 1)` at `x = 0` → 0, so the naked expression calls 0 a power of two.
9. `next_pow2(0)` via the smear → broken (0), special-case to 1.
10. C: `int x = 1 << 31;` → UB; `1u << 31` is fine.
11. C: `x >> 32` on a u32 → UB even though x86 hardware would execute `x >> 0`.
12. `popcount(0xF0F0F0F0)` → 16.
13. UB-free C rotate left → `(x << n) | (x >> (-n & 31))`.
14. `-7 >> 1` vs `-7 / 2` in C → $-4$ vs $-3$, floor vs truncate.
15. Number of submasks of `m` → $2^{\mathrm{popcount}(m)}$, enumerated by `s = (s - 1) & m` with a do-while.
16. `bswap32(0x12345678)` → `0x78563412`.
17. `reverse_bits32(1)` → `0x80000000`.
18. `__builtin_ctz(0)` → UB; Rust `u32::trailing_zeros(0)` → 32.
19. Hamming distance → `popcount(a ^ b)`.
20. Set bit `k` to `v` → `(x & ~(1 << k)) | (v << k)`.
21. Extract bits 23..16 of a u32 → `(x >> 16) & 0xFF`.
22. Fastest Python popcount → `x.bit_count()` (3.10+).
