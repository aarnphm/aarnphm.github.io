# 04 — representing numbers as byte streams

## core mental model

A byte stream has no numbers in it. It has bytes, and a _convention_ for mapping byte positions to significance. Every decoder is the polynomial

$$v = \sum_{i} b_i \cdot 256^{k(i)}$$

where $k(i)$ is decided by endianness. Little-endian: $k(i) = i$ (byte at lowest offset is the $256^0$ digit). Big-endian: $k(i) = n-1-i$. That is the entire topic; everything else is API and edge cases.

Python `int` is an unbounded abstraction with no width and no endianness — width appears only when you serialize. C integers have width always and endianness the moment they hit memory. Keep the two models separate in your head; the interview traps live at the boundary.

## endianness

Memory layout of the u32 `0x11223344` at address `a`:

```
offset:        a+0  a+1  a+2  a+3
little-endian:  44   33   22   11      (LSB first)
big-endian:     11   22   33   44      (MSB first, reads like the literal)
```

- **Network order is big-endian** (RFC 791/1700): protocol headers were designed to be human-auditable in dumps — a BE field reads left-to-right like the written number. `!` in `struct` means exactly this.
- **x86/ARM went little-endian** because LSB-at-lowest-address makes width changes free: a pointer to a u32 holding `7`, reinterpreted as `u16*` or `u8*`, still reads `7` at the _same address_. On BE you must add an offset to narrow. Early serial ALUs (8080 lineage) also wanted the low byte first for carry propagation. ARM is bi-endian on paper, LE in every OS you will touch.
- Consequence for you: the wire disagrees with your CPU exactly when parsing network protocols and BE file formats on commodity hardware, so byte swapping is the norm, never the exception.

Where it bites:

1. **File formats**: PNG/JPEG/AIFF lengths are BE; ZIP, BMP, ELF (mostly), WAV/RIFF, GGUF, and the safetensors u64 header length are LE. Same parser, opposite loops — misread a PNG chunk length as LE and you'll try to `read(0x0D000000)` instead of 13.
2. **Pointer-casting in C**: `*(uint32_t *)p` on a BE wire buffer gives the swapped value on x86 — and is UB anyway (strict aliasing + alignment). The blessed idiom is `memcpy(&v, p, 4)` or explicit shifts; compilers pattern-match both into a single `mov`/`bswap`.
3. **Hashing/memcmp on raw structs**: the digest depends on host endianness _and_ on padding bytes with indeterminate content. Serialize explicitly, then hash.
4. **numpy dtypes**: `'<f4'` vs `'>f4'` — loading a BE array with the native LE dtype silently produces garbage floats, no error.

Host detection: `sys.byteorder` in Python; in C, write `1` into an `int` and inspect the first byte via `char*`. `htonl`/`ntohl` are byte swaps on LE hosts, identity on BE.

## manual codecs (shifts only)

The form interviewers ask for. Loop and unrolled, both endians:

```python
def read_u32_le(b, o):  # v |= digit << position
  v = 0
  for i in range(4):
    v |= b[o + i] << (8 * i)
  return v


def read_u32_be(b, o):  # accumulator: shift up, or in
  v = 0
  for i in range(4):
    v = (v << 8) | b[o + i]
  return v


# unrolled LE / BE
v = b[o] | b[o + 1] << 8 | b[o + 2] << 16 | b[o + 3] << 24
v = b[o] << 24 | b[o + 1] << 16 | b[o + 2] << 8 | b[o + 3]


def write_u32_le(v):
  return bytes((v >> (8 * i)) & 0xFF for i in range(4))
```

Worked example, `read_u32_le(b'\x44\x33\x22\x11', 0)`:
`0x44 | 0x33<<8 | 0x22<<16 | 0x11<<24 = 0x44 + 0x3300 + 0x220000 + 0x11000000 = 0x11223344`.

C vs Python divergence — say these out loud in the interview:

| behavior            | Python                                                   | C                                                                                                                 |
| ------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| overflow            | never; you must mask `& 0xFFFFFFFF`                      | wraps (unsigned) / UB (signed)                                                                                    |
| `>>` on negative    | arithmetic, floors, infinite sign bits (`-1 >> 5 == -1`) | logical on unsigned; impl-defined on signed (pre-C++20)                                                           |
| `b[3] << 24`        | fine, unbounded int                                      | `char` promotes to _signed_ `int`; if `b[3] >= 0x80` the shift is UB — cast `(uint32_t)(unsigned char)b[3]` first |
| logical shift right | doesn't exist; mask first: `(x >> n) & mask`             | `>>` on unsigned                                                                                                  |
| unaligned load      | irrelevant, you index bytes                              | pointer-cast UB; traps on some ARM/SPARC; use `memcpy`                                                            |

The `char`-promotion row is the single most common C follow-up in this space. In Python the mirror trap is forgetting that intermediate values never truncate: after arithmetic you must mask to re-impose width.

### signed values: the two's-complement bridge

Width-$w$ two's complement is just unsigned arithmetic mod $2^w$ with the top half of the range relabeled negative.

- **encode**: `u = n & (2**64 - 1)` — masking _is_ the two's-complement conversion for negative `n` (Python's `&` acts on the infinite sign-extended form). Then write `u` as unsigned.
- **decode**: read unsigned `u`, then `u - 2**64 if u >= 2**63 else u`. Branchless: `(u ^ 2**63) - 2**63`.

Worked: `-2` as i16 → `-2 & 0xFFFF = 0xFFFE` → LE bytes `fe ff`. Decode `0xFFFE`: `≥ 0x8000`, so `0xFFFE - 0x10000 = -2`.

## the Python toolbox, ranked

1. **`int.from_bytes` / `int.to_bytes`** — the interview default, zero imports.
   `int.from_bytes(buf[o:o+4], 'little', signed=False)`; `n.to_bytes(8, 'little', signed=True)`. Raises `OverflowError` when `n` doesn't fit (including any negative without `signed=True`). Trap: since 3.11 the arguments are optional and **`byteorder` defaults to `'big'`** — `(258).to_bytes(2)` is `b'\x01\x02'`. Always pass it explicitly.

2. **`struct`** — for records and floats. Format chars you must know cold:

   | char | C type           | size | char | C type             | size |
   | ---- | ---------------- | ---- | ---- | ------------------ | ---- |
   | `b`  | signed char      | 1    | `B`  | unsigned char      | 1    |
   | `h`  | short            | 2    | `H`  | unsigned short     | 2    |
   | `i`  | int              | 4    | `I`  | unsigned int       | 4    |
   | `q`  | long long        | 8    | `Q`  | unsigned long long | 8    |
   | `f`  | float (binary32) | 4    | `d`  | double (binary64)  | 8    |

   Prefixes: `<` LE, `>` BE, `!` network (= BE) — all three use standard sizes, **no padding**. `=` native order, standard sizes, no padding. `@` (the default!) native order, native sizes, **with C alignment padding**: `calcsize('<BI') == 5` but `calcsize('@BI') == 8` (3 pad bytes after the `B`). Parsing wire data with the default prefix is the classic bug; always write `<` or `>`.
   `unpack` demands an exact-length buffer; `unpack_from(fmt, buf, offset)` doesn't, and `pack_into(fmt, buf, offset, ...)` writes into a `bytearray` in place — the pair to reach for when walking a record array. Errors are `struct.error`, not `ValueError`.

3. **`memoryview`** — zero-copy windows. `mv = memoryview(buf); mv[off:off+n]` slices without copying (a 100 MB parse loop with `bytes` slices is accidentally $O(n^2)$ in copies). `mv.cast('I')` reinterprets in _native_ order. `bytes(mv)` is the explicit copy at the end.

4. Bulk data: `numpy.frombuffer(buf, dtype='<u4')` — one line, but says "I know when to stop hand-rolling."

## IEEE 754 anatomy

```
float32:  [ s:1 ][ e:8  bias 127  ][ m:23 ]     value = (-1)^s · 1.m · 2^(e-127)
float64:  [ s:1 ][ e:11 bias 1023 ][ m:52 ]     value = (-1)^s · 1.m · 2^(e-1023)
```

Categories by raw exponent field:

| exponent | mantissa | meaning                                                                       |
| -------- | -------- | ----------------------------------------------------------------------------- |
| 0        | 0        | ±zero                                                                         |
| 0        | ≠0       | subnormal: $(-1)^s \cdot 0.m \cdot 2^{-126}$ (no hidden 1; gradual underflow) |
| 1..254   | any      | normal (hidden leading 1)                                                     |
| all ones | 0        | ±inf                                                                          |
| all ones | ≠0       | NaN — MSB of mantissa is the _quiet_ bit; the rest is payload                 |

Worked decompositions (float32):

- `1.0` = $1.0 \cdot 2^0$ → s=0, e=127=`0x7F`, m=0 → `0x3F800000`.
- `-2.5` = $-1.25 \cdot 2^1$ → s=1, e=128, m=`0.01` in binary = `0x200000` → `0xC0200000`.
- Smallest subnormal = bits `0x00000001` = $2^{-23} \cdot 2^{-126} = 2^{-149}$.
- Largest finite = `0x7F7FFFFF` = $(2 - 2^{-23}) \cdot 2^{127} \approx 3.40 \times 10^{38}$.

**float ↔ bits round-trip** (the bit-cast idiom, memorize):

```python
bits = struct.unpack('<Q', struct.pack('<d', f))[0]  # float64 -> u64
f = struct.unpack('<d', struct.pack('<Q', bits))[0]  # u64 -> float64
# float32: '<I' / '<f' — note pack('<f', x) also rounds x to nearest binary32
```

Endianness prefix must match on both sides; it cancels out, but mixing `<`/`>` across the pair byte-swaps your float.

- **Negative zero**: `0x80000000` (f32) / `0x8000_0000_0000_0000` (f64). `-0.0 == 0.0` is `True`, yet `1/-0.0 == -inf` and `struct` round-trips the sign bit. Detect with `math.copysign(1.0, x) < 0` or by comparing bits.
- **NaN**: `nan != nan` (breaks `==`-based tests, sort stability, dict-key sanity). Canonical quiet NaN f64 = `0x7FF8000000000000`. Test with `math.isnan`.
- **Why 0.1 is inexact**: $0.1 = 1/10$, and 10 has prime factor 5; base-2 expansions terminate only for denominators $2^k$, so 0.1 is the repeating binary `0.0001100110011…`. The stored double is `0x3FB999999999999A` ≈ 0.1000000000000000055511151231257827, hence `0.1 + 0.2 != 0.3`. Exactness needs `decimal`/`fractions` — or fixed point.
- ML aside: bfloat16 is literally the top 16 bits of float32 (same 8-bit exponent, mantissa truncated to 7); fp16 is 1/5/10 with bias 15 — tiny range, hence loss-scaling.

**Fixed-point Qm.n** — the deterministic alternative: store `round(x * 2**n)` in an integer; value is $i / 2^n$. Add/sub are exact integer ops; multiply is `(a * b) >> n` with a double-width intermediate; nothing depends on FPU rounding modes or fused ops, so results are bit-identical across machines — which is why lockstep game networking, DSP pipelines, and audio codecs use it. Q16.16 in an i32 gives range $\pm 32768$ at resolution $2^{-16}$.

## framing a stream of numbers

- **Fixed-width records** (`struct` array): record $i$ lives at `i * calcsize(fmt)` — $O(1)$ random access, mmap-friendly, trivially seekable. Cost: every field pays max width; schema changes shift every offset.
- **Varint-delimited / length-prefixed**: compact for small values, but decode is inherently sequential and a corrupted length desynchronizes everything after it. The classic frame is `u32 LE length ‖ payload` — validate the length against a sane cap _before_ allocating, or one bad frame is a 4 GB allocation.
- Delimiter/sentinel framing (e.g. NUL-terminated) needs escaping the moment the delimiter can appear in the payload; length prefixes don't.

## hexdump: the debugging skill

`hexdump -C file`, `xxd`, or `od -A x -t x1z`. Canonical row = offset column, 16 bytes of hex split 8+8, ASCII panel (`0x20..0x7E` printable, `.` otherwise):

```
00000000  89 50 4e 47 0d 0a 1a 0a  00 00 00 0d 49 48 44 52  |.PNG........IHDR|
```

Reading skills to demonstrate: spot ASCII magic in the right panel; read an LE u32 by taking four hex pairs right-to-left (`0d 00 00 00` → wait, that's `0x0000000d` _BE_ — here `00 00 00 0d` is the PNG length, BE 13); confirm alignment/padding by eyeballing runs of `00`. When your codec is wrong, diff hexdumps of expected vs actual before re-reading code.

## gotchas and interviewer follow-ups

1. `to_bytes()` with no `byteorder` is big-endian, and negative numbers raise `OverflowError` unless `signed=True`.
2. `buf[i]` on `bytes` yields `int`; `buf[i:i+1]` yields `bytes`. Iterating `bytes` yields ints. In C, `char` may be signed: `b'\xff'[0]` is 255 in Python, `(char)0xFF` is likely −1.
3. Python has no logical right shift; `-1 >> 1 == -1` forever. Mask _before_ shifting when simulating unsigned: `(x & 0xFFFFFFFF) >> n`.
4. `struct` default prefix `@` inserts alignment padding — never parse wire formats without an explicit `<`/`>`.
5. `struct.unpack` requires the exact byte length; slice or use `unpack_from`.
6. "First byte on the wire" means _least_ significant for LE. Interviewers love asking you to dictate the byte sequence of `0xDEADBEEF` both ways.
7. In C, building a u32 from `char*` without casting through `unsigned char` sign-extends `0x80..0xFF` and corrupts the value (and `<< 24` on a promoted negative int is UB).
8. Advance your cursor only after a successful read; on EOF raise _without_ moving, so callers can recover — interviewers probe this in streaming-decoder follow-ups.
9. Floats: never `==` NaN, never `memcmp` floats (−0.0 vs 0.0, NaN payloads); compare bits when you mean bits.
10. Follow-up bank: detect host endianness at runtime (`sys.byteorder`; C union trick) · why is hashing a raw struct wrong (padding + endianness) · make your decoder streaming (buffer partial input, resume — see module on streaming decoders) · what does the compiler emit for your shift codec (`mov` on matching endian, `bswap` otherwise — shifts are free) · how does `mmap` + fixed-width records give you zero-copy random access.

## rapid-fire drills

| Q                                        | A                                                                      |
| ---------------------------------------- | ---------------------------------------------------------------------- |
| Bytes of `0xDEADBEEF` LE?                | `ef be ad de`                                                          |
| `sys.byteorder` on x86-64/Apple Silicon? | `'little'` (both)                                                      |
| struct format for LE u64?                | `'<Q'`                                                                 |
| `calcsize('<BI')` vs `calcsize('@BI')`?  | 5 vs 8 (alignment padding)                                             |
| float32 bits of 1.0?                     | `0x3F800000`                                                           |
| float64 exponent bias?                   | 1023 (f32: 127)                                                        |
| `(-1) & 0xFF` in Python?                 | 255                                                                    |
| Detect −0.0?                             | `math.copysign(1.0, x) < 0`                                            |
| `int.from_bytes(b'\x00\x01', 'little')`? | 256                                                                    |
| NaN test on raw f32 bits?                | exponent all ones and mantissa ≠ 0                                     |
| Largest integer float64 holds exactly?   | $2^{53}$                                                               |
| Swap u16 halves of u32?                  | `(x >> 16) \| ((x & 0xFFFF) << 16)`                                    |
| `htonl` on a LE host?                    | full byte swap (bswap32); identity on BE                               |
| hexdump ASCII panel rule?                | `0x20`–`0x7E` literal, everything else `.`                             |
| Smallest positive float32?               | subnormal $2^{-149}$, bits `0x00000001`                                |
| Why is `0.1 + 0.2 != 0.3`?               | 1/10 is a repeating binary fraction; both addends carry rounding error |
