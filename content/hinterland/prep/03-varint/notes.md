# Varints: LEB128, zigzag, VLQ, PrefixVarint, and the rest of the family

## (a) Core mental model

A varint spends payload bits to describe its own length. Every design in the family answers three questions:

1. **Where do the length/tag bits live** — one continuation bit per byte (LEB128, VLQ, SQLite) or all tag bits packed into the first byte (UTF-8, PrefixVarint, group varint's shared tag byte).
2. **Group significance order** — least-significant group first (LEB128) or most-significant first (VLQ, SQLite).
3. **How the top of the range is capped** — a constrained final byte (LEB: 10th byte $\le$ 0x01 for u64) or a raw-tail trick (SQLite byte 9, PrefixVarint 0xFF prefix).

### LEB128 / protobuf uvarint (the one you will be asked to code)

Split $n$ into 7-bit groups from the least-significant end. Emit one byte per group, least-significant group first. Set bit 7 (0x80) on every byte except the last. Decoder reads until it sees a byte $<$ 0x80.

Density: 7 payload bits per 8-bit byte, so $k$ bytes hold $7k$ payload bits and $\text{len}(n) = \max(1, \lceil \operatorname{bitlen}(n)/7 \rceil)$. Size thresholds: 1 byte iff $n < 2^7$, 2 bytes iff $n < 2^{14}$, ..., $k$ bytes iff $n < 2^{7k}$.

**Worked: 300 → `AC 02`**

```text
300 = 0b1_0010_1100                 (9 bits)
low 7 bits : 0101100 = 0x2C   more coming -> 0x2C | 0x80 = 0xAC
next 7 bits: 0000010 = 0x02   done        -> 0x02
wire: AC 02
check: 0x2C + (0x02 << 7) = 44 + 256 = 300
```

**Worked: 16384 → `80 80 01`**

$16384 = 2^{14}$, groups LSB-first are 0000000, 0000000, 0000001:

```text
byte 1: 0x00 | 0x80 = 0x80          ("zero payload, continue")
byte 2: 0x00 | 0x80 = 0x80
byte 3: 0x01                        (bit 14 lands at shift 14)
wire: 80 80 01
```

Zero groups with the continuation bit set are legal and necessary in the middle; only a _trailing_ zero group is redundant (see overlong, below).

### Maximum lengths, and why

- u32: $\lceil 32/7 \rceil = 5$ bytes. Bytes 1–4 carry 28 bits; byte 5 carries bits 28–34, but u32 only has bits 28–31, so a valid final byte is $\le$ 0x0F (3 spare bits must be zero).
- u64: $\lceil 64/7 \rceil = 10$ bytes. Bytes 1–9 carry 63 bits; byte 10 carries bit 63 only, so a valid 10th byte is 0x00 or 0x01 — and 0x00 would be overlong, so a canonical 10-byte u64 varint always ends `01`. A lenient decoder that ignores this computes values $\ge 2^{64}$ (silently huge in Python, silently wrapped in C). Go's `binary.Uvarint` reports overflow for both "11th group needed" and "10th byte > 1".

$2^{64}-1$ → `FF FF FF FF FF FF FF FF FF 01` (nine 0x7F groups = 63 ones, then bit 63).

## (b) Idioms and techniques

### Encode loop — emit-at-least-once

The trap: $n = 0$ must emit exactly one byte `00`, so the loop body must run before the exit test (do-while shape).

```python
def encode_uvarint(n: int) -> bytes:
  out = bytearray()
  while True:
    g = n & 0x7F
    n >>= 7
    if n:
      out.append(g | 0x80)
    else:
      out.append(g)
      return bytes(out)
```

```c
size_t encode_uvarint(uint64_t n, uint8_t out[10]) {
    size_t i = 0;
    do {
        uint8_t g = n & 0x7F;
        n >>= 7;
        out[i++] = g | (n ? 0x80 : 0);
    } while (n);
    return i;
}
```

### Decode loop — accumulator + shift + three guards

```python
def decode_uvarint(buf: bytes, offset: int = 0) -> tuple[int, int]:
  value, shift, pos = 0, 0, offset
  while True:
    if pos - offset == 10:
      raise ValueError('too long')  # guard BEFORE reading byte 11
    if pos >= len(buf):
      raise ValueError('truncated')
    b = buf[pos]
    pos += 1
    value |= (b & 0x7F) << shift
    shift += 7
    if not b & 0x80:
      break
  if value > (1 << 64) - 1:
    raise ValueError('too long (u64 overflow)')  # 10th group was > 1
  if pos - offset > 1 and b == 0:
    raise ValueError('overlong')  # trailing zero group
  return value, pos - offset
```

Decode trace for `AC 02`:

| byte | group | shift | accumulator    |
| ---- | ----- | ----- | -------------- |
| AC   | 2C    | 0     | 44             |
| 02   | 02    | 7     | 44 + 256 = 300 |

Decode trace for `80 80 01`: acc stays 0, 0, then $1 \ll 14 = 16384$.

C version — the single most common C varint bug is the missing widening cast:

```c
uint64_t value = 0; unsigned shift = 0;
for (;;) {
    uint8_t b = *p++;
    value |= (uint64_t)(b & 0x7F) << shift;   /* without the cast, (b & 0x7F)
                                                 promotes to 32-bit int and
                                                 << 35 is UB */
    if (!(b & 0x80)) break;
    shift += 7;
}
```

Equivalent loop-limit guards, pick one and defend it: byte count `== 10`, `shift >= 64` (actually `shift >= 63` for the strict 10th-byte rule), or a post-loop `value > UINT64_MAX` check (only possible in Python where ints don't wrap).

### The three malformed classes (a strict decoder distinguishes all three)

1. **Truncated** — buffer ends while the last byte read still had the continuation bit set (empty buffer included). Recoverable: a streaming decoder reports "need more bytes" and waits. `80` alone is truncated, because `80 01` (= 128) is a legal completion.
2. **Too long** — unrecoverable regardless of future input: ten continuation bytes have been seen (an 11th group can never be valid for u64), or ten groups terminate but the 10th group exceeds 1 (value $\ge 2^{64}$). `80`×10 + `01` is too long. Report it even at end-of-buffer; do not mislabel it truncated.
3. **Overlong / non-canonical** — terminates, fits u64, but a shorter encoding of the same value exists. Criterion: more than one byte consumed and the final group is zero. `80 00` decodes to 0; canonical 0 is `00`. Minimality iff last group $\ne 0$ (or single byte).

Why canonicality matters — the interviewer follow-up you should volunteer:

- **Signatures / content addressing**: if `hash(bytes)` stands for `hash(value)`, two encodings of one value break dedup and let an attacker mutate bytes without changing meaning (malleability; Bitcoin Core has rejected non-minimal CompactSize with "non-canonical ReadCompactSize()" since 2012, and pre-SegWit transaction malleability is the war story).
- **Map/cache keys**: a cache keyed on raw encoded bytes treats `00` and `80 00` as different keys for the same logical key.
- **Consensus**: one lenient node and one strict node hash the "same" structure differently → fork. CBOR's deterministic encoding (RFC 8949 §4.2) mandates shortest-form integers for exactly this reason.
- **Protobuf's own answer**: parsers _accept_ overlong varints; canonicalization is nobody's job. Consequence: protobuf bytes are not canonical, so never hash or sign a serialized proto and expect cross-implementation stability (per-library "deterministic serialization" flags are explicitly not cross-library).
- **The counter-example that shows it's a policy choice**: WASM deliberately allows zero-padded LEB128 up to the max length ($\lceil N/7 \rceil$ bytes, spare bits in the final byte must be zero) so linkers can patch fixed-width 5-byte slots in place. LLVM emits padded LEBs for relocations.

### Signed integers: two strategies

**(a) Two's-complement-as-u64 (protobuf `int64`, also `int32`).** Reinterpret the signed value as u64 and uvarint it. $-1 = 2^{64}-1$ → `FF FF FF FF FF FF FF FF FF 01`: **every negative value costs 10 bytes**. This includes negative `int32`, which is sign-extended to 64 bits first (wire compatibility between int32/int64 fields). Classic screen gotcha: "how many bytes is $-1$ as protobuf int64?" — 10.

**(b) Zigzag (protobuf `sint64`).** Interleave signs so small magnitudes get small codes:

$$\operatorname{zz}(n) = (n \ll 1) \oplus (n \gg 63) \quad\text{in 64-bit two's-complement semantics}$$

The arithmetic right shift broadcasts the sign bit into an all-ones/all-zeros mask; XOR conditionally complements. Equivalently $\operatorname{zz}(n) = 2n$ for $n \ge 0$ and $2|n| - 1$ for $n < 0$.

| $n$        | $\operatorname{zz}(n)$ |
| ---------- | ---------------------- |
| 0          | 0                      |
| $-1$       | 1                      |
| 1          | 2                      |
| $-2$       | 3                      |
| 2          | 4                      |
| $2^{63}-1$ | $2^{64}-2$             |
| $-2^{63}$  | $2^{64}-1$             |

Decode: $(u \gg 1) \oplus -(u \wedge 1)$ — `-(u & 1)` is 0 or all-ones, again a conditional complement.

**Python vs C, explicitly:**

```python
MASK64 = (1 << 64) - 1


def zigzag_encode(n):  # n in [-2**63, 2**63)
  return ((n << 1) ^ (n >> 63)) & MASK64  # mask is mandatory


def zigzag_decode(u):  # u in [0, 2**64)
  return (u >> 1) ^ -(u & 1)  # no mask needed
```

Python ints are unbounded and `>>` on a negative is an arithmetic shift over _infinitely many_ sign bits (`-1 >> 63 == -1`), so the XOR flips infinitely many bits; `& MASK64` truncates back to the 64-bit result. Decode needs no mask: for $u \in [0, 2^{64})$ the formula lands exactly in $[-2^{63}, 2^{63})$.

```c
uint64_t zz_enc(int64_t n) { return ((uint64_t)n << 1) ^ (uint64_t)(n >> 63); }
int64_t  zz_dec(uint64_t u) { return (int64_t)((u >> 1) ^ (~(u & 1) + 1)); }
```

Do the `<<` on the _unsigned_ copy (signed left-shift overflow is UB in C); do the `>>` on the _signed_ value (you want the arithmetic shift; for signed negatives `>>` is implementation-defined in C but arithmetic on every mainstream compiler, and mandated arithmetic since C++20).

Choose `sint*` when the field is frequently negative; `int*` is fine when negatives are rare; `fixed64` when values are uniformly large.

### VLQ (MIDI flavor): most-significant group FIRST

Same 7-bit groups, same 0x80-continues convention, opposite group order. MIDI delta-times are the canonical spec (max `0F FF FF FF` → `FF FF FF 7F`).

```text
LEB128:  300 -> AC 2C? no: AC 02      (low group first)
VLQ:     300 -> 82 2C                 (high group first: groups [2, 0x2C])
LEB128:  128 -> 80 01
VLQ:     128 -> 81 00                 (the mirror pair)
```

Decode is the pretty direction — no shift counter: `value = (value << 7) | (b & 0x7F)` per byte. Encode must know the group count up front (extract groups LSB-first, then reverse). The overlong class mirrors LEB: a _leading_ `80` byte (zero top group) is redundant, whereas in LEB it is the _trailing_ `00`. Terminology trap: "VLQ" gets used loosely — MIDI is MSB-first; DWARF, protobuf, and WASM all use LEB128 (LSB-first). Git's ofs-delta varint is MSB-first _with a +1 bias per continuation step_, which makes every byte string decode to a distinct value: the overlong class is designed out of existence (bijective numeration) and range per length shifts up.

### UTF-8 viewed as a prefix code

| length | byte 1     | continuation bytes | payload bits |
| ------ | ---------- | ------------------ | ------------ |
| 1      | `0xxxxxxx` | —                  | 7            |
| 2      | `110xxxxx` | `10xxxxxx`         | 11           |
| 3      | `1110xxxx` | `10xxxxxx` ×2      | 16           |
| 4      | `11110xxx` | `10xxxxxx` ×3      | 21           |

The count of leading one bits in byte 1 announces the total length: length dispatch is one `clz`, not a per-byte loop. Continuation bytes carry their own `10` tag, which buys **self-synchronization**: from an arbitrary offset you find the next character boundary within 3 bytes (seek into the middle of a file, recover after corruption). The price is density — 6 payload bits per continuation byte, 21 bits max in 4 bytes where a tag-free scheme would carry 28. Overlong UTF-8 is a real CVE class: `C0 80` is an overlong NUL that smuggles past validators that scan for `00`; RFC 3629 requires rejection (Java's "modified UTF-8" uses `C0 80` internally, on purpose).

### PrefixVarint: UTF-8's tag placement without the per-byte tags

Keep "length from byte 1", make continuation bytes raw 8-bit payload:

```text
L in 1..8: byte 1 = (L-1) ones, one zero, then the top 8-L payload bits;
           then L-1 raw payload bytes.            capacity 7L bits
L = 9:     byte 1 = 0xFF, then 8 raw big-endian bytes.  capacity 64 bits
```

Same density as LEB128 through 56 bits ($7L$ bits in $L$ bytes), and u64 caps at **9 bytes vs LEB's 10** because the tag costs 1–8 bits total instead of 1 bit per byte. Decode: `clz` of `~byte1` gives the length, then one unaligned 8-byte load + mask — **one data-dependent branch per value** (or none, table-driven) versus LEB's conditional branch per _byte_, which mispredicts constantly on mixed-size streams. Cost: not self-synchronizing, and the fast path wants 8 bytes of overread headroom past the buffer end.

### SQLite's varint (a third answer to the cap question)

Big-endian groups like VLQ, hard-capped at 9 bytes: bytes 1–8 contribute their low 7 bits, and if a 9th byte is reached it contributes **all 8 bits** ($8 \times 7 + 8 = 64$ exactly). No wasted 10th byte, no 1-useful-bit tail; the decoder's loop bound is the fixed number 9.

### Google group varint

Batch four u32s behind one shared tag byte: four 2-bit fields give each value's byte length (1–4), then 4–16 data bytes. The tag indexes a 256-entry table of shuffle masks + total length → decode is table lookup + unaligned loads (`pshufb` in the SIMD version), zero per-value branches. Worst case 17 bytes per 4 values vs 20 for LEB u32s; loses on all-tiny data (minimum 5 bytes vs 4). This is the Jeff Dean search-postings trick and the ancestor of StreamVByte (which splits tag and data streams for cleaner SIMD).

### When each design wins

LEB128 wins on ubiquity and streaming simplicity (protobuf, WASM, DWARF, gRPC frame counts): byte-at-a-time, no overread, trivial in any language — and pays with a per-byte unpredictable branch, so scalar decode saturates around hundreds of MB/s. PrefixVarint spends the same space (identical $7L$ capacity through 56 bits, one byte shorter at the top) and moves all length information into one `clz`, so it wins when decode latency/branch predictability matters and you control both ends. Group varint/StreamVByte win bulk throughput on u32 columns — roughly an order of magnitude over scalar LEB decode — at 2 tag bits + minimum 1 byte per value, so they lose on sparse tiny values and need batching. SQLite's design wins when you want a hard small max length and a clean 64-bit cap. Zigzag is orthogonal: layer it under any of these when signed values cluster near zero. And when values are uniformly large ($\ge 2^{56}$), fixed-width 8-byte little-endian beats them all on both size and speed — varints are a bet on smallness.

## (c) Gotchas and interviewer follow-ups

1. **Negative input to the encode loop never terminates.** Python: `n >>= 7` on a negative approaches $-1$, not 0 (`-1 >> 7 == -1`), so `while n:` loops forever appending `FF`. C signed: same fixed point under arithmetic shift. State the domain up front and either raise or mask to u64 (`n & MASK64` is exactly protobuf int64 semantics).
2. **$n = 0$ must emit `00`.** A `while n:` pre-test loop emits nothing. Do-while shape.
3. **Truncated vs too-long are different protocol answers.** Truncated = "need more bytes" (resumable); too long = fatal even at EOF. After ten continuation bytes no suffix is valid — report too long, not truncated.
4. **Python has no unsigned; C has no unbounded.** Python: mask after zigzag encode, mask if you emulate wrapping. C: cast before shifting — `(uint64_t)(b & 0x7F) << shift`; `b & 0x7F` is `int`, and `int << 35` is UB.
5. **`char` signedness.** Buffers must be `uint8_t*`/`unsigned char*`. With signed `char`, `*p >= 0x80` is always false (range $[-128, 127]$) and sign extension corrupts `value |= *p << shift`.
6. **Return-value discipline.** Decide `(value, bytes_consumed)` vs `(value, new_offset)` and never mix them; the off-by-offset in the "decode a whole stream" follow-up is the most common live-coding bug. `consumed` composes: `offset += consumed`.
7. **Length-prefix DoS.** A uvarint length prefix like `FF FF FF FF 0F` claims 4 GiB. Validate against remaining buffer and a frame cap _before_ allocating.
8. **Protobuf trivia they probe:** the field header is itself a varint, `(field_number << 3) | wire_type`, wire type 0 = varint — so field 1 = 150 serializes as `08 96 01`. Negative `int32` costs 10 bytes; `sint32` fixes it; parsers accept overlong varints, hence no canonical proto bytes.
9. **"Little-endian" here means group order, not machine endianness.** Varints are byte streams; there is no host-order dependence anywhere. Saying this unprompted signals you actually know what endianness is.
10. **Python indexing:** `buf[i]` on `bytes` yields an `int` 0–255 (no `ord`); `buf[i:i+1]` yields `bytes`; slices copy, `memoryview(buf)` doesn't — mention it if they push on zero-copy stream decoding.
11. **Guard equivalences:** `shift >= 64`, byte-count $= 10$, and post-hoc `value >= 2**64` catch overlapping but not identical bad inputs; the strict rule is "at most 10 groups AND 10th group $\le 1$".
12. **Meta-move that scores:** before coding, state the domain (u64), the return convention, and the three error classes. Interviewers grade the error taxonomy more than the happy path.

## (d) Rapid-fire drills

1. Encode 150 → `96 01` (protobuf docs' `08 96 01` is field 1 = 150).
2. Decode `E5 8E 26` → $101 + 14 \cdot 2^7 + 38 \cdot 2^{14}$... compute: `0x65 | 0x0E<<7 | 0x26<<14` = 624485.
3. Max encoded length of u32 / u64? → 5 / 10 bytes ($\lceil 32/7 \rceil$, $\lceil 64/7 \rceil$).
4. Largest 1-byte / 2-byte values? → 127 / 16383.
5. Wire size of $-1$ as protobuf `int64` vs `sint64`? → 10 bytes vs 1 byte.
6. zigzag$(-3)$? → 5. zigzag$(2^{63}-1)$? → $2^{64}-2$. unzigzag$(7)$? → $-4$.
7. Is `FF 00` canonical? → No: overlong encoding of 127 (trailing zero group).
8. Valid values of the 10th byte of a u64 LEB varint? → 0x00 or 0x01, and 0x00 is overlong, so canonically 0x01.
9. MIDI VLQ of 128? → `81 00` (LEB is `80 01`).
10. UTF-8 lead byte `0xE2` → how long is the character? → 3 bytes.
11. PrefixVarint first byte `0xF0` → total length? → 5 bytes (4 leading ones).
12. Decode `80` ×10 then `01`? → error: too long (11th group needed).
13. Stream hands you `80 80` then blocks — what do you report? → truncated / need more bytes (resumable, not fatal).
14. Encode $2^{63}$ → `80 80 80 80 80 80 80 80 80 01`.
15. LEB bytes for 1,000,000? → 3 ($2^{14} \le 10^6 < 2^{21}$).
16. How many distinct values fit in $\le 2$ LEB bytes? → $2^{14} = 16384$.
17. SQLite varint max length, and byte 9's contribution? → 9 bytes; byte 9 carries 8 raw bits.
18. Group varint: bytes to encode four u32s, worst/best? → 17 / 5 (1 tag + 4×4 or 4×1).
19. Spot the C bug: `value |= (buf[i] & 0x7f) << shift;` with `uint64_t value` → missing `(uint64_t)` cast on the shifted operand; UB once `shift >= 32`.
20. Why is `80 00` worse than a wasted byte? → two byte-strings for one value: breaks hashing/signatures/map keys; strict decoders must reject.
