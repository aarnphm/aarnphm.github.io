# cheatsheet

One page for the night before. Every row is derived in full in the kit named next to it. Define `M32 = (1 << 32) - 1` and `M64 = (1 << 64) - 1` at the top of every solve.

## idioms (01-bits, 02-unsigned-alignment)

| expression                                                         | effect                                                                                                                 |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `x & -x`                                                           | isolate lowest set bit                                                                                                 |
| `x & (x - 1)`                                                      | clear lowest set bit; `x > 0 and x & (x - 1) == 0` tests power of two                                                  |
| `(word >> off) & ((1 << w) - 1)`                                   | extract a w-bit field at offset off                                                                                    |
| `(word & ~(m << off)) \| ((v & m) << off)` with `m = (1 << w) - 1` | insert a field: clear, then OR                                                                                         |
| `x & ~(a - 1)` (equals `x & -a`)                                   | align_down to power-of-two a                                                                                           |
| `(x + a - 1) & ~(a - 1)`                                           | align_up; idempotent; writing `+ a` instead of `+ a - 1` bumps already-aligned inputs                                  |
| `(x & (a - 1)) == 0`                                               | is_aligned                                                                                                             |
| `((n << 1) ^ (n >> 63)) & M64`                                     | zigzag encode; the mask is mandatory in Python because `>>` on a negative is arithmetic over infinitely many sign bits |
| `(u >> 1) ^ -(u & 1)`                                              | zigzag decode; needs no mask, lands exactly in $[-2^{63}, 2^{63})$                                                     |
| `((x & M) ^ s) - s` with `s = 1 << (bits - 1)`                     | to_signed: XOR swaps the halves, subtract re-centers                                                                   |
| `x & ((1 << bits) - 1)`                                            | to_unsigned; exactly C's signed-to-unsigned conversion rule                                                            |
| `(a + b) & M64`, `(x << s) & M64`                                  | mask after every op that can exceed the width; Python ints never wrap                                                  |
| `(x & M32) >> n`                                                   | logical right shift: mask before shifting                                                                              |
| `-r & 31`                                                          | rotate shift amount that stays legal at r = 0 (`x >> 32` is UB in C)                                                   |
| `x \|= x >> 1; ... x \|= x >> 16; x + 1` after `x -= 1`            | next_pow2 by bit smear (or `1 << (x - 1).bit_length()`)                                                                |

## uvarint / LEB128 worked vectors (03-varint)

7 payload bits per byte, least-significant group first, bit 7 set means more bytes follow. Length is $\max(1, \lceil \operatorname{bitlen}(n)/7 \rceil)$, so k bytes iff $n < 2^{7k}$.

| n            | bytes (hex)       | note                                                        |
| ------------ | ----------------- | ----------------------------------------------------------- |
| 0            | `00`              | the encode loop must emit at least once (do-while shape)    |
| 1            | `01`              |                                                             |
| 127          | `7F`              | last 1-byte value                                           |
| 128          | `80 01`           | first 2-byte value                                          |
| 300          | `AC 02`           | `0x2C \| 0x80`, then 2; check: $44 + (2 \ll 7) = 300$       |
| 16383        | `FF 7F`           | last 2-byte value ($2^{14} - 1$)                            |
| 16384        | `80 80 01`        | zero groups mid-stream with the continuation bit are legal  |
| $2^{64} - 1$ | `FF` x9 then `01` | 10 bytes; a valid 10th byte is only `00` (overlong) or `01` |

Decode loop: `acc |= (b & 0x7F) << shift; shift += 7`, stop when `b < 0x80`. Three guards, three error classes, keep them distinct:

1. **Truncated.** Buffer ends with the continuation bit still set (`80` alone). Recoverable: a streaming decoder holds `(acc, shift)` and waits instead of raising.
2. **Too long.** Ten continuation bytes seen, or the 10th group exceeds 1 so the value reaches $2^{64}$. Unrecoverable even at end of buffer; do not mislabel it truncated.
3. **Overlong.** More than one byte and the final group is zero (`80 00` for 0). Reject when the bytes are hashed, used as keys, or signed; protobuf itself accepts overlong, so canonicality is a policy you must state.

## zigzag (03-varint)

$\operatorname{zz}(n) = 2n$ for $n \ge 0$ and $2|n| - 1$ for $n < 0$. A signed varint is `uvarint(zigzag(n))`.

| n            | zz(n)        |
| ------------ | ------------ |
| 0            | 0            |
| $-1$         | 1            |
| 1            | 2            |
| $-2$         | 3            |
| 2            | 4            |
| $2^{63} - 1$ | $2^{64} - 2$ |
| $-2^{63}$    | $2^{64} - 1$ |

Follow-up they like to ask: protobuf `int64` (two's complement as u64, no zigzag) spends 10 bytes on $-1$; `sint64` (zigzag) spends 1.

## struct (04-byte-streams)

| char    | type     | size | char    | type     | size |
| ------- | -------- | ---- | ------- | -------- | ---- |
| `b`/`B` | i8/u8    | 1    | `h`/`H` | i16/u16  | 2    |
| `i`/`I` | i32/u32  | 4    | `q`/`Q` | i64/u64  | 8    |
| `f`     | binary32 | 4    | `d`     | binary64 | 8    |

Prefixes: `<` LE, `>` BE, `!` network (= BE), all with standard sizes and no padding. `=` is native order, standard sizes, no padding. `@` is the default and inserts native C padding: `calcsize('<BI')` is 5 while `calcsize('@BI')` is 8. Always write `<` or `>` for wire data. `unpack` demands an exact-length buffer; `unpack_from(fmt, buf, off)` does not. Errors are `struct.error`, not `ValueError`. Related trap: since 3.11 `int.to_bytes`/`from_bytes` arguments are optional and byteorder defaults to `'big'`, so pass it explicitly.

## IEEE 754 (04-byte-streams)

```
float32: [ s:1 ][ e:8  bias 127  ][ m:23 ]    value = (-1)^s * 1.m * 2^(e-127)
float64: [ s:1 ][ e:11 bias 1023 ][ m:52 ]    value = (-1)^s * 1.m * 2^(e-1023)
```

Exponent field 0 means zero (m = 0) or subnormal ($0.m \cdot 2^{-126}$, no hidden 1). All-ones exponent means inf (m = 0) or NaN (m nonzero). `1.0f` is `0x3F800000`. Bit cast both ways: `struct.unpack('<I', struct.pack('<f', x))[0]` and `struct.unpack('<f', struct.pack('<I', bits))[0]`.

## C vs Python (01, 02, 05)

| topic                | C                                                                | Python                                                 |
| -------------------- | ---------------------------------------------------------------- | ------------------------------------------------------ |
| `>>` on negatives    | implementation-defined (arithmetic on every mainstream compiler) | arithmetic over infinite sign bits: `-1 >> 63 == -1`   |
| signed `<<` overflow | UB; shift the unsigned copy                                      | fine, the int grows                                    |
| shift $\ge$ width    | UB                                                               | fine                                                   |
| unsigned wrap        | free, mod $2^n$ by definition                                    | never wraps; you must `& M`                            |
| `u8 << 24`           | promotes to signed `int` first, then sign-extends on widening    | no promotion                                           |
| `u16 * u16`          | promoted signed multiply can overflow: UB                        | fine                                                   |
| `-1 < 1u`            | false ($-1$ converts to a huge unsigned)                         | true                                                   |
| reading bytes        | `char` may be signed; use `uint8_t` at the decode boundary       | iterating `bytes` yields ints 0..255                   |
| varint accumulate    | `(uint64_t)(b & 0x7F) << shift`, the cast is mandatory           | unbounded; the 64-bit cap is policy you write yourself |

## clarifying questions for minutes 0 to 5

1. Signed or unsigned? If signed, zigzag or fixed-width two's complement?
2. Max magnitude: u32, u64, or arbitrary precision?
3. Output type: bytes/bytearray, or a string (full 0..255 alphabet or printable only)?
4. Endianness: required byte order, or mine to define?
5. Malformed input on decode: raise, return partial, or skip? Is input trusted?
6. Is canonical form required, i.e. must I reject overlong encodings like `80 00`?
7. One-shot buffer, or chunks (stateful streaming decoder)? Can a value straddle a chunk boundary?
8. Growable output buffer, or fixed pre-allocated (embedded flavor)?
9. Data distribution mostly small ints (varint wins) or uniformly large (fixed width wins)?
10. May I use `struct` and stdlib codecs, or is hand-rolling the point? (It is. Ask anyway.)

Then state the API and say the roundtrip property out loud, `decode(encode(xs)) == xs`, and write that assert before anything else.

## vectors worth memorizing

- Roundtrip set: `[]`, `[0]`, `[127, 128]`, `[255, 256]`, `[2**64 - 1]`.
- Malformed set: `80` (truncated), `80` x10 then `01` (too long), `80 00` (overlong).
- Delta stack: `[1000, 1005, 1004, 1010]` encodes to `D0 0F 0A 01 0C`, 5 bytes against 32 raw, a 6.4:1 cut. Deltas wrap mod $2^{64}$, so int64 min followed by int64 max round-trips through a 1-byte wrapped delta.
- UTF-8: `é` U+00E9 is `C3 A9`; `€` U+20AC is `E2 82 AC`; U+1F600 is `F0 9F 98 80`. Width thresholds 0x80 / 0x800 / 0x10000 / 0x10FFFF. Reject lead bytes C0, C1, F5+; surrogates U+D800..DFFF (after `ED`, continuation above `9F`); overlongs (after `E0`, continuation below `A0`; after `F0`, below `90`); beyond U+10FFFF (after `F4`, above `8F`).
- Alignment: `align_up(13, 8) = 16`, `align_up(16, 8) = 16`, `align_up(0, 8) = 0`.
- Wire-size argument to say at the wrap: telemetry-like data averages 1 to 2 varint bytes per int against 4 to 8 fixed, a 2-4x cut, which is why protobuf, Kafka, and Datadog tracers use it.

## sketches (07-stream-algorithms)

| structure       | space                                                                | guarantee                                                                      | one-line use                                                                    |
| --------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| reservoir-k     | $k$ items                                                            | every item kept with probability exactly $k/n$                                 | uniform sample, unknown length: $j = \text{randrange}(i+1)$, replace if $j < k$ |
| Misra-Gries     | $k-1$ counters                                                       | undercount $\le n/k$; everything with $f_x > n/k$ survives                     | heavy hitters above $1/k$ of the stream; second pass makes it exact             |
| Count-Min       | $\lceil e/\varepsilon \rceil \times \lceil \ln(1/\delta) \rceil$ u32 | overestimate only, excess $\le \varepsilon N$ w.p. $1 - \delta$; min over rows | point frequency queries; top-k needs a heap kept beside it                      |
| Bloom           | $\approx 9.6$ bits/element at 1% FP, $k = 7$                         | false positives only, never false negatives                                    | membership; no deletes, no resize                                               |
| HLL             | $2^{14}$ 6-bit registers = 12 KiB                                    | relative error $1.04/\sqrt{m} \approx 0.8\%$                                   | distinct count; union = register-wise max, refuse intersection                  |
| two-heap median | all $n$ values                                                       | exact; add $O(\log n)$, read $O(1)$                                            | running median; sizes within 1, even count = mean of tops                       |

## queueing (08-queueing)

Little, distribution-free, any boundary: $L = \lambda W$. 500 req/s at 100 ms means 50 in flight. M/M/1 sojourn is $W = E[S]/(1-\rho)$:

| $\rho$   | 0.50 | 0.80 | 0.90 | 0.95 | 0.99 |
| -------- | ---- | ---- | ---- | ---- | ---- |
| $W/E[S]$ | 2    | 5    | 10   | 20   | 100  |

Pollaczek-Khinchine: $W_q = \lambda E[S^2] / (2(1-\rho))$ — service variance, not the mean, drives the wait; 1% whales can 40x it at half-idle. Kingman: $W_q \approx \frac{\rho}{1-\rho} \cdot \frac{C_a^2 + C_s^2}{2} \cdot E[S]$, variability times utilization times time; $C_a^2 = 1$ recovers P-K, $C_a^2 = C_s^2 = 0$ gives zero (D/D/1 never queues).

- Token bucket $(r, B)$: lazy refill, two floats per key; policer, zero delay, admits burst $\le B$ then sustained $r$.
- Leaky bucket as queue: FIFO drained at exactly $r$; shaper, adds delay, perfectly paced output.
- Sliding-window counter: $\text{prev} \cdot (1 - e/\text{window}) + \text{curr}$, two ints per key; exact per aligned window, up to ~2x limit through a true trailing window.

Fan-out tail: $P(\text{any leg slow}) = 1 - p^n$. 100 leaves at per-leaf p99: $0.99^{100} \approx 37\%$ of requests dodge every slow leg, so the leaf p99 is roughly the root median. Hedge above p95 with a capped budget.
