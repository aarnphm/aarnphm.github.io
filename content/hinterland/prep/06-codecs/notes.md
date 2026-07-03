# Classic codecs

The concrete skins a 60-minute encoding screen dresses the same ideas in: UTF-8, base64, hex, RLE, delta+zigzag+varint, bitpacking, checksums. One skin per interview; the accumulator loop underneath never changes.

## Core mental model

Every codec here is a bit/byte-accumulator loop: pull units in, validate at each boundary, push units out. Four axes distinguish them:

| codec       | unit mapping           | length signaled by               | canonical form                         |
| ----------- | ---------------------- | -------------------------------- | -------------------------------------- |
| UTF-8       | codepoint → 1–4 bytes  | lead-byte prefix declares length | shortest form only (overlongs illegal) |
| base64      | 3 bytes → 4 sextets    | `=` padding to mod-4             | yes, if trailing bits forced to 0      |
| hex         | byte → 2 nibbles       | trivial 1:2                      | case choice                            |
| RLE pairs   | run → `[count][value]` | fixed 2-byte pairs               | greedy-maximal runs                    |
| uvarint     | uint64 → 1–10 bytes    | continuation MSB                 | shortest form (parsers rarely enforce) |
| bitpack     | k bits → shared bytes  | external `(k, n)`                | zero pad bits                          |
| Fletcher-16 | byte → 2 running sums  | n/a                              | n/a                                    |

Decoders are where screens are won. Every decoder is a validator, and the failure inputs (truncated, overlong, out-of-alphabet, out-of-range) are the actual test. State the format precisely first (bit order, error policy, canonical form), then write the error paths before the happy path.

### Python vs C, keep this loaded

| operation              | Python                                                 | C                                                                                                |
| ---------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| byte access            | `buf[i]` is already `int` in 0–255                     | `char` may be signed: `buf[i] << 12` sign-extends garbage into high bits. Use `const uint8_t *`  |
| right shift            | always arithmetic (ints are infinite two's complement) | unsigned → logical; signed negative → implementation-defined (arithmetic on every real compiler) |
| left shift of negative | fine, exact                                            | UB. Cast first: `((uint64_t)n << 1) ^ (uint64_t)(n >> 63)` is how protobuf writes zigzag         |
| shift by ≥ width       | fine                                                   | UB: `1ULL << 64` is not 0. Guard `k == 64` when building masks                                   |
| overflow               | never happens; emulate wrap with `& ((1 << 64) - 1)`   | unsigned wraps mod $2^w$ (defined); signed overflow UB                                           |
| `-1 % 255`             | `254`                                                  | `-1` — keep checksum accumulators unsigned                                                       |
| logical right shift    | mask first: `(x & MASK64) >> n`                        | `(uint64_t)x >> n`                                                                               |

## UTF-8 from scratch

Four byte templates; `x` bits carry the codepoint, most significant bits in the lead byte:

```
U+0000..U+007F     0xxxxxxx                              7 payload bits
U+0080..U+07FF     110xxxxx 10xxxxxx                    11 payload bits
U+0800..U+FFFF     1110xxxx 10xxxxxx 10xxxxxx           16 payload bits
U+10000..U+10FFFF  11110xxx 10xxxxxx 10xxxxxx 10xxxxxx  21 payload bits
```

Each length is REQUIRED to cover only what the previous cannot: 1 byte for cp < 0x80, 2 bytes for cp < 0x800, 3 bytes for cp < 0x10000, 4 bytes up to 0x10FFFF. Encoding `/` as C0 AF fits the 2-byte template bit-wise and is still illegal: overlong. Forbidden outright:

- surrogates U+D800–U+DFFF (UTF-16 pairing machinery, ill-formed in UTF-8),
- lead bytes C0, C1 (can only start overlong 2-byte forms),
- F5–FF (would encode > U+10FFFF, or are the 5/6-byte forms RFC 3629 amputated in 2003).

Worked: U+20AC (the euro sign) = `0010 0000 1010 1100` → 16 bits → 3 bytes → split 4|6|6 → `1110`+`0010`, `10`+`000010`, `10`+`101100` → **E2 82 AC**. Boundary vectors to memorize: 0x7F → `7F`; 0x80 → `C2 80`; 0x7FF → `DF BF`; 0x800 → `E0 A0 80`; 0xFFFF → `EF BF BF`; 0x10000 → `F0 90 80 80`; 0x10FFFF → `F4 8F BF BF`.

Decoder per sequence, in this order:

1. classify lead: `< 0x80` ASCII; `(b & 0xC0) == 0x80` lone continuation, reject; `> 0xF7` reject
2. truncation: all continuation bytes present?
3. each continuation satisfies `(c & 0xC0) == 0x80`
4. assemble: `cp = (cp << 6) | (c & 0x3F)`
5. value checks on assembled cp: overlong (`cp < min_for_length`), surrogate, `cp > 0x10FFFF`

Assembly, both languages:

```python
cp = (b0 & 0x0F) << 12 | (b1 & 0x3F) << 6 | (b2 & 0x3F)
```

```c
uint32_t cp = ((uint32_t)(p[0] & 0x0F) << 12)
            | ((uint32_t)(p[1] & 0x3F) << 6)
            |  (uint32_t)(p[2] & 0x3F);      /* p is unsigned char*, or you ship a bug */
```

Production decoders fold steps 2–5 into constraints on the second byte (E0 → second must be A0–BF, ED → 80–9F, F0 → 90–BF, F4 → 80–8F); Hoehrmann's DFA decoder is the canonical branchless table version, simdutf the vectorized one. ASCII fast path is SWAR: load 8 bytes, `if ((chunk & 0x8080808080808080) == 0)` copy the whole chunk.

## Base64

3 bytes = 24 bits = 4 sextets. Alphabet: `A–Z` (0–25), `a–z` (26–51), `0–9` (52–61), `+` (62), `/` (63); urlsafe swaps `+ /` for `- _`. Output length is always $4\lceil n/3 \rceil$ with padding.

Worked: `"Man"` = 4D 61 6E = `010011 010110 000101 101110` = 19, 22, 5, 46 → **TWFu**.

Tail rules: 1 leftover byte → 2 chars + `==` (the 4 dangling bits of the 2nd char are zero); 2 leftover bytes → 3 chars + `=` (2 dangling bits zero). Encoded length ≡ 1 (mod 4) is unconditionally invalid, padded or not: a lone sextet cannot produce a byte.

Decode validation, in order:

1. length mod 4 == 0
2. `=` legal only as the final one or two chars; `===` never
3. every remaining char through a 256-entry reverse table (unmapped → reject; strict decoders reject whitespace too, while `base64.b64decode` silently discards it by default — know which you are writing)
4. optionally, canonicality: `TQ==` and `TR==` both decode to `M` if you ignore the dangling bits; systems that sign, dedup, or content-address the encoded form must reject non-zero trailing bits or re-encode and compare

## Hex, manually

```python
HEX = '0123456789abcdef'
enc = ''.join(HEX[b >> 4] + HEX[b & 0x0F] for b in data)
```

Decode via reverse table (256 entries with a poison value in C, dict in Python); reject odd length and non-hex chars. Worked: `0xBE` → `"be"`; decode `"7f"` → `rev['7'] << 4 | rev['f']` = 0x7F. `bytes.fromhex` skips ASCII spaces; replicate or don't, but say so out loud.

## RLE, three ways

1. **Naive count–char, textual** (`"AAAB"` → `"3A1B"`): breaks on digit bytes in the data, unbounded counts. An interview trap, not a format.
2. **Byte pairs `[count][value]`**, count in 1..255, maximal runs split greedily: 300 × `A` → **FF 41 2D 41**. Round-trips arbitrary bytes; worst case 2x expansion (all runs length 1); pays off iff mean run length > 2.
3. **Escape/literal-run (PackBits, TIFF/Apple)**: control byte c in 0..127 → copy next c+1 literal bytes; 129..255 → repeat next byte 257−c times; 128 → no-op. Worst-case overhead 1/128 instead of 2x — this is the standard answer to "your RLE doubles random data".

Where RLE actually wins: bitmap indexes, sparse masks, definition/repetition levels in Parquet (its RLE + bit-pack hybrid), quantized sensor streams, fax (ITU T.4), BWT output (the stage that makes bzip2 work). The shared property: long runs of identical symbols, plus a need for O(1) state and branch-trivial decode.

## Capstone: delta → zigzag → uvarint

The timestamp-compression stack. Lineage: protobuf `sint64` wire format (zigzag), Kafka v2 record batches, Lucene posting-list gaps, Parquet DELTA_BINARY_PACKED, Gorilla → Prometheus/InfluxDB.

- **delta**: sorted-ish sequences have small gaps; store `x[i] - x[i-1]` with `x[-1]` taken as 0 (state your convention; storing the head raw is the other common one).
- **zigzag**: $z(n) = 2n$ for $n \ge 0$, $z(n) = -2n - 1$ for $n < 0$. Bit form `(n << 1) ^ (n >> 63)` (arithmetic shift), inverse `(z >> 1) ^ -(z & 1)`. Why: two's-complement −1 is FF…FF, a 10-byte uvarint; zigzag(−1) = 1 is one byte. Small magnitudes stay small in both signs.
- **uvarint (LEB128)**: 7 bits per byte, least-significant group first, MSB = "more bytes follow". u64 worst case $\lceil 64/7 \rceil = 10$ bytes. Same family as WebAssembly/DWARF LEB128. Contrast: SQLite's varint is big-endian, max 9 bytes, with the 9th byte carrying a full 8 bits.

Worked, `ints = [1000, 1005, 1004, 1010]`:

| x    | delta | zigzag | uvarint |
| ---- | ----- | ------ | ------- |
| 1000 | 1000  | 2000   | `D0 0F` |
| 1005 | 5     | 10     | `0A`    |
| 1004 | −1    | 1      | `01`    |
| 1010 | 6     | 12     | `0C`    |

Stream: **D0 0F 0A 01 0C** — 5 bytes against 32 as raw u64, 6.4:1. (2000 = `0b111_1101_0000`; low 7 bits `1010000` = 0x50, set the continuation bit → D0; remaining 15 → 0F.)

Decode-side hazards: buffer ends mid-varint (truncated); continuation bit still set after 10 bytes (too long — also a DoS vector if unbounded); a 10th byte contributing bits at or above $2^{64}$ (overflow; Go's `binary.Uvarint` flags exactly this); non-minimal `80 00` for 0 (protobuf parsers accept it, canonical/consensus formats must reject or normalize).

Delta overflow: for int64 inputs, $\max - \min$ can be $2^{64} - 1$, which does not fit int64. Real formats define deltas with wrapping mod-$2^{64}$ arithmetic (Parquet says so explicitly): C unsigned subtraction gives the wrap free, Python needs an explicit `& MASK64`, and the decoder recovers exact values because addition mod $2^{64}$ is a group. Signed overflow in C is UB — do the arithmetic unsigned.

## Fixed-width bitpacking

Pack n values of k bits each into $\lceil nk/8 \rceil$ bytes. **State the bit order before writing code.** Convention here (and in Parquet's RLE/bit-packed hybrid): LSB-first — value i owns stream bits $[ik, (i+1)k)$; stream bit j lives at bit $j \bmod 8$ of byte $\lfloor j/8 \rfloor$, bit 0 least significant; unused high bits of the final byte are zero. Parquet's deprecated BIT_PACKED level encoding is MSB-first, so both orders coexist inside one format and mixing them corrupts silently.

Encoder accumulator idiom:

```python
acc |= v << nbits
nbits += k
while nbits >= 8:
  out.append(acc & 0xFF)
  acc >>= 8
  nbits -= 8
```

Worked: `[1, 2, 3, 4, 5]`, k = 3 → 15 bits → 2 bytes:

```
stream bit:  0 1 2  3 4 5  6 7 | 8  9 10 11  12 13 14  (15 = pad 0)
value bits:  1 0 0  0 1 0  1 1 | 0  0  0  1   1  0  1
             [--1-] [--2-] [-3  3] [---4--]  [---5--]
byte:        0xD1                | 0x58
```

→ **D1 58**. Instant sanity checks that catch order bugs: k = 8 must be the identity on bytes; k = 1 of `[1,0,1,1,0,0,1,0]` must be `4D`.

Role in columnar formats: dictionary indices, def/rep levels, deltas after frame-of-reference. Blocks of 8/32/128 values get unrolled or SIMD unpack kernels (Lemire's FastPFor lineage); the win is $64/k$ per column of small ints before any general-purpose codec runs.

## Checksums for framing

You length-prefix a frame, then guard it. The ladder:

- **Additive sum** (`sum(buf) & 0xFF`): catches every single-byte change; blind to any reordering (commutative) and to balanced ±deltas across bytes.
- **Bare XOR**: strictly weaker — also commutative, plus any byte-pair XORed with the same mask cancels, and a duplicated block vanishes entirely. Two identical corruptions are invisible. This is the expected answer to "why is XOR weak".
- **Fletcher-16**: `s1 = (s1 + b) % 255; s2 = (s2 + s1) % 255`, result `(s2 << 8) | s1`. Since $s_2 = \sum_i (n - i + 1)\, b_i \bmod 255$, bytes are weighted by position → reordering detected, still two adds per byte and no tables. Modulus 255 rather than 256 so the sums can be computed with one's-complement carry folds (same trick as the RFC 1071 Internet checksum); the cost is $0x00 \equiv 0xFF \pmod{255}$ — a byte flipped between those two is invisible. Adler-32 is the same design mod 65521 with s1 starting at 1 (zlib).
- **CRC-32**: polynomial division over $GF(2)$; detects all burst errors up to the register width and all odd-weight errors with the right polynomial; Hamming-distance-vs-length tables are Koopman's. The step up when framing matters.
- Adversarial setting: none of the above — keyed MAC.

Worked, Fletcher-16 of `"abcde"`:

| byte | s1       | s2        |
| ---- | -------- | --------- |
| 97   | 97       | 97        |
| 98   | 195      | 292 → 37  |
| 99   | 294 → 39 | 76        |
| 100  | 139      | 215       |
| 101  | 240      | 455 → 200 |

Result `(200 << 8) | 240` = **0xC8F0**. More vectors: `"abcdef"` → 0x2057, `"abcdefgh"` → 0x0627.

Framing idiom: the checksum covers header + payload with the checksum field zeroed; the verifier recomputes with the field zeroed and compares.

## Gorilla-style XOR float compression, for flavor

Gorilla (Facebook, VLDB 2015) compresses f64 series by XORing each value with its predecessor: unchanged value → a single `0` bit; changed → `10` + reuse the previous leading/trailing-zero window, or `11` + 5 bits of leading-zero count + 6 bits of window length + the significant bits themselves. Timestamps get delta-of-delta bucketed into size classes. About 1.37 bytes per 16-byte point on production telemetry, roughly 12:1; Prometheus TSDB chunks and InfluxDB TSM are direct descendants. Interview relevance: it is the same accumulator machinery with a predictor bolted on — "previous value" is the model, XOR is the residual, and the window trick is bitpacking with an adaptive width.

## Gotchas and interviewer follow-ups

1. **Overlong is a CVE class, not pedantry.** C0 AF is `/` in two bytes; any system that validates strings before decoding and consumes them after is bypassable (the IIS/Nimda traversal, 2001). Structural fix: reject `cp < min_for_length`; never blocklist byte values.
2. **Java "modified UTF-8"** encodes U+0000 as C0 80 (so C strings never contain NUL) and astral codepoints as 6-byte surrogate pairs (CESU-8 style) — JNI and class files. Strict decoders reject both.
3. **WTF-8** deliberately admits unpaired surrogates so arbitrary Windows UTF-16 (e.g. filenames) round-trips; Rust's `OsStr` uses it. The name to drop when asked "how would you store possibly-invalid UTF-16".
4. **utf8mb3**: MySQL's legacy 3-byte cap; 4-byte codepoints (≥ U+10000, i.e. emoji) explode old schemas. Always test 0xFFFF/0x10000 across any boundary you build.
5. **Streaming follow-up, near-certain**: "input arrives in chunks and splits mid-sequence." Carry `(cp_so_far, bytes_still_needed)` across calls; emit on completion; pending state at EOF is a truncation error. Python ships this as `codecs.getincrementaldecoder("utf-8")`.
6. **Error policy follow-up**: raise vs replace with U+FFFD. The Unicode-recommended policy substitutes one U+FFFD per _maximal subpart_, then resyncs at the next plausible lead byte.
7. **Base64 length ≡ 1 (mod 4) can never be valid.** Unpadded base64url (JWT/JWS) recovers padding as `s + "=" * (-len(s) % 4)`; remainder 2 → 1 byte, remainder 3 → 2 bytes.
8. **Base64 malleability**: non-zero dangling bits give one byte string several encodings. Anything deduping, caching, or verifying signatures over the encoded form must canonicalize or reject.
9. **RLE worst case is 2x** for pair encoding — say it unprompted, then offer PackBits literals (1/128 overhead) as the fix. "RLE vs LZ?" — RLE when runs dominate and you need O(1) memory and trivial decode.
10. **Delta overflow**: int64 deltas need 65 bits in the worst case. Wrap mod $2^{64}$; C unsigned arithmetic is the free lunch, Python masks by hand, and un-wrapping in the decoder is exact because modular addition is a group.
11. **Varint traps**: forgetting to clear the continuation bit on the final byte; unbounded length (DoS via a stream of `80`); overflow bits in the 10th byte; non-minimal `80 00` breaking canonical-encoding assumptions in consensus or content-addressed systems.
12. **Bit order is a spec decision, not a detail.** LSB-first and MSB-first both live inside Parquet. Verify with the k = 8 identity and a k = 1 vector before trusting anything else. In C, mask building via `1u << k` is UB at k = width.
13. **Checksum field placement**: compute over the frame with the checksum field zeroed, or define the check over payload only; the verifier mirrors the choice. Fletcher's 0x00 ≡ 0xFF blindness is fair game, as is "when do you reach for CRC" (burst guarantees).
14. **Endianness and bit order are independent axes.** LEB128 is little-endian at the 7-bit-group level; SQLite's varint is big-endian; your bitpacker's fill order is a third independent choice. Qualify which axis you mean.

## Rapid-fire drills

1. Bytes for U+07FF vs U+0800? → 2 vs 3.
2. One-expression test for a continuation byte? → `(b & 0xC0) == 0x80`.
3. Why reject ED A0 80 though it matches the 3-byte template? → it decodes to U+D800, a surrogate: ill-formed in UTF-8.
4. Which lead bytes can never appear in valid UTF-8? → C0, C1, F5–FF.
5. Max UTF-8 sequence length since RFC 3629? → 4 bytes, cp ≤ 0x10FFFF.
6. How many codepoints does the 3-byte form legitimately cover? → $0x10000 - 0x800 - 2048 = 61440$ (surrogates excluded).
7. base64 output length for n bytes? → $4\lceil n/3 \rceil$ with padding.
8. How many `=` for 5 input bytes? → one (5 mod 3 = 2 leftover bytes).
9. zigzag(−3)? And the general map? → 5; negatives go to odds, non-negatives to evens.
10. Why not uvarint the raw two's complement of −1? → FF…FF costs 10 bytes; zigzag(−1) = 1 costs one.
11. Decode uvarint `AC 02`. → 0x2C + (2 << 7) = 300.
12. Max uvarint bytes for u64? → $\lceil 64/7 \rceil = 10$.
13. Bytes to bitpack 100 five-bit values? → $\lceil 500/8 \rceil = 63$.
14. bitpack with k = 8 should return what? → the input bytes verbatim; cheapest bit-order sanity check.
15. Additive checksum: always catches / never catches? → any single-byte change / any permutation of the bytes.
16. Fletcher-16 of `b"\x00"` vs `b"\xff"`? → both 0x0000; mod-255 blind spot.
17. Python `(-7) % 255` vs C? → 248 vs −7; keep C accumulators unsigned.
18. In Gorilla, what does a lone 0 bit mean? → XOR with the previous value is 0: exact repeat.
19. C pitfall when assembling codepoints from `char *`? → signed char sign-extension; use `unsigned char` / `uint8_t`.
