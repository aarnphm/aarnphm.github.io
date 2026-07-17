---
date: '2026-07-06'
id: study
modified: 2026-07-06 13:05:30 GMT-04:00
tags:
  - cs
title: study
---

# study plan

The goal is a usable interview reflex: define the wire contract, hand-run examples, code the happy path, add guards, then prove split and roundtrip behavior.

Every block follows the same small loop:

1. read the route and diagram.
2. solve one stub cold.
3. run `python3 test_problems.py`.
4. add the miss to a redo list.
5. review `notes.fc`.
6. re-solve misses from clean stubs.

## 5-day path

### day 1: machinery

**01-bits**

- Learn: fixed-width masks, logical shifts, bit fields, lowest-set-bit tricks.
- Solve: `pack_rgba`, `unpack_rgba`, `extract_field`, `insert_field`, `next_pow2`.
- Recite: why Python needs masks after `~`, `<<`, `+`, `-`, and `*`.

**02-unsigned-alignment**

- Learn: mod-$2^n$ arithmetic, sign bridges, power-of-two alignment.
- Solve: `align_up`, `align_down`, `is_aligned`, `to_unsigned`, `to_signed`.
- Recite: `align_up(16, 8) == 16`, because `+ a - 1` is the whole trick.

### day 2: likely live prompt

**03-varint**

- Learn: LEB128 uvarint, three malformed classes, zigzag.
- Solve: `encode_uvarint`, `decode_uvarint`, `zigzag_encode`, `zigzag_decode`, `decode_uvarint_seq`.
- Recite: `0 -> 00`, `127 -> 7f`, `128 -> 80 01`, `300 -> ac 02`, `16384 -> 80 80 01`.

Run a 60-minute mock using only the varint module:

| minute | action                                                             |
| ------ | ------------------------------------------------------------------ |
| 0-5    | ask contract questions: signedness, width, canonicality, streaming |
| 5-10   | hand-run 0, 127, 128, 300                                          |
| 10-30  | implement encode and one-shot decode                               |
| 30-45  | add truncated, too-long, overlong                                  |
| 45-55  | decode a sequence                                                  |
| 55-60  | say the roundtrip invariant and error taxonomy out loud            |

### day 3: chunk boundaries

**05-decode**

- Learn: split-invariance, `finish()`, sticky errors, bounded buffers.
- Solve: `StreamingVarintDecoder`, `FrameDeframer`, `TlvStreamParser`.
- Recite: `feed(a + b)` and `feed(a); feed(b)` must emit the same total output.

Stretch only after the core passes:

- `Utf8StreamValidator`
- `cobs_encode` and `cobs_decode`
- `ChunkedDecoder`

### day 4: bytes and adjacent algorithms

**04-byte-streams**

- Learn: endian polynomial, cursor discipline, `struct` prefixes.
- Solve: `read_u32_le`, `read_u32_be`, `write_i64_le`, `bswap32`, `BinaryReader`.
- Recite: `0xdeadbeef` is `ef be ad de` in little-endian, `de ad be ef` in big-endian.

**07-stream-algorithms and 08-queueing**

- Solve: `MovingAverage`, `StreamingMedian`, `sliding_window_max`, `reservoir_sample`, `TokenBucket`, `mm1_metrics`.
- Recite: `HLL: 2^14 registers ~= 12 KiB ~= 0.8%`, `Bloom 1% FP ~= 9.6 bits/item`, `M/M/1 W = E[S] / (1 - rho)`.

### day 5: codec breadth

**06-codecs**

- Learn: UTF-8 validation, base64 tails, RLE shape, delta-zigzag-uvarint.
- Solve: `rle_encode`, `rle_decode`, `utf8_encode`, `utf8_decode`, `delta_varint_encode`, `delta_varint_decode`.
- Recite: overlong UTF-8 is rejected after decoding the value or via first-continuation-byte constraints.

End with the redo list:

- Re-solve every miss from a clean stub.
- Review every deck once.
- Read [[hinterland/prep/cheatsheet]] once, slowly.
- Stop adding new topics.

## 1-day cram

Five hours. No heroics. The point is to be correct under time pressure.

| block     | work                                                                   |
| --------- | ---------------------------------------------------------------------- |
| 0:00-0:35 | read [[hinterland/prep/cheatsheet]] and hand-write the uvarint vectors |
| 0:35-1:50 | solve `03-varint` core stubs                                           |
| 1:50-2:50 | solve `05-decode` core stubs                                           |
| 2:50-3:10 | break, no screen                                                       |
| 3:10-4:10 | run the 60-minute varint mock from day 2                               |
| 4:10-4:40 | grade misses and re-solve one clean                                    |
| 4:40-5:00 | review all flashcards for 03, 05, 04, and 06                           |

If the real screen turns into UTF-8, RLE, buffered readers, sketches, or rate limits, use the cheatsheet rows as the fallback map.
