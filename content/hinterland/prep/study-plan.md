# study plan

Two variants. Pick by days remaining. Every solve follows the cold-solve rule in the README: stub first, harness as judge, solutions only after. Keep a redo list of every function you missed; a miss is cleared by re-solving from a clean stub, not by reading the fix.

Time boxes are fixed. When a day runs over, drop the stretch targets and keep the mock.

## 5-day plan

Daily shape: read the kit's `notes.md`, cold-solve the listed stubs in order, check failures against `solutions.py`, re-solve misses from clean stubs, finish with the rapid-fire drills in the notes. Mocks run in the evening under a real 60-minute timer and get graded with the rubric in `mock-screens.md` the same night.

### day 1: underpinnings (01-bits + 02-unsigned-alignment)

| block     | work                                                                                                                |
| --------- | ------------------------------------------------------------------------------------------------------------------- |
| 0:00-0:30 | read `01-bits/notes.md` sections 1-5                                                                                |
| 0:30-1:45 | cold-solve `pack_rgba`/`unpack_rgba`, `extract_field`/`insert_field`, `next_pow2`, `reverse_bits32`, `sar32`        |
| 1:45-2:15 | read `02-unsigned-alignment/notes.md` sections 1-2                                                                  |
| 2:15-3:15 | cold-solve `align_up`/`align_down`/`is_aligned`, `add32`/`sub32`/`mul32`/`shl32`/`shr32`, `to_unsigned`/`to_signed` |
| 3:15-3:45 | rapid-fire drills from both notes; misses go on the redo list                                                       |

Stretch: `popcount_swar`, `struct_layout`.

### day 2: 03-varint, then mock A

| block         | work                                                                                                                                                                                            |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0:00-0:45     | read `03-varint/notes.md` in full; the malformed-classes and canonicality section is the highest-yield prose in the kit                                                                         |
| 0:45-2:30     | cold-solve `encode_uvarint`, `decode_uvarint` (all three error tokens: truncated, too long, overlong), `zigzag_encode`/`zigzag_decode`, `encode_svarint`/`decode_svarint`, `decode_uvarint_seq` |
| 2:30-3:00     | re-solve misses from clean stubs; drills section (d)                                                                                                                                            |
| evening, 1:15 | mock screen A, then grade it                                                                                                                                                                    |

Stretch: `encode_vlq`/`decode_vlq`.

### day 3: 05-streaming

| block     | work                                                                                                                      |
| --------- | ------------------------------------------------------------------------------------------------------------------------- |
| 0:00-0:40 | read `05-streaming/notes.md`; internalize state = (accumulator, shift) and the all-splits invariance the harness enforces |
| 0:40-2:30 | cold-solve `StreamingVarintDecoder`, `FrameDeframer`, `TlvStreamParser`                                                   |
| 2:30-3:15 | `Utf8StreamValidator`; it is tagged hard, budget the whole block                                                          |
| 3:15-3:30 | drills                                                                                                                    |

Stretch: `cobs_encode`/`cobs_decode`, `ChunkedDecoder`.

### day 4: 04-byte-streams, the 07/08 screen-core set, then mock B

| block         | work                                                                                                                                                                         |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0:00-0:40     | read `04-byte-streams/notes.md`; memorize the struct format-char and prefix tables                                                                                           |
| 0:40-2:00     | cold-solve `read_u32_le`/`read_u32_be`, `write_i64_le`, `bswap32`, `BinaryReader`                                                                                            |
| 2:00-2:45     | `hexdump` (the format is byte-exact), then `float_to_bits`/`bits_to_float`                                                                                                   |
| 2:45-3:00     | drills                                                                                                                                                                       |
| 3:00-4:15     | cold-solve the 07/08 screen-core set: `MovingAverage`, `StreamingMedian`, `sliding_window_max` in `07-stream-algorithms`, then `TokenBucket`, `mm1_metrics` in `08-queueing` |
| evening, 1:15 | mock screen B, then grade it                                                                                                                                                 |

Stretch: `float32_parts`, `SlidingWindowCounter`.

### day 5: 06-codecs, mock C, consolidation

| block         | work                                                                                             |
| ------------- | ------------------------------------------------------------------------------------------------ |
| 0:00-0:40     | read `06-codecs/notes.md`                                                                        |
| 0:40-2:15     | cold-solve `rle_encode`/`rle_decode`, `utf8_encode`, `delta_varint_encode`/`delta_varint_decode` |
| 2:15-3:00     | `utf8_decode` (hard) or `b64_encode`/`b64_decode`; pick from your redo list                      |
| 3:00-3:30     | clear the redo list: re-solve every remaining miss of the week from clean stubs                  |
| evening, 1:15 | mock screen C, then grade it                                                                     |

Night before the screen: read `cheatsheet.md` once, slowly. Solve nothing new.

If you lose a day, fold day 1 into 15-minute drill blocks attached to the remaining days and keep all three mocks. The mocks predict the screen better than any single kit.

## 1-day cram

Five and a half hours. Covers the single most likely question shape (varint ladder plus streaming) and nothing else.

| block     | work                                                                                                                                                                                                       |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0:00-0:45 | read `cheatsheet.md`; hand-write the uvarint vector table and the zigzag table from memory until both are exact; skim the sketches and queueing sections once, as discussion ammunition, not solve targets |
| 0:45-2:15 | `03-varint` cold: `encode_uvarint`, `decode_uvarint`, `zigzag_encode`/`zigzag_decode`, `decode_uvarint_seq`                                                                                                |
| 2:15-3:15 | `05-streaming` cold: `StreamingVarintDecoder`, then `FrameDeframer` as far as the hour allows                                                                                                              |
| 3:15-3:30 | break, away from the screen                                                                                                                                                                                |
| 3:30-4:30 | mock screen A on a strict timer                                                                                                                                                                            |
| 4:30-5:00 | grade the mock; reread `00-recon/intel.md` sections 3, 4, and 5 (what graders reward, the clarifying questions, the red flags)                                                                             |
| 5:00-5:30 | cheatsheet again; say the three malformed classes and the roundtrip vector set out loud from memory                                                                                                        |

If the real screen deviates from the varint shape (UTF-8 validation, RLE, buffered writer), the cheatsheet rows plus the intel company table are the fallback briefing; both fit in the last half hour.
