# encoding screen prep

Prep for a 60-minute algorithmic code screen on encoding and decoding. Eight kits, 53 graded problems, 518 harness cases, all green against the reference solutions. Recon on who asks what, the most likely question ladder, what graders reward, and what fails candidates is in `00-recon/intel.md`. Read that file before anything else.

## directory map

| dir                     | topic                                                                                                                                 | problems | tests | screen-core problems                                                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------- | ----- | ------------------------------------------------------------------------------------------------------------------------------ |
| `01-bits`               | fixed-width register model, masks and fields, shifts, two's complement, popcount, rotates, XOR                                        | 7        | 62    | `pack_rgba`/`unpack_rgba`, `extract_field`/`insert_field`, `next_pow2`, `reverse_bits32`                                       |
| `02-unsigned-alignment` | mod-$2^n$ arithmetic, Python masking discipline, C promotion traps, sign bridges, alignment, C struct layout                          | 6        | 37    | `align_up`/`align_down`/`is_aligned`, `add32`/`sub32`/`mul32`/`shl32`/`shr32`, `to_unsigned`/`to_signed`, `struct_layout`      |
| `03-varint`             | LEB128/uvarint, zigzag, VLQ, PrefixVarint, the three-way malformed-input taxonomy                                                     | 7        | 156   | `encode_uvarint`, `decode_uvarint`, `zigzag_encode`/`zigzag_decode`, `encode_svarint`/`decode_svarint`, `decode_uvarint_seq`   |
| `04-byte-streams`       | endianness, shift-only codecs, the `struct`/`from_bytes` toolbox, IEEE 754, framing, hexdump                                          | 7        | 45    | `read_u32_le`/`read_u32_be`, `write_i64_le`, `bswap32`, `BinaryReader`, `hexdump`                                              |
| `05-streaming`          | incremental decoders as state machines, deframing, strict UTF-8, COBS, TLV, HTTP chunked                                              | 6        | 58    | `StreamingVarintDecoder`, `FrameDeframer`, `TlvStreamParser`                                                                   |
| `06-codecs`             | UTF-8, base64, hex, RLE, bitpacking, delta+zigzag+uvarint, Fletcher-16                                                                | 7        | 56    | `utf8_encode`/`utf8_decode`, `b64_encode`/`b64_decode`, `rle_encode`/`rle_decode`, `delta_varint_encode`/`delta_varint_decode` |
| `07-stream-algorithms`  | reservoir sampling, Misra–Gries, Count-Min, HLL, Bloom, two-heap median, SWAG windows, watermarks, Kafka/Flink semantics              | 7        | 44    | `MovingAverage`, `StreamingMedian`, `sliding_window_max`, `reservoir_sample`                                                   |
| `08-queueing`           | Little's law, the M/M/1 hockey stick, Pollaczek–Khinchine, Kingman/VUT, FIFO simulation, rate limiters, bounded queues, tail at scale | 6        | 60    | `TokenBucket`, `SlidingWindowCounter`, `mm1_metrics`, `pk_wq`, `kingman_wq`                                                    |

"screen-core" is the tag each problem docstring uses for problems likely to be asked live in the hour. The rest are tagged "depth" and pay off in the follow-up discussion after the code works.

## the practice loop

Each kit dir holds `notes.md` (theory and worked examples), `problems.py` (stubs you implement), `solutions.py` (reference), and `test_problems.py` (self-contained harness, stdlib only). From inside a kit dir:

```
python3 test_problems.py                             # runs your attempts in problems.py
PRACTICE_MODULE=solutions python3 test_problems.py   # runs the reference
```

The harness prints PASS or FAIL per case and exits 1 on any failure. Run the reference line once when you enter a kit to confirm the harness is healthy, then leave `solutions.py` closed until you have attempted the stubs.

The cold-solve rule: attempt every problem from its stub, against the harness, before reading any solution. Reading a solution first turns a recall exercise into a recognition exercise, and the screen tests recall. When a case fails and you are stuck, read only the failing function in `solutions.py`, then delete your attempt and re-solve from the clean stub.

## two orderings

Pedagogical, when you have five or more days: `01` then `02` then `03` then `04` then `05` then `06`. Each kit assumes the idioms of the ones before it.

Priority, when the screen is close: `03-varint` and `05-streaming` first, because the most likely live question is uvarint encode/decode with a streaming-decoder extension (see the composite flow in `00-recon/intel.md` section 2). Then `04-byte-streams` for `BinaryReader` and endianness, then `06-codecs` for the UTF-8, RLE, and delta-stack skins. Treat `01-bits` and `02-unsigned-alignment` as underpinnings and drill their rapid-fire sections whenever a mask or shift hesitation shows up in the later kits.

`07-stream-algorithms` and `08-queueing` sit outside both orderings: they are follow-up-discussion ammunition and systems-screen material, and the priority order above for the core encode/decode screen is unchanged. `MovingAverage`, `StreamingMedian`, and `TokenBucket` are themselves screen-core for the adjacent screen archetypes (Datadog runs a moving-average tech screen per `00-recon/intel.md`); mock screen D covers exactly those three.

## the other root files

- `cheatsheet.md` is the one page to reread the night before.
- `study-plan.md` is a 5-day plan plus a 1-day cram variant.
- `mock-screens.md` holds four timed 60-minute mocks with interviewer scripts, checkpoints, and a grading rubric.
