---
date: '2026-07-05'
id: intel
modified: 2026-07-06 12:41:11 GMT-04:00
tags:
  - seed
title: intel
---

# recon: 60-min encoding/decoding code screen

archetype: bit/byte ops, unsigned data + alignment, varint, integers-as-byte-streams, streaming decode. web recon run 2026-07-02. confidence marked per row; leetcode discuss bodies 403 direct fetch, so some rows rest on search-snippet corroboration + aggregators.

## company table

Use this for prompt priors. Sources sit below so the table stays readable.

| company                                  | likely shape                                            | follow-up ladder                                               | confidence  |
| ---------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------- | ----------- |
| Datadog                                  | encode unsigned ints to bytes, decode back              | one-byte codec, u64 varint, zigzag, streaming chunks           | medium      |
| Datadog                                  | buffered file writer                                    | partial writes, flush by size or time, thread safety, shutdown | high        |
| Datadog                                  | moving average or latency buckets over a metrics stream | bounded memory, streaming input                                | high        |
| Google                                   | UTF-8 validation from raw byte values                   | continuation counts, overlongs, ambiguous byte rules           | high        |
| Google / OpenAI / Meta                   | encode and decode string lists                          | length-prefix framing, UTF-8 safety, length overflow           | high        |
| OpenAI                                   | small systems with serialization                        | persistence to bytes, resumable iteration                      | medium-high |
| Microsoft                                | byte-output run-length encoding                         | runs above 255, decode direction                               | high        |
| Goldman / Oracle / Stack Overflow / LINE | textual RLE                                             | empty input, digit payloads, multi-digit counts                | high        |
| Stripe                                   | practical parser screen                                 | staged constraints and production edge cases                   | high        |
| Confluent                                | KV store with expiry and streamed timestamps            | ordering invariants, aggregate maintenance                     | high        |
| Cloudflare                               | bit-manipulation or parse/serialize screen              | network-ish byte formats                                       | medium      |
| Qualcomm / embedded                      | endian swap and bit fields                              | pointer walks, shifts, bit-field portability                   | high        |
| Arista                                   | C pointers, allocators, struct layout                   | alignment reasoning                                            | high        |
| Neuralink                                | embedded compression flavor                             | bitpacking, delta coding, ratio vs latency                     | low-medium  |
| MongoDB                                  | BSON-like length-prefixed binary docs                   | serialize a JSON subset                                        | low         |

Datadog sources:

- [interviewing.io](https://interviewing.io/datadog-interview-questions)
- [Blind posts](https://www.teamblind.com/company/Datadog/posts/datadog-interview)
- [dd-trace-dotnet varint helper](https://github.com/DataDog/dd-trace-dotnet/blob/635c879092089703ad6724bb70c96884f65cd631/tracer/src/Datadog.Trace/DataStreamsMonitoring/VarEncodingHelper.cs)
- [full-loop thread](https://leetcode.com/discuss/post/6376092/datadog-interview-full-loop-seniormid-le-x8uq/)
- [buffered writer](https://www.glassdoor.com/Interview/Code-a-Buffered-File-Writer-QTN_8013868.htm)
- [DesignGurus](https://www.designgurus.io/answers/detail/what-are-datadog-coding-interview-questions)
- [Prachub](https://prachub.com/interview-questions/implement-buffered-file-writer-with-concurrency-support)
- [tech-screening thread](https://leetcode.com/discuss/post/6375931/datadog-tech-screening-by-anonymous_user-x9aj/)
- [Ophy guide](https://ophyai.com/blog/company-guides/datadog-interview-guide)

Google / OpenAI / Meta sources:

- [LC 393](https://leetcode.com/problems/utf-8-validation/)
- [Taro UTF-8](https://www.jointaro.com/interviews/questions/utf-8-validation/)
- [HelloInterview UTF-8](https://www.hellointerview.com/community/questions/utf8-validation/cm5eh7nri04xx838ow2awa27x)
- [LC 271](https://leetcode.com/problems/encode-and-decode-strings/)
- [Glassdoor Google QTN_2605165](https://www.glassdoor.com/Interview/Encode-and-Decode-an-Array-of-strings-QTN_2605165.htm)
- [Taro OpenAI encode/decode](https://www.jointaro.com/interviews/openai/encode-and-decode-strings/)
- [igotanoffer OpenAI](https://igotanoffer.com/en/advice/openai-coding-interview)

RLE and parser sources:

- [Microsoft RLE](https://www.glassdoor.com/Interview/Run-length-encoding-write-compression-function-given-input-array-and-output-array-of-bytes-QTN_2275897.htm)
- [Goldman RLE](https://www.glassdoor.com/Interview/Implement-a-run-length-encoding-function-For-a-string-input-the-function-returns-output-encoded-as-follows-a-QTN_2454026.htm)
- [Oracle RLE thread](https://leetcode.com/discuss/interview-question/algorithms/124996/oracle-phone-screen-run-length-encoding)
- [Stripe questions](https://interviewing.io/stripe-interview-questions)
- [Stripe phone-screen gist](https://gist.github.com/pkafel/86470ca581350014bb89033fbffb9bb1)

Systems and embedded sources:

- [Confluent thread](https://leetcode.com/discuss/interview-question/759611/confluent-senior-software-engineer-phone-interview/)
- [Cloudflare Glassdoor report](https://www.glassdoor.sg/Interview/Phone-screen-interview-involved-bit-manipulation-coding-question-followed-by-discussion-about-the-company-and-the-work-At-QTN_1689832.htm)
- [Cloudflare prep guide](https://algocademy.com/blog/cloudflare-technical-interview-prep-a-comprehensive-guide/)
- [Qualcomm endian question](https://www.glassdoor.com/Interview/how-to-swap-big-endian-and-little-endian-QTN_112221.htm)
- [embedded bit manipulation prep](https://tonyfu97.github.io/Embedded-C-Interview-Prep/08_bit_manipulation/)
- [Arista set](https://www.geeksforgeeks.org/arista-network-interview-set-1/)
- [Arista Glassdoor](https://www.glassdoor.com/Interview/The-interviewer-started-with-some-small-talk-Talked-a-little-bit-about-my-research-work-Then-went-on-with-technical-quest-QTN_1145187.htm)
- [Neuralink Glassdoor](https://www.glassdoor.com/Interview/Neuralink-Interview-Questions-E1616853.htm)
- [Neuralink guide](https://www.interviewquery.com/interview-guides/neuralink-software-engineer)
- [BSON spec](https://www.mongodb.com/docs/manual/reference/bson-types/)
- [MongoDB Glassdoor](https://www.glassdoor.com/Interview/MongoDB-Interview-Questions-E433703.htm)

Background every ladder converges on: [protobuf varint encoding](https://protobuf.dev/programming-guides/encoding/) for 7-bit groups, continuation MSB, group order, and zigzag; [Let's Make a Varint](https://carlmastrangelo.com/blog/lets-make-a-varint) for the LEB128 vs prefix-varint design space.

## composite most-likely 60-minute flow

Datadog-shaped: one problem, ladder revealed one rung at a time. Later rungs depend on pace.

- 0:00-0:05 intro: one sentence on background.
- 0:05-0:12 clarify: ask the clarifying questions, state `encode(list[int]) -> bytes`, state `decode(bytes) -> list[int]`, then write `decode(encode(xs)) == xs`.
- 0:12-0:30 core: implement the one-byte codec for ints in `[0, 255]`. Name the 256 limitation before being prompted.
- 0:30-0:42 u64 extension: implement LEB128. Emit low 7 bits, set bit 7 when more groups remain, decode with `(b & 0x7f) << shift`.
- 0:42-0:52 signed or streaming extension: use zigzag for signed values, or a `Decoder.feed(chunk) -> list[int]` object holding `(accum, shift)` across chunks.
- 0:52-0:58 hardening: test `[]`, `[0]`, `[127, 128]`, `[255, 256]`, max u64, truncated final varint, too-long continuation runs, and overlong `0x80 0x00`.
- 0:58-1:00 wrap: O(total bytes) time, O(1) extra state, 1 to 10 bytes per u64, usually 2-4x smaller than fixed width on small telemetry-like ints.

If the screen is two-question format, expect one-byte plus u64 varint as Q1 and streaming or buffered writer as Q2, about 25 minutes each.

## what interviewers actually grade

- Roundtrip-first testing: write `assert decode(encode(xs)) == xs` before generalizing.
- Byte fluency: `& 0x7f`, `| 0x80`, shift-then-mask order, and the language's unsigned shift story.
- Edge cases named before prompting: zero, 127/128, 255/256, empty input, max width, truncation, garbage bytes.
- Streaming API design: explicit partial state, buffer ownership, and the return value for incomplete input.
- Encoding decision: name fixed-width, length-prefix, continuation-bit varint, zigzag, and pick with a reason.
- Malformed-input policy: choose raise, skip, or partial result up front, then enforce the guard.

## clarifying questions for the first five minutes

1. signed or unsigned? if signed, may I use zigzag, or do you want fixed-width two's complement?
2. max magnitude — fits in u32, u64, or arbitrary-precision?
3. output type: raw bytes/bytearray, or a string (and if string, full 0–255 alphabet or printable-only)?
4. endianness: any required byte order for multi-byte values, or is the scheme mine to define? (varint fixes little-endian group order by construction; fixed-width needs an explicit choice.)
5. malformed input on decode: raise, return partial + error, or skip? is input trusted?
6. is canonical form required — must I reject over-long encodings (`0x80 0x00` for 0)?
7. one-shot buffer, or bytes arriving in chunks (do you want a stateful/streaming decoder API)? can a value straddle a chunk boundary?
8. allocation constraints: free to append to a growable buffer, or fixed pre-allocated output (embedded flavor)? any memory ceiling relative to input size?
9. data distribution: mostly small integers? (decides varint vs fixed-width; say why it matters — wire size.)
10. am I allowed stdlib codecs (`struct.pack`, `DataView`, `varint` libs) or is hand-rolling the point? (it is the point; ask anyway, it shows you know libraries exist.)

## red flags that fail candidates

- Stringification dodge: `",".join(map(str, xs))` when byte output was specified.
- 7-bit and 8-bit confusion: off-by-one at 127/128 or 255/256.
- Sign and shift bugs: arithmetic shift where logical was needed, missing `& 0xff`, shifting past width, or JS numbers losing bits past $2^{53}$.
- Unbounded decode loop: malformed `0x80 0x80 0x80...` spins forever or overflows shift without the 10-byte u64 cap.
- Streaming state dropped at a chunk boundary: truncated varint state must stay in `(accum, shift)`.
- Encode-only testing: no `decode(encode(xs))`, no correctness argument.
- Delimiter schemes with no escaping: `0xFF` cannot separate values that may contain `0xFF`.
- Policy improvisation: the first malformed case returns `None`, the second raises, the third skips.
- Embedded sins: `char*` to `uint32_t*` at unaligned offsets, or assuming host endianness on wire data.
- No complexity story: bytes-per-value and distribution matter; `O(n)` alone says almost nothing.
