# Streaming / incremental decoding

This is the extension the interviewer bolts onto varint at minute 25: "great — now the bytes arrive over a socket in arbitrary chunks." Everything below is one idea instantiated six ways.

## Core mental model

A streaming decoder is a **state machine plus retained partial state**. `feed(chunk)` consumes as many complete items as the buffered bytes allow, emits them, and keeps the remainder (either raw bytes in a buffer, or digested state like `(accumulator, shift)`). The chunk boundary is noise imposed by the transport; your job is to make it invisible.

**The cardinal rule: output must be invariant under every split of the byte stream.** `feed(a + b)` must produce the same total output as `feed(a); feed(b)`, for every split point, including empty chunks. The property test that enforces it:

1. feed the whole stream in one call,
2. feed it byte-at-a-time,
3. feed it across every 2-part split `(stream[:i], stream[i:])` for `i` in `0..len`.

All three must yield identical concatenated output. If you write one test in the interview, write this one — it catches essentially every state-machine bug (state reset too early, boundary condition at buffer edge, consumed-but-not-emitted).

**Push vs pull.** Push API: caller hands you bytes (`feed(chunk) -> items`); this composes with event loops (`asyncio.Protocol.data_received`, epoll callbacks) and is what interviews want. Pull API: you call `read(n)` on a source and block until satisfied (`io.BufferedReader`, `recv` loops); easier to write straight-line but assumes you own the thread. `asyncio.StreamReader` is pull sugar built on a push core plus backpressure. You can mechanically convert pull to push with a generator and `send()`, worth mentioning, rarely worth writing live.

**Two failure modes, never conflated:**

- _Need more data_: the buffered prefix is a valid prefix of some longer valid stream. Buffer it, return what you have. Never raise.
- _Malformed_: no continuation of the buffered bytes can ever be valid (varint 11th byte, COBS block promising bytes past a zero, chunk-size line of `"xyz"`). Raise `ValueError` immediately — waiting for more bytes cannot fix it.

`finish()` / EOF closes the loophole: a dangling partial value is "need more data" right up until the stream ends, at which point it becomes malformed (truncated). Every stateful decoder needs an end-of-stream call that raises (or returns invalid) on dangling state. After a decoder raises, its state is undefined; real systems tear down the connection rather than resync (errors are sticky).

## Framing strategies compared

Streams are byte soup; framing recovers message boundaries. Four strategies, with the trade each makes:

| strategy                        | overhead                    | hostile-input risk                     | resync after corruption      |
| ------------------------------- | --------------------------- | -------------------------------------- | ---------------------------- |
| length-prefix (fixed or varint) | 1–10 B                      | declared length is attacker-controlled | none — desyncs forever       |
| delimiter + escaping            | worst case 2x expansion     | escape-state bugs                      | self-heals at next delimiter |
| COBS                            | 1 B per 254 B, +1 delimiter | truncated block                        | self-heals at next 0x00      |
| TLV (tag, length, value)        | tag + length per record     | same as length-prefix                  | none                         |

**Length-prefix** is the default. The one thing interviewers always probe: _a hostile declared length must hit a max-frame guard before you allocate or buffer, or you built a memory DoS._ Peer sends 5 bytes claiming a 1 GiB frame; a naive decoder happily buffers forever. Check `length > max_frame` the moment the length is decoded, not when the payload arrives. This guard is the streaming analogue of a bounds check.

**Delimiter + escaping** (SLIP, JSON-lines with embedded newlines escaped): delimiter `0x00`, escape `0xDB`, payload bytes equal to either get expanded to two-byte escape sequences. Worst case (all-delimiter payload) doubles the size — unacceptable when you've budgeted MTU. Decoder state: one bit ("last byte was escape").

**COBS** (consistent overhead byte stuffing) removes all `0x00` from the payload so `0x00` can frame it, with bounded overhead. Encoding: split the data at zeros; emit each zero-free run as `[code][run]` where `code = len(run) + 1` (1..0xFE); the zero itself is implied by the block boundary. A run of 254 non-zero bytes emits `code = 0xFF` and implies _no_ zero (the group continues). Walk `11 22 00 33`:

```
input : 11 22 00 33
blocks: [11 22] zero [33]
output: 03 11 22 02 33      # 03 = "2 bytes then a zero", 02 = "1 byte"
```

Decode reverses it: read `code`, copy `code-1` bytes, append `0x00` unless `code == 0xFF` or you are at end of input. Empty input encodes to `01`. A single `00` encodes to `01 01`. Overhead is exactly $\lceil n/254 \rceil$ bytes: 1 byte per 254, versus 2x worst case for escaping. Malformed cases: a zero byte inside encoded data, or a code promising more bytes than remain (truncated).

**TLV** — each record is (tag, length, value). Forward compatibility falls out for free: a reader that doesn't know tag 17 still knows its length, so it skips `length` bytes and keeps parsing. This is exactly how old protobuf readers survive new fields. **Protobuf wire format is varint-tag TLV**: the key is a varint `(field_number << 3) | wire_type`, and wire type 2 (`LEN`) is varint length + payload; wire types 0/1/5 make the "length" implicit (varint / 8 bytes / 4 bytes). Saying that sentence out loud is worth a lot in the follow-up discussion.

## Incremental decoder state, per format

### Varint: state = (accumulator, shift)

No buffer needed — partial bytes are digested immediately:

```python
for b in chunk:
  self._acc |= (b & 0x7F) << self._shift
  if b & 0x80:
    self._shift += 7  # need more data
  else:
    out.append(self._acc)  # value complete
    self._acc = self._shift = 0
```

Walk `0xAC 0x02` (= 300) split as `[0xAC] | [0x02]`: first feed digests `0xAC` → `acc = 0x2C, shift = 7`, emits nothing. Second feed: `acc |= 2 << 7` → `0x12C = 300`, emit. Bounds: for 64-bit values, at most 10 bytes; the 10th byte (shift 63) may contribute only bit 0, so `shift == 63 and b & 0x7E` is overflow and a continuation bit on the 10th byte is too-long. **Overlong** (non-canonical) encoding: a multi-byte varint whose final byte is `0x00` (e.g. `0x80 0x00` for 0) decodes to the same value as a shorter encoding; reject it when canonicity matters (hash keys, signatures, consensus). Know the split: protobuf itself _accepts_ non-canonical varints; strict-canonical is your policy layer.

C/Python divergence, since this is where it bites:

- Python ints are unbounded — `acc |= b << shift` can never overflow, so the 64-bit bound is policy you must write, not an error the machine gives you. In C, `uint64_t acc; acc |= (uint64_t)(b & 0x7F) << shift;` — the cast is mandatory (`int << 63` is UB) and `shift >= 64` is UB, so the bound check is for memory safety, not taste.
- Iterating `bytes` in Python yields `int` 0..255, already unsigned. In C, `char` may be signed: `char c = buf[i]; c & 0x80` sign-extends in comparisons and `c >> 1` drags the sign bit. Always `unsigned char` / `uint8_t` at the decode boundary.

### UTF-8: state = expected-continuation count (+ first-byte range constraint)

Lead byte tells you how many continuation bytes follow: `0xxxxxxx` → 0, `110xxxxx` → 1, `1110xxxx` → 2, `11110xxx` → 3. Streaming state is that count. But counting alone accepts garbage; the shortest-form and range checks all live in the **first continuation byte**:

| lead   | first continuation must be in | rejects                          |
| ------ | ----------------------------- | -------------------------------- |
| `E0`   | `A0..BF`                      | overlong 3-byte (would fit in 2) |
| `ED`   | `80..9F`                      | surrogates U+D800–DFFF           |
| `F0`   | `90..BF`                      | overlong 4-byte                  |
| `F4`   | `80..8F`                      | codepoints > U+10FFFF            |
| others | `80..BF`                      | —                                |

Plus: `C0`, `C1` are invalid outright (all their 2-byte forms are overlong), and `F5..FF` are invalid (> U+10FFFF). So streaming state is `(remaining, lo, hi)` where `(lo, hi)` constrains only the next byte and relaxes to `(0x80, 0xBF)` after it. The classic overlong exploit: `C0 80` decodes to `NUL` under a sloppy decoder, sailing past `strlen`-based checks — this is why overlong rejection is a security requirement (CVE-class, e.g. the IIS unicode traversal). Equivalent alternative: min-codepoint check — a 2/3/4-byte sequence must decode to $\ge$ U+0080 / U+0800 / U+10000; the range table is that check compiled into the first continuation byte.

### HTTP chunked transfer: a 5-state machine

```
SIZE_LINE -> DATA -> DATA_CRLF -> SIZE_LINE ... -> (size 0) -> TRAILERS -> DONE
```

Body = repeated `hex-size [;extensions] CRLF <size bytes> CRLF`, terminated by a `0` chunk, optional trailer header lines, and a final blank line. `4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n` → `Wikipedia`. The bugs interviewers fish for: CRLF torn across chunks (your state machine must hold "expecting `\n`" as state, or buffer the line), the _post-data_ CRLF forgotten entirely (data of length `size` is followed by its own CRLF that belongs to no line), the size line needs a length guard (a peer that never sends CRLF must not grow your line buffer unboundedly), and `size` is hex, attacker-controlled → same max-frame guard as length-prefix. Trailing data after DONE is malformed, not ignorable — smuggling attacks live in "ignorable" trailing bytes.

## Buffer management

The naive pattern is quadratic:

```python
self._buf += chunk  # fine, bytearray append is amortized O(len(chunk))
...
self._buf = self._buf[n:]  # BAD: copies the whole tail, O(len) per consume
```

Consume k items of size s from an n-byte buffer this way and you touch $O(n^2/s)$ bytes. Idioms that fix it:

- **Cursor + periodic compaction**: keep `self._pos`, read via index/slice from `pos`, and only `del self._buf[:pos]` when `pos` exceeds some threshold (e.g. `pos > 4096 and pos * 2 > len(buf)`). `del ba[:k]` is one `memmove` of the tail; compacting only when at least half the buffer is dead makes the cost amortized O(1) per byte.
- **`memoryview`** for zero-copy reads of payload slices: `bytes(memoryview(buf)[pos:pos+n])` copies once into the output object instead of slice-then-slice. In C this is just pointer arithmetic; Python makes you ask for it.
- `bytes` slicing always copies; `bytearray` supports in-place `del`; `collections.deque` of chunks + lazy join is the third option when frames are huge.

**Max-buffer guard**: independent of declared lengths, cap `len(buf) - pos`. A decoder whose state can grow without bound under adversarial input is incorrect, same as an unchecked array write.

## Gotchas and interviewer follow-ups

- "Feed it one byte at a time" — the split-invariance test. If your `feed` assumes a whole header is present, you fail here. Answer before being asked: state your invariant and the property test.
- "The length says 1 GiB" — max-frame guard _at length-decode time_. Bonus: note the guard belongs before allocation, and that `bytearray(huge)` is the allocation.
- "Stream ends mid-value" — `finish()`/EOF must raise (truncated), distinguishing it from mid-stream "need more data".
- "Why did your decoder get slow at 100 MB?" — quadratic re-slicing; answer with cursor + compaction.
- "Peer sends `0x80 0x00`" — overlong varint: decodes to 0, reject if canonical encoding is part of the contract; know protobuf accepts it.
- "How does an old reader skip a field it doesn't know?" — TLV: length is self-describing; protobuf wire types make every field skippable without schema.
- "Why COBS over escaping?" — bounded overhead (1/254 ≈ 0.4% vs 100% worst case), zero-free output, O(1) resync at next delimiter.
- "What do you do after a parse error?" — nothing recoverable in length-prefixed streams (desynced forever); delimiter/COBS resync at the next delimiter; real protocols close the connection.
- Empty chunk fed → must be a no-op, not a crash or a state reset.
- Emitting _copies_: return `bytes(view)`, never a live view into your internal buffer that compaction will invalidate.
- Sticky errors: once `feed` raises, subsequent behavior is undefined unless you specify otherwise; say so out loud.
- Python-specific: `for b in chunk` gives ints (fast path); `chunk[i:i+1]` gives `bytes` — comparing `b == b"\x80"` against an int is silently always-False. Type-confuse here and every test fails at once.

## Rapid-fire drills

| question                                | answer                                                                          |
| --------------------------------------- | ------------------------------------------------------------------------------- |
| Streaming varint state?                 | `(accumulator, shift)` — no byte buffer needed                                  |
| Max bytes for a 64-bit varint?          | 10; 10th byte may only be `0x00` or `0x01`                                      |
| Overlong varint?                        | multi-byte encoding with final byte `0x00`; decodes fine, non-canonical         |
| "Need more data" vs "malformed"?        | valid prefix → buffer & return; no valid continuation → raise now               |
| What does `finish()` do?                | turns dangling partial state into an error (truncated stream)                   |
| The one property test?                  | same output fed whole, byte-at-a-time, and across every 2-part split            |
| Hostile length prefix defense?          | `max_frame` check at length-decode time, before buffering/allocating            |
| COBS overhead?                          | 1 byte per 254 payload bytes ($\lceil n/254 \rceil$), plus the `0x00` delimiter |
| COBS code byte `0xFF` means?            | 254 data bytes follow, _no_ implied zero                                        |
| Delimiter+escape worst case?            | 2x expansion (every payload byte needs escaping)                                |
| UTF-8 streaming state?                  | expected continuation count + allowed range for the next byte                   |
| Why is `C0 80` rejected?                | overlong NUL — shortest-form violation, classic filter bypass                   |
| `ED A0 80`?                             | UTF-8-encoded surrogate U+D800 — invalid in UTF-8                               |
| Protobuf wire format in one phrase?     | varint-key TLV: key = `(field << 3) \| wire_type`                               |
| Chunked encoding states?                | size-line → data → CRLF → … → 0-chunk → trailers → done                         |
| Why is `buf = buf[n:]` per message bad? | O(n) copy per consume → quadratic; use cursor + compaction                      |
| `char` pitfall in C decoder?            | signed `char` sign-extends; use `uint8_t` at the byte boundary                  |
| Shift pitfall in C varint?              | `(b & 0x7F) << 63` without `uint64_t` cast is UB; `shift >= 64` is UB           |

## Where next

- `07-stream-algorithms` is the theory and patterns behind the incremental decoding here: the same feed/finish state-machine discipline, with the state a sketch (reservoir, Count-Min, HLL, two heaps) instead of an `(accumulator, shift)` pair, plus the pipeline vocabulary (watermarks, delivery semantics, Kafka/Flink) for when "decode a stream arriving in chunks" grows into "design the system those chunks flow through".
- `08-queueing` is what happens when the queue in front of your decoder fills: bounded buffers with an explicit overflow policy, backpressure instead of OOM, and the $1/(1-\rho)$ hockey stick that makes the max-buffer guard above a latency decision as much as a memory one.
