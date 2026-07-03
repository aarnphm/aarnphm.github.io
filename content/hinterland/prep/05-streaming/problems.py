"""Streaming/incremental decoding practice stubs.

Every stateful class must satisfy split invariance: for any byte stream,
total output is identical whether fed whole, byte-at-a-time, or across any
2-part split (including empty chunks). The test harness enforces this.
"""


class StreamingVarintDecoder:
  """Incremental unsigned LEB128 (protobuf-style) varint decoder.

  [difficulty: core] [screen-core — THE canonical streaming screen question]

  Wire format: each value is a little-endian base-128 integer. Each byte
  carries 7 payload bits (low bits); the high bit (0x80) is the
  continuation flag. Byte i contributes bits [7i, 7i+7). Values are
  unsigned and must fit in 64 bits, so an encoding is at most 10 bytes
  and the 10th byte may only be 0x00 or 0x01.

  API:
    feed(chunk: bytes) -> list[int]
        Consume chunk, return values COMPLETED during this call (possibly
        empty). A value torn across chunks is carried as internal state
        (accumulator, shift) — no byte buffer needed. Empty chunk is a
        no-op returning [].
    finish() -> None
        Signal end of stream. Raises ValueError if a partial value is
        dangling (truncated stream). No-op on a clean boundary.

  Error policy (raise ValueError, decoder state undefined afterwards):
    - too long: continuation bit set on the 10th byte (value would need
      an 11th byte);
    - overflow: 10th byte contributes bits above bit 63 (byte & 0x7E
      nonzero at shift 63);
    - overlong: multi-byte encoding whose final byte is 0x00 (e.g.
      b"\\x80\\x00" for 0) — non-canonical; note real protobuf accepts
      these, we reject by policy.
  """

  def __init__(self) -> None:
    raise NotImplementedError

  def feed(self, chunk: bytes) -> list[int]:
    raise NotImplementedError

  def finish(self) -> None:
    raise NotImplementedError


class FrameDeframer:
  """Incremental deframer for varint-length-prefixed frames.

  [difficulty: core] [screen-core — the standard follow-up to varint]

  Wire format: stream = repeated frames, each frame is
      <uvarint payload-length> <payload bytes>
  The length varint follows StreamingVarintDecoder rules (64-bit bound,
  canonical: reject overlong / too-long / overflow with ValueError).

  API:
    __init__(max_frame: int) — frames longer than max_frame are hostile.
    feed(chunk: bytes) -> list[bytes]
        Consume chunk, return payloads of frames COMPLETED this call, in
        order, each as an independent bytes object (never a view into
        internal buffers). Zero-length frames are valid and yield b"".
    finish() -> None
        Raises ValueError if a partial frame dangles (torn mid-length or
        mid-payload).

  Error policy: the moment a decoded length exceeds max_frame, raise
  ValueError IMMEDIATELY — before buffering any payload. A declared
  1 GiB length must never cause 1 GiB of buffering (memory DoS guard).

  Perf requirement: total work O(bytes fed); repeated buf = buf[n:]
  slicing per frame is quadratic and considered wrong.
  """

  def __init__(self, max_frame: int) -> None:
    raise NotImplementedError

  def feed(self, chunk: bytes) -> list[bytes]:
    raise NotImplementedError

  def finish(self) -> None:
    raise NotImplementedError


class Utf8StreamValidator:
  """Streaming strict UTF-8 validator (LeetCode 393 generalized, harder).

  [difficulty: hard] [depth — the streaming+strictness version is a
  follow-up discussion; LC393's single-buffer version is asked live]

  Validate that the concatenation of all chunks fed so far is a prefix
  of some valid UTF-8 byte sequence, under RFC 3629 strictness:
    - lead bytes 0x80..0xBF, 0xC0, 0xC1, 0xF5..0xFF are invalid;
    - continuation bytes must be 0x80..0xBF, with the FIRST continuation
      further restricted: after 0xE0 -> 0xA0..0xBF (overlong), after
      0xED -> 0x80..0x9F (surrogates U+D800..DFFF), after 0xF0 ->
      0x90..0xBF (overlong), after 0xF4 -> 0x80..0x8F (> U+10FFFF).

  API:
    feed(chunk: bytes) -> bool
        True if still valid so far (a torn multi-byte sequence at the
        chunk edge is VALID — need more data). Once invalid, sticky:
        every later call returns False.
    finish() -> bool
        True only if valid AND no dangling incomplete sequence.

  Returns bools rather than raising to match the LC-style contract;
  production code would raise UnicodeDecodeError.
  """

  def __init__(self) -> None:
    raise NotImplementedError

  def feed(self, chunk: bytes) -> bool:
    raise NotImplementedError

  def finish(self) -> bool:
    raise NotImplementedError


def cobs_encode(data: bytes) -> bytes:
  """COBS-encode data: output contains no 0x00 byte.

  [difficulty: warmup] [depth — framing-strategy discussion fodder]

  Encoding: split data into zero-free runs of at most 254 bytes. Emit
  each run as <code><run> where code = len(run) + 1 (0x01..0xFF).
  A code < 0xFF means the run was terminated by a zero in the input
  (the zero is implied, not emitted); code == 0xFF means 254 run bytes
  with NO implied zero (the run continues in the next block). The final
  block is terminated by end-of-input instead of a zero. No 0x00 frame
  delimiter is appended (caller's job).

  Canonical vectors (from the COBS paper / Wikipedia):
    b""                    -> b"\\x01"
    b"\\x00"               -> b"\\x01\\x01"
    b"\\x11\\x22\\x00\\x33" -> b"\\x03\\x11\\x22\\x02\\x33"
    254 nonzero bytes      -> b"\\xff" + those 254 bytes (255 total)

  Args: data — arbitrary bytes. Returns: encoded bytes, zero-free,
  len <= len(data) + ceil(len(data)/254) + (1 if data is empty or has
  structure requiring it — never more than 1 byte per 254).
  """
  raise NotImplementedError


def cobs_decode(data: bytes) -> bytes:
  """Inverse of cobs_encode.

  [difficulty: core] [depth]

  Decoding: repeatedly read code byte c (must be 0x01..0xFF), copy the
  next c-1 bytes to output, then append 0x00 unless c == 0xFF or input
  is exhausted.

  Error policy — raise ValueError on:
    - empty input (valid COBS is never empty; b"" encodes to b"\\x01");
    - any 0x00 byte anywhere in the input;
    - truncated block: code promises more bytes than remain.

  Round-trip law: cobs_decode(cobs_encode(d)) == d for all d.
  """
  raise NotImplementedError


class TlvStreamParser:
  """Incremental TLV (tag-length-value) record parser.

  [difficulty: core] [screen-core — protobuf wire format is exactly
  varint-tag TLV, so this is a natural live extension]

  Wire format: stream = repeated records, each record is
      <uvarint tag> <uvarint length> <length payload bytes>
  Both varints follow StreamingVarintDecoder rules (64-bit, canonical).

  API:
    __init__(max_length: int = 1 << 20)
    feed(chunk: bytes) -> list[tuple[int, bytes]]
        Records COMPLETED this call as (tag, payload) pairs, payload an
        independent bytes object. Unknown tags are still returned —
        skipping is the CALLER's forward-compat decision; the length
        field is what makes skipping possible at all.
    finish() -> None
        Raises ValueError on a dangling partial record (torn mid-tag,
        mid-length, or mid-payload).

  Error policy: ValueError immediately when a decoded length exceeds
  max_length (before buffering payload), or on malformed varints.
  """

  def __init__(self, max_length: int = 1 << 20) -> None:
    raise NotImplementedError

  def feed(self, chunk: bytes) -> list[tuple[int, bytes]]:
    raise NotImplementedError

  def finish(self) -> None:
    raise NotImplementedError


class ChunkedDecoder:
  """Incremental HTTP/1.1 chunked transfer-coding body decoder.

  [difficulty: hard] [depth — state-machine discussion; shows you have
  written a real protocol decoder]

  Wire format (RFC 9112): body = repeated chunks
      chunk-size-in-hex [;chunk-extensions] CRLF
      <chunk-size bytes of data> CRLF
  terminated by a 0-size chunk, then zero or more trailer lines, then a
  final empty line (CRLF). Chunk extensions after ';' are ignored.
  Trailer lines are consumed and discarded.

  API:
    __init__(max_chunk: int = 1 << 24)
    feed(chunk: bytes) -> bytes
        Decoded body bytes newly available from this call. After the
        terminator has been consumed, self.done is True and feeding any
        further nonempty data raises ValueError (trailing garbage is
        malformed, not ignorable — request smuggling lives there).
    done: bool attribute/property.
    finish() -> None
        Raises ValueError unless the terminal 0-chunk sequence was
        fully consumed.

  Error policy (ValueError):
    - non-hex size line (empty size, or invalid hex digits before ';');
    - size > max_chunk (hostile declared size — check at parse time);
    - size line exceeding 256 bytes without CRLF (unbounded-line guard);
    - missing CRLF after chunk data (byte discipline is exact);
    - data after completion.
  Line endings are strict CRLF; CR and LF may arrive in different
  chunks (state machine must tolerate every split).
  """

  def __init__(self, max_chunk: int = 1 << 24) -> None:
    raise NotImplementedError

  def feed(self, chunk: bytes) -> bytes:
    raise NotImplementedError

  def finish(self) -> None:
    raise NotImplementedError
