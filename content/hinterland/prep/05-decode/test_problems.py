"""Self-contained test harness for the streaming module.

Run against solutions:  PRACTICE_MODULE=solutions python3 test_problems.py
Run against your work:  python3 test_problems.py
"""

import importlib
import os
import sys

MOD = importlib.import_module(os.environ.get('PRACTICE_MODULE', 'problems'))

RESULTS = []


def case(name):
  def deco(fn):
    try:
      fn()
      print(f'PASS {name}')
      RESULTS.append(True)
    except Exception as e:
      print(f'FAIL {name}: {type(e).__name__}: {e}')
      RESULTS.append(False)
    return fn

  return deco


def expect(cond, msg):
  if not cond:
    raise AssertionError(msg)


def expect_raises(fn, msg):
  try:
    fn()
  except ValueError:
    return
  except Exception as e:
    raise AssertionError(
      f'{msg}: raised {type(e).__name__} instead of ValueError'
    )
  raise AssertionError(f'{msg}: no ValueError raised')


def uvarint(n):
  out = bytearray()
  while True:
    b = n & 0x7F
    n >>= 7
    if n:
      out.append(b | 0x80)
    else:
      out.append(b)
      return bytes(out)


def assert_all_splits(stream, runner, expected):
  whole = runner([stream])
  expect(whole == expected, f'fed whole: {whole!r} != {expected!r}')
  bb = runner([bytes([b]) for b in stream])
  expect(bb == expected, f'fed byte-at-a-time: {bb!r} != {expected!r}')
  for i in range(len(stream) + 1):
    got = runner([stream[:i], stream[i:]])
    expect(got == expected, f'2-part split at {i}: {got!r} != {expected!r}')


def run_varint(chunks):
  d = MOD.StreamingVarintDecoder()
  out = []
  for c in chunks:
    out.extend(d.feed(c))
  d.finish()
  return out


def make_run_deframer(max_frame):
  def run(chunks):
    d = MOD.FrameDeframer(max_frame)
    out = []
    for c in chunks:
      out.extend(d.feed(c))
    d.finish()
    return out

  return run


def run_utf8(chunks):
  v = MOD.Utf8StreamValidator()
  last = True
  for c in chunks:
    last = v.feed(c)
  return (last, v.finish())


def run_tlv(chunks):
  p = MOD.TlvStreamParser()
  out = []
  for c in chunks:
    out.extend(p.feed(c))
  p.finish()
  return out


def run_chunked(chunks):
  d = MOD.ChunkedDecoder()
  out = bytearray()
  for c in chunks:
    out += d.feed(c)
  d.finish()
  return (bytes(out), d.done)


# ---------------- StreamingVarintDecoder ----------------

VARINT_VALUES = [
  0,
  1,
  127,
  128,
  300,
  16383,
  16384,
  2**32 - 1,
  2**63,
  2**64 - 1,
]
VARINT_STREAM = b''.join(uvarint(v) for v in VARINT_VALUES)


@case('varint_golden_all_splits')
def _():
  assert_all_splits(VARINT_STREAM, run_varint, VARINT_VALUES)


@case('varint_per_call_emission')
def _():
  d = MOD.StreamingVarintDecoder()
  expect(d.feed(b'\xac') == [], 'partial value must emit nothing')
  expect(d.feed(b'\x02') == [300], 'completion must emit exactly [300]')
  d.finish()


@case('varint_empty_chunk_noop')
def _():
  d = MOD.StreamingVarintDecoder()
  expect(d.feed(b'') == [], 'empty chunk must return []')
  d.feed(b'\x80')
  expect(d.feed(b'') == [], 'empty chunk mid-value must return []')
  expect(d.feed(b'\x01') == [128], 'state must survive empty chunk')


@case('varint_max_u64_boundary')
def _():
  got = run_varint([b'\xff' * 9 + b'\x01'])
  expect(got == [2**64 - 1], f'max u64: {got!r}')


@case('varint_overflow_tenth_byte')
def _():
  expect_raises(
    lambda: run_varint([b'\xff' * 9 + b'\x02']),
    '10th byte with bits above bit 63 must raise',
  )


@case('varint_too_long_11_bytes')
def _():
  expect_raises(
    lambda: run_varint([b'\x80' * 10]), 'continuation on 10th byte must raise'
  )
  d = MOD.StreamingVarintDecoder()
  for i in range(9):
    d.feed(b'\xff')
  expect_raises(
    lambda: d.feed(b'\xff'), 'byte-wise too-long must raise on 10th feed'
  )


@case('varint_overlong_two_byte')
def _():
  expect_raises(lambda: run_varint([b'\x80\x00']), 'overlong 0 must raise')


@case('varint_overlong_multibyte')
def _():
  expect_raises(
    lambda: run_varint([b'\x80\x80\x00']), '3-byte overlong must raise'
  )
  expect_raises(
    lambda: run_varint([b'\xac\x80\x00']), 'trailing 0x00 group must raise'
  )


@case('varint_single_zero_ok')
def _():
  expect(run_varint([b'\x00']) == [0], 'single 0x00 is canonical zero')


@case('varint_dangling_finish_raises')
def _():
  d = MOD.StreamingVarintDecoder()
  d.feed(b'\xac')
  expect_raises(d.finish, 'finish mid-value must raise')
  d2 = MOD.StreamingVarintDecoder()
  d2.feed(b'\x80\x80')
  expect_raises(d2.finish, 'finish mid-value (zero payload bits) must raise')


@case('varint_finish_clean_noop')
def _():
  d = MOD.StreamingVarintDecoder()
  expect(d.feed(b'\x08\x96\x01') == [8, 150], 'two values in one chunk')
  d.finish()


# ---------------- FrameDeframer ----------------

DEFRAMER_FRAMES = [b'hello', b'', b'x' * 300, bytes(range(256))]
DEFRAMER_STREAM = b''.join(uvarint(len(p)) + p for p in DEFRAMER_FRAMES)


@case('deframer_golden_all_splits')
def _():
  assert_all_splits(
    DEFRAMER_STREAM, make_run_deframer(1 << 16), DEFRAMER_FRAMES
  )


@case('deframer_zero_length_frame')
def _():
  got = make_run_deframer(16)([b'\x00\x00\x00'])
  expect(got == [b'', b'', b''], f'three empty frames: {got!r}')


@case('deframer_per_call_emission')
def _():
  d = MOD.FrameDeframer(1 << 16)
  expect(d.feed(uvarint(3)) == [], 'length alone must emit nothing')
  expect(d.feed(b'ab') == [], 'partial payload must emit nothing')
  expect(d.feed(b'c') == [b'abc'], 'payload completion must emit frame')
  d.finish()


@case('deframer_gib_length_guard')
def _():
  d = MOD.FrameDeframer(1 << 16)
  expect_raises(
    lambda: d.feed(uvarint(1 << 30)),
    'declared 1 GiB frame must raise at length-decode time',
  )


@case('deframer_max_boundary_exact')
def _():
  got = make_run_deframer(8)([uvarint(8) + b'12345678'])
  expect(got == [b'12345678'], 'length == max_frame is allowed')
  d = MOD.FrameDeframer(8)
  expect_raises(lambda: d.feed(uvarint(9)), 'length == max_frame+1 must raise')


@case('deframer_torn_mid_length_finish_raises')
def _():
  d = MOD.FrameDeframer(1 << 16)
  expect(d.feed(uvarint(300)[:1]) == [], 'torn mid-length emits nothing')
  expect_raises(d.finish, 'finish torn mid-length must raise')


@case('deframer_torn_mid_payload_finish_raises')
def _():
  d = MOD.FrameDeframer(1 << 16)
  expect(d.feed(uvarint(5) + b'he') == [], 'torn mid-payload emits nothing')
  expect_raises(d.finish, 'finish torn mid-payload must raise')


@case('deframer_overlong_length_raises')
def _():
  d = MOD.FrameDeframer(1 << 16)
  expect_raises(
    lambda: d.feed(b'\x85\x00'), 'overlong length varint must raise'
  )


@case('deframer_payloads_are_bytes_copies')
def _():
  d = MOD.FrameDeframer(1 << 16)
  frames = d.feed(uvarint(3) + b'abc' + uvarint(2) + b'xy')
  expect(frames == [b'abc', b'xy'], f'got {frames!r}')
  expect(
    all(type(f) is bytes for f in frames), 'payloads must be bytes objects'
  )


@case('deframer_many_small_frames_compaction')
def _():
  frames = [bytes([65 + (i % 26)]) * (i % 7) for i in range(1000)]
  stream = b''.join(uvarint(len(p)) + p for p in frames)
  d = MOD.FrameDeframer(64)
  out = []
  for i in range(0, len(stream), 13):
    out.extend(d.feed(stream[i : i + 13]))
  d.finish()
  expect(out == frames, '1000 frames fed in 13-byte chunks must round-trip')


# ---------------- Utf8StreamValidator ----------------


@case('utf8_ascii_all_splits')
def _():
  assert_all_splits(b'hello, world', run_utf8, (True, True))


@case('utf8_multibyte_golden_all_splits')
def _():
  assert_all_splits('héllo wörld 你好 🎉'.encode(), run_utf8, (True, True))


@case('utf8_dangling_lead_all_splits')
def _():
  assert_all_splits(b'A\xc3', run_utf8, (True, False))
  assert_all_splits(b'ok\xf0\x9f\x8e', run_utf8, (True, False))


@case('utf8_overlong_c0_80_all_splits')
def _():
  assert_all_splits(b'\xc0\x80', run_utf8, (False, False))


@case('utf8_surrogate_rejected')
def _():
  assert_all_splits(b'\xed\xa0\x80', run_utf8, (False, False))
  expect(run_utf8([b'\xed\x9f\xbf']) == (True, True), 'U+D7FF must pass')
  expect(run_utf8([b'\xee\x80\x80']) == (True, True), 'U+E000 must pass')


@case('utf8_beyond_u10ffff_rejected')
def _():
  assert_all_splits(b'\xf4\x90\x80\x80', run_utf8, (False, False))
  expect(run_utf8([b'\xf4\x8f\xbf\xbf']) == (True, True), 'U+10FFFF must pass')
  expect(
    run_utf8([b'\xf5\x80\x80\x80']) == (False, False), '0xF5 lead must fail'
  )


@case('utf8_overlong_three_and_four_byte')
def _():
  expect(
    run_utf8([b'\xe0\x80\xaf']) == (False, False), 'overlong 3-byte must fail'
  )
  expect(
    run_utf8([b'\xf0\x80\x80\x80']) == (False, False),
    'overlong 4-byte must fail',
  )
  expect(run_utf8([b'\xe0\xa0\x80']) == (True, True), 'U+0800 must pass')
  expect(run_utf8([b'\xf0\x90\x80\x80']) == (True, True), 'U+10000 must pass')


@case('utf8_invalid_lead_bytes')
def _():
  for lead in (b'\x80', b'\xbf', b'\xc0', b'\xc1', b'\xf5', b'\xff'):
    v = MOD.Utf8StreamValidator()
    expect(v.feed(lead) is False, f'lead {lead!r} must be invalid')


@case('utf8_sticky_after_invalid')
def _():
  v = MOD.Utf8StreamValidator()
  expect(v.feed(b'\xc0\x80') is False, 'invalid input')
  expect(v.feed(b'plain ascii') is False, 'must stay False after invalid')
  expect(v.feed(b'') is False, 'empty chunk must stay False')
  expect(v.finish() is False, 'finish must stay False')


@case('utf8_torn_4byte_across_three_feeds')
def _():
  v = MOD.Utf8StreamValidator()
  expect(v.feed(b'\xf0') is True, 'lead alone is a valid prefix')
  expect(v.feed(b'\x9f\x8e') is True, 'mid-sequence still valid')
  expect(v.feed(b'\x89') is True, 'sequence completes')
  expect(v.finish() is True, 'no dangling state')


# ---------------- COBS ----------------

COBS_VECTORS = [
  (b'', b'\x01'),
  (b'\x00', b'\x01\x01'),
  (b'\x00\x00', b'\x01\x01\x01'),
  (b'\x11\x22\x00\x33', b'\x03\x11\x22\x02\x33'),
  (b'\x11\x22\x33\x44', b'\x05\x11\x22\x33\x44'),
  (b'\x11\x00\x00\x00', b'\x02\x11\x01\x01\x01'),
  (bytes(range(1, 255)), b'\xff' + bytes(range(1, 255))),
  (bytes(range(0, 255)), b'\x01\xff' + bytes(range(1, 255))),
  (bytes(range(1, 256)), b'\xff' + bytes(range(1, 255)) + b'\x02\xff'),
  (
    bytes(range(2, 256)) + b'\x00',
    b'\xff' + bytes(range(2, 256)) + b'\x01\x01',
  ),
]


@case('cobs_encode_canonical_vectors')
def _():
  for raw, enc in COBS_VECTORS:
    got = MOD.cobs_encode(raw)
    expect(
      got == enc,
      f'encode({raw[:8]!r}...len{len(raw)}): {got[:12]!r} != {enc[:12]!r}',
    )


@case('cobs_decode_canonical_vectors')
def _():
  for raw, enc in COBS_VECTORS:
    got = MOD.cobs_decode(enc)
    expect(got == raw, f'decode({enc[:8]!r}...): {got[:12]!r} != {raw[:12]!r}')


@case('cobs_roundtrip_suite')
def _():
  suite = [raw for raw, _ in COBS_VECTORS]
  suite += [b'\x00' * 254, bytes(range(256)) * 3, b'a' * 1000, b'\x01']
  for raw in suite:
    expect(
      MOD.cobs_decode(MOD.cobs_encode(raw)) == raw,
      f'round-trip failed for len {len(raw)}',
    )


@case('cobs_encoded_output_zero_free')
def _():
  for raw, _ in COBS_VECTORS:
    expect(
      0 not in MOD.cobs_encode(raw), f'zero byte in encoding of {raw[:8]!r}'
    )
  expect(
    0 not in MOD.cobs_encode(bytes(500)), 'zero byte in encoding of 500 zeros'
  )


@case('cobs_overhead_bound')
def _():
  raw = b'\x7f' * 1000
  enc = MOD.cobs_encode(raw)
  expect(
    len(enc) == 1000 + 4,
    f'1000 zero-free bytes need ceil(1000/254)=4 overhead, got {len(enc) - 1000}',
  )


@case('cobs_decode_empty_raises')
def _():
  expect_raises(lambda: MOD.cobs_decode(b''), 'empty input must raise')


@case('cobs_decode_zero_byte_raises')
def _():
  expect_raises(
    lambda: MOD.cobs_decode(b'\x00\x11'), 'zero code byte must raise'
  )
  expect_raises(
    lambda: MOD.cobs_decode(b'\x03\x11\x00'), 'zero inside block must raise'
  )


@case('cobs_decode_truncated_raises')
def _():
  expect_raises(
    lambda: MOD.cobs_decode(b'\x02'), 'code promising 1 byte with none left'
  )
  expect_raises(
    lambda: MOD.cobs_decode(b'\xff' + b'\x01' * 253),
    '0xFF block with only 253 bytes must raise',
  )


# ---------------- TlvStreamParser ----------------

TLV_RECORDS = [
  (1, b'hello'),
  (300, b''),
  (7, bytes(255)),
  (2**32, b'\xff\x00\xfe'),
]
TLV_STREAM = b''.join(uvarint(t) + uvarint(len(p)) + p for t, p in TLV_RECORDS)


@case('tlv_golden_all_splits')
def _():
  assert_all_splits(TLV_STREAM, run_tlv, TLV_RECORDS)


@case('tlv_per_call_emission')
def _():
  p = MOD.TlvStreamParser()
  expect(p.feed(uvarint(9)) == [], 'tag alone emits nothing')
  expect(p.feed(uvarint(2)) == [], 'tag+length emits nothing')
  expect(p.feed(b'h') == [], 'partial payload emits nothing')
  expect(p.feed(b'i') == [(9, b'hi')], 'completion emits the record')
  p.finish()


@case('tlv_hostile_length_guard')
def _():
  p = MOD.TlvStreamParser()
  expect_raises(
    lambda: p.feed(uvarint(5) + uvarint(1 << 40)),
    'declared 1 TiB payload must raise at length-decode time',
  )
  p2 = MOD.TlvStreamParser(max_length=16)
  expect_raises(
    lambda: p2.feed(uvarint(1) + uvarint(17)),
    'length just over max_length must raise',
  )


@case('tlv_torn_mid_tag_finish_raises')
def _():
  p = MOD.TlvStreamParser()
  expect(p.feed(b'\xac') == [], 'torn mid-tag emits nothing')
  expect_raises(p.finish, 'finish torn mid-tag must raise')


@case('tlv_torn_mid_length_finish_raises')
def _():
  p = MOD.TlvStreamParser()
  expect(p.feed(uvarint(1) + b'\x80') == [], 'torn mid-length emits nothing')
  expect_raises(p.finish, 'finish torn mid-length must raise')


@case('tlv_torn_mid_payload_finish_raises')
def _():
  p = MOD.TlvStreamParser()
  expect(
    p.feed(uvarint(1) + uvarint(4) + b'ab') == [],
    'torn mid-payload emits nothing',
  )
  expect_raises(p.finish, 'finish torn mid-payload must raise')


@case('tlv_zero_length_payload')
def _():
  got = run_tlv([uvarint(42) + uvarint(0)])
  expect(got == [(42, b'')], f'zero-length record: {got!r}')


# ---------------- ChunkedDecoder ----------------

WIKI = b'4\r\nWiki\r\n5\r\npedia\r\nE\r\n in\r\n\r\nchunks.\r\n0\r\n\r\n'
WIKI_BODY = b'Wikipedia in\r\n\r\nchunks.'


@case('chunked_wikipedia_all_splits')
def _():
  assert_all_splits(WIKI, run_chunked, (WIKI_BODY, True))


@case('chunked_extension_all_splits')
def _():
  assert_all_splits(
    b'4;name=val\r\nWiki\r\n0\r\n\r\n', run_chunked, (b'Wiki', True)
  )


@case('chunked_trailer_all_splits')
def _():
  assert_all_splits(
    b'5\r\nhello\r\n0\r\nX-Check: 1\r\n\r\n', run_chunked, (b'hello', True)
  )


@case('chunked_trailing_garbage_raises')
def _():
  expect_raises(
    lambda: run_chunked([WIKI + b'x']), 'byte after terminator must raise'
  )


@case('chunked_feed_after_done_raises')
def _():
  d = MOD.ChunkedDecoder()
  d.feed(WIKI)
  expect(d.done is True, 'done must be True after terminator')
  expect(d.feed(b'') == b'', 'empty feed after done is a no-op')
  expect_raises(lambda: d.feed(b'x'), 'nonempty feed after done must raise')


@case('chunked_bad_hex_raises')
def _():
  for line in (b'zz\r\n', b'+4\r\n', b'0x4\r\n', b' 4\r\n', b'\r\n'):
    d = MOD.ChunkedDecoder()
    expect_raises(lambda: d.feed(line), f'size line {line!r} must raise')


@case('chunked_missing_crlf_after_data_raises')
def _():
  d = MOD.ChunkedDecoder()
  expect_raises(
    lambda: d.feed(b'4\r\nWikiXY'), 'missing CRLF after chunk data'
  )


@case('chunked_gib_size_guard')
def _():
  d = MOD.ChunkedDecoder()
  expect_raises(
    lambda: d.feed(b'40000000\r\n'),
    'declared 1 GiB chunk must raise before buffering',
  )


@case('chunked_size_line_unbounded_guard')
def _():
  d = MOD.ChunkedDecoder()
  expect_raises(
    lambda: d.feed(b'1' * 300), 'size line without CRLF must hit guard'
  )


@case('chunked_bare_lf_raises')
def _():
  d = MOD.ChunkedDecoder()
  expect_raises(lambda: d.feed(b'4\nWiki'), 'bare LF size line must raise')


@case('chunked_truncated_finish_raises')
def _():
  d = MOD.ChunkedDecoder()
  d.feed(b'4\r\nWi')
  expect(d.done is False, 'not done mid-data')
  expect_raises(d.finish, 'finish mid-body must raise')
  d2 = MOD.ChunkedDecoder()
  d2.feed(b'4\r\nWiki\r\n0\r\n')
  expect_raises(d2.finish, 'finish before final CRLF must raise')


@case('chunked_incremental_body_emission')
def _():
  d = MOD.ChunkedDecoder()
  expect(d.feed(b'4\r\nWi') == b'Wi', 'data emitted as it arrives')
  expect(d.feed(b'ki\r\n') == b'ki', 'rest of chunk emitted')
  expect(d.feed(b'0\r\n\r\n') == b'', 'terminator emits nothing')
  expect(d.done is True, 'done after terminator')
  d.finish()


total = len(RESULTS)
passed = sum(RESULTS)
print(f'{passed}/{total} passed')
sys.exit(0 if passed == total else 1)
