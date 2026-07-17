"""Reference solutions for the streaming/incremental decoding module."""


class StreamingVarintDecoder:
  def __init__(self) -> None:
    self._acc = 0
    self._n = 0

  def feed(self, chunk: bytes) -> list[int]:
    out = []
    acc, n = self._acc, self._n
    for b in chunk:
      if n == 9:
        if b & 0x80:
          raise ValueError('varint too long (more than 10 bytes)')
        if b & 0x7E:
          raise ValueError('varint overflows 64 bits')
      acc |= (b & 0x7F) << (7 * n)
      if b & 0x80:
        n += 1
      else:
        if n > 0 and b == 0:
          raise ValueError('overlong varint encoding')
        out.append(acc)
        acc = 0
        n = 0
    self._acc, self._n = acc, n
    return out

  def finish(self) -> None:
    if self._n:
      raise ValueError('truncated varint at end of stream')


class FrameDeframer:
  def __init__(self, max_frame: int) -> None:
    self._max = max_frame
    self._buf = bytearray()
    self._pos = 0
    self._acc = 0
    self._n = 0
    self._need = -1

  def feed(self, chunk: bytes) -> list[bytes]:
    out = []
    buf = self._buf
    buf += chunk
    pos = self._pos
    while True:
      if self._need < 0:
        while pos < len(buf):
          b = buf[pos]
          pos += 1
          if self._n == 9:
            if b & 0x80:
              raise ValueError('frame length varint too long')
            if b & 0x7E:
              raise ValueError('frame length overflows 64 bits')
          self._acc |= (b & 0x7F) << (7 * self._n)
          if b & 0x80:
            self._n += 1
          else:
            if self._n > 0 and b == 0:
              raise ValueError('overlong frame length varint')
            length = self._acc
            self._acc = 0
            self._n = 0
            # DoS guard fires here, before any payload buffering
            if length > self._max:
              raise ValueError(
                f'frame length {length} exceeds max_frame {self._max}'
              )
            self._need = length
            break
        else:
          break
      if self._need >= 0:
        if len(buf) - pos < self._need:
          break
        out.append(bytes(memoryview(buf)[pos : pos + self._need]))
        pos += self._need
        self._need = -1
    self._pos = pos
    # compact only when at least half the buffer is dead: amortized O(1)/byte
    if pos > 4096 and pos * 2 > len(buf):
      del buf[:pos]
      self._pos = 0
    return out

  def finish(self) -> None:
    if self._n or self._need >= 0:
      raise ValueError('truncated frame at end of stream')


class Utf8StreamValidator:
  def __init__(self) -> None:
    self._remaining = 0
    self._lo = 0x80
    self._hi = 0xBF
    self._valid = True

  def feed(self, chunk: bytes) -> bool:
    if not self._valid:
      return False
    rem, lo, hi = self._remaining, self._lo, self._hi
    for b in chunk:
      if rem:
        if not (lo <= b <= hi):
          self._valid = False
          return False
        rem -= 1
        lo, hi = 0x80, 0xBF
      elif b < 0x80:
        pass
      elif 0xC2 <= b <= 0xDF:
        rem, lo, hi = 1, 0x80, 0xBF
      elif b == 0xE0:
        rem, lo, hi = 2, 0xA0, 0xBF
      elif b == 0xED:
        rem, lo, hi = 2, 0x80, 0x9F
      elif 0xE1 <= b <= 0xEF:
        rem, lo, hi = 2, 0x80, 0xBF
      elif b == 0xF0:
        rem, lo, hi = 3, 0x90, 0xBF
      elif b == 0xF4:
        rem, lo, hi = 3, 0x80, 0x8F
      elif 0xF1 <= b <= 0xF3:
        rem, lo, hi = 3, 0x80, 0xBF
      else:
        self._valid = False
        return False
    self._remaining, self._lo, self._hi = rem, lo, hi
    return True

  def finish(self) -> bool:
    return self._valid and self._remaining == 0


def cobs_encode(data: bytes) -> bytes:
  out = bytearray()
  idx = 0
  while True:
    end = min(idx + 254, len(data))
    z = data.find(0, idx, end)
    if z < 0:
      block = data[idx:end]
      if len(block) == 254:
        out.append(0xFF)
        out += block
        idx = end
        if idx == len(data):
          break
      else:
        out.append(len(block) + 1)
        out += block
        break
    else:
      out.append(z - idx + 1)
      out += data[idx:z]
      idx = z + 1
      if idx == len(data):
        out.append(0x01)
        break
  return bytes(out)


def cobs_decode(data: bytes) -> bytes:
  if not data:
    raise ValueError('empty input is not valid COBS')
  out = bytearray()
  i = 0
  n = len(data)
  while i < n:
    code = data[i]
    if code == 0:
      raise ValueError('zero byte inside COBS data')
    i += 1
    end = i + code - 1
    if end > n:
      raise ValueError('truncated COBS block')
    block = data[i:end]
    if 0 in block:
      raise ValueError('zero byte inside COBS data')
    out += block
    i = end
    if code != 0xFF and i < n:
      out.append(0)
  return bytes(out)


class TlvStreamParser:
  _TAG, _LEN, _PAYLOAD = range(3)

  def __init__(self, max_length: int = 1 << 20) -> None:
    self._max = max_length
    self._buf = bytearray()
    self._pos = 0
    self._acc = 0
    self._n = 0
    self._phase = self._TAG
    self._tag = 0
    self._need = 0

  def _varint_step(self, b: int):
    if self._n == 9:
      if b & 0x80:
        raise ValueError('varint too long (more than 10 bytes)')
      if b & 0x7E:
        raise ValueError('varint overflows 64 bits')
    self._acc |= (b & 0x7F) << (7 * self._n)
    if b & 0x80:
      self._n += 1
      return None
    if self._n > 0 and b == 0:
      raise ValueError('overlong varint encoding')
    v = self._acc
    self._acc = 0
    self._n = 0
    return v

  def feed(self, chunk: bytes) -> list[tuple[int, bytes]]:
    out = []
    buf = self._buf
    buf += chunk
    pos = self._pos
    while True:
      if self._phase == self._PAYLOAD:
        if len(buf) - pos < self._need:
          break
        out.append((self._tag, bytes(memoryview(buf)[pos : pos + self._need])))
        pos += self._need
        self._phase = self._TAG
      else:
        if pos >= len(buf):
          break
        v = self._varint_step(buf[pos])
        pos += 1
        if v is None:
          continue
        if self._phase == self._TAG:
          self._tag = v
          self._phase = self._LEN
        else:
          if v > self._max:
            raise ValueError(f'tlv length {v} exceeds max_length {self._max}')
          self._need = v
          self._phase = self._PAYLOAD
    self._pos = pos
    if pos > 4096 and pos * 2 > len(buf):
      del buf[:pos]
      self._pos = 0
    return out

  def finish(self) -> None:
    if self._phase != self._TAG or self._n:
      raise ValueError('truncated TLV record at end of stream')


class ChunkedDecoder:
  _SIZE, _DATA, _CR, _LF, _TRAILER, _DONE = range(6)

  def __init__(self, max_chunk: int = 1 << 24) -> None:
    self._max = max_chunk
    self._state = self._SIZE
    self._line = bytearray()
    self._need = 0

  @property
  def done(self) -> bool:
    return self._state == self._DONE

  def _parse_size(self, line: bytes) -> int:
    field = line.split(b';', 1)[0]
    if not field or any(c not in b'0123456789abcdefABCDEF' for c in field):
      raise ValueError(f'malformed chunk size line: {line!r}')
    size = int(field, 16)
    if size > self._max:
      raise ValueError(f'chunk size {size} exceeds max_chunk {self._max}')
    return size

  def feed(self, chunk: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(chunk)
    while i < n:
      st = self._state
      if st == self._DATA:
        take = min(self._need, n - i)
        out += chunk[i : i + take]
        i += take
        self._need -= take
        if self._need == 0:
          self._state = self._CR
      elif st == self._CR:
        if chunk[i] != 0x0D:
          raise ValueError('expected CR after chunk data')
        i += 1
        self._state = self._LF
      elif st == self._LF:
        if chunk[i] != 0x0A:
          raise ValueError('expected LF after chunk data')
        i += 1
        self._state = self._SIZE
      elif st == self._SIZE:
        b = chunk[i]
        i += 1
        self._line.append(b)
        if len(self._line) > 258:
          raise ValueError('chunk size line too long')
        if b == 0x0A:
          if len(self._line) < 2 or self._line[-2] != 0x0D:
            raise ValueError('bare LF in chunk size line')
          size = self._parse_size(bytes(self._line[:-2]))
          self._line.clear()
          if size == 0:
            self._state = self._TRAILER
          else:
            self._need = size
            self._state = self._DATA
      elif st == self._TRAILER:
        b = chunk[i]
        i += 1
        self._line.append(b)
        if len(self._line) > 4096:
          raise ValueError('trailer line too long')
        if b == 0x0A:
          if len(self._line) < 2 or self._line[-2] != 0x0D:
            raise ValueError('bare LF in trailer line')
          if len(self._line) == 2:
            self._state = self._DONE
          self._line.clear()
      else:
        raise ValueError('data after chunked body terminator')
    return bytes(out)

  def finish(self) -> None:
    if self._state != self._DONE:
      raise ValueError('truncated chunked body at end of stream')
