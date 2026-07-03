"""Reference implementations for the stream-algorithms problems. stdlib only."""

import collections
import hashlib
import heapq


class MovingAverage:
  def __init__(self, size: int) -> None:
    if size < 1:
      raise ValueError('size must be >= 1')
    self._buf = [0.0] * size
    self._size = size
    self._count = 0
    self._pos = 0
    self._sum = 0.0

  def next(self, x: float) -> float:
    if self._count < self._size:
      self._count += 1
    else:
      self._sum -= self._buf[self._pos]
    self._buf[self._pos] = x
    self._sum += x
    self._pos = (self._pos + 1) % self._size
    return self._sum / self._count


class StreamingMedian:
  def __init__(self) -> None:
    self._lo = []  # max-heap of the low half, stored negated
    self._hi = []  # min-heap of the high half

  def add(self, x: float) -> None:
    if self._lo and x > -self._lo[0]:
      heapq.heappush(self._hi, x)
    else:
      heapq.heappush(self._lo, -x)
    if len(self._lo) > len(self._hi) + 1:
      heapq.heappush(self._hi, -heapq.heappop(self._lo))
    elif len(self._hi) > len(self._lo):
      heapq.heappush(self._lo, -heapq.heappop(self._hi))

  def median(self) -> float:
    if not self._lo:
      raise ValueError('empty: no values added')
    if len(self._lo) > len(self._hi):
      return float(-self._lo[0])
    return (-self._lo[0] + self._hi[0]) / 2.0


def sliding_window_max(values, k: int) -> list:
  if k < 1 or k > len(values):
    raise ValueError('range: need 1 <= k <= len(values)')
  out = []
  dq = collections.deque()
  for i, v in enumerate(values):
    while dq and values[dq[-1]] <= v:
      dq.pop()
    dq.append(i)
    if dq[0] <= i - k:
      dq.popleft()
    if i >= k - 1:
      out.append(values[dq[0]])
  return out


def reservoir_sample(stream, k: int, rng) -> list:
  if k < 1:
    raise ValueError('k must be >= 1')
  reservoir = []
  for i, x in enumerate(stream):
    if i < k:
      reservoir.append(x)
    else:
      j = rng.randrange(i + 1)
      if j < k:
        reservoir[j] = x
  return reservoir


def misra_gries(stream, k: int) -> dict:
  if k < 2:
    raise ValueError('k must be >= 2')
  counters = {}
  for x in stream:
    if x in counters:
      counters[x] += 1
    elif len(counters) < k - 1:
      counters[x] = 1
    else:
      dead = []
      for key in counters:
        counters[key] -= 1
        if counters[key] == 0:
          dead.append(key)
      for key in dead:
        del counters[key]
  return counters


class CountMinSketch:
  def __init__(self, width: int, depth: int, seed: int) -> None:
    if width < 1 or depth < 1:
      raise ValueError('width and depth must be >= 1')
    self.width = width
    self.depth = depth
    self.total = 0
    self._rows = [[0] * width for _ in range(depth)]
    base = seed.to_bytes(8, 'big', signed=True)
    self._keys = [
      hashlib.blake2b(base + r.to_bytes(4, 'big'), digest_size=16).digest()
      for r in range(depth)
    ]

  @staticmethod
  def _data(item) -> bytes:
    if isinstance(item, bytes):
      return item
    if isinstance(item, str):
      return item.encode('utf-8')
    raise TypeError('item must be str or bytes')

  def _index(self, data: bytes, row: int) -> int:
    h = hashlib.blake2b(data, digest_size=8, key=self._keys[row])
    return int.from_bytes(h.digest(), 'big') % self.width

  def add(self, item, count: int = 1) -> None:
    if count < 1:
      raise ValueError('count must be >= 1')
    data = self._data(item)
    for r in range(self.depth):
      self._rows[r][self._index(data, r)] += count
    self.total += count

  def estimate(self, item) -> int:
    data = self._data(item)
    return min(self._rows[r][self._index(data, r)] for r in range(self.depth))


class SwagWindow:
  def __init__(self, op) -> None:
    self._op = op
    self._front = []  # (value, fold of this value..front bottom, in window order)
    self._back = []  # (value, fold of back bottom..this value, in window order)

  def push(self, x) -> None:
    agg = x if not self._back else self._op(self._back[-1][1], x)
    self._back.append((x, agg))

  def pop(self):
    if not self._front:
      if not self._back:
        raise IndexError('pop from empty window')
      while self._back:
        v = self._back.pop()[0]
        agg = v if not self._front else self._op(v, self._front[-1][1])
        self._front.append((v, agg))
    return self._front.pop()[0]

  def query(self):
    if not self._front and not self._back:
      raise ValueError('empty: query on empty window')
    if not self._front:
      return self._back[-1][1]
    if not self._back:
      return self._front[-1][1]
    return self._op(self._front[-1][1], self._back[-1][1])
