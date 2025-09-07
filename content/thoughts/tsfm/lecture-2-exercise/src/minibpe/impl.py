from __future__ import annotations

import dataclasses, json, os, multiprocessing, heapq

from .core import PRETOKENIZER


def _parse_merges_batch(lines: list[str]) -> dict[tuple[int, int], int]:
  out: dict[tuple[int, int], int] = {}
  for line in lines:
    s = line.strip()
    if not s:
      continue
    parts = s.split()
    if len(parts) < 3:
      parts = s.split(',')
      if len(parts) < 3:
        continue
    a = int(parts[0])
    b = int(parts[1])
    new_id = int(parts[2])
    out[a, b] = new_id
  return out


def _parse_vocab_batch(lines: list[str]) -> dict[tuple[int, ...], int]:
  out: dict[tuple[int, ...], int] = {}
  for line in lines:
    s = line.rstrip('\n')
    if not s:
      continue
    if '\t' in s:
      left, right = s.split('\t', 1)
      token_text = json.loads(left)
      tok_id = int(right.strip())
      symbol = tuple(token_text.encode('utf-8'))
      out[symbol] = tok_id
    else:
      parts = [int(x) for x in s.split()]
      if len(parts) < 2:
        continue
      *symbol, tok_id = parts
      out[tuple(symbol)] = tok_id
  return out


def _batched_line_reader(path: str, batch_size: int = 200_000):
  with open(path, 'r') as f:
    batch: list[str] = []
    for line in f:
      batch.append(line)
      if len(batch) >= batch_size:
        yield batch
        batch = []
    if batch:
      yield batch


@dataclasses.dataclass
class Tokenizer:
  merges: dict[tuple[int, int], int]
  vocab: dict[tuple[int, ...], int]
  id_to_sym: dict[int, tuple[int, ...]] = dataclasses.field(init=False)
  ranks: dict[tuple[int, int], int] = dataclasses.field(init=False)
  encode_cache: dict[bytes, tuple[int, ...]] = dataclasses.field(default_factory=dict, init=False)
  cache_max_token_bytes: int = 100

  def __post_init__(self) -> None:
    self.id_to_sym = {tok_id: tuple(symbol) for symbol, tok_id in self.vocab.items() if tok_id >= 256}
    self.ranks = {pair: rank for rank, (pair, _) in enumerate(sorted(self.merges.items(), key=lambda kv: kv[1]))}

  @classmethod
  def from_pretrained(cls, fp: str) -> 'Tokenizer':
    merges_fp = os.path.join(fp, 'merges.txt')
    vocab_fp = os.path.join(fp, 'vocab.txt')
    procs = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(procs) as pool:
      merges_parts = pool.imap_unordered(_parse_merges_batch, _batched_line_reader(merges_fp), chunksize=1)
      merges: dict[tuple[int, int], int] = {}
      for part in merges_parts:
        merges.update(part)
      vocab_parts = pool.imap_unordered(_parse_vocab_batch, _batched_line_reader(vocab_fp), chunksize=1)
      vocab: dict[tuple[int, ...], int] = {}
      for part in vocab_parts:
        vocab.update(part)
      pool.close()
      pool.join()
    return cls(merges=merges, vocab=vocab)

  def _get_pairs(self, symbols: list[int]) -> list[tuple[int, int]]:
    if len(symbols) < 2:
      return []
    return list(zip(symbols, symbols[1:]))

  def _apply_bpe(self, symbols: list[int]) -> list[int]:
    n = len(symbols)
    if n < 2:
      return symbols
    ids = list(symbols)
    next_idx = list(range(1, n)) + [-1]
    prev_idx = [-1] + list(range(0, n - 1))
    alive = [True] * n
    stamp = [0] * n
    heap: list[tuple[int, int, int, int, int]] = []

    ranks = self.ranks
    merges = self.merges
    heappush = heapq.heappush
    heappop = heapq.heappop

    def push(i: int) -> None:
      if i < 0 or i >= n:
        return
      j = next_idx[i]
      if j == -1:
        return
      pair = (ids[i], ids[j])
      r = ranks.get(pair)
      if r is None:
        return
      new_id = merges[pair]
      heappush(heap, (r, i, stamp[i], j, new_id))

    for i in range(n - 1):
      push(i)

    while heap:
      r, i, _, j, new_id = heappop(heap)
      if i < 0 or j < 0:
        continue
      if not (alive[i] and alive[j]):
        continue
      if next_idx[i] != j or prev_idx[j] != i:
        continue
      cur = ranks.get((ids[i], ids[j]))
      if cur is None or cur != r:
        push(i)
        continue
      ids[i] = new_id
      stamp[i] += 1
      alive[j] = False
      nj = next_idx[j]
      next_idx[i] = nj
      if nj != -1:
        prev_idx[nj] = i
      pi = prev_idx[i]
      if pi != -1:
        stamp[pi] += 1
        push(pi)
      if next_idx[i] != -1:
        push(i)

    out: list[int] = []
    k = 0
    while k != -1 and k < n:
      if alive[k]:
        out.append(ids[k])
      k = next_idx[k]
    return out

  def encode(self, text: str) -> list[int]:
    tokens: list[int] = []
    cache = self.encode_cache
    for m in PRETOKENIZER.finditer(text):
      b = m.group().encode('utf-8')
      if len(b) <= self.cache_max_token_bytes:
        cached = cache.get(b)
        if cached is None:
          merged = tuple(self._apply_bpe(list(b)))
          cache[b] = merged
        else:
          merged = cached
        tokens.extend(merged)
      else:
        tokens.extend(self._apply_bpe(list(b)))
    return tokens

  def encode_bytes(self, data: bytes) -> list[int]:
    return self._apply_bpe(list(data))

  def decode(self, ids: list[int]) -> str:
    def flatten(idx: int, out: list[int]) -> None:
      stack: list[int] = [idx]
      while stack:
        cur = stack.pop()
        if 0 <= cur < 256:
          out.append(cur)
          continue
        sym = self.id_to_sym.get(cur)
        if sym is None:
          continue
        for s in reversed(sym):
          stack.append(s)

    bytes_out: list[int] = []
    for idx in ids:
      flatten(idx, bytes_out)
    return bytes(bytes_out).decode('utf-8', errors='replace')
