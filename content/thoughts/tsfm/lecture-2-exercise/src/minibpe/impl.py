from __future__ import annotations

import dataclasses, json, os, multiprocessing

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
    if len(symbols) < 2:
      return symbols
    while True:
      pairs = self._get_pairs(symbols)
      candidates = [(pair, self.ranks[pair]) for pair in pairs if pair in self.ranks]
      if not candidates:
        break
      best_pair = min(candidates, key=lambda x: x[1])[0]
      merged: list[int] = []
      i = 0
      while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
          merged.append(self.merges[best_pair])
          i += 2
        else:
          merged.append(symbols[i])
          i += 1
      symbols = merged
      if len(symbols) < 2:
        break
    return symbols

  def encode(self, text: str) -> list[int]:
    tokens: list[int] = []
    for piece in PRETOKENIZER.findall(text):
      b = piece.encode('utf-8')
      symbols = list(b)
      merged = self._apply_bpe(symbols)
      tokens.extend(merged)
    return tokens

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
