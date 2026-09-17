from __future__ import annotations

import dataclasses
import heapq
import os
from pathlib import Path

from .patterns import PRETOKENIZER


@dataclasses.dataclass
class Tokenizer:
  merges: dict[tuple[int, int], int]
  vocab: dict[tuple[int, ...], int] = dataclasses.field(init=False)
  id_to_bytes: dict[int, bytes] = dataclasses.field(init=False)
  ranks: dict[tuple[int, int], int] = dataclasses.field(init=False)
  encode_cache: dict[bytes, tuple[int, ...]] = dataclasses.field(
    default_factory=dict, init=False
  )
  cache_max_token_bytes: int = 100

  def __post_init__(self) -> None:
    self.id_to_bytes = {i: bytes([i]) for i in range(256)}
    self.ranks = {}
    # Merge ids encode training order. The merge graph is the source of bytes,
    # including for old models whose display-only vocab.txt lost UTF-8 fragments.
    ordered = sorted(self.merges.items(), key=lambda entry: entry[1])
    for rank, (pair, tok_id) in enumerate(ordered):
      if not 256 <= tok_id <= 0xFFFFFFFF:
        raise ValueError(
          f'learned token id must fit an unsigned 32-bit integer: {tok_id}'
        )
      if tok_id in self.id_to_bytes:
        raise ValueError(f'duplicate or reserved token id: {tok_id}')
      if len(pair) != 2 or any(part not in self.id_to_bytes for part in pair):
        raise ValueError(f'merge refers to an undefined token: {pair}')
      self.id_to_bytes[tok_id] = (
        self.id_to_bytes[pair[0]] + self.id_to_bytes[pair[1]]
      )
      self.ranks[pair] = rank
    self.vocab = {(i,): i for i in range(256)} | dict(self.merges)

  @classmethod
  def from_pretrained(cls, fp: str | os.PathLike[str]) -> 'Tokenizer':
    merges: dict[tuple[int, int], int] = {}
    with (Path(fp) / 'merges.txt').open(encoding='utf-8') as handle:
      for line in handle:
        if not line.strip():
          continue
        parts = line.replace(',', ' ').split()
        if len(parts) != 3:
          raise ValueError(f'expected three merge ids: {line.strip()}')
        a, b, tok_id = map(int, parts)
        if (a, b) in merges:
          raise ValueError(f'duplicate merge pair: {(a, b)}')
        merges[a, b] = tok_id
    return cls(merges=merges)

  def save_pretrained(self, fp: str | os.PathLike[str]) -> None:
    directory = Path(fp)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'merges.txt').open('w', encoding='utf-8') as handle:
      for (a, b), tok_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
        handle.write(f'{a},{b},{tok_id}\n')
    with (directory / 'vocab.txt').open('w', encoding='utf-8') as handle:
      for tok_id, data in sorted(self.id_to_bytes.items()):
        # Decimal byte values preserve tokens that end inside a UTF-8 character.
        handle.write(' '.join(map(str, (*data, tok_id))) + '\n')

  def _apply_bpe(self, symbols: list[int]) -> list[int]:
    """Apply ranked merges using a heap and linked indices."""
    n = len(symbols)
    if n < 2:
      return symbols
    ids = list(symbols)
    next_idx = list(range(1, n)) + [-1]
    prev_idx = [-1] + list(range(0, n - 1))
    alive = [True] * n
    heap: list[tuple[int, int, int, int]] = []

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
      heappush(heap, (r, i, j, new_id))

    for i in range(n - 1):
      push(i)

    while heap:
      r, i, j, new_id = heappop(heap)
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
      alive[j] = False
      nj = next_idx[j]
      next_idx[i] = nj
      if nj != -1:
        prev_idx[nj] = i
      pi = prev_idx[i]
      if pi != -1:
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
    # Pretokenize to minimize cross-boundary merges, operate on raw UTF-8 bytes
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

  def decode_bytes(self, ids: list[int]) -> bytes:
    try:
      return b''.join(self.id_to_bytes[idx] for idx in ids)
    except KeyError as error:
      raise ValueError(f'unknown token id: {error.args[0]}') from error

  def decode(self, ids: list[int]) -> str:
    return self.decode_bytes(ids).decode('utf-8', errors='replace')

  def colorize_tokens(self, text: str) -> str:
    """
    Return a colorized two-line rendering of tokenization for `text`.

    Line 1: the original text.
    Line 2: colored blocks, one contiguous block per token span.

    This is useful for visually inspecting BPE segments. Uses ANSI colors.
    """
    # Choose a readable repeating 256-color palette for backgrounds
    palette = [
      196,
      202,
      208,
      214,
      220,
      154,
      118,
      82,
      46,
      51,
      45,
      39,
      27,
      63,
      99,
      135,
      171,
    ]
    reset = '\x1b[0m'
    # Slightly darker foreground over bright backgrounds
    fg = '\x1b[30m'
    block = '▁'  # low underline block; keeps text readable on the line above

    # Tokenize then decode each token id back to its string span
    token_ids = self.encode(text)
    spans: list[str] = [self.decode([tid]) for tid in token_ids]

    # Assemble underline blocks matching the visible length of each span
    underline_parts: list[str] = []
    for i, s in enumerate(spans):
      # Background color from palette, repeat length of span (fallback to 1)
      color = f'\x1b[48;5;{palette[i % len(palette)]}m'
      width = max(1, len(s))
      underline_parts.append(f'{color}{fg}{block * width}{reset}')

    text_line = text
    underline_line = ''.join(underline_parts)
    return f'{text_line}\n{underline_line}'

  def visualize_bpe(
    self,
    data: str | bytes,
    *,
    max_steps: int | None = None,
    show_candidates: int = 5,
  ) -> str:
    """
    Trace BPE merges step-by-step for a given input and return a readable log.

    - Shows the chosen pair at each merge and the resulting token sequence.
    - Also lists the top `show_candidates` current candidate pairs by rank at
      each step (recomputed from the live sequence for clarity).

    Args:
      data: Input text (str) or raw bytes to encode.
      max_steps: Optional cap on number of merges to display.
      show_candidates: Number of best-ranked pairs to display per step.
    """
    if isinstance(data, str):
      # For simplicity, operate on raw bytes for the whole string
      # (pretokenization complicates the display and is omitted here).
      symbols = list(data.encode('utf-8'))
    else:
      symbols = list(data)

    n = len(symbols)
    if n == 0:
      return '<empty>'
    if n == 1:
      return f'0 merges (single byte): {symbols}'

    # Local copy of the in-place merge state
    ids = list(symbols)
    next_idx = list(range(1, n)) + [-1]
    prev_idx = [-1] + list(range(0, n - 1))
    alive = [True] * n
    heap: list[tuple[int, int, int, int]] = []

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
      heappush(heap, (r, i, j, new_id))

    for i in range(n - 1):
      push(i)

    def live_indices() -> list[int]:
      out: list[int] = []
      k = 0
      while k != -1 and k < n:
        if alive[k]:
          out.append(k)
        k = next_idx[k]
      return out

    def snapshot_tokens() -> list[int]:
      toks: list[int] = []
      k = 0
      while k != -1 and k < n:
        if alive[k]:
          toks.append(ids[k])
        k = next_idx[k]
      return toks

    def render_tokens(toks: list[int]) -> str:
      return ' | '.join(repr(self.decode_bytes([tok_id])) for tok_id in toks)

    def current_candidates() -> list[tuple[int, tuple[int, int]]]:
      # Recompute live adjacent pairs and sort by rank (best first)
      cand: list[tuple[int, tuple[int, int]]] = []
      inds = live_indices()
      for i, j in zip(inds, inds[1:]):
        pair = (ids[i], ids[j])
        r = ranks.get(pair)
        if r is not None:
          cand.append((r, pair))
      cand.sort(key=lambda x: x[0])
      return cand[:show_candidates]

    lines: list[str] = []
    step = 0
    lines.append(f'input bytes: {symbols}')
    lines.append(
      f'tokens: {snapshot_tokens()} :: {render_tokens(snapshot_tokens())}'
    )
    while heap and (max_steps is None or step < max_steps):
      r, i, j, new_id = heappop(heap)
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

      # Log step before applying
      pair = (ids[i], ids[j])
      lines.append(f'step {step}: merge {pair} @ rank {r} -> {new_id}')
      cands = current_candidates()
      if cands:
        cand_str = ', '.join([f'{p}:{rk}' for rk, p in cands])
        lines.append(f'  top pairs: {cand_str}')

      # Apply merge
      ids[i] = new_id
      alive[j] = False
      nj = next_idx[j]
      next_idx[i] = nj
      if nj != -1:
        prev_idx[nj] = i
      pi = prev_idx[i]
      if pi != -1:
        push(pi)
      if next_idx[i] != -1:
        push(i)

      toks = snapshot_tokens()
      lines.append(f'  tokens: {toks} :: {render_tokens(toks)}')
      step += 1

    lines.append('final:')
    toks = snapshot_tokens()
    lines.append(f'  tokens: {toks} :: {render_tokens(toks)}')
    return '\n'.join(lines)
