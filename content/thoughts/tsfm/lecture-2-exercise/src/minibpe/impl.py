from __future__ import annotations

import dataclasses, json, os, multiprocessing, heapq

from .core import PRETOKENIZER

# Byte-level BPE encoder in pure Python
# Walkthrough:
# - Loads merges (pair -> new_id) and builds ranks (pair -> rank order)
# - Encodes by: pretokenize -> bytes -> merge loop
# - Merge loop keeps a linked list of indices and a min-heap of candidate pairs
#   ordered by rank. Each pop validates adjacency and staleness, merges in place,
#   then updates neighboring pairs. This avoids full rescans on every step.
# - Decoding expands ids back to bytes by recursively composing learned pairs.


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
    # Build id -> symbol table for decoding (only non-byte ids >=256)
    self.id_to_sym = {tok_id: tuple(symbol) for symbol, tok_id in self.vocab.items() if tok_id >= 256}
    # Convert merges table to ranks (lower rank merges first)
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
    """
    Apply byte pair encoding (BPE) merges to a sequence of byte ids.

    Background: Traditional BPE (Sennrich et al., 2015) starts from a
    symbolization of text as a sequence of bytes (or characters) and
    repeatedly merges the most frequent adjacent pair into a new symbol.

    Training records the order of pairs merged (or assigns each learned
    pair a rank). At inference/encoding time we then greedily apply those
    learned merges: at every step, among all adjacent pairs present in the
    current sequence, merge the one with the best (lowest) rank, and repeat
    until no mergeable pairs remain.

    Naïve application rescans the whole sequence to locate the next best
    pair after every merge, which is O(n) per step and can be costly.

    Implementation: heap + linked list
    - We maintain the sequence in-place using parallel next/prev index arrays to simulate a linked list.
      - Elements marked "dead" are skipped.
    - We push every mergeable adjacent pair into a min-heap keyed by its rank.
      - Each heap item also carries a stamp to help detect staleness.
    - When we pop a candidate:
      - validate that its two positions are still adjacent and alive and
        that the pair’s rank hasn’t changed;
      - otherwise we discard or re-push.
    - If valid, we merge them in-place, update neighbors, and push the newly formed adjacent pairs.
    - This avoids full rescans and keeps encoding efficient in practice.

    Note: this matches the greedy merge order of traditional BPE,
    just realized with data structures that make each step cheap.
    """
    # Heap + linked-list merge (similar idea to tiktoken)
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

  def decode(self, ids: list[int]) -> str:
    # Expand composed ids back to bytes using iterative stack; decode as utf-8
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

  def colorize_tokens(self, text: str) -> str:
    """
    Return a colorized two-line rendering of tokenization for `text`.

    Line 1: the original text.
    Line 2: colored blocks, one contiguous block per token span.

    This is useful for visually inspecting BPE segments. Uses ANSI colors.
    """
    # Choose a readable repeating 256-color palette for backgrounds
    palette = [196, 202, 208, 214, 220, 154, 118, 82, 46, 51, 45, 39, 27, 63, 99, 135, 171]
    reset = "\x1b[0m"
    # Slightly darker foreground over bright backgrounds
    fg = "\x1b[30m"
    block = "▁"  # low underline block; keeps text readable on the line above

    # Tokenize then decode each token id back to its string span
    token_ids = self.encode(text)
    spans: list[str] = [self.decode([tid]) for tid in token_ids]

    # Assemble underline blocks matching the visible length of each span
    underline_parts: list[str] = []
    for i, s in enumerate(spans):
      # Background color from palette, repeat length of span (fallback to 1)
      color = f"\x1b[48;5;{palette[i % len(palette)]}m"
      width = max(1, len(s))
      underline_parts.append(f"{color}{fg}{block * width}{reset}")

    text_line = ''.join(spans)
    underline_line = ''.join(underline_parts)
    return f"{text_line}\n{underline_line}"

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
      return "<empty>"
    if n == 1:
      return f"0 merges (single byte): {symbols}"

    # Local copy of the in-place merge state
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

    def flatten_token(tok_id: int) -> bytes:
      # Turn a token id back into bytes for display
      if 0 <= tok_id < 256:
        return bytes([tok_id])
      sym = self.id_to_sym.get(tok_id)
      if sym is None:
        return f"<{tok_id}>".encode()
      # Iterative stack to avoid recursion
      out: list[int] = []
      stack: list[int] = list(sym)
      while stack:
        t = stack.pop()
        if 0 <= t < 256:
          out.append(t)
        else:
          inner = self.id_to_sym.get(t)
          if inner is None:
            # Should not happen for well-formed vocabs; fall back
            return f"<{tok_id}>".encode()
          stack.extend(reversed(inner))
      return bytes(out)

    def render_tokens(toks: list[int]) -> str:
      parts: list[str] = [(flatten_token(t).decode('utf-8', errors='replace'))[::-1] for t in toks]
      return ' | '.join(parts)

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
    lines.append(f"input bytes: {symbols}")
    lines.append(f"tokens: {snapshot_tokens()} :: {render_tokens(snapshot_tokens())}")
    while heap and (max_steps is None or step < max_steps):
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

      # Log step before applying
      pair = (ids[i], ids[j])
      lines.append(f"step {step}: merge {pair} @ rank {r} -> {new_id}")
      cands = current_candidates()
      if cands:
        cand_str = ', '.join([f"{p}:{rk}" for rk, p in cands])
        lines.append(f"  top pairs: {cand_str}")

      # Apply merge
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

      toks = snapshot_tokens()
      lines.append(f"  tokens: {toks} :: {render_tokens(toks)}")
      step += 1

    lines.append("final:")
    toks = snapshot_tokens()
    lines.append(f"  tokens: {toks} :: {render_tokens(toks)}")
    return "\n".join(lines)
