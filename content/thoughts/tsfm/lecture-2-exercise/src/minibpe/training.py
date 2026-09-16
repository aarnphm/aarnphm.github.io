from collections import Counter


def train_bpe(
  pretokenized_freq: Counter[str], num_merges: int
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, ...], int]]:
  """Learn one merge at a time; tied counts choose the lowest token-id pair."""
  if num_merges < 0:
    raise ValueError('num_merges must be nonnegative')
  if any(count < 0 for count in pretokenized_freq.values()):
    raise ValueError('pretoken frequencies must be nonnegative')
  corpus = {
    tuple(token.encode('utf-8')): count
    for token, count in pretokenized_freq.items()
    if count > 0
  }
  vocab: dict[tuple[int, ...], int] = {(i,): i for i in range(256)}
  merges: dict[tuple[int, int], int] = {}
  for new_id in range(256, 256 + num_merges):
    counts: Counter[tuple[int, int]] = Counter()
    for symbols, freq in corpus.items():
      for pair in zip(symbols, symbols[1:]):
        counts[pair] += freq
    if not counts:
      break
    pair = min(counts, key=lambda p: (-counts[p], p))
    merges[pair] = new_id
    vocab[pair] = new_id
    updated: Counter[tuple[int, ...]] = Counter()
    for symbols, freq in corpus.items():
      merged: list[int] = []
      i = 0
      while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
          merged.append(new_id)
          i += 2
        else:
          merged.append(symbols[i])
          i += 1
      updated[tuple(merged)] += freq
    corpus = dict(updated)
  return merges, vocab
