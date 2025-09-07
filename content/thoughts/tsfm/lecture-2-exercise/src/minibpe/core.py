from __future__ import annotations

import mmap, os, time, json, typing as t, heapq, multiprocessing as mp
import regex as re, psutil, speedscope, fire

from collections import Counter, defaultdict
from tqdm import tqdm

from ._core import Tokenizer as TokenizerFast


# GPT-2 split pattern
PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKENIZER = re.compile(PRETOKENIZER_PATTERN)
BASEDIR = os.path.dirname(__file__)


# file_length: os.path.getsize(file_path)
_WORKER_FILE, _WORKER_MMAP, _WORKER_SPECIAL_TOKEN_BYTES = None, None, None


def _init_worker(fpath: str, special_token: str):
  global _WORKER_FILE, _WORKER_MMAP, _WORKER_SPECIAL_TOKEN_BYTES
  _WORKER_FILE = open(fpath, 'rb')
  _WORKER_MMAP = mmap.mmap(_WORKER_FILE.fileno(), 0, access=mmap.ACCESS_READ)
  _WORKER_SPECIAL_TOKEN_BYTES = special_token.encode('utf-8')


def pretokenize_chunk(start_end_indices: tuple[int, int]) -> Counter:
  start, end = start_end_indices
  assert (_WORKER_MMAP is not None) and (_WORKER_SPECIAL_TOKEN_BYTES is not None)
  chunk_bytes = _WORKER_MMAP[start:end]
  if chunk_bytes.startswith(_WORKER_SPECIAL_TOKEN_BYTES):
    chunk_bytes = chunk_bytes[len(_WORKER_SPECIAL_TOKEN_BYTES) :]
  content_chunk = chunk_bytes.decode('utf-8')
  pretokens = PRETOKENIZER.findall(content_chunk)
  return Counter(pretokens)


def pretokenize_batch(boundary_batch):
  total = Counter()
  for bounds in boundary_batch:
    total.update(pretokenize_chunk(bounds))
  return total, len(boundary_batch)


def chunk_text_file(
  file_path: str,
  num_processes: int,
  special_token: str,
  memory_interval: float = 1.0,
  memory_log_path: str | None = None,
  pt_batch: int = 256,
) -> Counter:
  with open(file_path, 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    special = special_token.encode('utf-8')
    chunk_starts = [0]
    start_index = 0
    while True:
      match_index = mm.find(special, start_index)
      if match_index == -1:
        break
      chunk_starts.append(match_index)
      start_index = match_index + len(special)
    chunk_ends = chunk_starts[1:]
    chunk_ends.append(len(mm))
    chunk_boundaries = list(zip(chunk_starts, chunk_ends))
    mm.close()

  memory_samples = [] if memory_log_path is not None else None
  process = psutil.Process(os.getpid())
  last_update = 0.0

  final_counts = Counter()
  if pt_batch <= 0:
    pt_batch = 1
  batched_boundaries = [chunk_boundaries[i : i + pt_batch] for i in range(0, len(chunk_boundaries), pt_batch)]

  if num_processes and num_processes > 1:
    chunksize = max(1, len(batched_boundaries) // (num_processes * 4) or 1)
    with mp.Pool(num_processes, initializer=_init_worker, initargs=(file_path, special_token)) as p:
      bar = tqdm(total=len(chunk_boundaries), desc='Pretokenizing')
      for counter, processed in p.imap_unordered(pretokenize_batch, batched_boundaries, chunksize=chunksize):
        final_counts.update(counter)
        bar.update(processed)

        # memory profiling
        now = time.time()
        if now - last_update >= memory_interval:
          rss_mb = process.memory_info().rss / (1024 * 1024)
          bar.set_postfix_str(f'RSS {rss_mb:.2f} MB', refresh=False)
          last_update = now
          if memory_samples is not None:
            memory_samples.append({'t': now, 'rss_mb': float(rss_mb)})
      bar.close()
      p.close()
      p.join()
  else:
    _init_worker(file_path, special_token)
    bar = tqdm(total=len(chunk_boundaries), desc='Pretokenizing')
    for boundary_batch in batched_boundaries:
      batch_total = Counter()
      for bounds in boundary_batch:
        batch_total.update(pretokenize_chunk(bounds))
      final_counts.update(batch_total)
      bar.update(len(boundary_batch))

      # memory profiling
      now = time.time()
      if now - last_update >= memory_interval:
        rss_mb = process.memory_info().rss / (1024 * 1024)
        bar.set_postfix_str(f'RSS {rss_mb:.2f} MB', refresh=False)
        last_update = now
        if memory_samples is not None:
          memory_samples.append({'t': now, 'rss_mb': float(rss_mb)})
    bar.close()

  print()
  if memory_samples is not None and memory_log_path is not None:
    try:
      with open(memory_log_path, 'w') as jf:
        json.dump({'samples': memory_samples}, jf)
    except Exception:
      pass
  return final_counts


def resolve_ds(ds: str) -> str:
  data = os.path.join(BASEDIR, 'data')
  aliases = {
    'toy': 'toy_data.txt',
    'tinygpt-train': 'TinyStoriesV2-GPT4-train.txt',
    'tinygpt-valid': 'TinyStoriesV2-GPT4-valid.txt',
  }
  if os.path.isabs(ds) and os.path.exists(ds):
    return ds
  if ds.lower() in aliases:
    return os.path.join(data, aliases[ds.lower()])
  return os.path.join(data, ds)


def _count_texts_batch(texts: list[str]) -> tuple[Counter, int]:
  c = Counter()
  total = 0
  for t_ in texts:
    toks = PRETOKENIZER.findall(t_)
    c.update(toks)
    total += len(toks)
  return c, total


def build_fineweb_parallel(
  token_budget: int = 1_000_000_000, split: str = 'train', processes: int = 8, batch_docs: int = 1000
) -> Counter:
  from datasets import load_dataset

  ds = load_dataset('HuggingFaceFW/fineweb', split=split, streaming=True)
  counts = Counter()
  total = 0
  batch: list[str] = []
  with mp.Pool(processes) as p:
    bar = tqdm(total=token_budget, desc='FineWeb pretok', unit='tok')

    def flush(b: list[str]):
      nonlocal counts, total
      if not b:
        return
      c, n = p.apply(_count_texts_batch, (b,))
      counts.update(c)
      total += n
      bar.update(n)

    for example in ds:
      text = example.get('text')
      if not text:
        continue
      batch.append(text)
      if len(batch) >= batch_docs:
        flush(batch)
        batch = []
      if total >= token_budget:
        break
    flush(batch)
    bar.close()
    p.close()
    p.join()
  return counts


def collect_fineweb_texts(token_budget: int = 1_000_000_000, split: str = 'train') -> list[str]:
  from datasets import load_dataset

  ds = load_dataset('HuggingFaceFW/fineweb', split=split, streaming=True)
  texts: list[str] = []
  total = 0
  for example in ds:
    text = example.get('text')
    if not text:
      continue
    n = len(PRETOKENIZER.findall(text))
    texts.append(text)
    total += n
    if total >= token_budget:
      break
  return texts


def train_bpe(pretokenized_freq: Counter[str], num_merges: int, batch_size: int = 1):
  # Represent every token as a list[int] so we can edit in place
  corpus = {tuple(token.encode()): count for token, count in pretokenized_freq.items()}
  vocab = {tuple([i]): i for i in range(256)}  # byte → id
  next_id = 256
  merges: dict[tuple[int, int], int] = {}

  remaining = num_merges
  pbar = tqdm(total=num_merges, desc='BPE merges', unit='merge')
  while remaining > 0:
    step = min(batch_size, remaining)
    pair_counts = defaultdict(int)
    for symbols, freq in corpus.items():
      if len(symbols) < 2:
        continue
      local = Counter(zip(symbols, symbols[1:]))
      if freq != 1:
        for pair, c in local.items():
          pair_counts[pair] += c * freq
      else:
        for pair, c in local.items():
          pair_counts[pair] += c

    if not pair_counts:
      break

    top_pairs = heapq.nlargest(step, pair_counts.keys(), key=pair_counts.get)
    selected_map: dict[tuple[int, int], int] = {}
    for pair in top_pairs:
      merges[pair] = next_id
      vocab[pair] = next_id
      selected_map[pair] = next_id
      next_id += 1

    # 3. Replace occurrences in every token
    updated_corpus = {}
    for symbols, freq in corpus.items():
      merged = []
      i = 0
      while i < len(symbols):
        if i < len(symbols) - 1:
          pair = (symbols[i], symbols[i + 1])
          new_tok = selected_map.get(pair)
          if new_tok is not None:
            merged.append(new_tok)
            i += 2
            continue
        merged.append(symbols[i])
        i += 1
      updated_corpus[tuple(merged)] = freq
    corpus = updated_corpus

    applied = len(top_pairs)
    if applied == 0:
      break
    remaining -= applied
    pbar.update(applied)

  pbar.close()

  return merges, vocab


def main(
  dataset: t.Literal['toy', 'tinygpt-train', 'tinygpt-valid', 'fineweb'] = 'toy',
  vocab_size: int = 131459,
  proc: int = 8,
  profile: t.Literal['none', 'speedscope'] = 'none',
  speedscope_outfile: str = 'tokenizer_profile.json',
  memory_interval: float = 1.0,
  memory_log_path: str | None = None,
  special_token: str = '<|endoftext|>',
  merges_filename: str = 'merges.txt',
  vocab_filename: str = 'vocab.txt',
  token_budget: int = 1_000_000_000,
  fineweb_split: str = 'train',
  batch_size: int = 1,
  fast: bool = False,
):
  os.makedirs((out_dir := os.path.join(BASEDIR, dataset)), exist_ok=True)
  merges = vocab_size - 256

  def _run():
    if dataset.lower().startswith('fineweb'):
      if fast:
        texts = collect_fineweb_texts(token_budget, split=fineweb_split)
        model = TokenizerFast.train_from_texts(texts, merges, proc)
        merges_tbl = {(a, b): nid for a, b, nid in model.merges_list()}
        vocab = {tuple(pair): nid for pair, nid in model.vocab_pairs()}
        return merges_tbl, vocab
      pretokenized_frequency_table = build_fineweb_parallel(token_budget, split=fineweb_split, processes=proc)
    else:
      datapath = resolve_ds(dataset)
      if fast:
        model = TokenizerFast.train_from_files([datapath], merges, proc)
        merges_tbl = {(a, b): nid for a, b, nid in model.merges_list()}
        vocab = {tuple(pair): nid for pair, nid in model.vocab_pairs()}
        return merges_tbl, vocab
      pretokenized_frequency_table = chunk_text_file(
        datapath, proc, special_token, memory_interval=memory_interval, memory_log_path=memory_log_path
      )

    merges_tbl, vocab = train_bpe(pretokenized_frequency_table, merges, batch_size=batch_size)
    return merges_tbl, vocab

  if profile.lower() == 'speedscope':
    with speedscope.track(speedscope_outfile):
      results = _run()
  else:
    results = _run()

  merges_tbl, vocab = results

  # csv format type beat
  with open(os.path.join(out_dir, merges_filename), 'w') as f1:
    for (a, b), new_id in sorted(merges_tbl.items(), key=lambda kv: kv[1]):
      f1.write(f'{a},{b},{new_id}\n')
  with open(os.path.join(out_dir, vocab_filename), 'w') as f2:
    id_to_pair = {new_id: pair for pair, new_id in merges_tbl.items()}
    cache: dict[int, tuple[int, ...]] = {}

    def expand_id_to_bytes(idx: int) -> tuple[int, ...]:
      if idx < 256:
        return (idx,)
      if idx in cache:
        return cache[idx]
      pair = id_to_pair.get(idx)
      if pair is None:
        return ()
      left = expand_id_to_bytes(pair[0])
      right = expand_id_to_bytes(pair[1])
      out = left + right
      cache[idx] = out
      return out

    for sym, tok_id in sorted(vocab.items(), key=lambda kv: kv[1]):
      bytes_seq: tuple[int, ...] = ()
      for x in sym:
        if x < 256:
          bytes_seq += (x,)
        else:
          bytes_seq += expand_id_to_bytes(x)
      token_text = bytes(bytes_seq).decode('utf-8', errors='replace')
      safe = json.dumps(token_text, ensure_ascii=False)
      f2.write(f'{safe}\t{tok_id}\n')

  print()


def cli():
  fire.Fire(main)
