from __future__ import annotations

import json
import mmap
import multiprocessing as mp
import os
import time
import typing as t
import psutil

from collections import Counter
from tqdm import tqdm
from .patterns import PRETOKENIZER
from .training import train_bpe
from .impl import Tokenizer


BASEDIR = os.path.dirname(__file__)


_WORKER_FILE, _WORKER_MMAP, _WORKER_SPECIAL_TOKEN_BYTES = None, None, None


def _init_worker(fpath: str, special_token: str):
  global _WORKER_FILE, _WORKER_MMAP, _WORKER_SPECIAL_TOKEN_BYTES
  _WORKER_FILE = open(fpath, 'rb')
  _WORKER_MMAP = mmap.mmap(_WORKER_FILE.fileno(), 0, access=mmap.ACCESS_READ)
  _WORKER_SPECIAL_TOKEN_BYTES = special_token.encode('utf-8')


def pretokenize_chunk(start_end_indices: tuple[int, int]) -> Counter[str]:
  start, end = start_end_indices
  assert (_WORKER_MMAP is not None) and (
    _WORKER_SPECIAL_TOKEN_BYTES is not None
  )
  chunk_bytes = _WORKER_MMAP[start:end]
  if chunk_bytes.startswith(_WORKER_SPECIAL_TOKEN_BYTES):
    chunk_bytes = chunk_bytes[len(_WORKER_SPECIAL_TOKEN_BYTES) :]
  content_chunk = chunk_bytes.decode('utf-8')
  pretokens = PRETOKENIZER.findall(content_chunk)
  return Counter(pretokens)


def pretokenize_batch(boundary_batch: list[tuple[int, int]]):
  total: Counter[str] = Counter()
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
) -> Counter[str]:
  if not special_token:
    raise ValueError('special_token must be nonempty')
  if os.path.getsize(file_path) == 0:
    return Counter()
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
  batched_boundaries = [
    chunk_boundaries[i : i + pt_batch]
    for i in range(0, len(chunk_boundaries), pt_batch)
  ]

  if num_processes and num_processes > 1:
    chunksize = max(1, len(batched_boundaries) // (num_processes * 4) or 1)
    with mp.Pool(
      num_processes,
      initializer=_init_worker,
      initargs=(file_path, special_token),
    ) as p:
      bar = tqdm(total=len(chunk_boundaries), desc='Pretokenizing')
      for counter, processed in p.imap_unordered(
        pretokenize_batch, batched_boundaries, chunksize=chunksize
      ):
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


def build_fineweb_counts(
  token_budget: int = 1_000_000_000,
  split: str = 'train',
  batch_docs: int = 1000,
) -> Counter:
  from datasets import load_dataset

  ds = load_dataset('HuggingFaceFW/fineweb', split=split, streaming=True)

  def _pretok_batch(batch: dict[str, list[str]]):
    texts: list[str] = batch.get('text', [])
    pretoks = [PRETOKENIZER.findall(t_) if t_ else [] for t_ in texts]
    lengths = [len(p) for p in pretoks]
    return {'_pretokens': pretoks, '_pretok_len': lengths}

  ds = ds.map(_pretok_batch, batched=True, batch_size=max(1, batch_docs))

  counts: Counter[str] = Counter()
  total = 0
  bar = tqdm(total=token_budget, desc='FineWeb pretok', unit='tok')

  for ex in ds:
    pretoks: list[str] | None = ex.get('_pretokens')
    if pretoks is None:
      # If batched transformation flattened, fall back to tokenizing a single text
      text = ex.get('text')
      if text:
        toks = PRETOKENIZER.findall(text)
        counts.update(toks)
        total += len(toks)
        bar.update(len(toks))
    else:
      counts.update(pretoks)
      total += len(pretoks)
      bar.update(len(pretoks))

    if total >= token_budget:
      break

  bar.close()
  return counts


def main(
  dataset: str = 'toy',
  vocab_size: int = 131459,
  proc: int = 8,
  profile: t.Literal['none', 'speedscope'] = 'none',
  speedscope_outfile: str = 'tokenizer_profile.json',
  memory_interval: float = 1.0,
  memory_log_path: str | None = None,
  special_token: str = '<|endoftext|>',
  token_budget: int = 1_000_000_000,
  fineweb_split: str = 'train',
  output_dir: str | None = None,
):
  if vocab_size < 256:
    raise ValueError('vocab_size must include all 256 byte tokens')
  out_dir = output_dir or os.path.join(
    BASEDIR, os.path.splitext(os.path.basename(dataset))[0]
  )
  merges = vocab_size - 256

  def _run():
    if dataset.lower().startswith('fineweb'):
      pretokenized_frequency_table = build_fineweb_counts(
        token_budget, split=fineweb_split
      )
    else:
      datapath = resolve_ds(dataset)
      pretokenized_frequency_table = chunk_text_file(
        datapath,
        proc,
        special_token,
        memory_interval=memory_interval,
        memory_log_path=memory_log_path,
      )

    return train_bpe(pretokenized_frequency_table, merges)

  if profile.lower() == 'speedscope':
    import speedscope

    with speedscope.track(speedscope_outfile):
      results = _run()
  else:
    results = _run()

  merges_tbl, _ = results

  Tokenizer(merges_tbl).save_pretrained(out_dir)


def cli():
  import fire

  fire.Fire(main)
