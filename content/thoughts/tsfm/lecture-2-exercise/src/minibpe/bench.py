from __future__ import annotations

import os, time, typing as t
import fire
from tqdm import tqdm

from .core import resolve_ds
from .impl import Tokenizer as PyTokenizer
from ._core import Tokenizer as RustTokenizer


def _load_valid_text(dataset: t.Literal['toy', 'tinygpt'] = 'toy') -> str:
  base_dir = os.path.dirname(__file__)
  if dataset == 'toy':
    path = os.path.join(base_dir, 'data', 'toy_data.txt')
  else:
    path = os.path.join(base_dir, 'data', 'TinyStoriesV2-GPT4-valid.txt')
  return open(path, 'r', encoding='utf-8', errors='ignore').read()

def _roundtrip_with_progress(model, text: str, label: str) -> float:
  enc_start = time.perf_counter()
  ids = model.encode(text)
  enc_s = time.perf_counter() - enc_start

  dec_start = time.perf_counter()
  bar = tqdm(total=len(ids), desc=f'{label} decode', unit='tok')
  # decode token-by-token to visualize progress
  parts: list[str] = []
  for i in ids:
    parts.append(model.decode([i]))
    bar.update(1)
  bar.close()
  _ = ''.join(parts)  # ensure similar work to full decode
  dec_s = time.perf_counter() - dec_start

  return enc_s + dec_s

def benchmark(
  dataset: t.Literal['toy', 'tinygpt-train'] = 'toy', merges: int = 500, processes: int = 4, batch_size: int = 1
) -> None:
  text = _load_valid_text(dataset)

  py_model = PyTokenizer.from_pretrained(os.path.join(os.path.dirname(__file__), dataset))
  r_model = RustTokenizer.from_pretrained(os.path.join(os.path.dirname(__file__), dataset))

  # train_path = resolve_ds(dataset)
  # # Train Python
  # start = time.perf_counter()
  # counts = chunk_text_file(train_path, processes, special_token='<|endoftext|>')
  # py_merges_map, py_vocab = train_bpe(counts, merges, batch_size=batch_size)
  # py_model = PyTokenizer(merges=py_merges_map, vocab=py_vocab)
  # py_train_s = time.perf_counter() - start
  #
  # # Train Rust
  # TODO: DOGSHIT LOL
  # start = time.perf_counter(); r_model = RustTokenizer.train_from_files([train_path], merges, processes); rust_train_s = time.perf_counter() - start

  # Python encode/decode
  py_rt_s = _roundtrip_with_progress(py_model, text, 'python')
  # Rust encode/decode
  r_rt_s = _roundtrip_with_progress(r_model, text, 'rust')

  print(f'dataset={dataset}')
  # print(f'python_train_s={py_train_s:.4f}')
  # print(f'rust_train_s={rust_train_s:.4f}')
  print(f'python_roundtrip_s={py_rt_s:.4f}')
  print(f'rust_roundtrip_s={r_rt_s:.4f}')


def cli():
  fire.Fire(benchmark)
