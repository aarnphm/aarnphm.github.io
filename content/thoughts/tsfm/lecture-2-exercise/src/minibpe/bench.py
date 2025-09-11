from __future__ import annotations

import os, time, typing as t
import fire

from .impl import Tokenizer as PyTokenizer
from ._core import Tokenizer as RustTokenizer


def _load_valid_text(dataset: t.Literal['toy', 'tinygpt', 'tinygpt-train'] = 'toy') -> str:
  base_dir = os.path.dirname(__file__)
  if dataset == 'toy':
    path = os.path.join(base_dir, 'data', 'toy_data.txt')
  else:
    path = os.path.join(base_dir, 'data', 'TinyStoriesV2-GPT4-valid.txt')
  return open(path, 'r', encoding='utf-8', errors='ignore').read()


def _timings_ms(model: PyTokenizer | RustTokenizer, text: str) -> tuple[float, float, float]:
  enc_start = time.perf_counter()
  ids = model.encode(text)
  enc_ms = (time.perf_counter() - enc_start) * 1000

  dec_start = time.perf_counter()
  _ = model.decode(ids)
  dec_ms = (time.perf_counter() - dec_start) * 1000

  return enc_ms, dec_ms, enc_ms + dec_ms


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

  py_enc_ms, py_dec_ms, py_rt_ms = _timings_ms(py_model, text)
  r_enc_ms, r_dec_ms, r_rt_ms = _timings_ms(r_model, text)

  print(f'dataset={dataset}')
  header = f'{"model":<10} {"encode_ms":>12} {"decode_ms":>12} {"roundtrip_ms":>14}'
  print(header)
  print('-' * len(header))
  print(f'{"python":<10} {py_enc_ms:12.2f} {py_dec_ms:12.2f} {py_rt_ms:14.2f}')
  print(f'{"rust":<10} {r_enc_ms:12.2f} {r_dec_ms:12.2f} {r_rt_ms:14.2f}')


def cli() -> None:
  fire.Fire(benchmark)
