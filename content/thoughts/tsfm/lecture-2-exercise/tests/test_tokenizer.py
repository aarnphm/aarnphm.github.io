from collections import Counter
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import random
import tempfile
import unittest

from minibpe import Tokenizer
from minibpe.patterns import PRETOKENIZER
from minibpe.training import train_bpe


def load_rust():
  path = os.environ.get('MINIBPE_EXTENSION_PATH')
  if path:
    loader = importlib.machinery.ExtensionFileLoader('minibpe._core', path)
    spec = importlib.util.spec_from_file_location('minibpe._core', path, loader=loader)
    if spec is None:
      raise RuntimeError(f'cannot load extension at {path}')
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.Tokenizer
  try:
    from minibpe._core import Tokenizer as RustTokenizer
  except ModuleNotFoundError as error:
    if error.name != 'minibpe._core':
      raise
    return None
  return RustTokenizer


RustTokenizer = load_rust()


class TokenizerTests(unittest.TestCase):
  def model(self, text: str, num_merges: int = 32) -> Tokenizer:
    merges, vocab = train_bpe(Counter(PRETOKENIZER.findall(text)), num_merges)
    return Tokenizer(merges)

  def test_ranked_training_recounts_after_each_merge(self):
    merges, vocab = train_bpe(Counter({'abc': 1, 'bc': 2}), 2)
    self.assertEqual(merges, {(98, 99): 256, (97, 256): 257})
    self.assertEqual(Tokenizer(merges).encode_bytes(b'abc'), [257])

  def test_ties_are_independent_of_counter_insertion_order(self):
    forward = train_bpe(Counter({'ab': 1, 'bc': 1}), 2)
    reverse = train_bpe(Counter({'bc': 1, 'ab': 1}), 2)
    self.assertEqual(forward, reverse)
    self.assertEqual(forward[0], {(97, 98): 256, (98, 99): 257})

  def test_nested_and_overlapping_merges(self):
    model = Tokenizer({(97, 97): 256, (256, 97): 257})
    self.assertEqual(model.encode_bytes(b'aaaaa'), [256, 257])
    self.assertEqual(model.decode_bytes([256, 257]), b'aaaaa')

  def test_empty_and_exhausted_training(self):
    self.assertEqual(train_bpe(Counter(), 100)[0], {})
    self.assertEqual(train_bpe(Counter({'x': 2}), 100)[0], {})
    self.assertEqual(len(train_bpe(Counter({'ab': 2}), 100)[0]), 1)
    self.assertEqual(train_bpe(Counter({'ab': 0}), 1)[0], {})
    for counts, budget in [(Counter({'ab': -1}), 1), (Counter(), -1)]:
      with self.assertRaises(ValueError):
        train_bpe(counts, budget)

  def test_arbitrary_bytes_and_unicode_roundtrip(self):
    model = self.model('café 你好 🦀 e\u0301 aaa café')
    rng = random.Random(916)
    payloads = [b'', bytes(range(256)), rng.randbytes(2048)]
    for payload in payloads:
      self.assertEqual(model.decode_bytes(model.encode_bytes(payload)), payload)
    for text in ['', 'café', '你好 🦀', 'e\u0301', 'a  b\n\t c', "I'm we'll"]:
      self.assertEqual(model.decode(model.encode(text)), text)

  def test_save_load_keeps_partial_utf8_tokens(self):
    model = Tokenizer({(226, 130): 256, (256, 172): 257})
    with tempfile.TemporaryDirectory() as directory:
      model.save_pretrained(directory)
      lines = (Path(directory) / 'vocab.txt').read_text().splitlines()
      self.assertIn('226 130 256', lines)
      self.assertIn('226 130 172 257', lines)
      restored = Tokenizer.from_pretrained(directory)
      self.assertEqual(restored.decode_bytes([256]), b'\xe2\x82')
      self.assertEqual(restored.decode([257]), '€')
      self.assertEqual(restored.encode('€'), [257])

  def test_legacy_lossy_vocab_does_not_override_merges(self):
    with tempfile.TemporaryDirectory() as directory:
      (Path(directory) / 'merges.txt').write_text('256 172 257\n226 130 256\n')
      (Path(directory) / 'vocab.txt').write_text('"�"\t256\n"€"\t257\n')
      restored = Tokenizer.from_pretrained(directory)
      self.assertEqual(restored.decode_bytes([256]), b'\xe2\x82')
      self.assertEqual(restored.decode([257]), '€')

  def test_malformed_models_and_unknown_ids_fail(self):
    malformed = [
      '97,98\n', 'no,98,256\n', '300,98,256\n', '97,98,97\n',
      '97,98,4294967296\n',
      '97,98,256\n98,99,256\n', '97,98,256\n97,98,257\n',
    ]
    with tempfile.TemporaryDirectory() as directory:
      for text in malformed:
        (Path(directory) / 'merges.txt').write_text(text)
        with self.subTest(text=text), self.assertRaises(ValueError):
          Tokenizer.from_pretrained(directory)
    with self.assertRaises(ValueError):
      Tokenizer({}).decode([256])

  def test_pretoken_boundaries_and_visualization(self):
    model = Tokenizer({(32, 32): 256, (97, 98): 257})
    self.assertEqual(model.encode('a  b'), [97, 32, 32, 98])
    self.assertEqual(model.encode_bytes(b'a  b'), [97, 256, 98])
    self.assertIn("b'ab'", model.visualize_bpe('ab'))
    self.assertEqual(model.colorize_tokens('€').splitlines()[0], '€')

  def test_file_training_saves_a_loadable_model(self):
    from minibpe.core import chunk_text_file, main

    with tempfile.TemporaryDirectory() as directory:
      source = Path(directory) / 'corpus.txt'
      output = Path(directory) / 'model'
      source.write_text('café café<|endoftext|>你好 你好', encoding='utf-8')
      main(dataset=str(source), output_dir=str(output), vocab_size=280, proc=1)
      model = Tokenizer.from_pretrained(output)
      self.assertGreater(len(model.merges), 0)
      self.assertEqual(model.decode(model.encode('café 你好')), 'café 你好')
      with self.assertRaises(ValueError):
        chunk_text_file(str(source), 1, '')
      source.write_text('')
      self.assertEqual(chunk_text_file(str(source), 1, '<|endoftext|>'), Counter())

  @unittest.skipIf(RustTokenizer is None, 'Rust extension is not built')
  def test_python_rust_parity_for_saved_learned_models(self):
    corpus = "café café 你好 🦀 e\u0301 a  b aaa aaaaa I'm we'll!\n\t "
    rng = random.Random(916)
    with tempfile.TemporaryDirectory() as directory:
      for budget in [0, 1, 8, 32, 128]:
        model = self.model(corpus, budget)
        model.save_pretrained(directory)
        rust = RustTokenizer.from_pretrained(directory)
        restored = Tokenizer.from_pretrained(directory)
        for text in ['', corpus, 'a  b', 'different 世界!\n', '€\x00']:
          ids = restored.encode(text)
          self.assertEqual(rust.encode(text), ids)
          self.assertEqual(rust.decode(ids), text)
        for payload in [bytes(range(256)), rng.randbytes(512), b'aaaaabcabc']:
          ids = restored.encode_bytes(payload)
          self.assertEqual(rust.encode_bytes(payload), ids)
          self.assertEqual(rust.decode_bytes(ids), payload)
          self.assertEqual(rust.decode(ids), restored.decode(ids))
        with self.assertRaises(ValueError):
          rust.decode([1000000])

  @unittest.skipIf(RustTokenizer is None, 'Rust extension is not built')
  def test_python_rust_reject_the_same_corrupt_models(self):
    with tempfile.TemporaryDirectory() as directory:
      for text in ['97,98\n', 'no,98,256\n', '300,98,256\n', '97,98,97\n',
      '97,98,4294967296\n',
                   '97,98,256\n98,99,256\n', '97,98,256\n97,98,257\n']:
        (Path(directory) / 'merges.txt').write_text(text)
        for backend in [Tokenizer, RustTokenizer]:
          with self.subTest(text=text, backend=backend), self.assertRaises(ValueError):
            backend.from_pretrained(directory)


if __name__ == '__main__':
  unittest.main()
