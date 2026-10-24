"""Earlier OCR pages must remain verifiable across the server transition."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from cache import ExtractionSpec, PageCache, canonical, normalize_page
from migration import activate_recipe


class RecipeMigrationTests(unittest.TestCase):
  def test_preserves_pages_provenance_and_repairs_interrupted_metadata(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / 'source.pdf'
      source.write_bytes(b'unchanged source')
      sha = hashlib.sha256(source.read_bytes()).hexdigest()
      old = ExtractionSpec('g', 'l', '0.1.5', 200, 'old')
      new = ExtractionSpec('g', 'l', '0.1.5', 200, 'direct-http')
      cache = PageCache(root, sha, old)
      page = normalize_page(
        2,
        'iv',
        [{'index': 0, 'label': 'text', 'content': 'First.\n\nSecond.'}],
      )
      cache.write(page)
      original = cache.path(2).read_bytes()
      (cache.directory / 'document.json').write_bytes(
        canonical({
          'source': str(source),
          'sourceSha256': sha,
          'recipe': asdict(old),
          'key': cache.key,
          'totalPages': 4,
          'pages': [],
        })
      )
      (root / 'recipe.json').write_bytes(canonical(asdict(old)))
      activate_recipe(root, new, previous_fingerprint='old')
      migrated = PageCache(root, sha, new)
      self.assertEqual(migrated.read(2), page)
      self.assertEqual(cache.path(2).read_bytes(), original)
      saved = json.loads(migrated.path(2).read_text())
      self.assertEqual(saved['origin']['recipe'], asdict(old))
      self.assertEqual(saved['origin']['key'], cache.key)
      self.assertEqual(
        json.loads((migrated.directory / 'document.json').read_text())[
          'pages'
        ],
        [2],
      )
      activate_recipe(root, new, previous_fingerprint='old')
      self.assertEqual(migrated.read(2), page)

  def test_rejects_corruption_or_unapproved_recipe_before_activation(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      old = ExtractionSpec('g', 'l', '0.1.5', 200, 'unknown')
      new = ExtractionSpec('g', 'l', '0.1.5', 200, 'direct-http')
      original = canonical(asdict(old))
      (root / 'recipe.json').write_bytes(original)
      with self.assertRaises(ValueError):
        activate_recipe(root, new, previous_fingerprint='old')
      self.assertEqual((root / 'recipe.json').read_bytes(), original)

  def test_model_transition_requires_the_exact_previous_weights(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      old = ExtractionSpec('unexpected', 'l', '0.1.5', 200, 'old')
      new = ExtractionSpec('g-8bit', 'l', '0.1.5', 200, 'direct-http')
      original = canonical(asdict(old))
      (root / 'recipe.json').write_bytes(original)
      with self.assertRaises(ValueError):
        activate_recipe(
          root, new, previous_fingerprint='old', previous_model_revision='g'
        )
      self.assertEqual((root / 'recipe.json').read_bytes(), original)


if __name__ == '__main__':
  unittest.main()
