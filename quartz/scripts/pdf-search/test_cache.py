from pathlib import Path
import tempfile
import unittest

from cache import ExtractionSpec, PageCache, normalize_page


class ExtractionContractTests(unittest.TestCase):
  def test_reading_order_paragraphs_and_physical_page(self):
    page = normalize_page(
      7,
      'iv',
      [
        {'index': 2, 'label': 'text', 'content': 'Second paragraph.'},
        {
          'index': 0,
          'label': 'text',
          'content': 'First paragraph.\n\nStill separate.',
        },
        {
          'index': 1,
          'label': 'image',
          'content': 'invented chart interpretation',
        },
      ],
    )
    self.assertEqual(page.page_number, 7)
    self.assertEqual(page.page_label, 'iv')
    self.assertEqual(
      page.text, 'First paragraph.\n\nStill separate.\n\nSecond paragraph.'
    )
    self.assertEqual(page.ignored_regions, 1)

  def test_empty_text_and_duplicate_order_are_failures(self):
    with self.assertRaises(ValueError):
      normalize_page(1, '1', [{'index': 0, 'label': 'text', 'content': ''}])
    with self.assertRaises(ValueError):
      normalize_page(
        1,
        '1',
        [
          {'index': 0, 'label': 'text', 'content': 'one'},
          {'index': 0, 'label': 'text', 'content': 'two'},
        ],
      )
    page = normalize_page(
      1, '1', [{'index': 0, 'label': 'image', 'content': ''}]
    )
    self.assertEqual(page.status, 'no-text')

  def test_source_and_recipe_invalidation_and_corrupt_cache(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      spec = ExtractionSpec(
        'glm-revision-a', 'layout-revision-a', 'sdk-a', 200
      )
      store = PageCache(root, 'source-a', spec)
      page = normalize_page(
        1, '1', [{'index': 0, 'label': 'text', 'content': 'Readable.'}]
      )
      store.write(page)
      self.assertEqual(store.read(1), page)
      self.assertIsNone(store.read(2))
      self.assertIsNone(PageCache(root, 'source-b', spec).read(1))
      changed = ExtractionSpec(
        'glm-revision-b', 'layout-revision-a', 'sdk-a', 200
      )
      self.assertIsNone(PageCache(root, 'source-a', changed).read(1))
      store.path(1).write_text('{broken', encoding='utf-8')
      self.assertIsNone(store.read(1))

  def test_page_identity_and_integrity_are_verified(self):
    with tempfile.TemporaryDirectory() as directory:
      store = PageCache(
        Path(directory), 'source', ExtractionSpec('g', 'l', 's', 200)
      )
      page = normalize_page(
        1, 'A', [{'index': 0, 'label': 'text', 'content': 'Original.'}]
      )
      store.write(page)
      store.path(2).write_bytes(store.path(1).read_bytes())
      self.assertIsNone(store.read(2))
      store.path(1).write_text(
        store.path(1).read_text().replace('Original.', 'Altered.')
      )
      self.assertIsNone(store.read(1))

  def test_runtime_and_generation_recipe_invalidates_pages(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      spec = ExtractionSpec('g', 'l', 's', 200, recipe_sha256='config-a')
      store = PageCache(root, 'source', spec)
      store.write(
        normalize_page(
          1, '1', [{'index': 0, 'label': 'text', 'content': 'Text.'}]
        )
      )
      changed = ExtractionSpec('g', 'l', 's', 200, recipe_sha256='config-b')
      self.assertIsNone(PageCache(root, 'source', changed).read(1))


if __name__ == '__main__':
  unittest.main()
