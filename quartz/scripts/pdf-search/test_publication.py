"""Publish both public paths when identical PDFs share a page cache."""

from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile
import unittest

import pymupdf

from build import PdfSource, read_passages
from cache import ExtractionSpec, PageCache, canonical, normalize_page


class SharedSourcePublicationTests(unittest.TestCase):
  def test_identical_pdf_paths_share_ocr_and_keep_distinct_passage_links(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      source, duplicate = root / 'original.pdf', root / 'duplicate.pdf'
      with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), 'Shared source paragraph.')
        pdf.save(source)
      duplicate.write_bytes(source.read_bytes())
      spec = ExtractionSpec('ocr', 'layout', 'sdk', 200)
      cache = PageCache(
        root / 'cache', hashlib.sha256(source.read_bytes()).hexdigest(), spec
      )
      cache.write(
        normalize_page(
          1,
          'i',
          [
            {'index': 0, 'label': 'doc_title', 'content': 'Shared title'},
            {
              'index': 1,
              'label': 'text',
              'content': 'Shared source paragraph.',
            },
          ],
        )
      )
      (cache.directory / 'document.json').write_bytes(
        canonical({
          'source': str(source),
          'sourceSha256': cache.source_sha256,
          'recipe': asdict(spec),
          'key': cache.key,
          'totalPages': 1,
          'pages': [1],
        })
      )
      (root / 'cache/recipe.json').write_bytes(canonical(asdict(spec)))
      inventory = {
        'original.pdf': PdfSource(source, 'Original', ()),
        'duplicate.pdf': PdfSource(duplicate, 'Duplicate', ()),
      }
      passages, documents, actual_spec = read_passages(
        root / 'cache', inventory
      )
      self.assertEqual(set(documents), set(inventory))
      self.assertEqual({p['slug'] for p in passages}, set(inventory))
      self.assertEqual({p['page'] for p in passages}, {1})
      self.assertEqual({p['pageLabel'] for p in passages}, {'i'})
      self.assertEqual(len({(p['title'], p['text']) for p in passages}), 1)
      self.assertEqual(actual_spec, spec)


if __name__ == '__main__':
  unittest.main()
