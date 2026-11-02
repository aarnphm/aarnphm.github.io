"""Check vector-aside evidence against real PDF geometry and native text."""

from pathlib import Path
import tempfile
import unittest

import pymupdf

from recovery import vector_aside, native_binary_text, native_region_text


class VectorAsideTests(unittest.TestCase):
  def test_native_prose_preserves_hex_and_rejects_hidden_or_clipped_text(self):
    with tempfile.TemporaryDirectory() as temporary:
      for hidden in (False, True):
        source = Path(temporary) / f'prose-{hidden}.pdf'
        with pymupdf.open() as pdf:
          page = pdf.new_page(width=1000, height=1000)
          page.insert_text(
            (100, 100),
            'X10 = 0 and X11 = 0xffffffffffffffff.',
            fontsize=10,
            render_mode=3 if hidden else 0,
          )
          pdf.save(source)
        text = native_region_text(source, 1, [90, 80, 500, 110])
        self.assertEqual(
          text, None if hidden else 'X10 = 0 and X11 = 0xffffffffffffffff.'
        )
        self.assertIsNone(native_region_text(source, 1, [90, 80, 200, 110]))
        self.assertIsNone(native_region_text(source, 1, [90, 180, 500, 210]))

  def test_native_binary_requires_complete_region_and_preserves_digits(self):
    with tempfile.TemporaryDirectory() as temporary:
      source = Path(temporary) / 'binary.pdf'
      digits = ' '.join(['00000000'] * 7 + ['00001001'])
      with pymupdf.open() as pdf:
        page = pdf.new_page(width=1000, height=1000)
        page.insert_text((100, 100), digits + ' = 9', fontsize=8)
        page.insert_text((100, 200), 'Ordinary prose', fontsize=8)
        pdf.save(source)
      self.assertEqual(
        native_binary_text(source, 1, [95, 85, 500, 110]), digits + ' = 9'
      )
      self.assertIsNone(native_binary_text(source, 1, [95, 85, 200, 110]))
      self.assertIsNone(native_binary_text(source, 1, [95, 185, 500, 210]))

  def test_coordinate_rounding_tolerance(self):
    with tempfile.TemporaryDirectory() as temporary:
      source = Path(temporary) / 'rounded.pdf'
      with pymupdf.open() as pdf:
        page = pdf.new_page(width=1000, height=1000)
        page.insert_text((200, 200), 'Native body text')
        page.draw_rect(pymupdf.Rect(50, 500, 100, 550.3))
        pdf.save(source)
      self.assertTrue(vector_aside(source, 1, [50, 500, 100, 550]))
      self.assertFalse(vector_aside(source, 1, [50, 500, 100, 548]))

  def test_requires_contained_drawing_and_native_text_without_overlap(self):
    with tempfile.TemporaryDirectory() as temporary:
      for name in ('icon', 'text', 'blank', 'oversized', 'scan'):
        path = Path(temporary) / f'{name}.pdf'
        with pymupdf.open() as pdf:
          page = pdf.new_page(width=1000, height=1000)
          if name != 'scan':
            page.insert_text((200, 200), 'Native body text')
          if name != 'blank':
            box = pymupdf.Rect(50, 500, 100, 550)
            if name == 'oversized':
              box = pymupdf.Rect(0, 0, 600, 600)
            page.draw_rect(box)
          if name == 'text':
            page.insert_text((55, 525), 'Note')
          pdf.save(path)
        self.assertEqual(
          vector_aside(path, 1, [48, 498, 102, 552]), name == 'icon', name
        )


if __name__ == '__main__':
  unittest.main()
