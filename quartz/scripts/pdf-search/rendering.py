"""Render PDF pages while avoiding enlargement of oversized full-page scans."""

from math import hypot

from PIL import Image
import pymupdf

RENDER_POLICY = 'native-oversized-scan-v1'


def render_page(page: pymupdf.Page, dpi: int) -> Image.Image:
  scale = dpi / 72
  if max(page.rect.width, page.rect.height) * scale > 4096:
    images = page.get_image_info()
    if (
      len(images) == 1
      and page.first_annot is None
      and page.first_widget is None
      and not page.get_drawings()
      and not any(
        span['type'] != 3 and span['opacity'] > 0
        for span in page.get_texttrace()
      )
    ):
      scan = images[0]
      bounds = pymupdf.Rect(scan['bbox'])
      # Image bounds use unrotated page coordinates, unlike page.rect.
      visible = page.rect * page.derotation_matrix
      if all(abs(a - b) < 0.5 for a, b in zip(bounds, visible)):
        a, b, c, d, _, _ = scan['transform']
        horizontal, vertical = hypot(a, b), hypot(c, d)
        if horizontal > 0 and vertical > 0:
          scale = min(
            scale, max(scan['width'] / horizontal, scan['height'] / vertical)
          )
  if scale == dpi / 72:
    pixels = page.get_pixmap(dpi=dpi, alpha=False)
  else:
    pixels = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
  return Image.frombytes('RGB', (pixels.width, pixels.height), pixels.samples)
