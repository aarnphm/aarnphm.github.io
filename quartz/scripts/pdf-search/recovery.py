"""Require native PDF evidence before recovering an empty marginal illustration."""

from pathlib import Path
import hashlib
import re

import pymupdf


def reviewed_region_text(
  source: Path, number: int, bounds: list[float], content: str | None
) -> dict[str, str] | None:
  if number == 96 and bounds == [155, 514, 222, 534]:
    if content not in (None, ''):
      return None
    text = 'Forms'
    models = 'gemini-4-argon-high,claude-opus-5-5-high'
  elif number == 12 and bounds == [256, 825, 766, 912]:
    text = (
      'begin  new max, min;\n'
      "  bop max ←'new a,b. if a>b then a else b';\n"
      "  bop min ←'new a,b. if a<b then a else b';"
    )
    models = 'gemini-4-argon-high'
  else:
    return None
  source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
  if source_hash != (
    '0caa6b96a86bb319a6a577500705cb8072da1b96341d46f35624ea48d8a80a49'
  ):
    return None
  reviewed = {
    'content': text,
    'recovery': 'reviewed-region-text-v1',
    'review_source_sha256': source_hash,
    'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md',
    'review_models': models,
    'review_approved_by': 'aarnphm',
    'review_date': '2026-11-02',
  }
  if number == 12:
    reviewed['review_recognition'] = 'approved-transcription-before-ocr'
  return reviewed


def reviewed_scan_artifact(
  source: Path, number: int, bounds: list[float]
) -> bool:
  # Physical page 18 has scan-edge scratches in this exact detector box.
  # Visual evidence and replay: docs/verification/ocr-aside-recovery.
  if number != 18 or bounds != [9, 269, 27, 468]:
    return False
  return hashlib.sha256(source.read_bytes()).hexdigest() == (
    '8a4f49282b4843078f7935c77b232a7ad06f258f4d2479691b74c38ed68e698a'
  )


def vector_aside(source: Path, number: int, bounds: list[float]) -> bool:
  with pymupdf.open(source) as pdf:
    page = pdf[number - 1]
    if not page.get_text().strip():
      return False
    area = (
      pymupdf.Rect(
        bounds[0] * page.rect.width / 1000,
        bounds[1] * page.rect.height / 1000,
        bounds[2] * page.rect.width / 1000,
        bounds[3] * page.rect.height / 1000,
      )
      * page.derotation_matrix
    )
    # Detector bounds are quantized to a 0-1000 grid. Allow one grid unit
    # at each edge, including in the native-text exclusion check.
    tolerance = (
      pymupdf.Rect(
        max(0, bounds[0] - 1) * page.rect.width / 1000,
        max(0, bounds[1] - 1) * page.rect.height / 1000,
        min(1000, bounds[2] + 1) * page.rect.width / 1000,
        min(1000, bounds[3] + 1) * page.rect.height / 1000,
      )
      * page.derotation_matrix
    )
    if area.is_empty or any(
      pymupdf.Rect(word[:4]).intersects(tolerance)
      for word in page.get_text('words')
    ):
      return False
    for drawing in page.get_drawings():
      box = drawing['rect']
      if tolerance.contains(box) and box.get_area() >= area.get_area() * 0.5:
        return True
    return False


def native_region_text(
  source: Path, number: int, bounds: list[float]
) -> str | None:
  """Recover complete visible native lines, retaining exponent/base markers."""

  with pymupdf.open(source) as pdf:
    page = pdf[number - 1]
    area = (
      pymupdf.Rect(
        bounds[0] * page.rect.width / 1000,
        bounds[1] * page.rect.height / 1000,
        bounds[2] * page.rect.width / 1000,
        bounds[3] * page.rect.height / 1000,
      )
      * page.derotation_matrix
    )
    margin = 8 * max(page.rect.width, page.rect.height) / 1000
    tolerance = pymupdf.Rect(
      area.x0 - margin, area.y0 - margin, area.x1 + margin, area.y1 + margin
    )
    for span in page.get_texttrace():
      if (span['type'] != 0 or span['opacity'] == 0) and pymupdf.Rect(
        span['bbox']
      ).intersects(area):
        return None
    for image in page.get_image_info():
      overlap = pymupdf.Rect(image['bbox']) & area
      if overlap.get_area() > area.get_area() * 0.05:
        return None
    lines: list[str] = []
    for block in page.get_text('dict')['blocks']:
      for line in block.get('lines', []):
        box = pymupdf.Rect(line['bbox'])
        center = (box.tl + box.br) / 2
        if not area.contains(center):
          continue
        if not tolerance.contains(box) or line['dir'] != (1.0, 0.0):
          return None
        spans = line['spans']
        baseline = max(spans, key=lambda span: span['size'])
        fragments: list[str] = []
        for span in spans:
          text = span['text']
          if '\ufffd' in text or any(ord(char) < 32 for char in text):
            return None
          if span['flags'] & 1:
            text = '^{' + text + '}'
          elif (
            span['size'] < baseline['size'] * 0.85
            and span['origin'][1]
            > baseline['origin'][1] + baseline['size'] * 0.1
          ):
            text = '_{' + text + '}'
          fragments.append(text)
        lines.append(''.join(fragments).strip())
    text = '\n'.join(lines).strip()
    return text or None


def native_binary_text(
  source: Path, number: int, bounds: list[float]
) -> str | None:
  text = native_region_text(source, number, bounds)
  if text is None or not re.search(r'(?:[01]{8}[ \t]+){7}[01]{8}', text):
    return None
  return text
