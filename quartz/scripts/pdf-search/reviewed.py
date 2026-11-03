"""Apply source-bound partial transcriptions without rewriting OCR evidence."""

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping, Sequence


def reviewed_regions(
  source_sha256: str, page: int, regions: Sequence[Mapping[str, object]]
) -> list[dict[str, object]] | None:
  if source_sha256 != (
    '67199d19ca1eed2eb62255e82711568cf5e55f57ce4f71d22453e6f11a9089c9'
  ) or page not in (274, 277):
    return None
  result = [dict(region) for region in regions]
  review = json.loads(
    (
      Path(__file__).with_name('reviewed') / f'semiology-{page}.json'
    ).read_text()
  )
  if (
    review['source_sha256'] != source_sha256
    or review['page'] != page
    or review['status'] != 'reviewed-partial'
    or '[unreadable:' not in review['text']
  ):
    raise RuntimeError('Invalid reviewed partial PDF transcription')
  for region in result:
    if region['index'] != review['region']:
      continue
    text = region['content']
    bounds = region.get('bbox_2d')
    if (
      not isinstance(text, str)
      or not isinstance(bounds, (list, tuple))
      or list(bounds) != review['bounds']
      or hashlib.sha256(text.encode()).hexdigest()
      != review['original_text_sha256']
    ):
      raise RuntimeError(f'Reviewed PDF region changed on page {page}')
    region['content'] = review['text']
    return result
  raise RuntimeError(f'Reviewed PDF region missing on page {page}')
