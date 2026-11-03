"""Apply source-bound partial transcriptions without rewriting OCR evidence."""

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping, Sequence


def reviewed_regions(
  source_sha256: str, page: int, regions: Sequence[Mapping[str, object]]
) -> list[dict[str, object]] | None:
  if source_sha256 == (
    '67199d19ca1eed2eb62255e82711568cf5e55f57ce4f71d22453e6f11a9089c9'
  ) and page in (274, 277):
    names = [f'semiology-{page}.json']
  elif (
    source_sha256
    == ('728e6d0ca3809d504f12f3302ef5303238a7324cf2824b303d64781d597dba7d')
    and page == 97
  ):
    names = [f'type97-{region}.json' for region in (2, 3, 5, 8, 10)]
  else:
    return None
  result = [dict(region) for region in regions]
  for name in names:
    review = json.loads(
      (Path(__file__).with_name('reviewed') / name).read_text()
    )
    apply_review(source_sha256, page, result, review)
  return result


def apply_review(
  source_sha256: str, page: int, regions: list[dict[str, object]], review: dict
) -> None:
  if (
    review['source_sha256'] != source_sha256
    or review['page'] != page
    or review['status'] not in ('reviewed-partial', 'reviewed-complete')
    or (
      review['status'] == 'reviewed-partial'
      and '[unreadable:' not in review['text']
    )
  ):
    raise RuntimeError('Invalid reviewed partial PDF transcription')
  for region in regions:
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
    return
  raise RuntimeError(f'Reviewed PDF region missing on page {page}')
