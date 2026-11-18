from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import yaml

SHA = re.compile(r'^[a-f0-9]{64}$')
VERSION = 1
BLOCK = 64
PREVIEW = 200
PREVIEW_BREAK = re.compile(r'</?(?:td|th|tr|br|table|caption|p|div)\b[^<>]*>')
FRONTMATTER = re.compile(r'\A---\n.*?\n---\n', re.S)

PREVIEW_DROP = re.compile(
  r'</?(?:thead|tbody|tfoot|sup|sub|u|i|em|b|strong|code|s|del|small|mark|kbd|span|font|a|center'
  r'|figure|figcaption|section|blockquote|img|wbr|hr|col|colgroup)\b[^<>]*>'
)
PART = (
  48  # chunks per text part; a 2,800-passage book must not be one 5 MB file
)
SAME_DOC = 4  # neighbours a row may keep inside its own document before cross-document ones win
AXES = 8
EPOCH = dt.date(1900, 1, 1)
WORD = re.compile(r'[a-z0-9]{3,}')
STOP = set(
  """the and for that with this from are was were have has not but all can its into than then
  they them their there these those which while when where what who will would could should
  also been being between each more most other some such only over under very your you our
  about after before because does doing done during here just like make made many much must
  new now off one out own same see set she him his her how may might per since still through
  too two upon use used using via way well without yet""".split()
)


def log(*args: object) -> None:
  print('[latent]', *args, file=sys.stderr, flush=True)


def sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def read_vectors(content: Path, vectors: dict, dims: int) -> np.ndarray:
  out = np.empty((vectors['rows'], dims), dtype=np.float32)
  for shard in vectors['shards']:
    path = content / shard['path'].lstrip('/')
    data = np.fromfile(path, dtype=np.float32)
    if data.size != shard['rows'] * dims:
      raise SystemExit(
        f'shard {path} holds {data.size} floats, expected {shard["rows"] * dims}'
      )
    out[shard['rowOffset'] : shard['rowOffset'] + shard['rows']] = (
      data.reshape(shard['rows'], dims)
    )
  return out


def read_shard_json(content: Path, shards: list[dict]) -> list[dict]:
  rows: list[dict] = []
  for shard in shards:
    with open(content / shard['path'].lstrip('/'), encoding='utf-8') as fh:
      rows.extend(json.load(fh))
  return rows


def classify(manifest: dict) -> str | None:
  if manifest.get('kind') == 'quartz-pdf-search-v1':
    return 'pdf'
  if manifest.get('modality') == 'image':
    return 'image'
  if manifest.get('version') == 2 and 'chunkMetadata' in manifest:
    return 'note'
  if manifest.get('version') == 2 and 'encoder' in manifest:
    return 'base'
  return None


def discover(content: Path, directory: Path) -> list[dict]:
  """Every manifest under the directory, deduplicated to its root pointer."""
  found: list[dict] = []
  for path in sorted(directory.rglob('manifest.json')):
    rel = path.relative_to(content)
    if 'latent' in rel.parts:
      continue
    raw = path.read_bytes()
    parent = path.parent
    if SHA.match(parent.name):
      pointer = parent.parent / 'manifest.json'
      if pointer.exists() and pointer.read_bytes() == raw:
        continue  # the pointer names this generation; one store
      if pointer.exists():
        continue  # an orphan generation beside a pointer that moved on
    manifest = json.loads(raw)
    kind = classify(manifest)
    store_id = str(rel.parent.as_posix())
    found.append(
      dict(
        id=store_id if store_id != '.' else 'notes',
        path=rel.as_posix(),
        kind=kind,
        manifest=manifest,
        digest=sha256(raw),
        rows=manifest.get('rows'),
        model=manifest.get('model'),
        dims=manifest.get('dims'),
        generation=manifest.get('generation')
        or manifest.get('passages', {}).get('generation'),
      )
    )
  return found


def days_since_epoch(date: str | None) -> int:
  if not date:
    return 0
  try:
    d = dt.date.fromisoformat(date[:10])
  except ValueError:
    return 0
  if d.year < 1900 or d > dt.date.today() + dt.timedelta(days=366):
    return 0
  return (d - EPOCH).days + 1


def preview(text: str, store: str) -> str:
  if store == 'pdf' and '<' in text:
    text = re.sub(
      r'[ \t]{2,}', ' ', PREVIEW_DROP.sub('', PREVIEW_BREAK.sub(' ', text))
    )
  return text[:PREVIEW]


def clean_title(title: str | None, slug: str) -> str:
  """A pdf store title can be a stray code fence from OCR; fall back to the file name."""
  t = re.sub(r'`+|\bmarkdown\b', '', title or '').strip()
  t = re.sub(r'\s+', ' ', t)
  if len(t) < 3:
    stem = os.path.splitext(os.path.basename(slug))[0]
    t = re.sub(r'[-_.]+', ' ', stem).strip() or slug
  return t


def sluggify(path: str) -> str:
  """quartz/util/path.ts sluggify: the slug Quartz gives a content path."""
  return '/'.join(
    re
    .sub(r'\s', '-', seg)
    .replace('&', '-and-')
    .replace('%', '-percent')
    .replace('?', '')
    .replace('#', '')
    for seg in path.split('/')
  ).rstrip('/')


def pdf_index_titles(content: Path) -> dict[str, str]:
  """pdf slug → the title its folder index gives it (`pdfTitles`, keyed by file name relative to the
  index's folder), read the way quartz/util/pdf-titles.ts reads it for folder pages and the pdf
  reader. It outranks the store's title, which is pdf metadata or OCR'd first-page text (a cited
  paper, a publisher line, a chapter heading)."""
  out: dict[str, str] = {}
  for path in sorted(content.rglob('index.md')):
    values = read_frontmatter(path).get('pdfTitles')
    if not isinstance(values, dict):
      continue
    folder = path.parent.relative_to(content).as_posix()
    folder = '' if folder == '.' else folder + '/'
    for name, value in values.items():
      if not isinstance(value, str):
        continue
      title = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', value)).strip()
      slug = sluggify(folder + str(name)).lstrip('/')
      if title and slug.lower().endswith('.pdf'):
        out[slug] = title
  return out


def read_frontmatter(path: Path) -> dict:
  try:
    text = path.read_text(encoding='utf-8', errors='replace')
  except OSError:
    return {}
  m = FRONTMATTER.match(text)
  if not m:
    return {}
  try:
    data = yaml.safe_load(m.group(0)[3:-4]) or {}
  except yaml.YAMLError:
    return {}
  return data if isinstance(data, dict) else {}


def tokens(text: str) -> set[str]:
  return {w for w in WORD.findall(text.lower()) if w not in STOP}


def resolve_box(
  regions: dict | None, page: int, text: str
) -> list[float] | None:
  if not regions:
    return None
  boxes = regions.get('pages', {}).get(str(page))
  if not boxes:
    return None
  want = tokens(text)
  if not want:
    return None
  best, score = None, 0.0
  for box in boxes:
    have = tokens(box.get('text', ''))
    if not have:
      continue
    overlap = len(want & have) / len(have)
    if overlap > score:
      best, score = box.get('bbox'), overlap
  return [round(float(v)) for v in best] if best and score >= 0.3 else None
