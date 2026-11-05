# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Restamp a generated sidecar without recomputing vectors, neighbours or models.

uv run quartz/scripts/latent/restamp.py --out docs/verification/latent-dates/artifacts/restamped
Output is a complete generation beside the live data, ready for verification.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import struct
from pathlib import Path

from timeline_dates import resolve_rows


def read(path: Path):
  return json.loads(path.read_text())


def write(path: Path, value: object) -> None:
  path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


def restamp(content: Path, out: Path) -> None:
  root = content / 'embeddings/latent'
  pointer = read(root / 'index.json')
  source = root / pointer['hash']
  dest = out / pointer['hash']
  if dest.exists():
    raise FileExistsError(dest)
  if read(source / 'index.json')['version'] != 1:
    raise ValueError('restamping supports sidecar version 1 only')
  docs = read(source / 'docs.json')
  rows = [
    dict(
      doc=d['id'],
      store=d['s'],
      date=d['date'],
      prec=d['prec'],
      dateSource=d.get('dateSource'),
    )
    for d in docs
  ]
  resolve_rows(content, rows)
  for doc, row in zip(docs, rows):
    doc.update({key: row.get(key) for key in ('date', 'prec', 'dateSource')})
  order = sorted(
    range(len(docs)),
    key=lambda i: (
      docs[i]['date'] is None,
      docs[i]['date'] or '',
      docs[i]['s'],
      docs[i]['id'],
    ),
  )
  remap = {old: new for new, old in enumerate(order)}
  shutil.copytree(source, dest)
  # Rewrite from the source, since renaming in place would overwrite another document.
  shutil.rmtree(dest / 'text')
  (dest / 'text').mkdir()
  for path in (source / 'text').glob('*.json'):
    payload = read(path)
    old = payload['d']
    payload['d'] = remap[old]
    if 'id' in payload:
      payload.update({
        key: docs[old].get(key) for key in ('date', 'prec', 'dateSource')
      })
    suffix = path.stem.partition('-')[2]
    name = str(remap[old]) + ('-' + suffix if suffix else '') + '.json'
    write(dest / 'text' / name, payload)
  for path in (source / 'prev').glob('*.json'):
    payload = read(path)
    for row in payload:
      row['d'] = remap[row['d']]
    write(dest / 'prev' / path.name, payload)
  probe = read(source / 'probe.json')
  for axis in probe['axes']:
    for entries in axis.get('poles', {}).values():
      for entry in entries:
        entry['d'] = remap[entry['d']]
  write(dest / 'probe.json', probe)
  tags_path = source / 'feat/tags.json'
  if tags_path.exists():
    tags = read(tags_path)
    tags['pred'] = sorted([
      [remap[d], predictions] for d, predictions in tags['pred']
    ])
    write(dest / 'feat/tags.json', tags)
  vectors = (source / 'vec/docs-256-int8.bin').read_bytes()
  width = len(vectors) // len(docs)
  assert width * len(docs) == len(vectors)
  (dest / 'vec/docs-256-int8.bin').write_bytes(
    b''.join(vectors[old * width : (old + 1) * width] for old in order)
  )
  rowdoc = (source / 'rowdoc.u32').read_bytes()
  (dest / 'rowdoc.u32').write_bytes(
    b''.join(
      struct.pack('<I', remap[old])
      for (old,) in struct.iter_unpack('<I', rowdoc)
    )
  )
  index = read(source / 'index.json')
  track = bytearray()
  sorted_docs = []
  for old in order:
    doc = docs[old]
    doc['d'] = remap[old]
    if 'copyOf' in doc:
      doc['copyOf'] = remap[doc['copyOf']]
    day = (
      (dt.date.fromisoformat(doc['date']) - dt.date(1900, 1, 1)).days + 1
      if doc['date']
      else 0
    )
    track.extend(
      struct.pack(
        '<HBBIH',
        day,
        index['kinds'].index(doc['s']),
        {'day': 0, 'month': 1, 'year': 2}.get(doc['prec'], 3),
        doc['r0'],
        min(doc['n'], 0xFFFF),
      )
    )
    sorted_docs.append(doc)
  write(dest / 'docs.json', sorted_docs)
  (dest / 'track.bin').write_bytes(track)
  index['datedDocs'] = sum(bool(d['date']) for d in docs)
  index['datesRestamped'] = dt.datetime.now(dt.timezone.utc).isoformat()
  write(dest / 'index.json', index)
  write(out / 'index.json', pointer)
  print(
    json.dumps(
      dict(output=str(dest), docs=len(docs), dated=index['datedDocs'])
    )
  )


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--content', type=Path, default=Path('content'))
  parser.add_argument('--out', type=Path, required=True)
  args = parser.parse_args()
  restamp(args.content.resolve(), args.out.resolve())
