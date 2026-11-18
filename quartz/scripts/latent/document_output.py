"""Write per-document text parts, the document table, and the packed timeline."""

from __future__ import annotations

import json
import struct
from pathlib import Path


def write_documents(
  out: Path,
  docs: list[dict],
  rows: list[dict],
  canon,
  unique,
  kinds: list[str],
  part_size: int,
) -> None:
  docs_out = []
  track = bytearray()
  for di, d in enumerate(docs):
    chunks = []
    for r in d['rows']:
      row = rows[r]
      chunk = dict(r=r, c=row['chunk'], text=row['text'])
      if 'md' in row:
        chunk['text'] = row['md']
        chunk['m'] = 1
      if row.get('same'):
        chunk['same'] = [row['same']['row'], row['same']['chunk'], row['why']]
      if row.get('into') is not None:
        chunk['j'] = row['into']
        chunk['text'] = ''
      if row['page']:
        chunk['p'] = row['page']
        chunk['l'] = row['label']
      if row.get('box'):
        chunk['box'] = row['box']
      if not unique[r]:
        chunk['copy'] = int(canon[r])
      chunks.append(chunk)
    payload = dict(
      d=di,
      id=d['id'],
      store=d['store'],
      title=d['title'],
      date=d['date'],
      prec=d['prec'],
      total=len(chunks),
      part=part_size,
      chunks=chunks[:part_size],
    )
    if d.get('dateSource'):
      payload['dateSource'] = d['dateSource']
    if d.get('sha'):
      payload['sha'] = d['sha']
      payload['pages'] = d.get('pages')
    with open(out / 'text' / f'{di}.json', 'w', encoding='utf-8') as fh:
      json.dump(payload, fh, ensure_ascii=False, separators=(',', ':'))
    for p in range(1, (len(chunks) + part_size - 1) // part_size):
      with open(
        out / 'text' / f'{di}-{p:03d}.json', 'w', encoding='utf-8'
      ) as fh:
        json.dump(
          dict(
            d=di, part=p, chunks=chunks[p * part_size : (p + 1) * part_size]
          ),
          fh,
          ensure_ascii=False,
          separators=(',', ':'),
        )

    first, count = d['rows'][0], len(d['rows'])
    docs_out.append(
      dict(
        d=di,
        id=d['id'],
        s=d['store'],
        t=d['title'],
        date=d['date'],
        **({'dateSource': d['dateSource']} if d.get('dateSource') else {}),
        prec=d['prec'],
        r0=first,
        n=count,
        **({'sha': d['sha'], 'pages': d['pages']} if d.get('sha') else {}),
        **({'copyOf': d['copyOf']} if 'copyOf' in d else {}),
        **(
          {'et': d['embedTitle']}
          if d.get('embedTitle') and d['embedTitle'] != d['title']
          else {}
        ),
      )
    )
    track += struct.pack(
      '<HBBIH',
      d['day'] & 0xFFFF,
      kinds.index(d['store']),
      {'day': 0, 'month': 1, 'year': 2}.get(d['prec'] or '', 3),
      first,
      min(count, 0xFFFF),
    )
  with open(out / 'docs.json', 'w', encoding='utf-8') as fh:
    json.dump(docs_out, fh, ensure_ascii=False, separators=(',', ':'))
  (out / 'track.bin').write_bytes(bytes(track))
