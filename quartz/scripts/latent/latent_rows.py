from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from build_common import (
  clean_title,
  log,
  pdf_index_titles,
  read_shard_json,
  read_vectors,
)
from generated_text import generated_text, note_sources, notebook_markdown
from latent_markdown import load_bib, note_markdown, restore_math
from paper_sources import paper_sources


def load_rows(
  content: Path,
  store: dict,
  offset: int,
  excluded: set[str] = frozenset(),
  arxiv: Path | None = None,
) -> tuple[list[dict], np.ndarray]:
  """Rows of one store in vector order: id, doc, title, chunk, page, label, text, date."""
  m = store['manifest']
  kind = store['kind']
  dims = m['dims']
  vectors = read_vectors(content, m['vectors'], dims)
  rows: list[dict] = []
  if kind == 'note':
    passages = (
      read_shard_json(content, m['passages']['shards'])
      if 'passages' in m
      else []
    )
    if passages and len(passages) != m['rows']:
      raise SystemExit('note passages do not align with the vectors')
    keep: list[int] = []
    for i, rid in enumerate(m['ids']):
      meta = m['chunkMetadata'].get(rid, {})
      if meta.get('parentSlug', rid.split('#')[0]) in excluded:
        continue
      keep.append(i)
      p = passages[i] if passages else {}
      rows.append(
        dict(
          id=rid,
          doc=meta.get('parentSlug', rid.split('#')[0]),
          title=m['titles'][i],
          chunk=meta.get('chunkId', 0),
          page=None,
          label='',
          text=p.get('text') or '',
          raw=p.get('text') or '',
          date=p.get('date'),
          dateSource=p.get('dateSource') or 'embedding-source',
          prec=p.get('datePrecision'),
          start=p.get('charStart'),
        )
      )
    if len(keep) != len(m['ids']):
      log(
        'note rows dropped (protected/private/draft)',
        len(m['ids']) - len(keep),
      )
      vectors = vectors[keep]
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
      by_doc.setdefault(r['doc'], []).append(r)
    sources = note_sources(content)
    by_name: dict[str, str] = {}
    for slug, path in sources.items():
      by_name.setdefault(path.stem.lower(), slug)
      by_name.setdefault(slug.rsplit('/', 1)[-1].lower(), slug)
    bib = load_bib(content / 'References.bib')
    restored = 0
    marked = 0
    generated = 0
    for doc, chunks in by_doc.items():
      source = sources.get(doc)
      if source is not None:
        md = (
          notebook_markdown(source)
          if source.suffix == '.ipynb'
          else source.read_text(encoding='utf-8', errors='replace')
        )
        restored += restore_math(chunks, md)
        marked += note_markdown(chunks, md, by_name, bib)
      generated += generated_text(chunks)
    log('note math spans restored', restored)
    log('notes read from markdown', marked, 'of', len(by_doc))
    log('note chunks rewritten from generated page text', generated)
  elif kind == 'pdf':
    passages = read_shard_json(content, m['passages']['shards'])
    if len(passages) != m['rows']:
      raise SystemExit('pdf passages do not align with the vectors')
    docs = m['documents']
    named = pdf_index_titles(content)
    log(
      'pdf titles from folder indexes',
      sum(slug in named for slug in docs),
      'of',
      len(docs),
    )
    for p in passages:
      d = docs.get(p['slug'], {})
      rows.append(
        dict(
          id=f'{p["slug"]}#page={p["page"]}&chunk={p["chunk"]}',
          doc=p['slug'],
          title=named.get(p['slug'])
          or clean_title(d.get('title') or p.get('title'), p['slug']),
          # the title the encoder saw (pdf-search/build.py prefixes the passage title)
          embedTitle=p.get('title'),
          chunk=p['chunk'],
          page=p['page'],
          label=p.get('pageLabel') or '',
          text=p['text'],
          date=d.get('date') or p.get('date'),
          dateSource=d.get('dateSource'),
          prec=d.get('datePrecision'),
          sha=d.get('sourceSha256'),
          pages=d.get('totalPages'),
        )
      )
  elif kind == 'image':
    for rid in m['ids']:
      rows.append(  # noqa: PERF401
        dict(
          id=rid,
          doc=rid,
          title=os.path.basename(rid),
          chunk=0,
          page=None,
          label='',
          text='',
          date=None,
          prec=None,
        )
      )
  elif kind == 'base':
    titles = m.get('titles') or m['ids']
    for i, rid in enumerate(m['ids']):
      rows.append(
        dict(
          id=rid,
          doc=rid,
          title=titles[i],
          chunk=0,
          page=None,
          label='',
          text='',
          date=None,
          prec=None,
        )
      )
  for i, r in enumerate(rows):
    r['row'] = offset + i
    r['store'] = kind
    r['storeId'] = store['id']
  if kind == 'pdf' and arxiv is not None:
    paper_sources(rows, arxiv)
  return rows, vectors
