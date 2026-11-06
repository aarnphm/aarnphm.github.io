from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import struct
import time
from collections import Counter
from pathlib import Path

import numpy as np

from dedup import canonical_rows, summary
from document_output import write_documents
from generated_text import note_sources
from build_common import (
  AXES,
  BLOCK,
  PART,
  SHA,
  VERSION,
  days_since_epoch,
  discover,
  log,
  preview,
  resolve_box,
  sha256,
)
from latent_analysis import (
  MIN_POLE,
  POLE_NOTES,
  POLE_ROWS,
  axes_spec,
  bimodality,
  excluded_notes,
  held_out_auc,
  knn,
  latent_page,
  pole_entry,
  pole_members,
  pole_rows,
  read_frontmatter,
  spec_json,
  tag_key,
  term_window,
  top_terms,
)
from latent_rows import load_rows
from timeline_dates import resolve_rows


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--content', default='content')
  ap.add_argument(
    '--directory', default='embeddings', help='relative to --content'
  )
  ap.add_argument('--k', type=int, default=16)
  ap.add_argument(
    '--arxiv',
    default='quartz/.quartz-cache/arxiv-src',
    help='arxiv.py output: <id>/paper.md, the TeX source of each arXiv pdf as markdown',
  )
  ap.add_argument(
    '--regions', default='embeddings/pdf-regions', help='relative to --content'
  )
  ap.add_argument(
    '--no-features',
    action='store_true',
    help='skip the sparse dictionaries (features.py)',
  )
  ap.add_argument(
    '--out',
    default=None,
    help='sidecar root instead of <directory>/latent (a dry run beside the live one)',
  )
  args = ap.parse_args()
  content = Path(args.content).resolve()
  directory = content / args.directory
  if not directory.is_dir():
    raise SystemExit(f'{directory} is not a directory')
  t0 = time.time()

  stores = discover(content, directory)
  if not stores:
    raise SystemExit('no manifest.json under the directory')
  reference = next((s for s in stores if s['kind'] == 'note'), stores[0])
  model, dims = reference['model'], reference['dims']
  for s in stores:
    s['included'] = (
      s['kind'] is not None and s['model'] == model and s['dims'] == dims
    )
    s['reason'] = (
      None
      if s['included']
      else (
        'unknown manifest kind'
        if s['kind'] is None
        else f'model {s["model"]} / {s["dims"]}-d is not {model} / {dims}-d'
      )
    )
  included = [s for s in stores if s['included']]
  digest_lines = sorted(f'{s["path"]}\t{s["digest"]}' for s in included)
  page, page_fm = latent_page(content, args.directory)
  spec = axes_spec(page_fm.get('axes'))
  if spec:
    digest_lines.append(f'axes\t{sha256(spec_json(spec).encode())}')
  log(
    'axes from', page.relative_to(content) if page else None, spec_json(spec)
  )
  excluded = excluded_notes(content)
  log('excluded notes (protected/private/draft)', len(excluded))
  sidecar_hash = sha256('\n'.join(digest_lines).encode())
  log(
    'stores', [(s['id'], s['kind'], s['rows'], s['included']) for s in stores]
  )
  log('hash', sidecar_hash)

  out_root = Path(args.out).resolve() if args.out else directory / 'latent'
  out = out_root / sidecar_hash
  if out.exists():
    shutil.rmtree(out)
  for sub in ('knn', 'prev', 'text', 'vec'):
    (out / sub).mkdir(parents=True)

  rows: list[dict] = []
  mats: list[np.ndarray] = []
  for s in included:
    s['rowOffset'] = len(rows)
    r, V = load_rows(
      content, s, len(rows), excluded, Path(args.arxiv).resolve()
    )
    rows.extend(r)
    mats.append(V)
    log(s['id'], len(r), 'rows')
  V = np.vstack(mats)
  del mats
  N = V.shape[0]
  resolve_rows(content, rows)

  # a document's rows are contiguous (docs.json carries r0 and n), so rows are permuted into
  # document order: a book that sits in two bases arrives as two separated runs otherwise
  first_seen: dict[str, int] = {}
  for r in rows:
    first_seen.setdefault(r['doc'], len(first_seen))
  perm = sorted(range(N), key=lambda i: (first_seen[rows[i]['doc']], i))
  if perm != list(range(N)):
    rows = [rows[i] for i in perm]
    V = V[perm]
    for i, r in enumerate(rows):
      r['row'] = i
    log('rows permuted into document order')
  log('rows', N, f'{time.time() - t0:.0f}s')
  # copied text: every row stays (a copied page still reads in full), but the graph, the axes and
  # the dictionaries count each text once, at its canonical row (dedup.py)
  canon = canonical_rows(rows, V)
  unique = canon == np.arange(N)
  copies = summary(rows, canon)
  log('copies', copies, f'{time.time() - t0:.0f}s')

  # documents, time-ordered: dated oldest -> newest, then undated by store and id
  doc_index: dict[str, int] = {}
  docs: list[dict] = []
  for r in rows:
    key = r['doc']
    if key not in doc_index:
      doc_index[key] = len(docs)
      docs.append(
        dict(
          id=key,
          store=r['store'],
          storeId=r['storeId'],
          title=r['title'],
          embedTitle=r.get('embedTitle'),
          date=r['date'],
          dateSource=r.get('dateSource'),
          prec=r['prec'],
          sha=r.get('sha'),
          pages=r.get('pages'),
          rows=[],
        )
      )
    docs[doc_index[key]]['rows'].append(r['row'])
  for d in docs:
    d['day'] = days_since_epoch(d['date'])
    if d['day'] == 0:
      d['date'] = None
  order = sorted(
    range(len(docs)),
    key=lambda i: (
      docs[i]['day'] == 0,
      docs[i]['day'],
      docs[i]['store'],
      docs[i]['id'],
    ),
  )
  docs = [docs[i] for i in order]
  doc_of_row = np.empty(N, dtype=np.uint32)
  for di, d in enumerate(docs):
    for r in d['rows']:
      doc_of_row[r] = di
    d['rows'].sort()
  log('docs', len(docs))
  # a document every row of which is a copy points at the document holding most of its text
  doc_own = np.array([bool(unique[d['rows']].any()) for d in docs])
  for di in np.flatnonzero(~doc_own):
    held = Counter(int(doc_of_row[canon[r]]) for r in docs[di]['rows'])
    docs[di]['copyOf'] = held.most_common(1)[0][0]

  # neighbours are canonical rows only; a copy lists its canonical row first, then that row's
  # neighbours, so the links pane of a copied page leads to the copy it repeats
  keep = np.flatnonzero(unique)
  idx = np.empty((N, args.k), dtype=np.uint32)
  sc = np.empty((N, args.k), dtype=np.float32)
  sub_idx, sc[keep] = knn(V[keep], args.k, doc_of_row[keep])
  idx[keep] = keep[sub_idx]
  for r in np.flatnonzero(~unique):
    c = canon[r]
    idx[r, 0], idx[r, 1:] = c, idx[c, :-1]
    sc[r, 0], sc[r, 1:] = float(V[r] @ V[c]), sc[c, :-1]
  degree = np.minimum(
    np.bincount(idx[keep].ravel(), minlength=N), 0xFFFF
  ).astype(np.uint16)
  log('knn', f'{time.time() - t0:.0f}s')

  # regions for pdf page silhouettes
  regions_dir = content / args.regions
  regions_cache: dict[str, dict | None] = {}

  def regions_for(sha: str | None) -> dict | None:
    if not sha:
      return None
    if sha not in regions_cache:
      path = regions_dir / f'{sha}.json'
      regions_cache[sha] = (
        json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
      )
    return regions_cache[sha]

  boxed = 0
  for r in rows:
    if r['store'] == 'pdf' and r['page']:
      r['box'] = resolve_box(regions_for(r.get('sha')), r['page'], r['text'])
      boxed += r['box'] is not None
  log('boxes', boxed)

  # vectors: 256-d MRL truncation, renormalised, int8; PCA axes over the truncated space
  V256 = V[:, :256].copy()
  V256 /= np.linalg.norm(V256, axis=1, keepdims=True) + 1e-9
  mean = V256[unique].mean(0)
  centred = V256 - mean
  cov = centred[unique].T @ centred[unique] / int(unique.sum())
  scale = float(np.abs(V256).max())
  q = np.clip(np.round(V256 / scale * 127), -127, 127).astype(np.int8)
  q.tofile(out / 'vec' / 'rows-256-int8.bin')
  doc_vecs = np.stack([V256[d['rows']].mean(0) for d in docs])
  doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9
  np.clip(np.round(doc_vecs / scale * 127), -127, 127).astype(np.int8).tofile(
    out / 'vec' / 'docs-256-int8.bin'
  )
  log('vectors', f'{time.time() - t0:.0f}s')

  # axes: the store contrast, the page's frontmatter axes, then principal directions of the rest
  store_arr = np.array([r['store'] for r in rows])
  axes: list[dict] = []

  def add_axis(
    aid: str, name: str, direction: np.ndarray, a: str, b: str, **extra: object
  ) -> None:
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    proj = centred @ direction
    axes.append(
      dict(
        id=aid,
        name=name,
        a=a,
        b=b,
        dir=direction.astype(np.float32),
        proj=proj,
        extra=extra,
      )
    )

  kinds = [
    k for k in ('note', 'pdf', 'image', 'base') if (store_arr == k).any()
  ]
  if 'note' in kinds and 'pdf' in kinds:
    add_axis(
      'store',
      'notes ↔ passages',
      V256[(store_arr == 'pdf') & unique].mean(0)
      - V256[(store_arr == 'note') & unique].mean(0),
      'notes',
      'pdf passages',
    )
  if spec:
    sources = note_sources(content)
    tags_of: dict[str, set[str]] = {}
    for d in docs:
      if d['store'] == 'note' and d['id'] in sources:
        t = read_frontmatter(sources[d['id']]).get('tags') or []
        tags_of[d['id']] = {
          tag_key(x)
          for x in (
            [t] if isinstance(t, str) else t if isinstance(t, list) else []
          )
        }
    doc_centred = doc_vecs - mean
    for ai, ax in enumerate(spec):
      ma, mb = (
        pole_members(docs, ax['a'], tags_of),
        pole_members(docs, ax['b'], tags_of),
      )
      a = sorted(d for d in ma - mb if doc_own[d])
      b = sorted(d for d in mb - ma if doc_own[d])
      label_a = ' · '.join(
        ax['a']['tags'] + [p + '/' for p in ax['a']['paths']]
      )
      label_b = ' · '.join(
        ax['b']['tags'] + [p + '/' for p in ax['b']['paths']]
      )
      if len(a) < MIN_POLE or len(b) < MIN_POLE:
        log(
          f'axis {label_a} ↔ {label_b} skipped: poles hold {len(a)} and {len(b)} documents'
        )
        continue
      sep = held_out_auc(doc_centred, a, b)
      log(
        f'axis {label_a} ↔ {label_b}',
        len(a),
        len(b),
        f'held-out auc {sep:.3f}',
      )
      add_axis(
        f'fm{ai + 1}',
        ax['name'] or f'{label_a} ↔ {label_b}',
        doc_vecs[b].mean(0) - doc_vecs[a].mean(0),
        label_a,
        label_b,
        source='frontmatter',
        spec=dict(a=ax['a'], b=ax['b']),
        docs=dict(a=len(a), b=len(b)),
        sep=round(sep, 3),
      )
  # principal directions of the variance the named axes leave: the store contrast and a
  # philosophy ↔ ml axis both lie close to the raw pc1, which would show the same split twice
  named = (
    np.stack([ax['dir'] for ax in axes]).T
    if axes
    else np.zeros((256, 0), dtype=np.float32)
  )
  if named.shape[1]:
    Q, _ = np.linalg.qr(named.astype(np.float64))
    P = np.eye(256) - Q @ Q.T
    cov_rest = P @ cov @ P
  else:
    cov_rest = cov
  evals, evecs = np.linalg.eigh(cov_rest)
  pcs = evecs[:, ::-1][:, :AXES].T.astype(np.float32)  # AXES x 256
  explained = (evals[::-1][:AXES] / np.trace(cov)).astype(np.float32)
  for i in range(AXES):
    if len(axes) >= AXES:
      break
    add_axis(
      f'pc{i + 1}',
      f'pc{i + 1} · {explained[i] * 100:.1f} %',
      pcs[i],
      '−',  # noqa: RUF001
      '+',
    )
  axes = axes[:AXES]
  rest_sample = np.random.default_rng(0).choice(
    keep, size=min(keep.size, 2000), replace=False
  )
  probe = []
  axis_i8 = np.zeros((N, AXES), dtype=np.int8)
  for ai, ax in enumerate(axes):
    proj = ax['proj']
    lim = float(np.abs(proj).max()) + 1e-9
    axis_i8[:, ai] = np.clip(np.round(proj / lim * 127), -127, 127)
    order = keep[np.argsort(proj[keep])]
    ends = dict(a=order, b=order[::-1])
    # the store axis has a notes end already; elsewhere notes get a list of their own
    note_ends = (
      {}
      if ax['id'] == 'store'
      else {
        side: pole_rows(rows, o, doc_of_row, POLE_NOTES, store='note')
        for side, o in ends.items()
      }
    )
    hist, _ = np.histogram(proj[keep], bins=32, range=(-lim, lim))
    probe.append(
      dict(
        id=ax['id'],
        name=ax['name'],
        a=ax['a'],
        b=ax['b'],
        **ax['extra'],
        bc=round(bimodality(proj[keep]), 3),
        scale=lim,
        hist=hist.astype(int).tolist(),
        terms={
          side: top_terms(rows, term_window(o, doc_of_row), rest_sample)
          for side, o in ends.items()
        },
        poles={
          side: [
            pole_entry(rows, doc_of_row, r)
            for r in pole_rows(rows, o, doc_of_row, POLE_ROWS)
          ]
          for side, o in ends.items()
        },
      )
    )
    if note_ends:
      probe[-1]['notes'] = {
        side: [pole_entry(rows, doc_of_row, r) for r in rs]
        for side, rs in note_ends.items()
      }
  with open(out / 'vec' / 'meta.json', 'w', encoding='utf-8') as fh:
    json.dump(
      dict(
        rows=N,
        docs=len(docs),
        dims=256,
        dtype='int8',
        scale=scale,
        mean=[round(float(v), 6) for v in mean],
        axes=[
          dict(
            id=a['id'],
            name=a['name'],
            dir=[round(float(v), 6) for v in a['dir']],
          )
          for a in axes
        ],
      ),
      fh,
    )
  with open(out / 'probe.json', 'w', encoding='utf-8') as fh:
    json.dump(dict(axes=probe), fh, ensure_ascii=False)
  log('axes', [a['id'] for a in axes], f'{time.time() - t0:.0f}s')

  # kNN blocks: per row 16 x (u32 idx, u8 score) then 8 x i8 axis projections
  stride = args.k * 5 + AXES
  score_u8 = np.clip(sc * 255, 0, 255).astype(np.uint8)
  for b in range(0, N, BLOCK):
    rows_in = min(BLOCK, N - b)
    buf = bytearray(rows_in * stride)
    for i in range(rows_in):
      base = i * stride
      r = b + i
      struct.pack_into(f'<{args.k}I', buf, base, *idx[r].tolist())
      buf[base + args.k * 4 : base + args.k * 5] = score_u8[r].tobytes()
      buf[base + args.k * 5 : base + stride] = axis_i8[r].tobytes()
    (out / 'knn' / f'{b // BLOCK:05d}.bin').write_bytes(bytes(buf))

  # preview blocks and per-document text
  for b in range(0, N, BLOCK):
    block = [
      dict(
        r=r['row'],
        d=int(doc_of_row[r['row']]),
        s=r['store'],
        t=r['title'],
        c=r['chunk'],
        p=r['page'],
        l=r['label'],
        x=''
        if r.get('same')
        else preview(r.get('plain') or r['text'], r['store']),
      )
      for r in rows[b : b + BLOCK]
    ]
    with open(
      out / 'prev' / f'{b // BLOCK:05d}.json', 'w', encoding='utf-8'
    ) as fh:
      json.dump(block, fh, ensure_ascii=False, separators=(',', ':'))
  # documents table and the time track
  write_documents(out, docs, rows, canon, unique, kinds, PART)
  degree.tofile(out / 'degree.u16')
  doc_of_row.tofile(out / 'rowdoc.u32')
  canon.astype('<u4').tofile(out / 'canon.u32')

  counts = Counter(r['store'] for r in rows)
  dated = sum(1 for d in docs if d['day'])
  index = dict(
    version=VERSION,
    hash=sidecar_hash,
    built=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
    model=model,
    dims=dims,
    k=args.k,
    block=BLOCK,
    part=PART,
    axes=AXES,
    rows=N,
    docs=len(docs),
    datedDocs=dated,
    kinds=kinds,
    axesPage=page.relative_to(content).as_posix() if page else None,
    axesSpec=spec,
    counts=dict(counts),
    copies=copies,
    stores=[
      dict(
        id=s['id'],
        kind=s['kind'],
        path='/' + s['path'],
        generation=s['generation'],
        rows=s['rows'],
        rowOffset=s.get('rowOffset'),
        model=s['model'],
        dims=s['dims'],
        included=s['included'],
        reason=s['reason'],
        coverage=s['manifest'].get('coverage'),
      )
      for s in stores
    ],
  )
  with open(out / 'index.json', 'w', encoding='utf-8') as fh:
    json.dump(index, fh, indent=1)
  if not args.no_features:
    # sparse dictionaries and tag probes over the rows just written; adds `features` and `tags`
    # to index.json and archives each dictionary under .models-archive/latent-<name>/
    from features import write_features

    write_features(
      out,
      Path('quartz/.quartz-cache').resolve(),
      archive=Path('.models-archive').resolve(),
      content=content,
    )
  with open(out_root / 'index.json', 'w', encoding='utf-8') as fh:
    json.dump(
      dict(version=VERSION, hash=sidecar_hash, directory=args.directory),
      fh,
      indent=1,
    )
  for entry in out_root.iterdir():
    if entry.is_dir() and SHA.match(entry.name) and entry.name != sidecar_hash:
      keep_for_remap(entry, out_root / 'remap' / entry.name)
      shutil.rmtree(entry)
  total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
  log('wrote', out, f'{total / 1e6:.1f} MB', f'{time.time() - t0:.0f}s')


REMAP_FILES = ('docs.json', 'vec/meta.json')
REMAP_DICT_FILES = ('features.json', 'meta.json', 'dec-int8.bin')


def keep_for_remap(old: Path, dest: Path) -> None:
  """
  What a later remap needs from a generation about to be pruned (docs/latent-embed.md §10):
  embeds in notes name its hash, rows and feature ids, so the doc table and each dictionary's
  labels and decoder stay under remap/<oldHash>/ (~3 MB) until remap files replace them.
  """
  if dest.exists():
    return
  for rel in REMAP_FILES:
    src = old / rel
    if src.is_file():
      (dest / rel).parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(src, dest / rel)
  feat = old / 'feat'
  if feat.is_dir():
    for d in feat.iterdir():
      for name in REMAP_DICT_FILES:
        src = d / name
        if d.is_dir() and src.is_file():
          (dest / 'feat' / d.name).mkdir(parents=True, exist_ok=True)
          shutil.copy2(src, dest / 'feat' / d.name / name)
  if dest.exists():
    log('kept', dest, 'for embeds that still name', old.name[:8])


if __name__ == '__main__':
  main()
