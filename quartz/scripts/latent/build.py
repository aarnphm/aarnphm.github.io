# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Build the `layout: latent` sidecar for every embedding store under a directory.

    uv run quartz/scripts/latent/build.py [--directory content/embeddings] [--k 16]

Discovery walks the directory for manifest.json files and classifies each one with the same
checks the browser readers use (quartz/util/pdf-search.ts, image-search.ts, semantic.worker.ts):
`kind: quartz-pdf-search-v1` is a pdf store, `modality: image` an image store, a v2 manifest with
`chunkMetadata` the notes store, a v2 manifest with `encoder` a base store. A root pointer and its
identical `<generation>/manifest.json` are one store, rooted at the pointer. A store with another
model or dimensionality is listed and excluded, never a second space. The sidecar is pinned to a
hash over every included manifest, so a regenerated store fails closed until this runs again.

Output: <directory>/latent/<hash>/ (see quartz/util/latent/sidecar.ts for the file contract) and
<directory>/latent/index.json pointing at it. Older hashes are pruned.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import shutil
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

SHA = re.compile(r'^[a-f0-9]{64}$')
VERSION = 1
BLOCK = 64
PREVIEW = 200
PART = 48  # chunks per text part; a 2,800-passage book must not be one 5 MB file
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
      raise SystemExit(f'shard {path} holds {data.size} floats, expected {shard["rows"] * dims}')
    out[shard['rowOffset'] : shard['rowOffset'] + shard['rows']] = data.reshape(shard['rows'], dims)
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
        generation=manifest.get('generation') or manifest.get('passages', {}).get('generation'),
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


def clean_title(title: str | None, slug: str) -> str:
  """A pdf store title can be a stray code fence from OCR; fall back to the file name."""
  t = re.sub(r'`+|\bmarkdown\b', '', title or '').strip()
  t = re.sub(r'\s+', ' ', t)
  if len(t) < 3:
    stem = os.path.splitext(os.path.basename(slug))[0]
    t = re.sub(r'[-_.]+', ' ', stem).strip() or slug
  return t


def tokens(text: str) -> set[str]:
  return {w for w in WORD.findall(text.lower()) if w not in STOP}


def resolve_box(regions: dict | None, page: int, text: str) -> list[float] | None:
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


# Note text reaches the embeddings as rendered html's text: KaTeX keeps its TeX source and drops the
# dollars. The reader renders `$…$` / `$$…$$`, so the dollars come back from the note's markdown:
# its math spans, in order, are found again in the document's text (rebuilt from charStart) with
# a forward cursor, and wrapped where a chunk holds the whole span.
FRONTMATTER = re.compile(r'\A---\n.*?\n---\n', re.S)
MD_FENCE = re.compile(r'^[ \t>]*(```|~~~)[^\n]*\n.*?^[ \t>]*\1[ \t]*$', re.M | re.S)
MD_CODE = re.compile(r'`[^`\n]+`')
MD_QUOTE = re.compile(r'^[ \t]*(?:>[ \t]?)+', re.M)
MD_DISPLAY = re.compile(r'\$\$([\s\S]+?)\$\$')
# pandoc's rule, as in quartz/util/markdown-renderer.ts: "$5 and $10" is not math
MD_INLINE = re.compile(r'\$(?![\s$])((?:\\.|[^\\$\n])+?)(?<!\s)\$(?!\d)')
MATH_WINDOW = 1500  # how far past the cursor a span may sit before it counts as dropped


def math_spans(md: str) -> list[tuple[str, bool]]:
  """(tex, display) for each math span of a note's markdown, in source order."""
  md = MD_QUOTE.sub('', MD_FENCE.sub('', FRONTMATTER.sub('', md)))
  spans: list[tuple[str, bool]] = []
  pos = 0
  for m in MD_DISPLAY.finditer(md):
    spans += [(t, False) for t in MD_INLINE.findall(MD_CODE.sub('', md[pos : m.start()]))]
    spans.append((m.group(1).strip(), True))
    pos = m.end()
  spans += [(t, False) for t in MD_INLINE.findall(MD_CODE.sub('', md[pos:]))]
  return [(t, d) for t, d in spans if t.strip()]


def tex_pattern(tex: str) -> re.Pattern[str]:
  # the passage text is html-escaped; whitespace may differ in runs
  body = r'\s+'.join(re.escape(html.escape(part, quote=False)) for part in tex.split())
  pre = r'(?<![\w\\])' if re.match(r'\w', tex) else ''
  post = r'(?!\w)' if re.search(r'\w$', tex) else ''
  return re.compile(pre + body + post)


def restore_math(chunks: list[dict], md: str) -> int:
  """Put the dollars back into one note's chunk texts; returns the spans restored."""
  placed = [(r['start'], r['text']) for r in chunks if r.get('start') is not None and r['text']]
  if not placed:
    return 0
  buf = [' '] * max(s + len(t) for s, t in placed)
  for s, t in placed:
    buf[s : s + len(t)] = t
  full = ''.join(buf)
  wraps: list[tuple[int, int, bool]] = []
  cursor = 0
  for tex, display in math_spans(md):
    pattern = tex_pattern(tex)
    m = pattern.search(full, cursor, min(len(full), cursor + MATH_WINDOW + 2 * len(tex)))
    # a long, distinctive span may sit past a dropped stretch (a callout the text skipped)
    if not m and len(tex) >= 12 and '\\' in tex:
      m = pattern.search(full, cursor)
    if not m:
      continue
    wraps.append((m.start(), m.end(), display))
    cursor = m.end()
  for r in chunks:
    s, t = r.get('start'), r['text']
    if s is None or not t:
      continue
    for a, b, display in reversed([w for w in wraps if w[0] >= s and w[1] <= s + len(t)]):
      fence = '$$' if display else '$'
      t = t[: a - s] + fence + t[a - s : b - s] + fence + t[b - s :]
    r['text'] = t
  return len(wraps)


def note_sources(content: Path) -> dict[str, Path]:
  """slug → markdown file, for the slugs Quartz derives from paths (spaces become dashes)."""
  out: dict[str, Path] = {}
  for path in content.rglob('*.md'):
    rel = path.relative_to(content).with_suffix('').as_posix()
    out.setdefault(rel.replace(' ', '-'), path)
  return out


def load_rows(content: Path, store: dict, offset: int) -> tuple[list[dict], np.ndarray]:
  """Rows of one store in vector order: id, doc, title, chunk, page, label, text, date."""
  m = store['manifest']
  kind = store['kind']
  dims = m['dims']
  vectors = read_vectors(content, m['vectors'], dims)
  rows: list[dict] = []
  if kind == 'note':
    passages = read_shard_json(content, m['passages']['shards']) if 'passages' in m else []
    if passages and len(passages) != m['rows']:
      raise SystemExit('note passages do not align with the vectors')
    for i, rid in enumerate(m['ids']):
      meta = m['chunkMetadata'].get(rid, {})
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
          date=p.get('date'),
          prec=p.get('datePrecision'),
          start=p.get('charStart'),
        )
      )
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
      by_doc.setdefault(r['doc'], []).append(r)
    sources = note_sources(content)
    restored = 0
    for doc, chunks in by_doc.items():
      source = sources.get(doc)
      if source is not None:
        restored += restore_math(chunks, source.read_text(encoding='utf-8', errors='replace'))
    log('note math spans restored', restored)
  elif kind == 'pdf':
    passages = read_shard_json(content, m['passages']['shards'])
    if len(passages) != m['rows']:
      raise SystemExit('pdf passages do not align with the vectors')
    docs = m['documents']
    for p in passages:
      d = docs.get(p['slug'], {})
      rows.append(
        dict(
          id=f'{p["slug"]}#page={p["page"]}&chunk={p["chunk"]}',
          doc=p['slug'],
          title=clean_title(d.get('title') or p.get('title'), p['slug']),
          chunk=p['chunk'],
          page=p['page'],
          label=p.get('pageLabel') or '',
          text=p['text'],
          date=d.get('date') or p.get('date'),
          prec=d.get('datePrecision'),
          sha=d.get('sourceSha256'),
          pages=d.get('totalPages'),
        )
      )
  elif kind == 'image':
    for rid in m['ids']:
      rows.append(
        dict(id=rid, doc=rid, title=os.path.basename(rid), chunk=0, page=None, label='', text='', date=None, prec=None)
      )
  elif kind == 'base':
    titles = m.get('titles') or m['ids']
    for i, rid in enumerate(m['ids']):
      rows.append(dict(id=rid, doc=rid, title=titles[i], chunk=0, page=None, label='', text='', date=None, prec=None))
  for i, r in enumerate(rows):
    r['row'] = offset + i
    r['store'] = kind
    r['storeId'] = store['id']
  return rows, vectors


def knn(V: np.ndarray, k: int, doc_of_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Exact top-k by cosine, with at most SAME_DOC neighbours from the row's own document: a
  dictionary's entries otherwise fill every slot with the dictionary and the links pane never
  leaves the book. Candidates come from the top 8k; when fewer cross-document rows are there the
  slots fall back to same-document ones, so k is always met."""
  n = V.shape[0]
  idx = np.empty((n, k), dtype=np.uint32)
  sc = np.empty((n, k), dtype=np.float32)
  block = 1024
  wide = min(8 * k, n - 1)
  for a in range(0, n, block):
    S = V[a : a + block] @ V.T
    for j in range(S.shape[0]):
      S[j, a + j] = -2.0
    part = np.argpartition(-S, wide, axis=1)[:, :wide]
    scores = np.take_along_axis(S, part, axis=1)
    order = np.argsort(-scores, axis=1)
    cand = np.take_along_axis(part, order, axis=1)
    cand_sc = np.take_along_axis(scores, order, axis=1)
    same = doc_of_row[cand] == doc_of_row[a : a + block][:, None]
    for j in range(cand.shape[0]):
      keep: list[int] = []
      spare: list[int] = []
      own = 0
      for c in range(wide):
        if same[j, c]:
          if own < SAME_DOC:
            own += 1
            keep.append(c)
          else:
            spare.append(c)
        else:
          keep.append(c)
        if len(keep) == k:
          break
      if len(keep) < k:
        keep = sorted(keep + spare[: k - len(keep)])
      idx[a + j] = cand[j, keep]
      sc[a + j] = cand_sc[j, keep]
  return idx, sc


def bimodality(values: np.ndarray) -> float:
  """Sarle's bimodality coefficient: > 0.555 suggests two modes."""
  n = values.size
  if n < 4:
    return 0.0
  m = values.mean()
  s = values.std()
  if s == 0:
    return 0.0
  z = (values - m) / s
  g = float((z**3).mean())
  k = float((z**4).mean()) - 3.0
  return (g * g + 1.0) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def top_terms(rows: list[dict], order: np.ndarray, rest: np.ndarray, n: int = 5) -> list[str]:
  pole = Counter()
  for r in order:
    pole.update(tokens(rows[int(r)]['title'] + ' ' + rows[int(r)]['text'][:400]))
  other = Counter()
  for r in rest:
    other.update(tokens(rows[int(r)]['title'] + ' ' + rows[int(r)]['text'][:400]))
  scored = {w: c / (1 + other.get(w, 0) * len(order) / max(1, len(rest))) for w, c in pole.items() if c >= 3}
  return [w for w, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:n]]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--content', default='content')
  ap.add_argument('--directory', default='embeddings', help='relative to --content')
  ap.add_argument('--k', type=int, default=16)
  ap.add_argument('--regions', default='embeddings/pdf-regions', help='relative to --content')
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
    s['included'] = s['kind'] is not None and s['model'] == model and s['dims'] == dims
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
  sidecar_hash = sha256('\n'.join(digest_lines).encode())
  log('stores', [(s['id'], s['kind'], s['rows'], s['included']) for s in stores])
  log('hash', sidecar_hash)

  out_root = directory / 'latent'
  out = out_root / sidecar_hash
  if out.exists():
    shutil.rmtree(out)
  for sub in ('knn', 'prev', 'text', 'vec'):
    (out / sub).mkdir(parents=True)

  rows: list[dict] = []
  mats: list[np.ndarray] = []
  for s in included:
    s['rowOffset'] = len(rows)
    r, V = load_rows(content, s, len(rows))
    rows.extend(r)
    mats.append(V)
    log(s['id'], len(r), 'rows')
  V = np.vstack(mats)
  del mats
  N = V.shape[0]

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
          date=r['date'],
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
    range(len(docs)), key=lambda i: (docs[i]['day'] == 0, docs[i]['day'], docs[i]['store'], docs[i]['id'])
  )
  docs = [docs[i] for i in order]
  doc_of_row = np.empty(N, dtype=np.uint32)
  for di, d in enumerate(docs):
    for r in d['rows']:
      doc_of_row[r] = di
    d['rows'].sort()
  log('docs', len(docs))

  idx, sc = knn(V, args.k, doc_of_row)
  degree = np.minimum(np.bincount(idx.ravel(), minlength=N), 0xFFFF).astype(np.uint16)
  log('knn', f'{time.time() - t0:.0f}s')

  # regions for pdf page silhouettes
  regions_dir = content / args.regions
  regions_cache: dict[str, dict | None] = {}

  def regions_for(sha: str | None) -> dict | None:
    if not sha:
      return None
    if sha not in regions_cache:
      path = regions_dir / f'{sha}.json'
      regions_cache[sha] = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
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
  mean = V256.mean(0)
  centred = V256 - mean
  cov = centred.T @ centred / N
  evals, evecs = np.linalg.eigh(cov)
  pcs = evecs[:, ::-1][:, :AXES].T.astype(np.float32)  # AXES x 256
  explained = (evals[::-1][:AXES] / evals.sum()).astype(np.float32)
  scale = float(np.abs(V256).max())
  q = np.clip(np.round(V256 / scale * 127), -127, 127).astype(np.int8)
  q.tofile(out / 'vec' / 'rows-256-int8.bin')
  doc_vecs = np.stack([V256[d['rows']].mean(0) for d in docs])
  doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9
  np.clip(np.round(doc_vecs / scale * 127), -127, 127).astype(np.int8).tofile(out / 'vec' / 'docs-256-int8.bin')
  log('vectors', f'{time.time() - t0:.0f}s')

  # axes: store contrasts first, then principal directions
  store_arr = np.array([r['store'] for r in rows])
  axes: list[dict] = []

  def add_axis(aid: str, name: str, direction: np.ndarray, a: str, b: str) -> None:
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    proj = centred @ direction
    axes.append(dict(id=aid, name=name, a=a, b=b, dir=direction.astype(np.float32), proj=proj))

  kinds = [k for k in ('note', 'pdf', 'image', 'base') if (store_arr == k).any()]
  if 'note' in kinds and 'pdf' in kinds:
    add_axis(
      'store',
      'notes ↔ passages',
      V256[store_arr == 'pdf'].mean(0) - V256[store_arr == 'note'].mean(0),
      'notes',
      'pdf passages',
    )
  fr = np.array([r['doc'].startswith('fr/') for r in rows])
  if fr.sum() >= 20 and (~fr & (store_arr == 'note')).sum() >= 20:
    add_axis(
      'lang',
      'français ↔ english',
      V256[~fr & (store_arr == 'note')].mean(0) - V256[fr].mean(0),
      'fr/',
      'english notes',
    )
  for i in range(AXES):
    if len(axes) >= AXES:
      break
    add_axis(f'pc{i + 1}', f'pc{i + 1} · {explained[i] * 100:.1f} %', pcs[i], '−', '+')
  axes = axes[:AXES]
  rest_sample = np.random.default_rng(0).choice(N, size=min(N, 2000), replace=False)
  probe = []
  axis_i8 = np.zeros((N, AXES), dtype=np.int8)
  for ai, ax in enumerate(axes):
    proj = ax['proj']
    lim = float(np.abs(proj).max()) + 1e-9
    axis_i8[:, ai] = np.clip(np.round(proj / lim * 127), -127, 127)
    order = np.argsort(proj)
    neg, pos = order[:12], order[-12:][::-1]
    hist, _ = np.histogram(proj, bins=32, range=(-lim, lim))
    probe.append(
      dict(
        id=ax['id'],
        name=ax['name'],
        a=ax['a'],
        b=ax['b'],
        bc=round(bimodality(proj), 3),
        scale=lim,
        hist=hist.astype(int).tolist(),
        terms=dict(a=top_terms(rows, order[:200], rest_sample), b=top_terms(rows, order[-200:], rest_sample)),
        poles=dict(
          a=[dict(r=int(r), d=int(doc_of_row[r]), t=rows[int(r)]['title'], s=rows[int(r)]['store']) for r in neg],
          b=[dict(r=int(r), d=int(doc_of_row[r]), t=rows[int(r)]['title'], s=rows[int(r)]['store']) for r in pos],
        ),
      )
    )
  with open(out / 'vec' / 'meta.json', 'w', encoding='utf-8') as fh:
    json.dump(
      dict(
        rows=N,
        docs=len(docs),
        dims=256,
        dtype='int8',
        scale=scale,
        mean=[round(float(v), 6) for v in mean],
        axes=[dict(id=a['id'], name=a['name'], dir=[round(float(v), 6) for v in a['dir']]) for a in axes],
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
        x=r['text'][:PREVIEW],
      )
      for r in rows[b : b + BLOCK]
    ]
    with open(out / 'prev' / f'{b // BLOCK:05d}.json', 'w', encoding='utf-8') as fh:
      json.dump(block, fh, ensure_ascii=False, separators=(',', ':'))
  for di, d in enumerate(docs):
    chunks = []
    for r in d['rows']:
      row = rows[r]
      chunk = dict(r=r, c=row['chunk'], text=row['text'])
      if row['page']:
        chunk['p'] = row['page']
        chunk['l'] = row['label']
      if row.get('box'):
        chunk['box'] = row['box']
      chunks.append(chunk)
    payload = dict(
      d=di,
      id=d['id'],
      store=d['store'],
      title=d['title'],
      date=d['date'],
      prec=d['prec'],
      total=len(chunks),
      part=PART,
      chunks=chunks[:PART],
    )
    if d.get('sha'):
      payload['sha'] = d['sha']
      payload['pages'] = d.get('pages')
    with open(out / 'text' / f'{di}.json', 'w', encoding='utf-8') as fh:
      json.dump(payload, fh, ensure_ascii=False, separators=(',', ':'))
    for p in range(1, (len(chunks) + PART - 1) // PART):
      with open(out / 'text' / f'{di}-{p:03d}.json', 'w', encoding='utf-8') as fh:
        json.dump(
          dict(d=di, part=p, chunks=chunks[p * PART : (p + 1) * PART]), fh, ensure_ascii=False, separators=(',', ':')
        )

  # documents table and the time track
  docs_out = []
  track = bytearray()
  for di, d in enumerate(docs):
    first, count = d['rows'][0], len(d['rows'])
    docs_out.append(
      dict(
        d=di,
        id=d['id'],
        s=d['store'],
        t=d['title'],
        date=d['date'],
        prec=d['prec'],
        r0=first,
        n=count,
        **({'sha': d['sha'], 'pages': d['pages']} if d.get('sha') else {}),
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
  degree.tofile(out / 'degree.u16')
  doc_of_row.tofile(out / 'rowdoc.u32')

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
    counts=dict(counts),
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
  with open(out_root / 'index.json', 'w', encoding='utf-8') as fh:
    json.dump(dict(version=VERSION, hash=sidecar_hash, directory=args.directory), fh, indent=1)
  for entry in out_root.iterdir():
    if entry.is_dir() and SHA.match(entry.name) and entry.name != sidecar_hash:
      shutil.rmtree(entry)
  total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
  log('wrote', out, f'{total / 1e6:.1f} MB', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
  main()
