from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from build_common import AXES, SAME_DOC, log, read_frontmatter, tokens
from generated_text import note_sources


def knn(
  V: np.ndarray, k: int, doc_of_row: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


# Pole ends. A 2,800-passage book holds every one of the furthest dozen rows of an axis it leans
# along (pc2's low end was twelve rows of the Arcades Project), and a course notebook copied into
# eleven folders is eleven documents with one title. So the exemplars walk inward from the end
# and keep one row per document and per title, as dedup.py keeps one row per text; notes get
# their own short list, since pdf passages outnumber note chunks and crowd them off the ends.
# The pole's words read a window of the furthest rows that takes at most TERM_DOC_CAP rows from
# one document.
POLE_ROWS = 12  # distinct documents per pole end
POLE_NOTES = 6  # distinct notes per pole end
TERM_ROWS = 200
TERM_DOC_CAP = 8
TITLE_KEY = re.compile(r'[^0-9a-z]+')
# 'Structure and Interpretation of Signals and Systems, Second Edition' and the first edition
EDITION = re.compile(
  r' (?:\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth) ed(?:ition)?\b.*$'
)


def title_key(title: str | None) -> str:
  return EDITION.sub('', TITLE_KEY.sub(' ', (title or '').lower()).strip())


def pole_rows(
  rows: list[dict],
  order: np.ndarray,
  doc_of_row: np.ndarray,
  limit: int,
  store: str | None = None,
) -> list[int]:
  """Rows from the pole end inward, the first of each document and each title."""
  docs: set[int] = set()
  titles: set[str] = set()
  out: list[int] = []
  for r in order:
    r = int(r)
    row = rows[r]
    if store is not None and row['store'] != store:
      continue
    d = int(doc_of_row[r])
    t = title_key(row.get('title'))
    if t and row['store'] == 'note':
      # a book's base entry and its pdf are one work; a note titled 'linear algebra' is not the book
      t = 'note\t' + t
    if d in docs or (t and t in titles):
      continue
    docs.add(d)
    if t:
      titles.add(t)
    out.append(r)
    if len(out) >= limit:
      break
  return out


def term_window(order: np.ndarray, doc_of_row: np.ndarray) -> np.ndarray:
  """The TERM_ROWS furthest rows, at most TERM_DOC_CAP of them from one document."""
  held: Counter = Counter()
  out: list[int] = []
  for r in order:
    d = int(doc_of_row[r])
    if held[d] >= TERM_DOC_CAP:
      continue
    held[d] += 1
    out.append(int(r))
    if len(out) >= TERM_ROWS:
      break
  return np.asarray(out, dtype=np.int64)


def pole_entry(rows: list[dict], doc_of_row: np.ndarray, r: int) -> dict:
  return dict(
    r=int(r),
    d=int(doc_of_row[r]),
    t=rows[int(r)]['title'],
    s=rows[int(r)]['store'],
  )


def top_terms(
  rows: list[dict], order: np.ndarray, rest: np.ndarray, n: int = 5
) -> list[str]:
  pole = Counter()
  for r in order:
    pole.update(
      tokens(rows[int(r)]['title'] + ' ' + rows[int(r)]['text'][:400])
    )
  other = Counter()
  for r in rest:
    other.update(
      tokens(rows[int(r)]['title'] + ' ' + rows[int(r)]['text'][:400])
    )
  scored = {
    w: c / (1 + other.get(w, 0) * len(order) / max(1, len(rest)))
    for w, c in pole.items()
    if c >= 3
  }
  return [w for w, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:n]]


# Frontmatter axes. The `layout: latent` page names its probe axes by their poles:
#
#   axes:
#     - name: philosophy ↔ ml
#       a: [philosophy, pattern]          # tags (a tag also matches its children, `math` → `math/linalg`)
#       b: { tags: [ml], paths: [thoughts/papers] }  # or tags and path prefixes of document ids
#
# A pole's direction is the mean of its documents' mean vectors, so a 2,800-passage book counts
# once, like a note; a document in both poles counts in neither. The normalised spec is folded
# into the sidecar hash (quartz/util/latent/sidecar.ts `axesSpec` builds the same JSON), so a
# page whose axes changed reads the sidecar as behind and a browser never mixes two axis sets.
MIN_POLE = 8


# protected (encrypted on the site, plaintext in the store), private and draft notes never enter the
# sidecar: no rows, text, neighbours, dictionaries, labels or steering. embed_build.py prunes them
# from the note store too; this keeps a stale store from leaking them back in.
EXCLUDE_KEYS = ('protected', 'private', 'draft')


def excluded_notes(content: Path) -> set[str]:
  out: set[str] = set()
  for slug, path in note_sources(content).items():
    if path.suffix != '.md':
      continue
    fm = read_frontmatter(path)
    if any(fm.get(key) for key in EXCLUDE_KEYS):
      out.add(slug)
  return out


def link_target(value: object) -> str:
  """`[[embeddings]]`, `[[thoughts/LLMs|LLMs]]`, `embeddings/` → the bare path."""
  if not isinstance(value, str):
    return ''
  inner = re.sub(r'^\[\[|\]\]$', '', value.strip())
  return inner.split('|')[0].split('#')[0].strip().strip('/')


def tag_key(tag: object) -> str:
  # Quartz's slugTag turns spaces into dashes; matching ignores case
  return re.sub(r'\s', '-', str(tag).strip().lstrip('#')).lower()


def pole_spec(value: object) -> dict:
  tags: object = []
  paths: object = []
  if isinstance(value, (str, list)):
    tags = value
  elif isinstance(value, dict):
    tags, paths = value.get('tags') or [], value.get('paths') or []
  tags = (
    [tags] if isinstance(tags, str) else tags if isinstance(tags, list) else []
  )
  paths = (
    [paths]
    if isinstance(paths, str)
    else paths
    if isinstance(paths, list)
    else []
  )
  return dict(
    paths=sorted({link_target(p) for p in paths if link_target(p)}),
    tags=sorted({tag_key(t) for t in tags if tag_key(t)}),
  )


def axes_spec(raw: object) -> list[dict]:
  out: list[dict] = []
  for ax in raw if isinstance(raw, list) else []:
    if not isinstance(ax, dict):
      continue
    a, b = pole_spec(ax.get('a')), pole_spec(ax.get('b'))
    if not (a['tags'] or a['paths']) or not (b['tags'] or b['paths']):
      log('axis without two poles skipped', ax)
      continue
    name = ax.get('name')
    out.append(dict(a=a, b=b, name=str(name).strip() if name else None))
  return out[
    : AXES - 2
  ]  # the store contrast and at least one principal direction stay


def spec_json(spec: list[dict]) -> str:
  return json.dumps(
    spec, ensure_ascii=False, separators=(',', ':'), sort_keys=True
  )


def latent_page(content: Path, directory: str) -> tuple[Path | None, dict]:
  """The `layout: latent` page reading this directory; its folder index wins a tie."""
  want = directory.strip('/')
  hits: list[tuple[Path, dict]] = []
  for path in sorted(content.rglob('*.md')):
    if 'latent' in path.relative_to(content).parts[:-1]:
      continue
    try:
      with open(path, encoding='utf-8', errors='replace') as fh:
        head = fh.read(2048)
    except OSError:
      continue
    if not head.startswith('---') or not re.search(
      r'^layout:\s*[\'"]?latent\b', head, re.M
    ):
      continue
    fm = read_frontmatter(path)
    if (link_target(fm.get('directory')) or 'embeddings') == want:
      hits.append((path, fm))
  if not hits:
    return None, {}
  hits.sort(key=lambda h: h[0] != content / want / 'index.md')
  if len(hits) > 1:
    log(
      'several latent pages read',
      want,
      '; axes from',
      hits[0][0].relative_to(content),
    )
  return hits[0]


def pole_members(
  docs: list[dict], spec: dict, tags_of: dict[str, set[str]]
) -> set[int]:
  out: set[int] = set()
  for di, d in enumerate(docs):
    doc_id = d['id']
    if any(doc_id == p or doc_id.startswith(p + '/') for p in spec['paths']):
      out.add(di)
      continue
    have = tags_of.get(doc_id)
    if have and any(
      t == want or t.startswith(want + '/')
      for t in have
      for want in spec['tags']
    ):
      out.add(di)
  return out


def held_out_auc(doc_vecs: np.ndarray, a: list[int], b: list[int]) -> float:
  """Direction from every other document of each pole, AUC on the rest: does the axis generalise."""
  fa, ta, fb, tb = a[::2], a[1::2], b[::2], b[1::2]
  w = doc_vecs[fb].mean(0) - doc_vecs[fa].mean(0)
  pa, pb = doc_vecs[ta] @ w, doc_vecs[tb] @ w
  return float(
    (pb[:, None] > pa[None, :]).mean()
    + 0.5 * (pb[:, None] == pa[None, :]).mean()
  )
