"""Canonical rows: each passage text is counted once by the kNN graph, the axes, the sparse
dictionaries and the labels.

The garden repeats text. A course notebook is copied into each lab and assignment that extends it
(the P0 compiler sits in eleven folders), OCR writes the same `$$` or "this page intentionally
left blank" thousands of times, and a pdf can sit under two slugs. Every copy is a row with a
vector of its own, so a copied passage pulls a sparse feature, a neighbour list and a label
towards itself as many times as it was copied.

`canonical_rows` maps each row to the row that holds its text first: itself for a unique row.
Rows are visited notes first, then by document id and chunk, so the copy in the lecture folder
(`05 Construction of a Parser`) is canonical over the ones in `88 Assignment 7`. A row is a copy
when its text, squashed, is the same as an earlier row's, or when its vector lies within COS of an
earlier row and their word shingles overlap by JACCARD or more (a notebook cell edited in a few
lines, a chunk cut a sentence later). Rows without text (images, base entries) are their own
canonical. The sidecar keeps every row, so a copied page still opens and reads in full; build.py
writes the map as canon.u32 and the consumers skip a row whose canonical is another row.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

COS = 0.95  # vector cosine for a near copy candidate
JACCARD = 0.8  # 5-word shingle overlap that confirms it
SHINGLE = 5
CANDIDATES = 8  # nearest earlier rows checked per row
TAG = re.compile(r'<[^<>]+>')
SPACE = re.compile(r'\s+')


def squash(text: str) -> str:
  return SPACE.sub(' ', TAG.sub(' ', text)).strip().lower()


def shingles(words: list[str]) -> set[int]:
  return {
    hash(' '.join(words[i : i + SHINGLE]))
    for i in range(len(words) - SHINGLE + 1)
  }


def canonical_rows(rows: list[dict], V: np.ndarray) -> np.ndarray:
  """canon[r] is the first row in visit order with r's text; canon[r] == r for a unique row."""
  N = len(rows)
  canon = np.arange(N, dtype=np.uint32)
  text_rows = [
    i
    for i, r in enumerate(rows)
    if r['store'] in ('note', 'pdf') and (r.get('raw') or r['text'])
  ]
  text_rows.sort(
    key=lambda i: (
      rows[i]['store'] != 'note',
      rows[i]['doc'],
      rows[i]['chunk'],
      rows[i].get('page') or 0,
      i,
    )
  )
  squashed = [squash(rows[i].get('raw') or rows[i]['text']) for i in text_rows]
  first: dict[str, int] = {}
  sets: list[set[int] | None] = []
  for pos, (i, t) in enumerate(zip(text_rows, squashed)):
    key = hashlib.sha1(t.encode()).hexdigest()
    if not t:
      sets.append(None)
      continue
    if key in first:
      canon[i] = canon[text_rows[first[key]]]
    else:
      first[key] = pos
    words = t.split()
    sets.append(shingles(words) if len(words) >= 2 * SHINGLE else None)

  # near copies: earlier rows in visit order whose vectors and shingles both agree
  order = np.asarray(text_rows, dtype=np.int64)
  U = V[order]
  n = len(order)
  block = 2048
  wide = min(CANDIDATES, n - 1)
  for a in range(0, n, block):
    S = U[a : a + block] @ U.T
    for j in range(S.shape[0]):
      S[j, a + j :] = -1.0  # only an earlier row can be canonical
    cand = np.argpartition(-S, wide, axis=1)[:, :wide]
    cs = np.take_along_axis(S, cand, axis=1)
    for j in range(S.shape[0]):
      pos = a + j
      i = int(order[pos])
      mine = sets[pos]
      if canon[i] != i or mine is None:
        continue
      for c in cand[j][np.argsort(-cs[j])]:
        if S[j, c] < COS:
          break
        theirs = sets[int(c)]
        if theirs and len(mine & theirs) / len(mine | theirs) >= JACCARD:
          canon[i] = canon[int(order[c])]
          break
  # an exact group's first row can itself turn out a near copy later in the pass
  while True:
    up = canon[canon]
    if np.array_equal(up, canon):
      return canon
    canon = up


def summary(rows: list[dict], canon: np.ndarray) -> dict:
  """Counts for the build log and index.json: copied rows by store, documents with no row of
  their own."""
  N = len(rows)
  copied = canon != np.arange(N)
  by_store: dict[str, int] = {}
  whole: dict[str, bool] = {}
  for i, r in enumerate(rows):
    if copied[i]:
      by_store[r['store']] = by_store.get(r['store'], 0) + 1
    whole[r['doc']] = whole.get(r['doc'], True) and bool(copied[i])
  return dict(
    rows=int(copied.sum()),
    stores=by_store,
    docs=sum(whole.values()),
    cos=COS,
    jaccard=JACCARD,
  )
