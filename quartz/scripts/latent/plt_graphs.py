# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "mlx-lm"]
# ///
"""Bake one attribution graph per note passage for /embeddings' circuit tab.

    uv run quartz/scripts/latent/plt_graphs.py --plt quartz/.quartz-cache/latent-plt/full-w8192-k64 \\
      [--limit 200] [--nodes 200] [--ctx 32] [--row 3664 --row 120]

Per passage (the same [bos] + chunk tokens plt_bake.py reads), one library forward finds the
token Gemma predicts best that is worth tracing: its top-1 guess is right, it starts a word of
three or more letters, it is not a copy of a token already in the context, and p >= --min-p;
the most confident such token wins (then any right top-1 word piece, else the row is skipped),
provided the --ctx window alone still predicts it. Its graph is attribute.build over the --ctx tokens before it, in-process, so the model and the
transcoders load once.

Output, next to the bake: quartz/.quartz-cache/latent-bake/<sidecar[:8]>/graphs/<row>.json,
attribute.py's graph plus passage = {row, start, end, target, spans, token, p}: offsets in UTF-16
units of the chunk text (the context, the traced token, and each graph position's token from 1,
for the reader's highlight), and
graphs/summary.jsonl, one line per passage (target, p, influence by kind, seconds).
Resumable: rows with a graph are skipped.

Between passages it answers the reader's live requests (plt_serve.py POST /graph), so the model
and transcoders sit in memory once: live/req-<id>.json in, live/<id>.json out (live_graph).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def log(*args: object) -> None:
  print('[graphs]', *args, flush=True)


# function words (en, fr): Gemma predicts them confidently everywhere, and their circuits are
# grammar, not the passage. They rank after every content word.
STOP = set(
  'the and for that with this from are was were has have had not but you your our their they'
  ' them its his her she him who whom which what when where why how all any can could would'
  ' should will may might must one two than then there here into onto over under about also'
  ' been being does did done just more most much many some such only very each other these'
  ' those les des une est que qui pour dans par sur pas avec sont nous vous elle elles ils'
  ' mais ses son sa leur aux cette ces être avoir fait'.split()
)


def wordlike(s: str, initial: bool) -> bool:
  w = s.strip()
  return (not initial or s[:1] == ' ') and len(w) >= 3 and w.isalpha()


def pick(model, t, ids: list[int], ctx: int, min_p: float):
  """(position of the traced token in ids, its probability in the window) or None. Candidates
  come from the whole passage; the first that the --ctx window alone still predicts top-1 wins,
  since the graph only sees the window."""
  import mlx.core as mx

  def probs(seq):
    logits = model(mx.array([seq], dtype=mx.int32))[0].astype(mx.float32)
    nxt = mx.array(seq[1:], dtype=mx.int32)
    lp = logits[:-1]
    p = mx.exp(
      mx.take_along_axis(lp, nxt[:, None], axis=-1)[:, 0]
      - mx.logsumexp(lp, axis=-1)
    )
    top = mx.argmax(lp, axis=-1)
    mx.eval(p, top)
    return np.array(p), np.array(top)

  p, top = probs(ids)
  cands = []
  for pos in range(
    17, len(ids)
  ):  # ids[pos] is predicted at pos-1; >= 16 tokens of context
    q = float(p[pos - 1])
    if top[pos - 1] != ids[pos] or q < min_p:
      continue
    if ids[pos] in ids[max(1, pos - ctx) : pos]:
      continue
    s = t.decode([ids[pos]])
    if wordlike(s, False):
      cands.append((s.strip().lower() in STOP, not wordlike(s, True), -q, pos))
  for *_, pos in sorted(cands)[:6]:
    win = [ids[0], *ids[max(1, pos - ctx) : pos + 1]]
    pw, tw = probs(win)
    if tw[-1] == ids[pos] and pw[-1] >= min_p:
      return pos, float(pw[-1])
  return None


def live_graph(
  model, lm, t, tok, tcs, mean, req: dict, *, plt, ctx: int, nodes: int
) -> dict:
  """The graph of the token after a reader selection: req = {row, text, end}, `text` the chunk's
  raw markdown and `end` the selection's end in UTF-16 units of it. The window is the --ctx tokens
  ending at the selection's last token, tokenised as the bake does."""
  from attribute import build
  from plt_bake import tokenize

  ids, spans, _ = tokenize(tok, req['text'], 1024)
  end = int(req['end'])
  # spans index tokens after bos: ids[i] is spans[i - 1]; i = the last token starting before end
  i = max((k + 1 for k, (a, _) in enumerate(spans) if a < end), default=0)
  if i < 1:
    raise ValueError('no token before the selection end')
  start = max(1, i - ctx + 1)
  g = build(
    model,
    lm,
    t,
    tcs,
    [ids[0], *ids[start : i + 1]],
    prompt=t.decode(ids[start : i + 1]),
    plt=plt,
    nodes_=nodes,
    mean=mean,
  )
  after = (
    spans[i] if i < len(spans) else None
  )  # the token the text really has next
  g['passage'] = dict(
    row=int(req['row']),
    start=int(spans[start - 1][0]),
    end=int(spans[i - 1][1]),
    target=[int(after[0]), int(after[1])]
    if after is not None
    else [int(spans[i - 1][1])] * 2,
    spans=[[int(a), int(b)] for a, b in spans[start - 1 : i]],
    token=t.decode([ids[i + 1]]) if i + 1 < len(ids) else '',
    live=True,
  )
  return g


def drain(live: Path, **kw) -> int:
  """Answer every pending live request; an error goes back as {error}."""
  n = 0
  for req_path in sorted(live.glob('req-*.json')):
    rid = req_path.stem[4:]
    try:
      body = live_graph(**kw, req=json.loads(req_path.read_text()))
    except Exception as e:  # noqa: BLE001 - the reader shows it
      body = dict(error=f'{type(e).__name__}: {e}')
    tmp = live / f'.{rid}.json'
    tmp.write_text(json.dumps(body, ensure_ascii=False))
    tmp.rename(live / f'{rid}.json')
    req_path.unlink(missing_ok=True)
    n += 1
    log(
      f'live graph {rid}',
      'error' if 'error' in body else f'→ {body["logits"][0]["token"]!r}',
    )
  return n


def spread_key(content: Path, directory: str):
  """Sort key over (row, text) that walks every document breadth-first: a passage whose rank in
  its document has more trailing zero bits comes first (0, then 64, 32, 96, ...), documents
  interleaved. 54k pdf passages in 243 pdfs would take ~180 h; this way the first few hundred
  graphs already give every pdf one circuit."""
  from features import load_side

  root = content / directory / 'latent'
  digest = json.loads((root / 'index.json').read_text())['hash']
  _, _, _, doc_of, _ = load_side(root / digest)
  doc_of = np.asarray(doc_of)
  first = {}
  for r, d in enumerate(doc_of.tolist()):
    first.setdefault(d, r)

  def key(item):
    r = item[0]
    d = int(doc_of[r])
    k = r - first[d]
    tz = 64 if k == 0 else (k & -k).bit_length() - 1
    return (-tz, d, k)

  return key


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--plt', type=Path, required=True)
  ap.add_argument('--content', type=Path, default=Path('content'))
  ap.add_argument('--directory', default='embeddings')
  ap.add_argument('--kinds', default='note')
  ap.add_argument('--out', type=Path, default=None, help='the bake folder')
  ap.add_argument(
    '--ctx', type=int, default=32, help='tokens of context before the target'
  )
  ap.add_argument(
    '--passage', type=int, default=256, help='tokens of the chunk searched'
  )
  ap.add_argument('--nodes', type=int, default=200)
  ap.add_argument('--min-p', type=float, default=0.3)
  ap.add_argument('--limit', type=int, default=0)
  ap.add_argument('--row', type=int, action='append', default=[])
  ap.add_argument(
    '--spread',
    action='store_true',
    help='cover every document before going deeper: passage 0 of each, then the middles, ...',
  )
  a = ap.parse_args()

  import mlx.core as mx
  from mlx_lm import load

  import acts
  from attribute import build, load_plt, unembed_mean
  from label import MODEL, REVISION, resolve
  from plt_bake import tokenize

  digest, rows = acts.corpus(
    a.content.resolve(), a.directory, 0, a.kinds.split(',')
  )
  out = (
    a.out or Path('quartz/.quartz-cache/latent-bake') / digest[:8]
  ) / 'graphs'
  out.mkdir(parents=True, exist_ok=True)
  if a.row:
    want = set(a.row)
    rows = [(r, x) for r, x in rows if r in want]
  todo = [(r, x) for r, x in rows if not (out / f'{r}.json').exists()]
  if a.spread:
    todo.sort(key=spread_key(a.content.resolve(), a.directory))
  if a.limit:
    todo = todo[: a.limit]
  log(
    f'{len(rows)} {a.kinds} rows in sidecar {digest[:8]}, {len(todo)} to trace'
  )

  mx.set_cache_limit(2 << 30)
  model, tok = load(resolve(MODEL, REVISION))
  lm = acts.text_model(model)
  t = getattr(tok, '_tokenizer', tok)
  tcs = load_plt(a.plt, len(lm.model.layers))
  mean = unembed_mean(lm)
  summary = (out / 'summary.jsonl').open('a')
  t0, done = time.time(), 0
  live = out.parent / 'live'
  live.mkdir(exist_ok=True)
  served = dict(
    model=model,
    lm=lm,
    t=t,
    tok=tok,
    tcs=tcs,
    mean=mean,
    plt=a.plt,
    ctx=a.ctx,
    nodes=a.nodes,
  )
  for r, text in todo:
    if drain(live, **served):
      mx.clear_cache()
    ids, spans, _ = tokenize(tok, text, a.passage)
    hit = pick(model, t, ids, a.ctx, a.min_p)
    if hit is None:
      summary.write(
        json.dumps(dict(row=r, skipped='no confident target')) + '\n'
      )
      summary.flush()
      continue
    pos, q = hit
    start = max(1, pos - a.ctx)
    g = build(
      model,
      lm,
      t,
      tcs,
      [ids[0], *ids[start:pos]],
      prompt=t.decode(ids[start:pos]),
      plt=a.plt,
      nodes_=a.nodes,
      mean=mean,
    )
    # spans index tokens after bos: ids[i] is spans[i - 1]
    g['passage'] = dict(
      row=r,
      start=int(spans[start - 1][0]),
      end=int(spans[pos - 2][1]),
      target=[int(spans[pos - 1][0]), int(spans[pos - 1][1])],
      spans=[
        [int(a), int(b)] for a, b in spans[start - 1 : pos - 1]
      ],  # graph pos 1, 2, ...
      token=t.decode([ids[pos]]),
      p=round(q, 4),
    )
    (out / f'{r}.json').write_text(json.dumps(g, ensure_ascii=False) + '\n')
    kinds: dict[str, float] = {}
    for n in g['nodes']:
      if n['kind'] != 'logit':
        kinds[n['kind']] = kinds.get(n['kind'], 0.0) + n.get('influence', 0.0)
    summary.write(
      json.dumps(
        dict(
          row=r,
          token=g['passage']['token'],
          p=g['passage']['p'],
          top=g['logits'][0]['token'],
          **{k: round(v, 3) for k, v in kinds.items()},
          check=g['check'],
          seconds=g['seconds'],
        ),
        ensure_ascii=False,
      )
      + '\n'
    )
    summary.flush()
    done += 1
    del g
    mx.clear_cache()
    log(
      f'{done}/{len(todo)} row {r} → {t.decode([ids[pos]])!r} p {q:.2f},'
      f' {(time.time() - t0) / done:.0f}s a graph'
    )
  log('done', out)


if __name__ == '__main__':
  main()
