"""Label the transcoder features that appear in attribution graphs, with the same local Gemma.

    .venv/bin/python quartz/scripts/latent/plt_label.py --plt quartz/.quartz-cache/latent-plt/full-w8192-k64

1. The features to label are every (layer, feature) node in the graphs under --graphs.
2. Their strongest contexts come from streaming garden tokens (notes + pdfs, --tokens of them)
   through Gemma 4 and the transcoders. Each feature keeps its top activations, at most two
   per row so one passage cannot fill the list.
3. Gemma is shown each feature's top snippets, with the firing token in «», plus the tokens its
   decoder promotes when read straight through the unembedding. It answers {label, description}.
4. Each label is scored by detection (Paulo et al. 2024), as label.py does: four held-out firing
   snippets and four random garden tokens, balanced accuracy.

Labels go to <plt>/labels.json as {"L<ll>": {"<f>": {label, description, score, n, top, promotes,
model, at}}}, and are copied onto the graph nodes as `label` (graph files are rewritten in place).
A rerun skips features already labelled unless --force.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ASK = """You are naming one feature of a transcoder (a sparse dictionary over one MLP layer) of a \
small language model, Gemma 4, layer {layer} of 42. It was read on a personal research garden: \
notes on machine learning, philosophy, mathematics and literature, plus the papers and books \
they cite.

Below are the snippets where the feature fires most strongly, strongest first. The token it \
fires on is marked «like this»; the text before it is the context the model had.

{rows}
{promotes}
Name what makes it fire on the marked token: a word or token type, a topic, a syntactic role, \
a position in a name or formula, or what it predicts next. Be as precise as the snippets allow.

Reply with JSON only: {{"label": "<2 to 6 words, lowercase except proper nouns>", \
"description": "<at most 25 words: when the feature fires and what it seems to do>"}}"""

SCORE = """A feature of a language model is described as:
label: {label}
description: {description}

For each numbered snippet below, answer 1 if the feature would fire on the token marked «», \
else 0.

{rows}

Reply with JSON only: a list of {n} numbers, 0 or 1, in order."""

TOP = 24  # activations kept per feature (shown + held out)
SHOWN = 12
HELD = 4
PER_ROW = 2
BEFORE, AFTER = 28, 4
LATE = 30  # layers whose decoder rows are shown to Gemma as promoted tokens  # tokens of context around the firing token


def log(*args: object) -> None:
  print('[plt-label]', *args, flush=True)


def graph_features(graphs: list[Path], top: int = 0) -> dict[int, list[int]]:
  """(layer, feature) nodes of the graphs. With top, only the features the circuit tab names:
  each graph's 10 most influential, plus the `top` with the most influence summed over all
  graphs (5,800 passage graphs hold ~100k features, 2.5 s a label)."""
  infl: dict[tuple[int, int], float] = {}
  named: set[tuple[int, int]] = set()
  for g in graphs:
    feats = [n for n in json.loads(g.read_text())['nodes'] if n['kind'] == 'feature']
    for n in feats:
      k = (int(n['layer']), int(n['feature']))
      infl[k] = infl.get(k, 0.0) + float(n.get('influence', 0.0))
    feats.sort(key=lambda n: -float(n.get('influence', 0.0)))
    named.update((int(n['layer']), int(n['feature'])) for n in feats[:10])
  keys = set(infl) if not top else named | set(sorted(infl, key=lambda k: -infl[k])[:top])
  want: dict[int, set[int]] = {}
  for l, f in keys:
    want.setdefault(l, set()).add(f)
  return {l: sorted(fs) for l, fs in sorted(want.items())}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--plt', required=True)
  ap.add_argument('--graphs', default='quartz/.quartz-cache/latent-graphs')
  ap.add_argument('--content', default='content')
  ap.add_argument('--tokens', type=float, default=4e5)
  ap.add_argument('--kinds', default='note,pdf')
  ap.add_argument('--ctx', type=int, default=1024)
  ap.add_argument('--batch-tokens', type=int, default=8192)
  ap.add_argument('--limit', type=int, default=0)
  ap.add_argument('--force', action='store_true')
  ap.add_argument('--no-score', action='store_true')
  ap.add_argument(
    '--top', type=int, default=0, help='only each graph\'s top 10 + the N most influential'
  )
  args = ap.parse_args()

  import mlx.core as mx
  from mlx_lm import load

  import acts
  from label import MODEL, REVISION, Gemma, first_json, resolve, squash
  from transcoders import Transcoder

  mx.set_cache_limit(4 << 30)
  plt = Path(args.plt)
  graphs = sorted(Path(args.graphs).rglob('*.json'))
  graphs = [g for g in graphs if g.name != 'summary.json']
  want = graph_features(graphs, args.top)
  out_path = plt / 'labels.json'
  store = json.loads(out_path.read_text()) if out_path.exists() else {}
  if not args.force:
    want = {
      l: [f for f in fs if str(f) not in store.get(f'L{l:02d}', {})]
      for l, fs in want.items()
    }
    want = {l: fs for l, fs in want.items() if fs}
  total = sum(len(v) for v in want.values())
  log(f'{total} features over {len(want)} layers from {len(graphs)} graphs')
  if not total:
    return

  path = resolve(MODEL, REVISION)
  model, tok = load(path)
  lm = acts.text_model(model)
  m = lm.model
  tcs = {l: Transcoder.load(plt / f'L{l:02d}', 0) for l in want}
  sel = {l: mx.array(np.array(fs, np.int32)) for l, fs in want.items()}

  # 2. top activations over a stream of garden tokens
  _, rows = acts.corpus(
    Path(args.content).resolve(), 'embeddings', 11, args.kinds.split(',')
  )
  text_of = dict(rows)
  best = {
    l: (
      np.full((len(fs), TOP * 4), -1.0, np.float32),
      np.zeros((len(fs), TOP * 4, 2), np.int64),
    )
    for l, fs in want.items()
  }
  fires = {l: np.zeros(len(fs), np.int64) for l, fs in want.items()}
  rand: list[tuple[int, int]] = []
  rng = np.random.default_rng(0)
  seen, t0 = 0, time.time()
  taps = {l: ('in',) for l in want}
  for got, toks, rw, pos, _ in acts.batches(
    lm, tok, rows, taps, args.ctx, args.batch_tokens, 1.0, 0
  ):
    n = len(toks)
    for l, tc in tcs.items():
      x = got[l, 'in'].astype(mx.float32) / tc.sx - tc.b_pre
      _, z = tc.code(tc.p, x)
      a = np.array(z[:, sel[l]]).T  # (features, n)
      fires[l] += (a > 0).sum(1)
      vals, where = best[l]
      cat_v = np.concatenate([vals, a], 1)
      cat_w = np.concatenate(
        [
          where,
          np.broadcast_to(np.stack([rw, pos], 1)[None], (a.shape[0], n, 2)),
        ],
        1,
      )
      keep = np.argpartition(-cat_v, TOP * 4 - 1, axis=1)[:, : TOP * 4]
      best[l] = (
        np.take_along_axis(cat_v, keep, 1),
        np.take_along_axis(cat_w, keep[:, :, None], 1),
      )
    rand.extend(
      (int(rw[j]), int(pos[j]))
      for j in rng.choice(n, size=min(4, n), replace=False)
    )
    seen += n
    if seen >= args.tokens:
      break
  log(f'{seen} tokens in {time.time() - t0:.0f}s')

  cache: dict[int, list[int]] = {}

  def ids_of(r: int) -> list[int]:
    if r not in cache:
      cache[r] = acts.encode(tok, text_of[r], args.ctx)
    return cache[r]

  def snippet(r: int, p: int) -> str:
    ids = ids_of(r)
    pre = tok.decode(ids[max(1, p - BEFORE) : p])
    hit = tok.decode([ids[p]])
    post = tok.decode(ids[p + 1 : p + 1 + AFTER])
    return squash(f'…{pre}«{hit}»{post}')

  norm_w = m.norm.weight.astype(mx.float32)

  def promotes(tc, f: int) -> list[str]:
    v = (tc.p['W_dec'][f] * norm_w)[None, None]
    lg = m.embed_tokens.as_linear(v.astype(mx.bfloat16)).astype(mx.float32)[
      0, 0
    ]
    top = np.argsort(-np.array(lg))[:10]
    return [tok.decode([int(t)]) for t in top]

  # 3 + 4. label and score
  gemma = Gemma(path)
  gemma.lm = (model, tok)
  jobs = []
  for l, fs in want.items():
    vals, where = best[l]
    for i, f in enumerate(fs):
      order = np.argsort(-vals[i])
      picked, per = [], {}
      for j in order:
        if vals[i, j] <= 0:
          break
        r, p = (int(x) for x in where[i, j])
        if per.get(r, 0) >= PER_ROW:
          continue
        per[r] = per.get(r, 0) + 1
        picked.append((r, p, float(vals[i, j])))
        if len(picked) >= TOP:
          break
      if len(picked) < 3:
        log(f'L{l:02d} f{f}: fires {len(picked)}x in the sample, skipped')
        continue
      jobs.append((l, f, picked, int(fires[l][i])))
  if args.limit:
    jobs = jobs[: args.limit]
  log(f'{len(jobs)} features to label')
  t1 = time.time()
  for a in range(0, len(jobs), 16):
    chunk = jobs[a : a + 16]
    prompts, extra = [], []
    for l, f, picked, nf in chunk:
      tc = tcs[l]
      prom = promotes(tc, f)
      rows_txt = '\n'.join(
        f'{j + 1}. {snippet(r, p)}'
        for j, (r, p, _) in enumerate(picked[:SHOWN])
      )
      prompts.append(
        ASK.format(
          layer=l,
          rows=rows_txt,
          # read straight through the unembedding, a decoder direction only means something near
          # the top of the stack; earlier layers get no hint rather than a misleading one
          promotes=(
            '\nWhen the feature fires it pushes the model toward these next tokens: '
            + ', '.join(repr(t) for t in prom)
            + '\n'
            if l >= LATE
            else ''
          ),
        )
      )
      extra.append(prom)
    raws = gemma.text(prompts, max_tokens=110)
    asks, mixes = [], []
    for (l, f, picked, nf), prom, raw in zip(chunk, extra, raws):
      got = first_json(raw)
      if not isinstance(got, dict) or not str(got.get('label') or '').strip():
        log(f'L{l:02d} f{f}: unparsed {raw[:100]!r}')
        continue
      rec = dict(
        label=squash(str(got['label']))[:60],
        description=squash(str(got.get('description') or ''))[:240],
        n=nf,
        top=[[r, p, round(v, 3)] for r, p, v in picked[:SHOWN]],
        promotes=prom,
        model=MODEL,
        at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
      )
      store.setdefault(f'L{l:02d}', {})[str(f)] = rec
      held = picked[SHOWN : SHOWN + HELD]
      if args.no_score or len(held) < 2:
        continue
      hit_rows = {r for r, _, _ in picked}
      negs = [rp for rp in rand if rp[0] not in hit_rows]
      neg = [
        negs[int(k)]
        for k in rng.choice(len(negs), size=len(held), replace=False)
      ]
      mix = [((r, p), 1) for r, p, _ in held] + [(rp, 0) for rp in neg]
      rng.shuffle(mix)
      asks.append(
        SCORE.format(
          label=rec['label'],
          description=rec['description'],
          rows='\n'.join(
            f'{j + 1}. {snippet(*rp)}' for j, (rp, _) in enumerate(mix)
          ),
          n=len(mix),
        )
      )
      mixes.append((rec, mix))
    for (rec, mix), raw in zip(
      mixes, gemma.text(asks, max_tokens=40) if asks else []
    ):
      ans = first_json(raw)
      if isinstance(ans, list) and len(ans) == len(mix):
        y = np.array([t for _, t in mix])
        g = np.array([
          1 if str(v).strip() in ('1', 'True', 'true') else 0 for v in ans
        ])
        rec['score'] = round(
          float(0.5 * (g[y == 1].mean() + (1 - g[y == 0]).mean())), 2
        )
    out_path.write_text(
      json.dumps(store, ensure_ascii=False, indent=1, sort_keys=True) + '\n'
    )
    done = min(a + 16, len(jobs))
    last = store.get(f'L{chunk[-1][0]:02d}', {}).get(str(chunk[-1][1]), {})
    log(
      f'{done}/{len(jobs)}, {(time.time() - t1) / done:.1f} s each, '
      f'L{chunk[-1][0]:02d} f{chunk[-1][1]} "{last.get("label")}" ({last.get("score")})'
    )

  # labels onto the graph nodes
  for g in graphs:
    data = json.loads(g.read_text())
    for nd in data['nodes']:
      if nd['kind'] == 'feature':
        rec = store.get(f'L{int(nd["layer"]):02d}', {}).get(str(nd['feature']))
        if rec:
          nd['label'] = rec['label']
    g.write_text(json.dumps(data, ensure_ascii=False))
  scored = [
    v['score'] for d in store.values() for v in d.values() if 'score' in v
  ]
  if scored:
    log(f'detection score mean {np.mean(scored):.2f} over {len(scored)}')
  log(f'done -> {out_path}')


if __name__ == '__main__':
  main()
