# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx>=0.32.3", "mlx-vlm==0.7.10", "transformers>=5.14", "numpy"]
# ///
"""The words that make a text feature fire (docs/latent-steering.md, item 5), for the explorer's
feature card: integrated gradients from embeddinggemma-2's input tokens to the feature's
pre-activation, on the feature's strongest rows.

    uv run quartz/scripts/latent/words.py [--rows 3 --steps 8 --limit 0]

Why gradients: the pooled embedding is a mean of per-token outputs, so a direct split per position
is exact, but it credits the slots where attention gathered the chunk (<eos>, the `title | text`
template) rather than the words that fed them. Integrated gradients from a baseline where every
token but <bos>/<eos> is <pad> (a zero input is off-manifold and breaks the normalisation) push the
credit back to the source tokens, and its total is close to z(x) - z(baseline) (0.8-1.2 at 16 steps;
the default 4 steps rank the words the same and cost a quarter). The
path's steps run as one batch, so a (row, feature) costs one batched forward and backward.

Results append to quartz/.quartz-cache/latent-words-<sidecar>.jsonl as they finish, so a stopped run
resumes; feat/text/words.json is rewritten every 100 features and at the end. Coarse features
(Matryoshka level 0, then 1) and frequent ones go first.

words.json: {model, steps, rows, features: {f: {w: [[word, share]], r: [[row, [[word, share]]]]}}}
`share` is the word's part of the positive attribution in that row; `w` averages over the rows.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import STOP, WORD  # noqa: E402

MODEL_ID, REV = (
  'mlx-community/embeddinggemma-2-bf16',
  '1a4ffddb7905d3f63486748deabe091a01fb6201',
)
TEMPLATE = {'title', ':', '▁|', '▁text', '|', 'text'}
PUNCT = re.compile(r'^[\W_]+$')
MAX_TOKENS = 512


def log(*args: object) -> None:
  print('[words]', *args, flush=True)


def words_of(
  toks: list[str], ig: np.ndarray, skip: set[int]
) -> list[tuple[str, float]]:
  """Merge sentencepiece pieces into words; a piece starting with ▁ or a newline opens a word."""
  out: list[list] = []
  for i, t in enumerate(toks):
    if i in skip:
      out.append(None)  # type: ignore[arg-type]
      continue
    opens = (
      t.startswith('▁') or t.startswith('\n') or not out or out[-1] is None
    )
    piece = t.replace('▁', ' ').strip()
    if opens or PUNCT.match(piece or ' '):
      out.append([piece, float(ig[i])])
    else:
      out[-1][0] += piece
      out[-1][1] += float(ig[i])
  merged: dict[str, float] = {}
  for item in out:
    if item is None or not item[0]:
      continue
    key = item[0].lower().strip('.,;:!?()[]{}"\'`*_')
    # the card shows words: markup, numbers, single letters and stopwords are attribution the
    # reader cannot use (they still count in the row's total, so shares stay honest)
    if not WORD.fullmatch(key) or key in STOP:
      continue
    merged[key] = merged.get(key, 0.0) + item[1]
  return sorted(merged.items(), key=lambda kv: -kv[1])


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--content', default='content')
  ap.add_argument('--directory', default='embeddings')
  ap.add_argument('--cache', default='quartz/.quartz-cache')
  ap.add_argument(
    '--rows', type=int, default=3, help='strongest rows per feature'
  )
  ap.add_argument('--steps', type=int, default=4)
  ap.add_argument(
    '--limit', type=int, default=0, help='features to do this run (0 = all)'
  )
  args = ap.parse_args()

  import mlx.core as mx
  from mlx_vlm import load

  root = Path(args.content) / args.directory / 'latent'
  side = root / json.loads((root / 'index.json').read_text())['hash']
  docs = json.loads((side / 'docs.json').read_text())
  rowdoc = np.fromfile(side / 'rowdoc.u32', dtype=np.uint32)
  tb = side / 'feat' / 'text'
  meta = json.loads((tb / 'meta.json').read_text())
  table = json.loads((tb / 'features.json').read_text())
  M, D = meta['features'], meta['dims']
  enc = (
    np
    .fromfile(tb / 'enc-int8.bin', dtype=np.int8)
    .reshape(M, D)
    .astype(np.float32)
  )
  enc *= np.array(meta['encScale'], np.float32)[:, None]
  mean = mx.array(np.array(meta['mean'], np.float32))
  norm, bias = meta['norm'], np.array(meta['bias'], np.float32)
  part = json.loads((side / 'index.json').read_text())['part']

  cache = Path(args.cache) / f'latent-words-v2-{side.name[:16]}.jsonl'
  cache.parent.mkdir(parents=True, exist_ok=True)
  done: dict[int, dict] = {}
  if cache.exists():
    for line in cache.read_text().splitlines():
      try:
        item = json.loads(line)
        done[item['f']] = item
      except ValueError:
        pass
  order = sorted(
    (t for t in table if t['n'] >= 3),
    key=lambda t: (t.get('level', 0), -t['n']),
  )
  todo = [t for t in order if t['f'] not in done]
  if args.limit:
    todo = todo[: args.limit]
  log(
    f'{len(done)} features cached, {len(todo)} to do, {args.rows} rows × {args.steps} steps'
  )

  model, processor = load(MODEL_ID, revision=REV)
  tok = processor.tokenizer
  lm = model.language_model
  scale = model.config.text_config.hidden_size**0.5
  texts: dict[tuple[int, int], dict] = {}

  def chunk(r: int) -> tuple[str, str]:
    d = int(rowdoc[r])
    doc = docs[d]
    c = r - doc['r0']
    p = c // part
    key = (d, p)
    if key not in texts:
      name = f'{d}.json' if p == 0 else f'{d}-{p:03d}.json'
      texts[key] = {
        ch['r']: ch['text']
        for ch in json.loads((side / 'text' / name).read_text())['chunks']
      }
    return doc['t'] or 'none', texts[key].get(r, '')

  def attribute(r: int, f: int) -> tuple[list[str], np.ndarray, set[int]]:
    title, text = chunk(r)
    prompt = f'title: {title} | text: {text}'
    inputs = processor(text=[prompt], return_tensors='mlx')
    ids = np.array(inputs['input_ids'])[:, :MAX_TOKENS]
    mask = inputs['attention_mask'][:, :MAX_TOKENS]
    keep = np.isin(ids, [tok.bos_token_id, tok.eos_token_id])
    x = (lm.embed_tokens(mx.array(ids)) * scale).astype(mx.float32)
    x0 = (
      lm.embed_tokens(mx.array(np.where(keep, ids, tok.pad_token_id))) * scale
    ).astype(mx.float32)
    alphas = mx.array(
      ((np.arange(args.steps) + 0.5) / args.steps).astype(np.float32)
    )[:, None, None]
    path = x0 + (x - x0) * alphas  # steps x T x H
    maskb = mx.broadcast_to(mask, (args.steps, mask.shape[1]))
    e_f = mx.array(enc[f])

    def total(p):
      H = lm(p.astype(mx.bfloat16), maskb).astype(mx.float32)[:, :, :256]
      v = H.mean(1)
      v = v / mx.linalg.norm(v, axis=1, keepdims=True)
      return mx.sum(((v - mean) / norm) @ e_f)

    g = mx.grad(total)(path)
    ig = mx.sum(mx.mean(g, axis=0) * (x - x0)[0], axis=-1)
    mx.eval(ig)
    toks = tok.convert_ids_to_tokens(ids[0].tolist())
    prefix = len(
      tok(f'title: {title} | text:', add_special_tokens=True)['input_ids']
    )
    skip = {
      i
      for i, t in enumerate(toks)
      if keep[0, i] or (i < prefix and t in TEMPLATE)
    }
    return toks, np.array(ig), skip

  t0 = time.time()
  with cache.open('a') as fh:
    for n, row in enumerate(todo, 1):
      f = row['f']
      rows = [
        r
        for r, _ in row['top']
        if docs[int(rowdoc[r])]['s'] in ('note', 'pdf')
      ][: args.rows]
      per = []
      for r in rows:
        try:
          toks, ig, skip = attribute(r, f)
        except Exception as err:  # a malformed chunk should not stop the run
          log(f'#{f} row {r}: {err}')
          continue
        pos = float(ig.clip(min=0).sum()) or 1.0
        ws = [(w, s / pos) for w, s in words_of(toks, ig, skip) if s > 0][:5]
        per.append([int(r), [[w, round(s, 3)] for w, s in ws]])
      agg: dict[str, float] = {}
      for _, ws in per:
        for w, s in ws:
          agg[w] = agg.get(w, 0.0) + s / max(1, len(per))
      item = dict(
        f=f,
        w=[
          [w, round(s, 3)]
          for w, s in sorted(agg.items(), key=lambda kv: -kv[1])[:8]
        ],
        r=per,
      )
      done[f] = item
      fh.write(json.dumps(item, ensure_ascii=False) + '\n')
      fh.flush()
      if n % 100 == 0 or n == len(todo):
        write(tb, done, args)
        rate = (time.time() - t0) / n
        log(
          f'{n}/{len(todo)} features, {rate:.1f}s each, about {rate * (len(todo) - n) / 3600:.1f} h left'
        )
  write(tb, done, args)


def write(tb: Path, done: dict[int, dict], args) -> None:
  out = dict(
    model=MODEL_ID,
    steps=args.steps,
    rows=args.rows,
    features={
      str(f): dict(w=v['w'], r=v['r']) for f, v in sorted(done.items())
    },
  )
  tmp = tb / 'words.json.tmp'
  tmp.write_text(
    json.dumps(out, ensure_ascii=False, separators=(',', ':')),
    encoding='utf-8',
  )
  tmp.replace(tb / 'words.json')


if __name__ == '__main__':
  main()
