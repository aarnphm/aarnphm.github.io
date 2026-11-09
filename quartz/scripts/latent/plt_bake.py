# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "mlx-lm"]
# ///
"""Bake Gemma 4 E4B's per-token transcoder codes and attention for /embeddings' selection view.

    uv run quartz/scripts/latent/plt_bake.py --plt quartz/.quartz-cache/latent-plt/full-w8192-k64
    uv run quartz/scripts/latent/plt_bake.py --plt ... --kinds pdf          # later, ~14 h
    uv run quartz/scripts/latent/plt_bake.py --plt ... --index              # inverted index only

The browser never runs Gemma (5 GB), so the selection inspector reads what this writes. Each
sidecar row is read alone, [bos] + chunk cut at --ctx, through the same tapped 4-bit forward the
transcoders were trained on (acts.forward). Per row, for every token after bos:

  codes  the --top strongest transcoder features at each of the 42 layers (id, activation),
         encoded exactly as attribute.py does (x/sx - b_pre, TopK)
  attn   at the global-attention layers, the --attn-top positions it reads most (attention
         probability averaged over heads; bos, the attention sink, left out)
  spans  its [start, end) in the chunk text in UTF-16 code units, so the client can map a DOM
         selection (JS strings index UTF-16) onto tokens

Output, quartz/.quartz-cache/latent-bake/<sidecar[:8]>/:
  meta.json      model, revision, sidecar, plt, layers, top, attn layers, attn-top, ctx, rows
                 {row: [tokens, cut]} (cut = the chunk ran past --ctx)
  rows/<row>.bin little-endian: u32 magic 'PLTB', u32 version 1, u32 T, u32 L, u32 K, u32 A, u32 KA,
                 then u32 spans[T,2], u16 ids[T,L,K], f16 acts[T,L,K], u16 src[T,A,KA],
                 f16 w[T,A,KA]; token t is model position t+1, src indexes the same t space
  index/L<ll>.bin  per feature, its --index-top strongest (row u32, pos u16, act f16) over the bake,
                 sorted by feature then activation: u32 count per feature [M], then the entries
  pooled.bin     per row and layer, the max activation of its 32 strongest features (bridges)

Resumable: rows already written are skipped. --limit bakes a few rows for a check.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MAGIC = b'PLTB'
VERSION = 1
POOL = 32
GLOSS = 3


def log(*args: object) -> None:
  print('[bake]', *args, flush=True)


def utf16_spans(text: str, offsets) -> np.ndarray:
  """Code-point (start, end) offsets -> UTF-16 code-unit offsets."""
  units = np.zeros(len(text) + 1, np.uint32)
  if text:
    units[1:] = np.cumsum([2 if ord(c) > 0xFFFF else 1 for c in text])
  o = np.asarray(offsets, np.int64).reshape(-1, 2)
  return units[np.clip(o, 0, len(text))].astype(np.uint32)


def tokenize(tok, text: str, ctx: int):
  """[bos] + ids cut at ctx, with each id's UTF-16 span; the ids match acts.encode."""
  t = getattr(tok, '_tokenizer', tok)
  enc = t(text, add_special_tokens=False, return_offsets_mapping=True)
  ids = list(enc['input_ids'])
  cut = len(ids) > ctx - 1
  ids = ids[: ctx - 1]
  spans = utf16_spans(text, enc['offset_mapping'][: len(ids)])
  return [t.bos_token_id, *ids], spans, cut


def write_row(path: Path, spans, ids, acts_, src, w) -> None:
  T, L, K = ids.shape
  A, KA = src.shape[1], src.shape[2]
  with path.open('wb') as f:
    f.write(MAGIC + struct.pack('<6I', VERSION, T, L, K, A, KA))
    f.write(spans.astype('<u4').tobytes())
    f.write(ids.astype('<u2').tobytes())
    f.write(acts_.astype('<f2').tobytes())
    f.write(src.astype('<u2').tobytes())
    f.write(w.astype('<f2').tobytes())


def read_row(path: Path):
  raw = path.read_bytes()
  assert raw[:4] == MAGIC, path
  _, T, L, K, A, KA = struct.unpack('<6I', raw[4:28])
  o = 28
  out = {}
  for name, dt, shape in (
    ('spans', '<u4', (T, 2)),
    ('ids', '<u2', (T, L, K)),
    ('acts', '<f2', (T, L, K)),
    ('src', '<u2', (T, A, KA)),
    ('w', '<f2', (T, A, KA)),
  ):
    n = int(np.prod(shape)) * np.dtype(dt).itemsize
    out[name] = np.frombuffer(raw[o : o + n], dt).reshape(shape)
    o += n
  return out


def bake_batch(lm, tcs, grid, lens, glob, top, attn_top):
  """One padded forward: per row, (ids, acts, src, w) numpy arrays over its tokens after bos."""
  import mlx.core as mx
  import mlx.nn as nn


  m = lm.model
  n = len(m.layers)
  inputs = mx.array(grid)
  B, P = grid.shape
  emb = m.embed_tokens(inputs)
  h = emb * m.embed_scale
  ple = m._project_per_layer_inputs(h, m._get_per_layer_inputs(inputs, emb))
  masks = m._make_masks(h, [None] * n)
  inter = [(None, None)] * n
  i_ = mx.arange(P)[:, None]
  j_ = mx.arange(P)[None, :]
  causal = mx.where(j_ <= i_, 0.0, -1e30).astype(mx.float32)
  codes_i, codes_a, att_i, att_w = [], [], [], []
  for i, (layer, mask, prev) in enumerate(
    zip(m.layers, masks, m.previous_kvs)
  ):
    kvs, offset = inter[prev]
    xn = layer.input_layernorm(h)
    a, kvs, offset = layer.self_attn(
      xn, mask, None, shared_kv=kvs, offset=offset
    )
    if i in glob:
      att = layer.self_attn
      q = att.q_norm(att.q_proj(xn).reshape(B, P, att.n_heads, att.head_dim))
      q = att.rope(q.transpose(0, 2, 1, 3), offset=0)
      k = kvs[0]
      k = mx.repeat(k, att.n_heads // k.shape[1], axis=1)
      s = (
        q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)
      ) * att.scale
      p = mx.softmax(s + causal, axis=-1).mean(
        axis=1
      )  # (B, P, P), heads averaged
      p = p[:, 1:, 1:]  # drop bos as query and as source
      kk = min(attn_top, P - 1)
      idx = mx.argpartition(-p, kth=kk - 1, axis=-1)[..., :kk]
      ww = mx.take_along_axis(p, idx, axis=-1)
      mx.eval(idx, ww)
      att_i.append(np.array(idx))
      att_w.append(np.array(ww))
    mid = h + layer.post_attention_layernorm(a)
    x = layer.pre_feedforward_layernorm(mid)
    y = layer.post_feedforward_layernorm(layer.mlp(x))
    h = mid + y
    if layer.per_layer_input_gate is not None:
      g = nn.gelu_approx(layer.per_layer_input_gate(h)) * ple[:, :, i, :]
      h = h + layer.post_per_layer_input_norm(layer.per_layer_projection(g))
    if layer.layer_scalar is not None:
      h = h * layer.layer_scalar
    inter[i] = (kvs, offset)
    # attribute.encode_tc without its TopK: the --top strongest of the k a token keeps are the
    # --top largest pre-activations, so z itself is never needed (a third of the layer's time)
    tc = tcs[i]
    xc = x[:, 1:].reshape(B * (P - 1), -1).astype(mx.float32) / tc['sx'] - tc['b_pre']
    pre = xc @ tc['W_enc'] + tc['b_enc']
    idx = mx.argpartition(-pre, kth=top - 1, axis=-1)[:, :top]
    za = mx.maximum(mx.take_along_axis(pre, idx, axis=-1), 0)
    mx.eval(idx, za, h)
    # copy out at once: idx is a view of the whole (tokens, M) argpartition, and holding it
    # kept 42 of those per batch alive, 11 GB at 8192 tokens (the 2026-11-09 swap)
    codes_i.append(np.array(idx).reshape(B, P - 1, top))
    codes_a.append(np.array(za).reshape(B, P - 1, top))
    del idx, za, pre, xc
  ci = np.stack(codes_i, axis=2)  # (B, P-1, L, K)
  ca = np.stack(codes_a, axis=2)
  ai = np.stack(att_i, axis=2)  # (B, P-1, A, KA)
  aw = np.stack(att_w, axis=2)
  out = []
  for b, T in enumerate(lens):
    t = T - 1  # tokens after bos
    o = np.argsort(-ca[b, :t], axis=-1)  # strongest first
    ids = np.take_along_axis(ci[b, :t], o, -1)
    acts_ = np.take_along_axis(ca[b, :t], o, -1)
    oa = np.argsort(-aw[b, :t], axis=-1)
    src = np.take_along_axis(ai[b, :t], oa, -1)
    w = np.take_along_axis(aw[b, :t], oa, -1)
    w = np.where(
      src < np.arange(t)[:, None, None] + 1, w, 0
    )  # causal: src <= t
    out.append((ids, acts_, src, w))
  return out


def gloss_of(text16: bytes, spans: np.ndarray, rows, poss, acts_) -> str:
  """The distinct tokens a feature fires on hardest, as text: a label for features no one named."""
  seen: list[str] = []
  for r, p in zip(rows, poss):
    s, e = spans[r][p]
    raw = text16[r][2 * s : 2 * e].decode('utf-16-le', 'replace')
    t = raw.strip() or ('↵' if '\n' in raw else '␣')
    if t and t.lower() not in (x.lower() for x in seen):
      seen.append(t if len(t) <= 24 else f'{t[:23]}…')
    if len(seen) == GLOSS:
      break
  return ' · '.join(seen)


def build_index(
  out: Path,
  n_layers: int,
  M: int,
  keep: int,
  texts: dict[int, str] | None = None,
  group: int = 7,
) -> None:
  """index/L<ll>.bin, gloss/L<ll>.json and pooled.bin from rows/, in passes of `group` layers
  to bound memory. A gloss is a feature's strongest distinct tokens, from the row texts."""
  rows = sorted(out.glob('rows/*.bin'), key=lambda p: int(p.stem))
  log(f'index over {len(rows)} rows')
  (out / 'index').mkdir(exist_ok=True)
  (out / 'gloss').mkdir(exist_ok=True)
  text16 = {r: t.encode('utf-16-le') for r, t in (texts or {}).items()}
  spans: dict[int, np.ndarray] = {}
  pooled = {}
  for g0 in range(0, n_layers, group):
    layers = range(g0, min(g0 + group, n_layers))
    per = {l: [] for l in layers}
    for p in rows:
      r = int(p.stem)
      d = read_row(p)
      if g0 == 0:
        spans[r] = d['spans'].copy()
      T, _, K = d['ids'].shape
      pos = np.repeat(np.arange(T, dtype=np.uint16), K)
      for l in layers:
        f = d['ids'][:, l].ravel()
        a = d['acts'][:, l].ravel().astype(np.float32)
        live = a > 0
        f, a, ps = f[live], a[live], pos[live]
        per[l].append((f, np.full(len(f), r, np.uint32), ps, a))
        o = np.argsort(-a, kind='stable')
        uf, first = np.unique(f[o], return_index=True)
        best = np.argsort(-a[o][first], kind='stable')[:POOL]
        pooled.setdefault(r, {})[l] = (uf[best], a[o][first][best])
    for l in layers:
      f, r, pos, a = (np.concatenate(c) for c in zip(*per.pop(l)))
      o = np.lexsort((-a, f))
      f, r, pos, a = f[o], r[o], pos[o], a[o]
      starts = np.searchsorted(f, np.arange(M))
      k = (np.arange(len(f)) - starts[f]) < keep
      counts = np.bincount(f[k], minlength=M).astype('<u4')
      with (out / 'index' / f'L{l:02d}.bin').open('wb') as fh:
        fh.write(counts.tobytes())
        fh.write(r[k].astype('<u4').tobytes())
        fh.write(pos[k].astype('<u2').tobytes())
        fh.write(a[k].astype('<f2').tobytes())
      if text16:
        fk, rk, pk, ak = f[k], r[k], pos[k], a[k]
        bounds = np.searchsorted(fk, np.arange(M + 1))
        gloss = {}
        for fi in np.flatnonzero(counts):
          lo, hi = bounds[fi], bounds[fi + 1]
          rows_ = [int(x) for x in rk[lo:hi] if int(x) in text16]
          if rows_:
            sel = [i for i in range(lo, hi) if int(rk[i]) in text16]
            gloss[int(fi)] = gloss_of(text16, spans, rk[sel], pk[sel], ak[sel])
        (out / 'gloss' / f'L{l:02d}.json').write_text(
          json.dumps(gloss, ensure_ascii=False)
        )
    log(f'index layers {layers.start}-{layers.stop - 1}')
  with (out / 'pooled.bin').open('wb') as fh:
    order = sorted(pooled)
    fh.write(struct.pack('<3I', len(order), n_layers, POOL))
    for r in order:
      fh.write(struct.pack('<I', r))
      for l in range(n_layers):
        ids, acts_ = pooled[r][l]
        pad = POOL - len(ids)
        fh.write(np.pad(ids, (0, pad)).astype('<u2').tobytes())
        fh.write(np.pad(acts_, (0, pad)).astype('<f2').tobytes())
  log('index written')


def load_encoders(path: Path, n_layers: int):
  """The encoder half of each transcoder, evaluated now so the decoder side (W_dec, W_skip:
  4.6 of the 8 GB in fp32) is dropped with the file and never sits on the GPU. With it the bake
  held 22 GB on the 32 GB Mac and swapped down to 55 tok/s (2026-11-09)."""
  import mlx.core as mx

  tcs = []
  for l in range(n_layers):
    f = path / f'L{l:02d}.safetensors'
    if not f.exists():
      raise SystemExit(f'{path} has no transcoder for layer {l}: train all layers first')
    w = mx.load(str(f))
    meta = json.loads(f.with_suffix('.json').read_text())
    tc = dict(
      W_enc=w['W_enc'].astype(mx.float32),
      b_enc=w['b_enc'].astype(mx.float32),
      b_pre=(w['b_pre'] if 'b_pre' in w else mx.zeros_like(w['b_dec'])).astype(
        mx.float32
      ),
      sx=float(w['sx'][0]),
      k=int(meta['k']),
    )
    mx.eval(tc['W_enc'], tc['b_enc'], tc['b_pre'])
    del w
    tcs.append(tc)
  mx.clear_cache()
  return tcs


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--plt', type=Path, required=True)
  ap.add_argument('--content', type=Path, default=Path('content'))
  ap.add_argument('--directory', default='embeddings')
  ap.add_argument('--kinds', default='note')
  ap.add_argument('--out', type=Path, default=None)
  ap.add_argument('--ctx', type=int, default=1024)
  ap.add_argument('--top', type=int, default=4)
  ap.add_argument('--attn-top', type=int, default=4)
  ap.add_argument('--batch-tokens', type=int, default=8192)
  ap.add_argument('--index-top', type=int, default=32)
  ap.add_argument('--limit', type=int, default=0)
  ap.add_argument(
    '--index', action='store_true', help='only rebuild index/ and pooled.bin'
  )
  a = ap.parse_args()

  import acts
  from label import MODEL, REVISION, resolve

  digest, rows = acts.corpus(
    a.content.resolve(), a.directory, 0, a.kinds.split(',')
  )
  out = a.out or Path('quartz/.quartz-cache/latent-bake') / digest[:8]
  (out / 'rows').mkdir(parents=True, exist_ok=True)
  meta_path = out / 'meta.json'
  meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

  if a.index:
    build_index(out, meta['layers'], meta['width'], a.index_top, dict(rows))
    return

  import mlx.core as mx
  from mlx_lm import load

  mx.set_cache_limit(2 << 30)
  model, tok = load(resolve(MODEL, REVISION))
  lm = acts.text_model(model)
  n = len(lm.model.layers)
  glob = [
    i
    for i, layer in enumerate(lm.model.layers)
    if layer.layer_type == 'full_attention'
  ]
  tcs = load_encoders(a.plt, n)
  M = int(tcs[0]['W_enc'].shape[1])
  meta.update(
    model=MODEL,
    revision=REVISION,
    sidecar=digest,
    plt=str(a.plt),
    layers=n,
    width=M,
    top=a.top,
    attnLayers=glob,
    attnTop=a.attn_top,
    ctx=a.ctx,
    spans='utf16',
  )
  meta.setdefault('rows', {})
  todo = [(r, t) for r, t in rows if not (out / 'rows' / f'{r}.bin').exists()]
  if a.limit:
    todo = todo[: a.limit]
  log(
    f'{len(rows)} {a.kinds} rows in sidecar {digest[:8]}, {len(todo)} to bake, attention at {glob}'
  )
  enc = [(r, *tokenize(tok, t, a.ctx)) for r, t in todo]
  enc.sort(key=lambda e: len(e[1]))
  t0, done_tok, last = time.time(), 0, 0.0
  i = 0
  while i < len(enc):
    L = len(enc[i][1])
    j = i + 1
    while j < len(enc) and (j - i + 1) * len(enc[j][1]) <= a.batch_tokens:
      j += 1
    group = enc[i:j]
    P = max(len(e[1]) for e in group)
    grid = np.zeros((len(group), P), np.int32)
    for b, e in enumerate(group):
      grid[b, : len(e[1])] = e[1]
    res = bake_batch(
      lm, tcs, grid, [len(e[1]) for e in group], glob, a.top, a.attn_top
    )
    for (r, ids, spans, cut), (fi, fa, src, w) in zip(group, res):
      write_row(out / 'rows' / f'{r}.bin', spans, fi, fa, src, w)
      meta['rows'][str(r)] = [len(ids) - 1, bool(cut)]
      done_tok += len(ids) - 1
    del res
    # every batch has its own (B, P): mlx's buffer cache keeps each shape's buffers, and over
    # thousands of shapes that filled 35 GB of swap on the 32 GB Mac (2026-11-09)
    mx.clear_cache()
    i = j
    if time.time() - last > 30 or i >= len(enc):
      last = time.time()
      meta_path.write_text(json.dumps(meta) + '\n')
      rate = done_tok / max(time.time() - t0, 1e-9)
      gb = 1 << 30
      log(
        f'{i}/{len(enc)} rows, {done_tok} tokens, {rate:.0f} tok/s, longest {L}..{P},'
        f' mlx {mx.get_active_memory() / gb:.1f} GB active {mx.get_cache_memory() / gb:.1f}'
        f' cache {mx.get_peak_memory() / gb:.1f} peak'
      )
      mx.reset_peak_memory()
  meta_path.write_text(json.dumps(meta) + '\n')
  if not a.limit:
    build_index(out, n, M, a.index_top, dict(rows))
  log('done', out)


if __name__ == '__main__':
  main()
