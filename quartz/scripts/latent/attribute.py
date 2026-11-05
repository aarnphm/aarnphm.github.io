# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "mlx-lm"]
# ///
"""Attribution graphs for Gemma 4 E4B over its garden transcoders (step 3).

    uv run quartz/scripts/latent/attribute.py --plt quartz/.quartz-cache/latent-plt/<dir> \\
      --prompt "The author of Gödel, Escher, Bach is" [--nodes 400]

The method is Ameisen et al. 2025 ("Circuit tracing", transformer-circuits.pub) with per-layer
transcoders: replace every layer's MLP (and here its PLE gate) by its transcoder plus an error
term that makes the replacement exact on this prompt, then freeze everything else that is not
linear on this prompt: attention patterns, the 1/rms of every RMSNorm (the four sandwich norms,
q/k/v norms, the final norm). What is left is linear in the sources, so every target splits
exactly into contributions:

  sources  token embeddings (one per position), active transcoder features (layer, position,
           feature: activation a times its decoder row), error nodes (layer, position), and the
           transcoder biases b_dec (one per layer)
  targets  the top logits at the last position (logit minus the mean logit, before the softcap)
           and the pre-activations of the features kept in the graph

A target's row comes from one backward pass through the frozen model (batched: one copy of the
prompt per target), and the edge from source s is <gradient at s's write, s's write>. The skip
path of each transcoder (W_skip, linear) stays in the frozen model, so it carries gradient
instead of being a node. Completeness, the sum of a row = its target, is checked on every run.

Features enter the graph greedily by influence on the logits (circuit-tracer's scheme): rows for
the logits first, then batches of the features with the largest influence so far, until --nodes.
Influence is the logit-weighted sum over paths of |normalised edges|, (I - A)^-1 on the kept
set. Pruning then keeps the nodes holding --node-share of influence and the edges holding
--edge-share.

Output: quartz/.quartz-cache/latent-graphs/<slug>.json
  {prompt, tokens, model, plt, nodes: [{id, kind, layer, pos, feature, act, influence}],
   links: [{source, target, weight}], logits: [{token, prob}], check: {...}}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path('quartz/.quartz-cache/latent-graphs')


def log(*args: object) -> None:
  print('[graph]', *args, flush=True)


def inv_rms(x, eps: float):
  import mlx.core as mx

  x = x.astype(mx.float32)
  return mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


def weight(norm):
  return norm.weight if 'weight' in norm else None


def nrm(x, s, w):
  """RMSNorm with its 1/rms frozen to s."""
  return x * s if w is None else x * s * w


def load_plt(path: Path, n_layers: int):
  import mlx.core as mx

  tcs = []
  for l in range(n_layers):
    f = path / f'L{l:02d}.safetensors'
    if not f.exists():
      raise SystemExit(
        f'{path} has no transcoder for layer {l}: train all layers first'
      )
    w = mx.load(str(f))
    meta = json.loads(f.with_suffix('.json').read_text())
    tcs.append(
      dict(
        W_enc=w['W_enc'].astype(mx.float32),
        b_enc=w['b_enc'].astype(mx.float32),
        W_dec=w['W_dec'].astype(mx.float32),
        W_skip=w['W_skip'].astype(mx.float32),
        b_dec=w['b_dec'].astype(mx.float32),
        sx=float(w['sx'][0]),
        sy=float(w['sy'][0]),
        k=int(meta['k']),
        fvu=meta.get('fvu'),
      )
    )
  return tcs


def encode_tc(tc, x):
  """Transcoder code of x (P, d) in model units: (pre-activation, z)."""
  import mlx.core as mx

  pre = (x.astype(mx.float32) / tc['sx']) @ tc['W_enc'] + tc['b_enc']
  thr = mx.topk(pre, tc['k'], axis=-1).min(axis=-1, keepdims=True)
  return pre, mx.where(pre >= thr, mx.maximum(pre, 0), 0)


def masks(lm, P: int):
  import mlx.core as mx

  i = mx.arange(P)[:, None]
  j = mx.arange(P)[None, :]
  causal = j <= i
  w = lm.model.window_size
  neg = mx.array(-1e30, mx.float32)
  return {
    'full_attention': mx.where(causal, 0.0, neg),
    'sliding_attention': mx.where(causal & (i - j < w), 0.0, neg),
  }


def record(lm, tcs, ids: list[int]):
  """The true forward on one prompt, attention written out so its pattern can be kept, with
  every quantity the frozen model needs. Returns (frozen, sources, logits)."""
  import mlx.core as mx
  import mlx.nn as nn

  m = lm.model
  eps = lm.args.rms_norm_eps
  x_ids = mx.array([ids], dtype=mx.int32)
  P = len(ids)
  emb = m.embed_tokens(x_ids)
  h = (emb * m.embed_scale).astype(mx.float32)
  h0 = h
  ple = m._project_per_layer_inputs(
    emb * m.embed_scale, m._get_per_layer_inputs(x_ids, emb)
  ).astype(mx.float32)
  mk = masks(lm, P)
  kvs = {}
  F = []  # per layer: frozen factors and pattern
  S = []  # per layer: transcoder sources
  for i, layer in enumerate(m.layers):
    att = layer.self_attn
    H, hd = att.n_heads, att.head_dim
    s_in = inv_rms(h, eps)
    hn = nrm(h, s_in, weight(layer.input_layernorm))
    q = att.q_norm(att.q_proj(hn).reshape(1, P, H, hd)).transpose(0, 2, 1, 3)
    q = att.rope(q, offset=0)
    if att.has_kv:
      KV = att.n_kv_heads
      k = att.k_norm(att.k_proj(hn).reshape(1, P, KV, hd)).transpose(
        0, 2, 1, 3
      )
      k = att.rope(k, offset=0)
      vraw = att.v_proj(hn).reshape(1, P, KV, hd).astype(mx.float32)
      s_v = inv_rms(vraw, eps)
      v = (vraw * s_v).transpose(0, 2, 1, 3)
      kvs[i] = (k, v, i, s_v)
      src = i
    else:
      src = m.previous_kvs[i]
      k, v, _, s_v = kvs[src]
    rep = H // k.shape[1]
    kk = mx.repeat(k, rep, axis=1).astype(mx.float32)
    vv = mx.repeat(v, rep, axis=1)
    scores = (q.astype(mx.float32) @ kk.transpose(0, 1, 3, 2)) * att.scale
    probs = mx.softmax(scores + mk[layer.layer_type], axis=-1)
    a = (probs @ vv).transpose(0, 2, 1, 3).reshape(1, P, -1)
    a = att.o_proj(a)
    s_pa = inv_rms(a, eps)
    mid = h + nrm(a, s_pa, weight(layer.post_attention_layernorm))
    s_pf = inv_rms(mid, eps)
    x = nrm(mid, s_pf, weight(layer.pre_feedforward_layernorm))
    mo = layer.mlp(x)
    s_po = inv_rms(mo, eps)
    y = nrm(mo, s_po, weight(layer.post_feedforward_layernorm))
    h1 = mid + y
    if layer.per_layer_input_gate is not None:
      g = nn.gelu_approx(layer.per_layer_input_gate(h1)) * ple[:, :, i, :]
      g = layer.post_per_layer_input_norm(layer.per_layer_projection(g))
      y = y + g
      h1 = h1 + g
    scalar = layer.layer_scalar if layer.layer_scalar is not None else 1.0
    h = h1 * scalar
    tc = tcs[i]
    pre, z = encode_tc(tc, x[0])
    skip = (x[0] / tc['sx']) @ tc['W_skip'] * tc['sy']
    yhat = (z @ tc['W_dec'] + tc['b_dec']) * tc['sy'] + skip
    err = y[0] - yhat
    F.append(
      dict(
        s_in=s_in,
        s_v=s_v,
        kv_src=src,
        probs=probs,
        s_pa=s_pa,
        s_pf=s_pf,
        scalar=scalar,
      )
    )
    S.append(dict(z=z, pre=pre, err=err, x=x[0], y=y[0]))
  s_fin = inv_rms(h, eps)
  hf = nrm(h, s_fin, weight(m.norm))
  return dict(layers=F, s_fin=s_fin, h0=h0), S, hf


def linear(lm, tcs, frozen, h0, writes, stop: int | None = None):
  """The frozen replacement model: h0 (B, P, d), writes[l] (B, P, d) = everything a layer adds
  besides its skip path. Returns the final normed residual (B, P, d), or with stop=l the MLP
  input of layer l."""
  m = lm.model
  h = h0
  vs = {}
  for i, layer in enumerate(m.layers):
    f = frozen['layers'][i]
    att = layer.self_attn
    B, P, _ = h.shape
    if att.has_kv:
      hn = nrm(h, f['s_in'], weight(layer.input_layernorm))
      KV, hd = att.n_kv_heads, att.head_dim
      v = (att.v_proj(hn).reshape(B, P, KV, hd) * f['s_v']).transpose(
        0, 2, 1, 3
      )
      vs[i] = v
    v = vs[f['kv_src']]
    import mlx.core as mx

    vv = mx.repeat(v, att.n_heads // v.shape[1], axis=1)
    a = (f['probs'] @ vv).transpose(0, 2, 1, 3).reshape(B, P, -1)
    a = att.o_proj(a)
    mid = h + nrm(a, f['s_pa'], weight(layer.post_attention_layernorm))
    x = nrm(mid, f['s_pf'], weight(layer.pre_feedforward_layernorm))
    if stop == i:
      return x
    tc = tcs[i]
    h = (mid + writes[i] + (x / tc['sx']) @ tc['W_skip'] * tc['sy']) * f[
      'scalar'
    ]
  return nrm(h, frozen['s_fin'], weight(m.norm))


def unembed_rows(lm, ids: list[int]):
  import mlx.core as mx

  return lm.model.embed_tokens(mx.array(ids, dtype=mx.int32)).astype(
    mx.float32
  )


def unembed_mean(lm, chunk: int = 16384):
  import mlx.core as mx

  V = lm.args.vocab_size
  acc = mx.zeros((lm.args.hidden_size,), mx.float32)
  for a in range(0, V, chunk):
    rows = lm.model.embed_tokens(mx.arange(a, min(V, a + chunk))).astype(
      mx.float32
    )
    acc = acc + rows.sum(axis=0)
    mx.eval(acc)
  return acc / V


class Graph:
  """Sources and rows of one prompt's attribution graph."""

  def __init__(self, lm, tcs, ids: list[int]):
    import mlx.core as mx

    self.lm, self.tcs, self.ids = lm, tcs, ids
    self.frozen, self.S, self.hf = record(lm, tcs, ids)
    self.L, self.P = len(tcs), len(ids)
    self.writes = [
      ((s['z'] @ tc['W_dec']) + tc['b_dec']) * tc['sy'] + s['err']
      for s, tc in zip(self.S, tcs)
    ]
    mx.eval(self.writes)
    # active features: (layer, pos, feature, act)
    act, bos = [], []
    for l, s in enumerate(self.S):
      z = np.array(s['z'])
      for p, f in zip(*np.nonzero(z)):
        # bos features are an attention sink, constant across prompts: they fold into the bos
        # node (embedding position 0) instead of entering the graph one by one
        (bos if p == 0 else act).append((l, int(p), int(f), float(z[p, f])))
    self.active = act
    self.bos = bos
    self.index = {(l, p, f): j for j, (l, p, f, _) in enumerate(act)}

  def rows(self, kind: str, items: list, batch: int = 32):
    """Rows over sources for targets. kind 'logit': items are unembedding directions (d,);
    kind 'feature': items are (layer, pos, feature). Returns (feature contributions (T, n),
    error (T, L, P), bias (T, L), embedding (T, P), target values (T,))."""
    import mlx.core as mx

    lm, tcs, L, P = self.lm, self.tcs, self.L, self.P
    n = len(self.active)
    groups: list[tuple[int | None, list[int]]] = []
    if kind == 'logit':
      groups = [
        (None, list(range(a, min(len(items), a + batch))))
        for a in range(0, len(items), batch)
      ]
    else:
      by: dict[int, list[int]] = {}
      for j, (l, _, _) in enumerate(items):
        by.setdefault(l, []).append(j)
      groups = [
        (l, js[a : a + batch])
        for l, js in sorted(by.items())
        for a in range(0, len(js), batch)
      ]
    res = [None] * len(items)
    for stop, js in groups:
      B = len(js)
      h0 = mx.broadcast_to(self.frozen['h0'], (B, P, lm.args.hidden_size))
      ws = [mx.broadcast_to(w[None], (B, P, w.shape[-1])) for w in self.writes]
      if kind == 'logit':
        dirs = mx.stack([items[j] for j in js])  # (B, d)
        pos = mx.array([P - 1] * B)

        def tgt(h0, *ws):
          out = linear(lm, tcs, self.frozen, h0, list(ws))
          return mx.sum(out[mx.arange(B), pos] * dirs)

      else:
        tc = tcs[stop]
        ps = mx.array([items[j][1] for j in js])
        enc = mx.stack([tc['W_enc'][:, items[j][2]] for j in js])  # (B, d)

        def tgt(h0, *ws):
          x = linear(lm, tcs, self.frozen, h0, list(ws), stop=stop)
          return mx.sum((x[mx.arange(B), ps] / tc['sx']) * enc)

      val, grads = mx.value_and_grad(tgt, argnums=list(range(L + 1)))(h0, *ws)
      g0, gw = grads[0], grads[1:]
      emb = mx.sum(g0 * self.frozen['h0'], axis=-1)  # (B, P)
      err = mx.stack(
        [mx.sum(g * s['err'], axis=-1) for g, s in zip(gw, self.S)], 1
      )
      bias = mx.stack(
        [
          mx.sum(g * (tc['b_dec'] * tc['sy']), axis=(-1, -2))
          for g, tc in zip(gw, tcs)
        ],
        1,
      )
      # features: a * sy * <W_dec_f, g[l, p]>, gathered for the active set
      feat = np.zeros((B, n), np.float32)
      for l in range(L if stop is None else stop):
        rows_l = [j for j, (al, *_) in enumerate(self.active) if al == l]
        if not rows_l:
          continue
        pp = mx.array([self.active[j][1] for j in rows_l])
        ff = mx.array([self.active[j][2] for j in rows_l])
        aa = mx.array([self.active[j][3] for j in rows_l])
        dec = tcs[l]['W_dec'][ff] * tcs[l]['sy']  # (r, d)
        gl = gw[l][:, pp, :]  # (B, r, d)
        feat[:, rows_l] = np.array(mx.sum(gl * dec[None], axis=-1) * aa[None])
      for l in range(L if stop is None else stop):
        here = [b for b in self.bos if b[0] == l]
        if here:
          ff = mx.array([b[2] for b in here])
          aa = mx.array([b[3] for b in here])
          dec = (tcs[l]['W_dec'][ff] * tcs[l]['sy'] * aa[:, None]).sum(0)
          emb = emb + mx.concatenate(
            [mx.sum(gw[l][:, :1, :] * dec, axis=-1), mx.zeros((B, P - 1))],
            axis=1,
          )
      mx.eval(emb, err, bias)
      tv = self.target_values(kind, [items[j] for j in js], stop)
      for b, j in enumerate(js):
        res[j] = (
          feat[b],
          np.array(err[b]),
          np.array(bias[b]),
          np.array(emb[b]),
          tv[b],
        )
      _ = val
    F = np.stack([r[0] for r in res])
    E = np.stack([r[1] for r in res])
    Bs = np.stack([r[2] for r in res])
    Em = np.stack([r[3] for r in res])
    T = np.array([r[4] for r in res])
    return F, E, Bs, Em, T

  def target_values(self, kind, items, stop):
    if kind == 'logit':
      hf = np.array(self.hf[0, -1])
      return [float(hf @ np.array(d)) for d in items]
    tc = self.tcs[stop]
    out = []
    for l, p, f in items:
      x = np.array(self.S[l]['x'][p]) / tc['sx']
      out.append(float(x @ np.array(tc['W_enc'][:, f])))
    return out


def influence(A: np.ndarray, w: np.ndarray) -> np.ndarray:
  """Logit-weighted influence of every node over paths: w^T (I - Â)^{-1}, with Â the
  |row-normalised| adjacency (rows = targets, cols = sources, square on the node set)."""
  Ah = np.abs(A)
  Ah = Ah / np.maximum(Ah.sum(1, keepdims=True), 1e-12)
  n = len(A)
  return np.linalg.solve((np.eye(n) - Ah).T, w)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument(
    '--plt', required=True, help='a transcoders.py output folder (all layers)'
  )
  ap.add_argument('--prompt', required=True)
  ap.add_argument(
    '--chat', action='store_true', help='wrap the prompt in the chat template'
  )
  ap.add_argument(
    '--logits', type=int, default=8, help='at most this many target logits'
  )
  ap.add_argument('--logit-mass', type=float, default=0.95)
  ap.add_argument(
    '--nodes', type=int, default=400, help='feature nodes to expand'
  )
  ap.add_argument('--batch', type=int, default=32)
  ap.add_argument('--node-share', type=float, default=0.8)
  ap.add_argument('--edge-share', type=float, default=0.98)
  ap.add_argument('--quantised', action='store_true')
  ap.add_argument('--out', default='')
  args = ap.parse_args()

  import mlx.core as mx
  from label import MODEL, REVISION, resolve
  from mlx_lm import load

  import acts

  model, tok = load(resolve(MODEL, REVISION))
  lm = acts.text_model(model)
  if not args.quantised:
    acts.densify(lm)
  tcs = load_plt(Path(args.plt), len(lm.model.layers))
  t = getattr(tok, '_tokenizer', tok)
  if args.chat:
    ids = list(
      t.apply_chat_template(
        [{'role': 'user', 'content': args.prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
      )
    )
  else:
    ids = acts.encode(tok, args.prompt, 512)
  t0 = time.time()
  G = Graph(lm, tcs, ids)
  # exactness: the written-out forward is the library's, and the frozen replacement with its
  # error nodes is the written-out forward
  P = len(ids)
  ref = model(mx.array([ids], dtype=mx.int32)).astype(mx.float32)[0, -1]
  probs = mx.softmax(ref, axis=-1)
  order = np.argsort(-np.array(probs))
  keep, mass = [], 0.0
  for tid in order[: args.logits]:
    keep.append(int(tid))
    mass += float(probs[int(tid)].item())
    if mass >= args.logit_mass:
      break
  E = unembed_rows(lm, keep)
  c = lm.final_logit_softcapping or 0
  mine = G.hf[0, -1] @ E.T
  mine = mx.tanh(mine / c) * c if c else mine
  rec_err = float(mx.max(mx.abs(mine - ref[mx.array(keep)])).item()) / max(
    1e-6, float(mx.max(mx.abs(ref[mx.array(keep)])).item())
  )
  rep = linear(lm, tcs, G.frozen, G.frozen['h0'], [w[None] for w in G.writes])
  exact = float(mx.max(mx.abs(rep - G.hf)).item()) / max(
    1e-6, float(mx.max(mx.abs(G.hf)).item())
  )
  log(
    f'{P} tokens, {len(G.active)} active features; written-out vs library logits {rec_err:.2e},'
    f' replacement vs written-out residual {exact:.2e}'
  )
  mean = unembed_mean(lm)
  dirs = [E[i] - mean for i in range(len(keep))]

  n = len(G.active)
  lw = np.array([float(probs[tid].item()) for tid in keep])
  lw = lw / lw.sum()
  LF, LE, LB, LEm, LT = G.rows('logit', dirs, args.batch)
  done_rows: dict[int, tuple] = {}
  score = (
    np.abs(LF).T @ lw
  )  # first-order influence of each feature on the logits
  order_f: list[int] = []
  while len(order_f) < min(args.nodes, n):
    cand = [j for j in np.argsort(-score) if j not in done_rows][: args.batch]
    if not cand or score[cand[0]] <= 0:
      break
    items = [G.active[j][:3] for j in cand]
    F, E, Bs, Em, T = G.rows('feature', items, args.batch)
    for b, j in enumerate(cand):
      done_rows[j] = (F[b], E[b], Bs[b], Em[b], T[b])
      order_f.append(j)
    # second-order: propagate through the rows computed so far
    S = np.array(order_f)
    A = np.stack([done_rows[j][0][S] for j in S])
    inf_nodes = np.abs(LF[:, S]).T @ lw
    infl = influence(A, inf_nodes)
    score = np.abs(LF).T @ lw
    for j_i, j in enumerate(S):
      score += (
        infl[j_i]
        * np.abs(done_rows[j][0])
        / max(1e-12, np.abs(done_rows[j][0]).sum())
      )
    log(f'{len(order_f)} feature rows, {time.time() - t0:.0f}s')

  # completeness: every row sums to its target
  def total(F, E, Bs, Em):
    return F.sum(1) + E.reshape(len(E), -1).sum(1) + Bs.sum(1) + Em.sum(1)

  comp_logit = np.max(
    np.abs(total(LF, LE, LB, LEm) - LT) / np.maximum(np.abs(LT), 1e-6)
  )
  S = np.array(order_f)
  FR = np.stack([done_rows[j][0] for j in S]) if len(S) else np.zeros((0, n))
  ER = (
    np.stack([done_rows[j][1] for j in S]) if len(S) else np.zeros((0, G.L, P))
  )
  BR = np.stack([done_rows[j][2] for j in S]) if len(S) else np.zeros((0, G.L))
  EmR = np.stack([done_rows[j][3] for j in S]) if len(S) else np.zeros((0, P))
  TR = np.array([done_rows[j][4] for j in S])
  comp_feat = (
    float(
      np.max(
        np.abs(total(FR, ER, BR, EmR) - TR) / np.maximum(np.abs(TR), 1e-3)
      )
    )
    if len(S)
    else 0.0
  )
  log(f'completeness: logits {comp_logit:.2e}, features {comp_feat:.2e}')

  # node set: logits, kept features, errors (layer, pos), embeddings (pos); bias folded out
  nodes = []
  for i, tid in enumerate(keep):
    nodes.append(
      dict(
        id=f'logit:{tid}',
        kind='logit',
        layer=G.L,
        pos=P - 1,
        token=t.decode([tid]),
        prob=round(float(lw[i]), 4),
      )
    )
  for j in S:
    l, p, f, a = G.active[j]
    nodes.append(
      dict(
        id=f'{l}.{p}.{f}',
        kind='feature',
        layer=l,
        pos=p,
        feature=f,
        act=round(a, 4),
      )
    )
  err_ids = [(l, p) for l in range(G.L) for p in range(P)]
  nodes.extend(
    dict(id=f'err:{l}.{p}', kind='error', layer=l, pos=p) for l, p in err_ids
  )
  for p in range(P):
    nodes.append(
      dict(
        id='bos' if p == 0 else f'emb:{p}',
        kind='embedding',
        layer=-1,
        pos=p,
        token=t.decode([ids[p]]),
      )
    )
  K, nf = len(keep), len(S)
  N = len(nodes)
  A = np.zeros((N, N), np.float32)
  A[:K, K : K + nf] = LF[:, S]
  A[:K, K + nf : K + nf + len(err_ids)] = LE.reshape(K, -1)
  A[:K, K + nf + len(err_ids) :] = LEm
  if nf:
    A[K : K + nf, K : K + nf] = FR[:, S]
    A[K : K + nf, K + nf : K + nf + len(err_ids)] = ER.reshape(nf, -1)
    A[K : K + nf, K + nf + len(err_ids) :] = EmR
  w = np.zeros(N)
  w[:K] = lw
  infl = influence(A, w)
  # prune nodes: keep sources holding node-share of the influence (logits always stay)
  src = np.arange(K, N)
  ordr = src[np.argsort(-infl[src])]
  cum = np.cumsum(infl[ordr]) / max(1e-12, infl[src].sum())
  kept_src = set(
    ordr[: int(np.searchsorted(cum, args.node_share)) + 1].tolist()
  )
  alive = [i for i in range(N) if i < K or i in kept_src]
  # prune edges by |weight| x target influence, keep edge-share
  sub = A[np.ix_(alive, alive)]
  ti = np.array([infl[i] if i >= K else w[i] for i in alive])
  sc = np.abs(sub) * ti[:, None]
  flat = np.sort(sc.ravel())[::-1]
  cut = (
    flat[
      min(
        len(flat) - 1,
        int(
          np.searchsorted(
            np.cumsum(flat) / max(1e-12, flat.sum()), args.edge_share
          )
        ),
      )
    ]
    if flat.size
    else 0
  )
  links = []
  for a_i, ia in enumerate(alive):
    for b_i, ib in enumerate(alive):
      if sub[a_i, b_i] != 0 and sc[a_i, b_i] >= cut:
        links.append(
          dict(
            source=nodes[ib]['id'],
            target=nodes[ia]['id'],
            weight=round(float(sub[a_i, b_i]), 5),
          )
        )
  for i in alive:
    nodes[i]['influence'] = round(float(infl[i] if i >= K else w[i]), 5)
  graph = dict(
    prompt=args.prompt,
    tokens=[t.decode([x]) for x in ids],
    model=MODEL,
    revision=REVISION,
    plt=str(args.plt),
    fvu={l: tc['fvu'] for l, tc in enumerate(tcs)},
    logits=[
      dict(token=t.decode([tid]), prob=round(float(probs[tid].item()), 4))
      for tid in keep
    ],
    nodes=[nodes[i] for i in alive],
    links=links,
    check=dict(
      forward=rec_err,
      replacement=exact,
      completeness_logits=float(comp_logit),
      completeness_features=comp_feat,
    ),
    seconds=round(time.time() - t0, 1),
  )
  slug = (
    re.sub(r'[^a-z0-9]+', '-', args.prompt.lower()).strip('-')[:60] or 'graph'
  )
  out = Path(args.out) if args.out else OUT / f'{slug}.json'
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(graph, ensure_ascii=False) + '\n')
  nf_alive = sum(1 for i in alive if nodes[i]['kind'] == 'feature')
  log(
    f'{len(alive)} nodes ({nf_alive} features), {len(links)} links -> {out} in {graph["seconds"]}s'
  )


if __name__ == '__main__':
  main()
