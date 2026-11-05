# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "mlx-lm", "huggingface-hub", "safetensors"]
# ///
"""Per-token activations of Gemma 4 E4B over the garden: step one past the embedding SAE.

    uv run quartz/scripts/latent/acts.py [--layers 5,11,17,23,29,35,41] [--budget-gb 20]
    uv run quartz/scripts/latent/acts.py --check            # hooked forward == mlx-lm forward
    uv run quartz/scripts/latent/acts.py --bench 64         # tokens/s over 64 rows, no writes
    uv run quartz/scripts/latent/acts.py --sae decoderesearch/gemma-4-saes:<folder> --site resid

The embedding SAE (features.py) splits one 256-d vector per passage. This reads the passages
with the same Gemma that writes the labels (label.py: MODEL @ REVISION, 4-bit, mlx), one token
at a time, and keeps what a transcoder or an attribution graph needs.

Two sites per layer l, both 2560-d:
  mlp   `in`  = pre_feedforward_layernorm(resid_mid), what the MLP reads
        `out` = post_feedforward_layernorm(mlp(in)) + the per-layer-embedding (PLE) gate's write,
                i.e. resid_post / layer_scalar - resid_mid: everything the layer adds after
                attention. A per-layer transcoder maps in -> out; folding the PLE gate into
                `out` keeps the replacement model complete (PLE is a second gated nonlinearity
                per layer, fed by a token-indexed embedding, that an MLP-only transcoder would
                leave in the error node).
  resid `post` = the layer's output (after layer_scalar), the hook of the published residual
                SAEs (decoderesearch/gemma-4-saes, layers 7/21/35).

Rows are the sidecar's canonical note and pdf chunks (protected, private and draft notes are
already out of the store), each read alone: [bos] + chunk, cut at --ctx. Position 0 (bos, an
attention sink with outlier norms) is never kept; every other token is kept with probability
--keep, the same tokens at every layer, so a cross-layer transcoder can train on the cache too.
Rows are visited in a seeded shuffle, so a cache cut short by --budget-gb is a uniform sample of
the garden, not its first folders.

Output, quartz/.quartz-cache/latent-acts/<sidecar[:8]>-<site>/:
  meta.json        model, revision, sidecar, site, layers, d, keep, ctx, seed, tokens, rows done,
                   per-layer mean norms (for scaling a transcoder's input), tokens/s
  L<ll>.<part>.bf16  tokens x d raw bfloat16 (uint16 bits; `bf16()` below decodes)
  tok.u32 row.u32 pos.u16   per kept token: token id, sidecar row, position in the row

Disk: d x 2 bytes x parts per token per layer, 10 KB for the mlp site. All 42 layers at both mlp
parts is 430 KB a token, so a full-depth cache of the garden does not fit a laptop; `stream()`
hands the same activations to a trainer batch by batch instead, and the cache is for a few
layers (sweeps, per-token feature inspection, checking a published SAE).

Readers take the token count from meta.json, not from file lengths (a run stopped by the budget
leaves its cut window's tail past it). The run resumes: meta.json is rewritten after every flush, and a restart with the same settings
truncates the files to the recorded token count and skips the rows already read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

D_SITES = {'mlp': ('in', 'out'), 'resid': ('post',)}
CACHE = Path('quartz/.quartz-cache/latent-acts')


def log(*args: object) -> None:
  print('[acts]', *args, flush=True)


def bf16(raw: np.ndarray) -> np.ndarray:
  """uint16 bfloat16 bits -> float32."""
  return (raw.astype(np.uint32) << 16).view(np.float32)


def text_model(model):
  """gemma4.Model (the multimodal checkpoint mlx-lm loads) -> its gemma4_text.Model."""
  return getattr(model, 'language_model', model)


def forward(
  lm, inputs, want: dict[int, tuple[str, ...]], logits: bool = False
):
  """Gemma 4 text forward with taps, mirroring mlx_lm.models.gemma4_text (Gemma4TextModel and
  DecoderLayer.__call__) step for step; `--check` holds it to the library's own logits.

  inputs: (B, L) int32. want: {layer: parts} with parts from 'in', 'out', 'post'.
  Returns ({(layer, part): (B, L, d)}, logits or None).
  """
  import mlx.core as mx
  import mlx.nn as nn

  m = lm.model
  n = len(m.layers)
  emb = m.embed_tokens(inputs)
  h = emb * m.embed_scale
  ple = None
  if m.hidden_size_per_layer_input:
    ple = m._project_per_layer_inputs(h, m._get_per_layer_inputs(inputs, emb))
  masks = m._make_masks(h, [None] * n)
  inter = [(None, None)] * n
  taps: dict[tuple[int, str], object] = {}
  for i, (layer, mask, prev) in enumerate(
    zip(m.layers, masks, m.previous_kvs)
  ):
    if layer.enable_moe:
      raise SystemExit(
        'acts.py taps dense Gemma 4 layers; this checkpoint has MoE blocks'
      )
    kvs, offset = inter[prev]
    a, kvs, offset = layer.self_attn(
      layer.input_layernorm(h), mask, None, shared_kv=kvs, offset=offset
    )
    mid = h + layer.post_attention_layernorm(a)
    x = layer.pre_feedforward_layernorm(mid)
    y = layer.post_feedforward_layernorm(layer.mlp(x))
    h = mid + y
    if layer.per_layer_input_gate is not None and ple is not None:
      g = nn.gelu_approx(layer.per_layer_input_gate(h)) * ple[:, :, i, :]
      g = layer.post_per_layer_input_norm(layer.per_layer_projection(g))
      h = h + g
      y = y + g
    if layer.layer_scalar is not None:
      h = h * layer.layer_scalar
    inter[i] = (kvs, offset)
    for part in want.get(i, ()):
      taps[i, part] = {'in': x, 'out': y, 'post': h}[part]
    if not logits and i >= max(want, default=-1):
      break  # nothing past the deepest tap is needed
  out = None
  if logits:
    out = (
      m.embed_tokens.as_linear(m.norm(h))
      if lm.tie_word_embeddings
      else lm.lm_head(m.norm(h))
    )
    if lm.final_logit_softcapping is not None:
      c = lm.final_logit_softcapping
      out = mx.tanh(out / c) * c
  return taps, out


def densify(lm) -> int:
  """Swap the decoder layers' 4-bit linears for bf16 copies of the same weights (the function
  is unchanged: these are the dequantised 4-bit weights, not the original bf16 ones). Batched
  prefill is compute bound, where a plain matmul beats dequantise-and-multiply. Embedding tables
  stay quantised (gathers, 2.8 B parameters of PLE). Returns bytes added."""
  import mlx.core as mx
  import mlx.nn as nn

  added = 0
  for layer in lm.model.layers:
    for parent in (layer.self_attn, layer.mlp):
      for name, child in list(parent.children().items()):
        if isinstance(child, nn.QuantizedLinear):
          w = mx.dequantize(
            child.weight,
            child.scales,
            child.get('biases'),
            child.group_size,
            child.bits,
          ).astype(child.scales.dtype)
          dense = nn.Linear(w.shape[1], w.shape[0], bias='bias' in child)
          dense.weight = w
          if 'bias' in child:
            dense.bias = child.bias
          setattr(parent, name, dense)
          added += w.nbytes
    for name in ('per_layer_input_gate', 'per_layer_projection'):
      child = getattr(layer, name, None)
      if isinstance(child, nn.QuantizedLinear):
        w = mx.dequantize(
          child.weight,
          child.scales,
          child.get('biases'),
          child.group_size,
          child.bits,
        ).astype(child.scales.dtype)
        dense = nn.Linear(w.shape[1], w.shape[0], bias=False)
        dense.weight = w
        setattr(layer, name, dense)
        added += w.nbytes
  mx.eval(lm.parameters())
  return added


def check(lm, ids: list[int]) -> float:
  """Max |logit difference| between forward() and the library forward, relative to the logit
  range, on one short row; a mismatch means mlx-lm moved under us."""
  import mlx.core as mx

  x = mx.array([ids], dtype=mx.int32)
  _, mine = forward(lm, x, {}, logits=True)
  ref = lm(x)
  diff = mx.max(
    mx.abs(mine.astype(mx.float32) - ref.astype(mx.float32))
  ).item()
  span = mx.max(mx.abs(ref.astype(mx.float32))).item()
  return diff / max(span, 1e-6)


def corpus(content: Path, directory: str, seed: int):
  """(sidecar hash, [(row, text)]): canonical note and pdf chunks of the live sidecar, shuffled."""
  from features import Texts, load_canon, load_side

  root = content / directory / 'latent'
  digest = json.loads((root / 'index.json').read_text())['hash']
  side = root / digest
  index, docs, _, doc_of, kinds = load_side(side)
  canon = load_canon(side, len(doc_of))
  texts = Texts(side, docs, index['part'], kinds)
  rows = np.flatnonzero(
    (canon == np.arange(len(doc_of))) & np.isin(kinds, ['note', 'pdf'])
  )
  rows = np.random.default_rng(seed).permutation(rows)
  out = []
  for r in rows:
    t = texts.text(int(r)).strip()
    if t:
      out.append((int(r), t))
  return digest, out


def encode(tok, text: str, ctx: int) -> list[int]:
  t = getattr(tok, '_tokenizer', tok)
  ids = t.encode(text, add_special_tokens=False)
  return [t.bos_token_id, *list(ids)[: ctx - 1]]


def batches(
  lm,
  tok,
  rows,
  want,
  ctx: int,
  batch_tokens: int,
  keep: float,
  seed: int,
  start: int = 0,
):
  """Yield (taps {(layer, part): (n, d) mx.array}, tok (n,), row (n,), pos (n,), done) per
  forward batch, n = the kept tokens and done = the rows read in full so far (it advances once a
  window's last batch lands, so a resume restarts at a window boundary). Rows are bucketed by length (windows of 256) so a batch
  pads little; right padding is invisible to the causal tokens before it."""
  import mlx.core as mx

  rng = np.random.default_rng(seed + start)
  i = start
  while i < len(rows):
    window = rows[i : i + 256]
    enc = [(r, encode(tok, t, ctx)) for r, t in window]
    order = sorted(range(len(enc)), key=lambda j: len(enc[j][1]))
    a = 0
    while a < len(order):
      group = [order[a]]
      L = len(enc[order[a]][1])
      while a + len(group) < len(order):
        nxt = len(enc[order[a + len(group)]][1])
        if (len(group) + 1) * max(L, nxt) > batch_tokens:
          break
        group.append(order[a + len(group)])
        L = max(L, nxt)
      a += len(group)
      B = len(group)
      grid = np.zeros((B, L), np.int32)
      live = np.zeros((B, L), bool)
      for b, j in enumerate(group):
        ids = enc[j][1]
        grid[b, : len(ids)] = ids
        live[b, 1 : len(ids)] = True  # never bos
      pick = live & (rng.random((B, L)) < keep)
      flat = np.flatnonzero(pick.ravel())
      taps, _ = forward(lm, mx.array(grid), want)
      idx = mx.array(flat.astype(np.int32))
      got = {k: v.reshape(B * L, -1)[idx] for k, v in taps.items()}
      mx.eval(got)
      bb, ll = np.divmod(flat, L)
      yield (
        got,
        grid.ravel()[flat].astype(np.uint32),
        np.array([enc[group[b]][0] for b in bb], np.uint32),
        ll.astype(np.uint16),
        i + len(window) if a >= len(order) else i,
      )
    i += len(window)


def stream(
  model,
  tok,
  rows,
  layers,
  site='mlp',
  ctx=1024,
  batch_tokens=8192,
  keep=1.0,
  seed=0,
):
  """For a trainer: an endless generator of {(layer, part): (n, d)} batches over the garden,
  one pass after another with a fresh token sample each pass."""
  lm = text_model(model)
  want = {l: D_SITES[site] for l in layers}
  epoch = 0
  while True:
    for got, *_ in batches(
      lm, tok, rows, want, ctx, batch_tokens, keep, seed + 7919 * epoch
    ):
      yield got
    epoch += 1


def load_sae(spec: str):
  """`repo:folder` or a local folder holding cfg.json + *.safetensors in SAELens layout
  (W_enc d x M, b_enc, W_dec M x d, b_dec, optional threshold for JumpReLU / k in cfg for TopK)."""
  from safetensors.numpy import load_file

  if ':' in spec and not Path(spec).exists():
    from huggingface_hub import snapshot_download

    repo, folder = spec.split(':', 1)
    path = (
      Path(snapshot_download(repo, allow_patterns=[f'{folder}/*'])) / folder
    )
  else:
    path = Path(spec)
  cfg = json.loads((path / 'cfg.json').read_text())
  w = {}
  for f in sorted(path.glob('*.safetensors')):
    if f.name != 'sparsity.safetensors':
      w.update(load_file(str(f)))
  return cfg, {k: v.astype(np.float32) for k, v in w.items()}


def sae_fvu(
  cfg: dict, w: dict, X: np.ndarray, chunk: int = 1024
) -> tuple[float, float]:
  """Fraction of variance unexplained and mean L0 of a residual SAE on rows X (n, d), in
  chunks (a 65k-wide code is 256 KB a row)."""
  err, l0 = 0.0, 0.0
  b_dec = w.get('b_dec', 0)
  for a in range(0, len(X), chunk):
    x = X[a : a + chunk]
    pre = (x - b_dec if cfg.get('apply_b_dec_to_input', True) else x) @ w[
      'W_enc'
    ] + w['b_enc']
    if 'threshold' in w:
      z = np.where(pre > w['threshold'], pre, 0)
    elif cfg.get('k'):
      k = int(cfg['k'])
      z = np.zeros_like(pre)
      top = np.argpartition(-pre, k, axis=1)[:, :k]
      np.put_along_axis(
        z, top, np.maximum(np.take_along_axis(pre, top, 1), 0), 1
      )
    else:
      z = np.maximum(pre, 0)
    rec = z @ w['W_dec'] + b_dec
    err += float(((x - rec) ** 2).sum())
    l0 += float((z > 0).sum())
  fvu = err / float(((X - X.mean(0)) ** 2).sum())
  return fvu, l0 / len(X)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--content', default='content')
  ap.add_argument('--directory', default='embeddings')
  ap.add_argument('--site', choices=sorted(D_SITES), default='mlp')
  ap.add_argument(
    '--layers',
    default='5,11,17,23,29,35,41',
    help="comma-separated 0-based layers, or 'all'; the default is the seven global-attention layers",
  )
  ap.add_argument(
    '--keep', type=float, default=0.25, help='share of non-bos tokens kept'
  )
  ap.add_argument('--ctx', type=int, default=1024)
  ap.add_argument('--batch-tokens', type=int, default=8192)
  ap.add_argument('--budget-gb', type=float, default=20.0)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument(
    '--flush', type=int, default=64, help='batches between flushes'
  )
  ap.add_argument(
    '--check',
    action='store_true',
    help='compare the tapped forward to mlx-lm and exit',
  )
  ap.add_argument(
    '--bench',
    type=int,
    default=0,
    help='forward this many rows, report tokens/s, write nothing',
  )
  ap.add_argument(
    '--sae',
    default='',
    help='repo:folder of a residual SAE to score on the cached tokens',
  )
  ap.add_argument(
    '--quantised',
    action='store_true',
    help='keep the 4-bit linears (default: bf16 copies of the same weights, faster prefill)',
  )
  ap.add_argument('--out', default='')
  args = ap.parse_args()

  from label import MODEL, REVISION, resolve
  from mlx_lm import load

  import mlx.core as mx

  path = resolve(MODEL, REVISION)
  model, tok = load(path)
  lm = text_model(model)
  n_layers = len(lm.model.layers)
  d = lm.args.hidden_size
  layers = (
    list(range(n_layers))
    if args.layers == 'all'
    else [int(x) for x in args.layers.split(',') if x]
  )
  if any(not 0 <= l < n_layers for l in layers):
    raise SystemExit(f'layers run 0..{n_layers - 1}')
  parts = D_SITES[args.site]
  want = {l: parts for l in layers}

  digest, rows = corpus(
    Path(args.content).resolve(), args.directory, args.seed
  )
  log(f'{len(rows)} rows from sidecar {digest[:8]}, {n_layers} layers, d {d}')

  probe = [encode(tok, t, 64) for _, t in rows[:3]]
  if args.check:
    # the 4-bit library forward is the reference for both the taps and the dense copies
    refs = []
    for ids in probe:
      x = mx.array([ids], dtype=mx.int32)
      refs.append(lm(x).astype(mx.float32))
      rel = check(lm, ids)
      log(f'4-bit: max |dlogit| / max |logit| = {rel:.2e}')
      if rel > 1e-2:
        raise SystemExit('tapped forward drifted from mlx-lm')
  if not args.quantised:
    log(f'dense layers: +{densify(lm) / 1e9:.1f} GB')
  if args.check:
    if not args.quantised:
      for ids, ref in zip(probe, refs):
        _, mine = forward(lm, mx.array([ids], dtype=mx.int32), {}, logits=True)
        rel = (
          mx.max(mx.abs(mine.astype(mx.float32) - ref)) / mx.max(mx.abs(ref))
        ).item()
        top = (
          (mx.argmax(mine, -1) == mx.argmax(ref, -1))
          .astype(mx.float32)
          .mean()
          .item()
        )
        log(
          f'dense vs 4-bit: max |dlogit| / max |logit| = {rel:.2e}, top-1 agreement {top:.1%}'
        )
    log('tapped forward matches mlx-lm')
    return

  if args.bench:
    t0, n = time.time(), 0
    for got, toks, *_ in batches(
      lm,
      tok,
      rows[: args.bench],
      want,
      args.ctx,
      args.batch_tokens,
      1.0,
      args.seed,
    ):
      n += len(toks)
    dt = time.time() - t0
    lens = [len(encode(tok, t, 10**9)) for _, t in rows[: args.bench]]
    log(
      f'{n} tokens in {dt:.1f}s = {n / dt:.0f} tokens/s through layer {max(layers)};'
      f' mean row {np.mean(lens):.0f} tokens, {np.mean(np.array(lens) > args.ctx):.1%} cut at ctx'
    )
    return

  out = Path(args.out) if args.out else CACHE / f'{digest[:8]}-{args.site}'
  out.mkdir(parents=True, exist_ok=True)
  conf = dict(
    model=MODEL,
    revision=REVISION,
    sidecar=digest,
    site=args.site,
    layers=layers,
    parts=list(parts),
    d=d,
    keep=args.keep,
    ctx=args.ctx,
    seed=args.seed,
  )
  per_token = d * 2 * len(parts) * len(layers) + 4 + 4 + 2
  cap = int(args.budget_gb * 1e9 / per_token)
  meta_path = out / 'meta.json'
  meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
  if meta and any(meta.get(k) != v for k, v in conf.items()):
    raise SystemExit(
      f'{out} holds a cache with other settings; pass --out or remove it'
    )
  tokens, done = int(meta.get('tokens', 0)), int(meta.get('rows', 0))
  sums = meta.get('norms') or {f'{l}.{p}': 0.0 for l in layers for p in parts}
  sums = {k: v * tokens for k, v in sums.items()}  # stored as means
  files = {(l, p): out / f'L{l:02d}.{p}.bf16' for l in layers for p in parts}
  side_files = {
    'tok': (out / 'tok.u32', 4),
    'row': (out / 'row.u32', 4),
    'pos': (out / 'pos.u16', 2),
  }
  for (l, p), f in files.items():
    with open(f, 'ab') as fh:
      fh.truncate(tokens * d * 2)
  for f, size in side_files.values():
    with open(f, 'ab') as fh:
      fh.truncate(tokens * size)
  if done >= len(rows):
    log(f'{out}: {tokens} tokens already, nothing to do')
  else:
    log(
      f'{out}: resuming at row {done}, {tokens} tokens; cap {cap} tokens ({args.budget_gb} GB)'
    )
  handles = {k: open(f, 'ab') for k, f in files.items()}
  sides = {k: open(f, 'ab') for k, (f, _) in side_files.items()}
  t0, n0 = time.time(), tokens
  committed = tokens  # tokens up to the last whole window: what a resume keeps

  def flush() -> None:
    for fh in (*handles.values(), *sides.values()):
      fh.flush()
      os.fsync(fh.fileno())
    rate = (tokens - n0) / max(1e-9, time.time() - t0)
    meta_path.write_text(
      json.dumps(
        dict(
          conf,
          tokens=committed,
          rows=done,
          of=len(rows),
          tokensPerSecond=round(rate),
          norms={k: v / max(1, tokens) for k, v in sums.items()},
          bytes=tokens * per_token,
          at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        ),
        indent=1,
      )
      + '\n'
    )
    log(
      f'{tokens}/{cap} tokens, row {done}/{len(rows)}, {rate:.0f} kept tokens/s'
    )

  k = 0
  if tokens < cap:
    for got, toks, rws, pos, upto in batches(
      lm,
      tok,
      rows,
      want,
      args.ctx,
      args.batch_tokens,
      args.keep,
      args.seed,
      start=done,
    ):
      take = min(len(toks), cap - tokens)
      for (l, p), v in got.items():
        v = v[:take]
        handles[l, p].write(
          np.array(mx.view(v.astype(mx.bfloat16), mx.uint16)).tobytes()
        )
        sums[f'{l}.{p}'] += float(
          mx.sum(mx.sqrt(mx.sum(v.astype(mx.float32) ** 2, axis=-1))).item()
        )
      sides['tok'].write(toks[:take].tobytes())
      sides['row'].write(rws[:take].tobytes())
      sides['pos'].write(pos[:take].tobytes())
      tokens += take
      if upto > done:
        done, committed = upto, tokens
      k += 1
      if k % args.flush == 0 or tokens >= cap:
        flush()
      if tokens >= cap:
        break  # meta keeps the last whole window; a rerun redoes the cut one, a bigger budget goes on
    flush()
  for fh in (*handles.values(), *sides.values()):
    fh.close()

  if args.sae:
    if args.site != 'resid' or len(layers) != 1:
      raise SystemExit(
        '--sae scores one residual layer: pass --site resid --layers <l>'
      )
    cfg, w = load_sae(args.sae)
    l = layers[0]
    raw = np.memmap(files[l, 'post'], dtype=np.uint16, mode='r').reshape(-1, d)
    pick = np.random.default_rng(0).choice(
      len(raw), size=min(len(raw), 8192), replace=False
    )
    fvu, l0 = sae_fvu(cfg, w, bf16(np.asarray(raw[np.sort(pick)])))
    log(
      f'{args.sae} on layer {l}: fvu {fvu:.3f}, L0 {l0:.0f} over {len(pick)} garden tokens'
    )


if __name__ == '__main__':
  main()
