"""CPU checks for garden_vllm on a random 6-layer Gemma 4 (gemma_torch.tiny: the real layer
pattern with sliding/global layers, KV sharing and PLE). No vLLM, no GPU, no checkpoint.

    uv run --with torch==2.13.0 --with transformers==5.10.1 --with safetensors --with pytest \
      pytest quartz/scripts/latent/garden_vllm/test_garden_vllm.py -q

What they pin down:
1. the garden body with nothing armed is the stock HF layer, bit for bit;
2. readout codes equal gemma_torch.encode at the same site;
3. a steer through the bank equals gemma_torch.Taps' steer hook;
4. the taps are the contract's sites (mlp.out = resid_post / layer_scalar - mid);
5. the planner's slots, tap rows and records for a mixed prefill + decode batch;
6. a compiled forward reads buffer contents at run time (the graph-safety property);
7. a clt.py checkpoint loads: readout equals clt.encode, steer rows equal its decoders.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # gemma_torch, clt
sys.path.insert(0, str(HERE.parent))

import gemma_torch as gt
from garden_vllm import hf
from garden_vllm.dicts import Dictionary
from garden_vllm.state import Garden, GardenConfig, Planner, Req, sink_for


def tiny():
  model, _, tcs = gt.tiny(None)
  model = model.float()
  return model, tcs


def write_plt(root: Path, tcs: dict, labels: dict | None = None) -> Path:
  """gemma_torch.tiny's random transcoders as a plt/v1 directory."""
  from safetensors.torch import save_file

  root.mkdir(parents=True, exist_ok=True)
  for l, tc in tcs.items():
    d, M = tc['W_enc'].shape
    save_file(
      dict(
        W_enc=tc['W_enc'],
        b_enc=tc['b_enc'].float() + 0.1 * torch.randn(M),
        W_dec=tc['W_dec'],
        W_skip=torch.zeros(d, d, dtype=torch.bfloat16),
        b_pre=0.1 * torch.randn(d),
        b_dec=torch.zeros(d),
        sx=torch.tensor([1.5]),
        sy=torch.tensor([0.5]),
      ),
      str(root / f'L{l:02d}.safetensors'),
    )
  import json

  (root / 'manifest.json').write_text(
    json.dumps(
      dict(arch=dict(kind='plt', d=d, M=M, k=4), engine='torch/bf16-dequant')
    )
  )
  if labels:
    (root / 'labels.json').write_text(json.dumps(labels))
  return root


def ids_of(n=12, seed=1):
  return torch.randint(
    0, 200, (1, n), generator=torch.Generator().manual_seed(seed)
  )


def test_body_is_the_stock_layer():
  model, _ = tiny()
  ids = ids_of()
  ref = model(input_ids=ids, use_cache=False).logits
  hf.install(
    model, GardenConfig(), n_max=64
  )  # no dictionary: no readout, steer or taps
  got = model(input_ids=ids, use_cache=False).logits
  assert torch.equal(ref, got)
  hf.uninstall(model)
  assert torch.equal(model(input_ids=ids, use_cache=False).logits, ref)


def test_readout_matches_gemma_torch(tmp_path):
  model, tcs = tiny()
  root = write_plt(tmp_path / 'plt', tcs)
  D = Dictionary(root)
  layers = (1, 5)
  ids = ids_of()
  # gemma_torch's hook at pre_feedforward_layernorm, last position, its encode()
  taps = gt.Taps(
    model, {l: gt.load_transcoders(root, [l], 4)[l] for l in layers}
  )
  taps.arm(layers, {}, None)
  model(input_ids=ids, use_cache=False)
  want = {l: gt.encode(taps.tcs[l], taps.x[l], 4) for l in layers}
  taps.disarm()
  taps.close()
  g = hf.install(
    model, GardenConfig(readout=layers, top=4), n_max=64, dictionary=D
  )
  model(input_ids=ids, use_cache=False)
  n = ids.shape[1]
  for i, l in enumerate(layers):
    got = [
      (int(f), round(float(v), 3))
      for f, v in zip(g.ro_ids[i, n - 1], g.ro_vals[i, n - 1].float())
      if v > 0
    ]
    assert [f for f, _ in got] == [f for f, _ in want[l]], (l, got, want[l])
    assert np.allclose(
      [v for _, v in got], [v for _, v in want[l]], rtol=2e-3, atol=2e-3
    )


def test_steer_matches_gemma_torch(tmp_path):
  model, tcs = tiny()
  labels = {'L03': {'2': {'top': [[0, 0, 4.0]]}}}
  root = write_plt(tmp_path / 'plt', tcs, labels)
  D = Dictionary(root)
  spec = [dict(layer=3, id=2, s=1.5)]
  ids = ids_of()
  # gemma_torch: v = s * a_max * W_dec[f] * sy, added to the layer output times layer_scalar
  tc = gt.load_transcoders(root, [3], 4)[3]
  v = 1.5 * gt.a_max(labels, 3, 2) * tc['W_dec'][2].float() * tc['sy']
  taps = gt.Taps(model, {})
  taps.arm([], {3: v}, None)
  want = model(input_ids=ids, use_cache=False).logits
  taps.disarm()
  plain = model(input_ids=ids, use_cache=False).logits
  taps.close()
  assert not torch.allclose(want, plain)

  g = hf.install(
    model, GardenConfig(readout=(), steer_slots=2), n_max=64, dictionary=D
  )
  vecs = D.steer_vectors(spec)
  assert set(vecs) == {3} and torch.allclose(vecs[3], v, atol=1e-6)
  pl = Planner(g)
  pl.plan([
    Req('a', {'steer': spec}, start=0, n=ids.shape[1], prompt=ids.shape[1])
  ])
  got = model(input_ids=ids, use_cache=False).logits
  assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
  pl.plan(
    [Req('b', {}, start=0, n=ids.shape[1], prompt=ids.shape[1])],
    finished=['a'],
  )
  assert torch.equal(model(input_ids=ids, use_cache=False).logits, plain)
  assert pl.free == [2, 1] or sorted(pl.free) == [1, 2]


def test_taps_are_the_contract_sites(tmp_path):
  model, tcs = tiny()
  root = write_plt(tmp_path / 'plt', tcs)
  ids = ids_of()
  n = ids.shape[1]
  ref = {}
  hooks = []
  for l, layer in enumerate(model.model.layers):
    hooks.append(
      layer.pre_feedforward_layernorm.register_forward_hook(
        lambda m, a, o, l=l: ref.__setitem__((l, 'in'), o[0].clone())
      )
    )
    hooks.append(
      layer.pre_feedforward_layernorm.register_forward_pre_hook(
        lambda m, a, l=l: ref.__setitem__((l, 'mid'), a[0][0].clone())
      )
    )
    hooks.append(
      layer.register_forward_hook(
        lambda m, a, o, l=l: ref.__setitem__((l, 'post'), o[0].clone())
      )
    )
  model(input_ids=ids, use_cache=False)
  for h in hooks:
    h.remove()
  cfg = GardenConfig(
    readout=(),
    steer_slots=0,
    taps=('mlp.in', 'mlp.out', 'resid.mid', 'resid.post'),
  )
  g = hf.install(model, cfg, n_max=64, dictionary=Dictionary(root))
  pl = Planner(g)
  pl.plan([Req('h', {'tap': 'all'}, start=0, n=n, prompt=n)])
  model(input_ids=ids, use_cache=False)
  rows = slice(1, n)  # bos dropped
  for l, layer in enumerate(model.model.layers):
    scalar = float(layer.layer_scalar)
    assert torch.allclose(
      g.tap_out(l, 'mlp.in')[: n - 1].float(),
      ref[l, 'in'][rows],
      atol=2e-2,
      rtol=8e-3,
    )
    assert torch.allclose(
      g.tap_out(l, 'resid.mid')[: n - 1].float(),
      ref[l, 'mid'][rows],
      atol=2e-2,
      rtol=8e-3,
    )
    assert torch.allclose(
      g.tap_out(l, 'resid.post')[: n - 1].float(),
      ref[l, 'post'][rows],
      atol=2e-2,
      rtol=8e-3,
    )
    out = ref[l, 'post'][rows] / scalar - ref[l, 'mid'][rows]
    assert torch.allclose(
      g.tap_out(l, 'mlp.out')[: n - 1].float(), out, atol=3e-2, rtol=1e-2
    )


def test_planner_mixed_batch(tmp_path):
  _, tcs = tiny()
  root = write_plt(
    tmp_path / 'plt', tcs, {'L01': {'0': {'top': [[0, 0, 2.0]]}}}
  )
  D = Dictionary(root)
  sink = str(tmp_path / 'sink')
  g = Garden(
    GardenConfig(
      readout=(1, 5),
      top=2,
      steer_slots=2,
      taps=('mlp.in',),
      tap_layers=(1,),
      sink=sink,
    ),
    L=6,
    d=64,
    n_max=32,
    dictionary=D,
  )
  pl = Planner(g)
  # a: prefill chunk of a 10-token prompt (positions 4..9), steered from decode, readout decode
  # b: one decode token at position 20, steered everywhere, readout all
  # c: unarmed, 3 tokens; d: harvest of positions 2 and 5 in a 6-token prompt
  reqs = [
    Req(
      'a',
      {
        'tag': 'A',
        'steer': [dict(layer=1, id=0, s=1)],
        'steer_from': 'decode',
        'readout': 'decode',
      },
      4,
      6,
      10,
    ),
    Req(
      'b',
      {'tag': 'B', 'steer': [dict(layer=1, id=0, s=-1)], 'readout': 'all'},
      20,
      1,
      15,
    ),
    Req('c', None, 0, 3, 3),
    Req('d', {'tag': 'D', 'tap': [2, 5, 9]}, 0, 6, 6),
  ]
  p = pl.plan(reqs)
  assert p.n == 16 and p.taps == 2
  slot = g.slot[:16].tolist()
  sa, sb = pl.slots['a'], pl.slots['b']
  assert sa != sb
  assert slot[:6] == [0, 0, 0, 0, 0, sa]  # only position 9 = prompt - 1 of a
  assert slot[6] == sb and slot[7:] == [0] * 9
  assert torch.allclose(
    g.bank[1, sa], 2.0 * D.writes(1, 0)[1]
  ) and torch.allclose(g.bank[1, sb], -g.bank[1, sa])
  assert g.bank[0].abs().sum() == 0 and g.bank[:, 0].abs().sum() == 0
  assert g.tap_rows[:2].tolist() == [10 + 2, 10 + 5]
  # fake a forward: codes equal to the row index, taps equal to their row
  g.ro_ids[:, :16] = torch.arange(16, dtype=torch.int32)[None, :, None]
  g.ro_vals[:, :16] = 1.0
  g.tap_out(1, 'mlp.in')[:2] = torch.tensor([[12.0], [15.0]]).to(
    torch.bfloat16
  )
  pl.collect(p)
  import json

  a = [
    json.loads(x) for x in (Path(sink) / 'A.ndjson').read_text().splitlines()
  ]
  assert [r[0] for r in a[0]['readout']] == [9]  # prompt - 1 onward
  assert a[0]['readout'][0][1][0] == [1, 5, 1.0]  # layer 1, row 5 of the batch
  b = [
    json.loads(x) for x in (Path(sink) / 'B.ndjson').read_text().splitlines()
  ]
  assert [r[0] for r in b[0]['readout']] == [20] and b[0]['readout'][0][1][0][
    1
  ] == 6
  dl = [
    json.loads(x) for x in (Path(sink) / 'D.ndjson').read_text().splitlines()
  ]
  z = np.load(Path(sink) / dl[0]['npz'])
  assert z['pos'].tolist() == [2, 5]
  assert not (Path(sink) / 'c.ndjson').exists()
  # next step: a finished, its slot comes back; b steps on with the same slot
  p2 = pl.plan(
    [Req('b', {'tag': 'B', 'steer': [dict(layer=1, id=0, s=-1)]}, 21, 1, 15)],
    finished=['a'],
  )
  assert (
    pl.slots == {'b': sb} and sa in pl.free and g.slot[0] == sb and p2.n == 1
  )


def test_compiled_graph_reads_buffers(tmp_path):
  """What a CUDA graph needs: one trace, behaviour from buffer contents. aot_eager runs the ops
  through dynamo + functionalisation as vLLM's compile does, without a C compiler."""
  model, tcs = tiny()
  model.requires_grad_(False)
  root = write_plt(tmp_path / 'plt', tcs)
  g = hf.install(
    model,
    GardenConfig(readout=(5,), top=2, steer_slots=1),
    n_max=16,
    dictionary=Dictionary(root),
  )
  ids = ids_of(n=16)
  pl = Planner(g)
  eager_plain = model(input_ids=ids, use_cache=False).logits
  pl.plan([Req('s', {'steer': [dict(layer=2, id=1, s=3)]}, 0, 16, 16)])
  eager_steer = model(input_ids=ids, use_cache=False).logits
  ro_eager = g.ro_ids.clone()
  torch._dynamo.reset()
  f = torch.compile(
    lambda x: model(input_ids=x, use_cache=False).logits,
    backend='aot_eager',
    fullgraph=True,
  )
  g.ro_ids.zero_()
  assert torch.allclose(f(ids), eager_steer, atol=1e-5)
  assert torch.equal(
    g.ro_ids, ro_eager
  )  # the compiled readout wrote the buffer
  pl.plan(
    [Req('t', {}, 0, 16, 16)], finished=['s']
  )  # same shapes, slots now all zero
  assert torch.allclose(f(ids), eager_plain, atol=1e-5)


def test_clt_checkpoint(tmp_path):
  clt = pytest.importorskip('clt')
  L, d, M, k, S = 6, 64, 32, 4, 3
  m = clt.CLT(L, d, M, k, S)
  with torch.no_grad():
    for p in [m.W_enc, m.b_enc, *m.W_dec, m.b_dec]:
      p.normal_()
    m.sx.uniform_(0.5, 2)
    m.sy.uniform_(0.5, 2)
    m.b_pre.normal_()
  out = tmp_path / 'clt'
  clt.save(m, out, dict(engine='torch/bf16-dequant'))
  D = Dictionary(out)
  assert (D.kind, D.span, D.L, D.M, D.k) == ('clt', S, L, M, k)
  # steering rows: every target of the source, in model units
  w = D.writes(4, 7)
  assert sorted(w) == [4, 5]  # S_4 = min(3, 6 - 4)
  for t, v in w.items():
    ref = m.W_dec[4][7, t - 4].to(torch.bfloat16).float() * float(m.sy[t])
    assert torch.allclose(v, ref, rtol=1e-3, atol=1e-5)
  # readout: the same top codes as clt.encode on the same input
  x = torch.randn(5, d)
  g = Garden(
    GardenConfig(readout=(2,), top=3, readout_dtype='float32'),
    L,
    d,
    8,
    dictionary=D,
  )
  from garden_vllm import ops

  ops.readout(
    x, g.W_enc[0], g.b_enc[0], g.b_pre[0], g.sx[0], g.ro_ids[0], g.ro_vals[0]
  )
  xc = x / m.sx[2] - m.b_pre[2]
  W = m.W_enc[2].to(torch.bfloat16).float()  # what the file holds
  z = torch.relu((xc @ W) + m.b_enc[2])
  top = torch.topk(z, 3, dim=-1)
  assert torch.equal(g.ro_ids[0, :5].long(), top.indices)
  assert torch.allclose(
    g.ro_vals[0, :5].float(), top.values, rtol=2e-3, atol=2e-3
  )


def test_sink_memory():
  s = sink_for('memory')
  q = s.open('x')
  s.put('x', {'step': 1, 'readout': []})
  assert q.get_nowait()['step'] == 1
  s.close('x')


class _Tok:
  """Byte-level stand-in for the Gemma tokenizer: enough for acts.encode's two calls."""

  bos_token_id = 2

  def encode(self, text, add_special_tokens=False):
    return [3 + b % 190 for b in text.encode()]


def test_stream_feeds_clt(tmp_path):
  """The torch-engine harvest end to end: hf.stream -> clt.shuffled -> clt.train -> clt/v1 ->
  this plugin's Dictionary. The handshake with the circuits thread's trainer."""
  clt = pytest.importorskip('clt')
  model, _ = tiny()
  model.requires_grad_(False)
  L, d = model.config.num_hidden_layers, model.config.hidden_size
  g = hf.install(
    model,
    GardenConfig(readout=(), steer_slots=0, taps=('mlp.in', 'mlp.out')),
    n_max=96,
  )
  rows = [(i, 'the garden reads ' * (1 + i % 4)) for i in range(40)]
  got = list(
    hf.stream(
      model,
      _Tok(),
      g,
      rows,
      ctx=48,
      batch_tokens=96,
      keep=0.5,
      seed=0,
      meta=True,
    )
  )
  assert got and all(
    set(b) == {(l, p) for l in range(L) for p in ('in', 'out')} for b, _ in got
  )
  n = sum(m['tok'].size for _, m in got)
  assert n == sum(b[0, 'in'].shape[0] for b, _ in got) and 0 < n
  assert all((m['pos'] >= 1).all() for _, m in got)  # bos never kept
  # the same draw as acts.batches: a second run is identical
  again = list(
    hf.stream(
      model,
      _Tok(),
      g,
      rows,
      ctx=48,
      batch_tokens=96,
      keep=0.5,
      seed=0,
      meta=True,
    )
  )
  assert all(
    np.array_equal(a['pos'], b['pos']) and np.array_equal(a['row'], b['row'])
    for (_, a), (_, b) in zip(got, again)
  )
  batches = (b for b, _ in got * 20)
  m = clt.CLT(L, d, 32, 4, 2)
  out = tmp_path / 'clt'
  clt.train(
    m,
    clt.shuffled(batches, 512, 128),
    out,
    steps=3,
    fit_tokens=256,
    info=dict(engine='torch/bf16-dequant'),
  )
  D = Dictionary(out)
  assert (D.kind, D.span, D.engine) == ('clt', 2, 'torch/bf16-dequant')
  assert sorted(D.steer_vectors([dict(layer=0, id=1, s=1.0)])) == [0, 1]
