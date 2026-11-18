"""The garden body on an HF Gemma4ForCausalLM: the torch engine.

The CPU tests check the plugin's layer body here against the stock HF layer and gemma_torch.py's
hooks, and the CLT harvest runs here on a GPU: a trainer in the same process reads the taps
without an engine in between (docs/latent-interp-contract.md §4).

    garden = install(model, GardenConfig(dict_path=..., taps=('mlp.in', 'mlp.out')), n_max=16384)
    for batch in stream(model, tok, garden, rows, ctx=1024, batch_tokens=16384, keep=0.25):
        ...   # {(layer, 'in' | 'out'): (n, d) bf16 on the model's device}: clt.train's Batch
"""

from __future__ import annotations

import types
from collections.abc import Iterator

import numpy as np
import torch

from .body import garden_layer
from .state import Garden, GardenConfig, Planner, Req


def install(model, cfg: GardenConfig, n_max: int, dictionary=None) -> Garden:
  """Swap every decoder layer's forward for the garden body; returns the Garden, registered on
  the model as `model.model.garden` so it moves with .to()."""
  inner = model.model
  conf = model.config
  dev = next(model.parameters()).device
  g = Garden(
    cfg, conf.num_hidden_layers, conf.hidden_size, n_max, dev, dictionary
  )
  inner.garden = g

  def forward(self, hidden_states, per_layer_input=None, **kw):
    def attn(x):
      return self.self_attn(hidden_states=x, **kw)[0]

    return garden_layer(
      self, self.layer_idx, hidden_states, per_layer_input, attn, g
    )

  for layer in inner.layers:
    layer.forward = types.MethodType(forward, layer)
  return g


def uninstall(model) -> None:
  for layer in model.model.layers:
    layer.__dict__.pop('forward', None)
  if hasattr(model.model, 'garden'):
    del model.model.garden


def encode(tok, text: str, ctx: int) -> list[int]:
  """acts.encode: [bos] + the text's tokens, cut at ctx."""
  t = getattr(tok, '_tokenizer', tok)
  ids = t.encode(text, add_special_tokens=False)
  return [t.bos_token_id, *list(ids)[: ctx - 1]]


@torch.inference_mode()
def stream(
  model,
  tok,
  garden: Garden,
  rows: list[tuple[int, str]],
  *,
  ctx: int,
  batch_tokens: int,
  keep: float,
  seed: int,
  start: int = 0,
  meta: bool = False,
) -> Iterator:
  """acts.batches on the torch engine: the same length-bucketed windows of 256 rows, the same
  right-padded grid and the same seeded keep draw, so for one tokenizer and seed it keeps the
  same (row, pos) as the mlx cache. Yields the contract's batch dict, {(layer, 'in' | 'out'):
  (n, d)} for the layers tapped at mlp.in / mlp.out, which clt.train and clt.shuffled take as is.
  With meta=True it yields (batch, {'tok', 'row', 'pos', 'done'}) for a shard writer."""
  planner = Planner(garden)
  dev = next(model.parameters()).device
  layers = sorted({l for l, _ in garden.tap_sites})
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
      grid = np.zeros((B, L), np.int64)
      live = np.zeros((B, L), bool)
      for b, j in enumerate(group):
        ids = enc[j][1]
        grid[b, : len(ids)] = ids
        live[b, 1 : len(ids)] = True  # never bos
      pick = live & (rng.random((B, L)) < keep)
      flat = np.flatnonzero(pick.ravel())
      if B * L > garden.n_max:
        raise RuntimeError(f'batch {B}x{L} > n_max {garden.n_max}')
      # the flat grid as one request a row: the tap rows are the kept cells
      reqs = [
        Req(
          id=str(b),
          spec={'tap': np.flatnonzero(pick[b]).tolist()},
          start=0,
          n=L,
          prompt=L,
        )
        for b in range(B)
      ]
      plan = planner.plan(reqs)
      model(input_ids=torch.from_numpy(grid).to(dev), use_cache=False)
      n = plan.taps
      got = {}
      for l in layers:
        for site, part in (('mlp.in', 'in'), ('mlp.out', 'out')):
          if (l, site) in garden.tap_sites:
            got[l, part] = garden.tap_out(l, site)[:n].clone()
      if not meta:
        yield got
        continue
      bb, ll = np.divmod(flat, L)
      yield (
        got,
        dict(
          tok=grid.ravel()[flat].astype(np.uint32),
          row=np.array([enc[group[b]][0] for b in bb], np.uint32),
          pos=ll.astype(np.uint16),
          done=i + len(window) if a >= len(order) else i,
        ),
      )
    i += len(window)
