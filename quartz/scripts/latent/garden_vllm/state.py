"""Device buffers the ops read and write, and the host side that fills them per step.

Two halves, so the host half tests without a GPU or vLLM:

- `Garden` (an nn.Module, registered on the model): readout encoders and code buffers, the
  steering bank and per-token slots, tap rows and tap outputs. Shapes are fixed at load from
  `n_max` (vLLM's max_num_batched_tokens), so CUDA graphs see stable addresses.
- `Planner`: given the step's requests in batch order (id, garden spec, first position, token
  count, prompt length), writes slots and tap rows, keeps the steering bank in step with the
  live requests, and after the forward turns the code and tap buffers into per-request records.

A request asks for interp through `SamplingParams(extra_args={'garden': spec})`:

    {'tag': 'r1',                          # sink key (vLLM may suffix request ids)
     'readout': 'decode' | 'all' | False,  # positions to report: from the last prompt token, or all but bos
     'steer': [{'layer': 21, 'id': 837, 's': 2.0}],
     'steer_from': 'all' | 'decode' | <position>,   # 'all' = gemma_torch.py semantics
     'tap': [positions] | 'all'}           # harvest: rows of every configured tap site

Readout at position p is the forward that predicts token p+1, so generated token i (0-based)
pairs with position prompt_len - 1 + i, as gemma_torch.read pairs a token with the forward that
wrote it.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import ops
from .dicts import Dictionary

SITES = ('mlp.in', 'mlp.out', 'resid.mid', 'resid.post')


@dataclass
class GardenConfig:
  dict_path: str | None = None
  readout: tuple[int, ...] = (7, 21, 35)
  top: int = 4  # r: 4 = the PLTB bake row
  readout_dtype: str = 'bfloat16'  # 'float32' for the contract's fp32 encode
  steer_slots: int = 8  # concurrent steered requests
  taps: tuple[
    str, ...
  ] = ()  # SITES, e.g. ('mlp.in', 'mlp.out') for a CLT harvest
  tap_layers: tuple[int, ...] | None = None  # None = every layer
  sink: str = (
    'memory'  # or a directory: one ndjson file a tag (multi-process engines)
  )

  @classmethod
  def from_dict(cls, d: dict | None) -> 'GardenConfig':
    d = dict(d or {})
    for k in ('readout', 'taps', 'tap_layers'):
      if d.get(k) is not None:
        d[k] = tuple(d[k])
    return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Garden(torch.nn.Module):
  def __init__(
    self,
    cfg: GardenConfig,
    L: int,
    d: int,
    n_max: int,
    device=None,
    dictionary: Dictionary | None = None,
  ):
    super().__init__()
    self.cfg, self.L, self.d, self.n_max = cfg, L, d, n_max
    self.dict = dictionary or (
      Dictionary(cfg.dict_path) if cfg.dict_path else None
    )

    def buf(name, t):
      self.register_buffer(name, t.to(device), persistent=False)

    self.readout_layers = tuple(cfg.readout) if self.dict else ()
    self.ro_index = {l: i for i, l in enumerate(self.readout_layers)}
    self.sx: list[float] = []
    if self.readout_layers:
      dt = getattr(torch, cfg.readout_dtype)
      encs = [self.dict.encoder(l, dt) for l in self.readout_layers]
      buf('W_enc', torch.stack([e['W_enc'] for e in encs]))
      buf('b_enc', torch.stack([e['b_enc'] for e in encs]))
      buf('b_pre', torch.stack([e['b_pre'] for e in encs]))
      self.sx = [e['sx'] for e in encs]
      R = len(encs)
      buf('ro_ids', torch.zeros(R, n_max, cfg.top, dtype=torch.int32))
      buf('ro_vals', torch.zeros(R, n_max, cfg.top, dtype=torch.float16))

    self.steering = bool(self.dict) and cfg.steer_slots > 0
    if self.steering:
      buf('bank', torch.zeros(L, cfg.steer_slots + 1, d))
      buf('slot', torch.zeros(n_max, dtype=torch.long))

    layers = cfg.tap_layers if cfg.tap_layers is not None else range(L)
    self.tap_sites = {(l, s) for l in layers for s in cfg.taps}
    for s in cfg.taps:
      if s not in SITES:
        raise ValueError(f'unknown tap site {s}; want one of {SITES}')
    if self.tap_sites:
      buf('tap_rows', torch.zeros(n_max, dtype=torch.long))
      for l, s in sorted(self.tap_sites):
        buf(self._tap_name(l, s), torch.zeros(n_max, d, dtype=torch.bfloat16))

  @staticmethod
  def _tap_name(l: int, site: str) -> str:
    return f'tap_L{l:02d}_' + site.replace('.', '_')

  def tap_out(self, l: int, site: str) -> torch.Tensor:
    return getattr(self, self._tap_name(l, site))

  # --- called inside the forward (graph-safe: no host reads, no data-dependent shapes) -------

  def _tap(self, l: int, site: str, t: torch.Tensor) -> None:
    if (l, site) in self.tap_sites:
      ops.tap(t, self.tap_rows, self.tap_out(l, site))

  def layer(self, l, mid, x, y, g, out, scalar):
    """The garden part of decoder layer l, on flat (n, d) tensors: returns the new `out`
    (mid + y + g, before layer_scalar). Python branches here depend only on load-time config,
    so a traced graph has the same ops every step."""
    i = self.ro_index.get(l)
    if i is not None:
      ops.readout(
        x,
        self.W_enc[i],
        self.b_enc[i],
        self.b_pre[i],
        self.sx[i],
        self.ro_ids[i],
        self.ro_vals[i],
      )
    self._tap(l, 'mlp.in', x)
    self._tap(l, 'resid.mid', mid)
    if self.steering:
      out = ops.steer(out, self.bank[l], self.slot)
    if (l, 'mlp.out') in self.tap_sites:
      write = y if g is None else y + g
      if self.steering:  # what the layer really adds: its write plus the steer
        write = write + self.bank[l].index_select(
          0, self.slot[: out.shape[0]]
        ).to(write.dtype)
      self._tap(l, 'mlp.out', write)
    if (l, 'resid.post') in self.tap_sites:
      self._tap(l, 'resid.post', out * scalar)
    return out


# --- host side -------------------------------------------------------------------------------


@dataclass
class Req:
  id: str
  spec: dict | None
  start: int  # position of the step's first token (num_computed_tokens)
  n: int  # tokens scheduled this step
  prompt: int  # prompt length


@dataclass
class Entry:
  tag: str
  offset: int  # first row in the flat batch
  start: int
  n: int
  readout_from: int | None  # first reported position, None = no readout
  tap: np.ndarray | None = None  # positions captured this step
  tap_at: int = 0  # their first row in the tap buffers


@dataclass
class Plan:
  n: int
  entries: list[Entry] = field(default_factory=list)
  taps: int = 0


def _from(spec_value, prompt: int, default: int | None) -> int | None:
  if spec_value is None:
    return default
  if spec_value is False:
    return None
  if spec_value == 'all' or spec_value is True:
    return 1
  if spec_value == 'decode':
    return max(prompt - 1, 1)
  return int(spec_value)


def _np(v) -> np.ndarray:
  """bf16 as its uint16 bits (acts.py's cache convention), anything else as is."""
  if isinstance(v, torch.Tensor):
    v = v.detach().cpu()
    return (
      v.view(torch.int16).numpy().view(np.uint16)
      if v.dtype == torch.bfloat16
      else v.numpy()
    )
  return np.asarray(v)


class Sink:
  """Per-tag queues of records, for an engine in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0);
  with a directory, one ndjson file a tag instead, for a server in another process to tail."""

  def __init__(self, where: str = 'memory'):
    self.dir = None if where == 'memory' else Path(where)
    if self.dir:
      self.dir.mkdir(parents=True, exist_ok=True)
    self.q: dict[str, queue.SimpleQueue] = {}
    self.lock = threading.Lock()

  def open(self, tag: str) -> queue.SimpleQueue:
    with self.lock:
      return self.q.setdefault(tag, queue.SimpleQueue())

  def put(self, tag: str, item: dict) -> None:
    if self.dir:
      tensors = {
        k: v
        for k, v in item.items()
        if isinstance(v, (torch.Tensor, np.ndarray))
      }
      if tensors:  # taps are big: an npz beside the line
        name = f'{tag}.{item.get("step", 0)}.npz'
        np.savez(self.dir / name, **{k: _np(v) for k, v in tensors.items()})
        item = {k: v for k, v in item.items() if k not in tensors} | {
          'npz': name
        }
      with (self.dir / f'{tag}.ndjson').open('a') as f:
        f.write(json.dumps(item) + '\n')
      return
    self.open(tag).put(item)

  def close(self, tag: str) -> None:
    with self.lock:
      self.q.pop(tag, None)


SINKS: dict[str, Sink] = {}


def sink_for(where: str) -> Sink:
  if where not in SINKS:
    SINKS[where] = Sink(where)
  return SINKS[where]


class Planner:
  def __init__(self, garden: Garden):
    self.g = garden
    self.sink = sink_for(garden.cfg.sink)
    self.slots: dict[str, int] = {}  # request id -> bank row
    self.free = (
      list(range(garden.cfg.steer_slots, 0, -1)) if garden.steering else []
    )
    self.step = 0
    self._slot = np.zeros(garden.n_max, np.int64)
    self._rows = np.zeros(garden.n_max, np.int64)

  def release(self, rid: str) -> None:
    s = self.slots.pop(rid, None)
    if s is not None:
      self.free.append(s)

  def _slot_for(self, r: Req) -> int:
    if r.id in self.slots:
      return self.slots[r.id]
    if not self.free:
      raise RuntimeError(
        'garden: every steering slot is taken; raise steer_slots'
      )
    vecs = self.g.dict.steer_vectors(r.spec['steer'])
    s = self.free.pop()
    with torch.no_grad():
      self.g.bank[:, s].zero_()
      for t, v in vecs.items():
        self.g.bank[t, s] = v.to(self.g.bank.device)
    self.slots[r.id] = s
    return s

  def plan(self, reqs: list[Req], finished=()) -> Plan:
    """Fill the slot and tap-row buffers for this step's flat batch (requests in batch order,
    each request's tokens contiguous). Returns what collect() needs."""
    for rid in finished:
      self.release(rid)
    g, p = self.g, Plan(n=sum(r.n for r in reqs))
    if p.n > g.n_max:
      raise RuntimeError(f'garden: {p.n} tokens > n_max {g.n_max}')
    self._slot[: p.n] = 0
    off = 0
    for r in reqs:
      spec = r.spec or {}
      pos = np.arange(r.start, r.start + r.n)
      if g.steering and spec.get('steer'):
        s = self._slot_for(r)
        on = pos >= _from(spec.get('steer_from'), r.prompt, 0)
        self._slot[off : off + r.n][on] = s
      e = Entry(
        tag=str(spec.get('tag') or r.id),
        offset=off,
        start=r.start,
        n=r.n,
        readout_from=_from(spec.get('readout'), r.prompt, None)
        if g.readout_layers
        else None,
      )
      want = spec.get('tap')
      if g.tap_sites and want is not None:
        sel = (
          pos[pos >= 1]
          if want == 'all'
          else np.intersect1d(pos, np.asarray(want, np.int64))
        )
        if len(sel):
          e.tap, e.tap_at = sel, p.taps
          self._rows[p.taps : p.taps + len(sel)] = off + (sel - r.start)
          p.taps += len(sel)
      if spec:
        p.entries.append(e)
      off += r.n
    with torch.no_grad():
      if g.steering:
        g.slot[: p.n].copy_(
          torch.from_numpy(self._slot[: p.n]), non_blocking=True
        )
      if g.tap_sites:
        self._rows[p.taps : p.n] = 0
        g.tap_rows[: p.n].copy_(
          torch.from_numpy(self._rows[: p.n]), non_blocking=True
        )
    return p

  def collect(self, p: Plan) -> None:
    """After the forward: per request, the codes of its reported positions and its tap rows, to
    the sink. One device-to-host copy a step, only of the rows the step computed."""
    g = self.g
    self.step += 1
    if not p.entries:
      return
    ids = vals = None
    if g.readout_layers and any(e.readout_from is not None for e in p.entries):
      ids = g.ro_ids[:, : p.n].cpu().numpy()
      vals = g.ro_vals[:, : p.n].float().cpu().numpy()
    tapped = {}
    if p.taps:
      tapped = {k: g.tap_out(*k)[: p.taps] for k in sorted(g.tap_sites)}
    for e in p.entries:
      item: dict = {'step': self.step}
      if ids is not None and e.readout_from is not None:
        a = max(e.readout_from - e.start, 0)
        rows = []
        for j in range(a, e.n):
          feat = [
            [
              l,
              int(ids[i, e.offset + j, q]),
              round(float(vals[i, e.offset + j, q]), 3),
            ]
            for i, l in enumerate(g.readout_layers)
            for q in range(ids.shape[-1])
            if vals[i, e.offset + j, q] > 0
          ]
          rows.append([e.start + j, feat])
        if rows:
          item['readout'] = rows
      if e.tap is not None:
        item['pos'] = e.tap
        for (l, s), t in tapped.items():
          item[f'L{l:02d}.{s}'] = t[e.tap_at : e.tap_at + len(e.tap)].clone()
      if len(item) > 1:
        self.sink.put(e.tag, item)
