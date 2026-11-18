# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.9", "transformers==5.10.1", "safetensors", "numpy"]
# ///
"""Gemma 4 E4B in torch, for serving /read off the Mac (modal_gemma.py runs it on an L40S).

The weights are the mlx 4-bit checkpoint the transcoders were trained on
(mlx-community/gemma-4-e4b-it-4bit @ label.REVISION), dequantised group by group into an HF
Gemma4ForCausalLM. No other copy of Gemma is used: a bf16 checkpoint from Google agrees with the
4-bit one on only 95-98% of top-1 tokens, and the transcoders would read the difference as error.

/read follows /mnt/project-files/edge-aaron-llm/mac-read-spec.md and the client in
quartz/components/scripts/palette/intelligence.ts. Per generated token it reports:

- cites: the share of attention (7 global layers x 8 heads, bos excluded) that the forward that
  wrote the token paid to each passage's tokens;
- feat: the top transcoder codes at the readout layers, read at the transcoder's own site
  (pre_feedforward_layernorm output) on that same forward.

Interventions: `steer` adds s * a_max * W_dec[id] * sy to a layer's MLP+PLE write (a_max from
labels.json), `mask` adds a bias over a passage's positions at the global layers.

    python gemma_torch.py --model <mlx dir> --plt <transcoder dir> --query "..." --passages p.json
    python gemma_torch.py --tiny   # random 4-layer model, checks the taps and the stream shape
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Iterator
from pathlib import Path

import torch

SYSTEM = (
  "You read Aaron's garden, his public notes, and answer from the numbered passages only. "
  'Write two to four plain sentences in lowercase. Put the passage number in brackets, like [3837], '
  'right after the claim it supports. Say so when the passages do not answer the question. If more '
  'reading would help, end with at most two lines of the form `search: <query>`.'
)
MAX_TOKENS = 200
MAX_PASSAGES = 10


# --- weights ------------------------------------------------------------------------------------


def dequant(
  w: torch.Tensor,
  scales: torch.Tensor,
  biases: torch.Tensor,
  bits: int,
  group: int,
):
  """mlx affine quantisation, undone: w packs 32/bits values per uint32, low bits first;
  value = q * scale + bias per `group` inputs. Computed in fp32."""
  per = 32 // bits
  w = w.view(torch.int32) if w.dtype != torch.int32 else w
  shifts = torch.arange(per, dtype=torch.int32) * bits
  q = (w.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)  # (out, in/per, per)
  q = q.reshape(w.shape[0], -1).float()
  g = q.reshape(q.shape[0], -1, group)
  out = g * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)
  return out.reshape(q.shape)


def mlx_state(
  model_dir: Path, dtype=torch.float32
) -> tuple[dict, dict[str, torch.Tensor]]:
  """(text config, HF-named state dict) from an mlx-community Gemma 4 snapshot; each tensor is
  dequantised in fp32, then cast to dtype, so peak memory is one fp32 tensor."""
  from safetensors.torch import load_file

  cfg = json.loads((model_dir / 'config.json').read_text())
  quant = cfg.get('quantization') or {}
  bits, group = int(quant.get('bits', 4)), int(quant.get('group_size', 64))
  raw: dict[str, torch.Tensor] = {}
  for f in sorted(model_dir.glob('model*.safetensors')):
    raw.update(load_file(str(f)))
  out: dict[str, torch.Tensor] = {}
  for k, v in raw.items():
    if not k.startswith('language_model.model.') or k.endswith((
      '.scales',
      '.biases',
    )):
      continue
    base = k.removesuffix('.weight')
    if k.endswith('.weight') and f'{base}.scales' in raw:
      v = dequant(v, raw[f'{base}.scales'], raw[f'{base}.biases'], bits, group)
    out['model.' + k.removeprefix('language_model.model.')] = v.to(dtype)
  return cfg.get('text_config', cfg), out


def load_model(model_dir: Path, dtype=torch.bfloat16, device='cpu'):
  """HF Gemma4ForCausalLM with eager attention (the taps read attention weights) and the
  tokenizer of the same snapshot."""
  from transformers import AutoTokenizer, Gemma4ForCausalLM, Gemma4TextConfig

  text_cfg, state = mlx_state(model_dir, dtype)
  config = Gemma4TextConfig(**{
    k: v for k, v in text_cfg.items() if k != 'quantization'
  })
  config._attn_implementation = 'eager'
  with torch.device('meta'):
    model = Gemma4ForCausalLM(config)
  missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
  missing = [
    k for k in missing if k != 'lm_head.weight'
  ]  # tied to embed_tokens
  if missing or unexpected:
    raise SystemExit(
      f'weight mismatch: missing {missing[:8]} unexpected {unexpected[:8]}'
    )
  model.tie_weights()
  # the non-persistent buffers are not in any checkpoint and stay on meta: rebuild them
  from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextRotaryEmbedding,
  )

  inner = model.model
  inner.rotary_emb = Gemma4TextRotaryEmbedding(
    config
  )  # placeholder until the cast below
  for emb, scale in (
    (inner.embed_tokens, config.hidden_size**0.5),
    (
      getattr(inner, 'embed_tokens_per_layer', None),
      (config.hidden_size_per_layer_input or 0) ** 0.5,
    ),
  ):
    if emb is not None:
      emb.register_buffer('embed_scale', torch.tensor(scale), persistent=False)
  left = [
    n
    for n, t in [*model.named_parameters(), *model.named_buffers()]
    if t.is_meta
  ]
  if left:
    raise SystemExit(f'still on meta after loading: {left[:8]}')
  model = model.to(device=device, dtype=dtype).eval()
  # RoPE frequencies stay fp32, as from_pretrained leaves them; .to(dtype) would round them
  inner.rotary_emb = Gemma4TextRotaryEmbedding(config).to(device)
  return model, AutoTokenizer.from_pretrained(str(model_dir))


def load_transcoders(
  plt_dir: Path, layers, k: int, device='cpu'
) -> dict[int, dict]:
  """Encoder side (and the decoder rows steering needs) of the published per-layer transcoders."""
  from safetensors.torch import load_file

  tcs = {}
  for l in layers:
    w = load_file(str(plt_dir / f'L{l:02d}.safetensors'))
    tcs[l] = dict(
      W_enc=w['W_enc'].to(device, torch.bfloat16),
      b_enc=w['b_enc'].to(device, torch.float32),
      b_pre=w.get('b_pre', torch.zeros_like(w['b_dec'])).to(
        device, torch.float32
      ),
      W_dec=w['W_dec'].to(device, torch.bfloat16),
      sx=float(w['sx'].reshape(-1)[0]),
      sy=float(w['sy'].reshape(-1)[0]),
      k=k,
    )
  return tcs


def encode(tc: dict, x: torch.Tensor, top: int = 3) -> list[tuple[int, float]]:
  """Top `top` TopK-ReLU codes of one vector x (d,) in model units: [(feature, z)]."""
  xc = x.float() / tc['sx'] - tc['b_pre']
  pre = (xc.to(tc['W_enc'].dtype) @ tc['W_enc']).float() + tc['b_enc']
  v, i = torch.topk(pre, tc['k'])
  keep = v > 0
  v, i = v[keep][:top], i[keep][:top]
  return [(int(a), round(float(b), 3)) for a, b in zip(i, v)]


# --- prompt -------------------------------------------------------------------------------------


def build_prompt(tok, query: str, passages: list[dict], think: bool):
  """Token ids of the chat prompt, and each passage's [start, end) token span."""
  blocks = [f'[{int(p["row"])}] {str(p["text"]).strip()}' for p in passages]
  user = '\n\n'.join(blocks) + f'\n\nquestion: {query}'
  system = ('<|think|>' if think else '') + SYSTEM
  text = tok.apply_chat_template(
    [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
    tokenize=False,
    add_generation_prompt=True,
  )
  enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
  ids, offsets = enc['input_ids'], enc['offset_mapping']
  spans: dict[int, tuple[int, int]] = {}
  at = text.find(user)
  for p, block in zip(passages, blocks):
    a = text.find(block, at)
    b = a + len(block)
    at = b
    toks = [i for i, (s, e) in enumerate(offsets) if e > a and s < b]
    if toks:
      spans[int(p['row'])] = (toks[0], toks[-1] + 1)
  return ids, spans


# --- taps ---------------------------------------------------------------------------------------


class Taps:
  """Hooks on the HF modules, armed per request. Every forward leaves, for its last position:
  self.attn[l] (heads, kv) at the global layers and self.x[l] (d,) at the readout layers."""

  def __init__(self, model, tcs: dict[int, dict]):
    self.model = model
    self.tcs = tcs
    cfg = model.config
    self.layers = model.model.layers
    self.full = [
      i for i, t in enumerate(cfg.layer_types) if t == 'full_attention'
    ]
    self.readout: list[int] = []
    self.steer: dict[int, torch.Tensor] = {}
    self.bias: torch.Tensor | None = (
      None  # (kv_max,) additive, global layers only
    )
    self.attn: dict[int, torch.Tensor] = {}
    self.x: dict[int, torch.Tensor] = {}
    self.handles = []
    for l in self.full:
      attn = self.layers[l].self_attn
      self.handles.append(
        attn.register_forward_pre_hook(self._mask, with_kwargs=True)
      )
      self.handles.append(attn.register_forward_hook(self._attn(l)))
    for l, layer in enumerate(self.layers):
      self.handles.append(
        layer.pre_feedforward_layernorm.register_forward_hook(self._x(l))
      )
      self.handles.append(layer.register_forward_hook(self._steer(l)))

  def arm(self, readout, steer, bias):
    self.readout, self.steer, self.bias = list(readout), steer, bias
    self.attn, self.x = {}, {}

  def disarm(self):
    self.arm([], {}, None)

  def _mask(self, module, args, kwargs):
    if self.bias is None:
      return None
    hs = kwargs.get('hidden_states', args[0] if args else None)
    m = kwargs.get('attention_mask')
    q = hs.shape[1]
    kv = (
      m.shape[-1]
      if m is not None
      else int(kwargs['past_key_values'].get_seq_length()) + q
    )
    dt = hs.dtype
    if m is None:
      m = torch.zeros(1, 1, q, kv, dtype=dt, device=hs.device)
    elif m.dtype == torch.bool:
      m = torch.zeros(m.shape, dtype=dt, device=hs.device).masked_fill(
        ~m, torch.finfo(dt).min
      )
    kwargs['attention_mask'] = m + self.bias[:kv].to(dt)
    return args, kwargs

  def _attn(self, l: int):
    def hook(module, args, out):
      w = out[1] if isinstance(out, tuple) else None
      if w is not None:
        self.attn[l] = w[0, :, -1, :].float()

    return hook

  def _x(self, l: int):
    def hook(module, args, out):
      if l in self.readout:
        self.x[l] = out[0, -1].detach()

    return hook

  def _steer(self, l: int):
    def hook(module, args, out):
      v = self.steer.get(l)
      if v is None:
        return None
      # the layer's write y is scaled by layer_scalar with the residual; add v to y
      return out + (v * module.layer_scalar).to(out.dtype)

    return hook

  def close(self):
    for h in self.handles:
      h.remove()


def cites_of(
  attn: dict[int, torch.Tensor],
  spans: dict[int, tuple[int, int]],
  top: int = 3,
):
  """Share of the global layers' attention (bos excluded) on each passage, strongest first."""
  if not attn:
    return []
  mass = torch.stack([a.sum(0) for a in attn.values()]).sum(0)  # (kv,)
  total = float(mass[1:].sum()) or 1.0
  got = [
    (row, float(mass[a:b].sum()) / total) for row, (a, b) in spans.items()
  ]
  got = [
    (r, round(m, 3)) for r, m in sorted(got, key=lambda t: -t[1]) if m >= 0.005
  ]
  return got[:top]


# --- /read --------------------------------------------------------------------------------------


def split_hops(text: str) -> list[str]:
  hops = []
  for line in reversed(text.rstrip().split('\n')):
    m = re.match(r'^\s*search:\s*(.+?)\s*$', line, re.I)
    if not m:
      break
    hops.insert(0, m.group(1))
  return hops


def stop_ids(model_dir: Path | None, tok) -> set[int]:
  ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
  if model_dir and (model_dir / 'generation_config.json').exists():
    e = json.loads((model_dir / 'generation_config.json').read_text()).get(
      'eos_token_id'
    )
    ids |= set(e if isinstance(e, list) else [e] if e is not None else [])
  for t in ('<end_of_turn>', '<turn|>'):
    i = tok.convert_tokens_to_ids(t)
    if isinstance(i, int) and i != tok.unk_token_id:
      ids.add(i)
  return ids


def a_max(labels: dict, layer: int, feature: int) -> float:
  e = labels.get(f'L{layer:02d}', {}).get(str(feature))
  if e and e.get('top'):
    return float(e['top'][0][2])
  return 8.0  # about the median top activation in labels.json


@torch.inference_mode()
def read(
  model, tok, taps: Taps, req: dict, *, labels: dict, stops: set[int], device
) -> Iterator[dict]:
  """The /read stream: one event per generated token, then a done event."""
  from transformers import DynamicCache

  query = str(req['query'])[:400]
  passages = list(req.get('passages') or [])[:MAX_PASSAGES]
  n = min(int(req.get('max_tokens') or 120), MAX_TOKENS)
  readout = [
    int(l) for l in (req.get('readout') or [7, 21, 35]) if int(l) in taps.tcs
  ][:4]
  ids, spans = build_prompt(tok, query, passages, bool(req.get('think')))

  steer: dict[int, torch.Tensor] = {}
  for s in req.get('steer') or []:
    l, f, k = int(s['layer']), int(s['id']), float(s['s'])
    tc = taps.tcs.get(l)
    if tc is None:
      continue
    v = k * a_max(labels, l, f) * tc['W_dec'][f].float() * tc['sy']
    steer[l] = steer.get(l, 0) + v
  bias = None
  if req.get('mask'):
    bias = torch.zeros(len(ids) + n + 1, device=device)
    for m in req['mask']:
      span = spans.get(int(m['row']))
      if span:
        bias[span[0] : span[1]] += max(float(m['bias']), -1e30)

  taps.arm(readout, steer, bias)
  try:
    cache = DynamicCache(config=model.config)
    t0 = time.perf_counter()
    inp = torch.tensor([ids], device=device)
    out = model(input_ids=inp, past_key_values=cache, use_cache=True)
    prefill = time.perf_counter() - t0
    text, events = '', []
    t1 = time.perf_counter()
    for i in range(n):
      nxt = int(out.logits[0, -1].argmax())
      if nxt in stops:
        break
      piece = tok.decode([nxt])
      text += piece
      ev = dict(
        t=piece, i=i, cites=[list(c) for c in cites_of(taps.attn, spans)]
      )
      ev['feat'] = [
        [l, f, z]
        for l in readout
        if l in taps.x
        for f, z in encode(taps.tcs[l], taps.x[l], 1)
      ]
      events.append(ev)
      yield ev
      taps.attn, taps.x = {}, {}
      out = model(
        input_ids=torch.tensor([[nxt]], device=device),
        past_key_values=cache,
        use_cache=True,
      )
    decode = time.perf_counter() - t1
    total: dict[int, float] = {}
    for ev in events:
      for row, m in ev['cites']:
        total[row] = total.get(row, 0.0) + m
    s = sum(total.values()) or 1.0
    yield dict(
      done=True,
      hops=split_hops(text),
      cites=[
        [r, round(m / s, 3)]
        for r, m in sorted(total.items(), key=lambda t: -t[1])
      ],
      prefill_ms=round(prefill * 1000),
      decode_tps=round(len(events) / decode, 1) if decode > 0 else None,
      tokens=dict(prompt=len(ids), out=len(events)),
    )
  finally:
    taps.disarm()


# --- checks -------------------------------------------------------------------------------------


@torch.inference_mode()
def compare(
  model, tcs: dict[int, dict], ref: dict, device, batch: int = 10
) -> dict:
  """The numerics gate against the Mac (docs/verification/latent-modal/ref.py): next-token
  agreement with the mlx 4-bit model, and how close the transcoder-site activations and their
  TopK codes are at the readout layers. Position 0 (bos) is left out."""
  import numpy as np

  ids = torch.tensor(ref['ids'], dtype=torch.long)
  layers = [l for l in tcs if f'x{l}' in ref]
  got: dict[int, list] = {l: [] for l in layers}
  hooks = [
    model.model.layers[l].pre_feedforward_layernorm.register_forward_hook(
      lambda m, a, out, l=l: got[l].append(out.float().cpu())
    )
    for l in layers
  ]
  top1 = []
  try:
    for b in range(0, len(ids), batch):
      out = model(input_ids=ids[b : b + batch].to(device), use_cache=False)
      top1.append(out.logits.float().argmax(-1).cpu())
  finally:
    for h in hooks:
      h.remove()
  t1 = torch.cat(top1)[:, 1:].numpy()
  m5 = ref['top5'][:, 1:]
  res = dict(
    positions=int(t1.size),
    top1_agree=round(float((t1 == m5[..., 0]).mean()), 4),
    top1_in_mlx_top5=round(float((t1[..., None] == m5).any(-1).mean()), 4),
  )
  for l in layers:
    xt = torch.cat(got[l])[:, 1:].reshape(-1, model.config.hidden_size)
    xm = torch.from_numpy(ref[f'x{l}'][:, 1:].astype(np.float32)).reshape(
      xt.shape
    )
    rel = ((xt - xm).norm(dim=-1) / xm.norm(dim=-1).clamp_min(1e-6)).median()
    tc = tcs[l]
    codes = []
    for x in (xt, xm):
      xc = x.to(device) / tc['sx'] - tc['b_pre']
      pre = (xc.to(tc['W_enc'].dtype) @ tc['W_enc']).float() + tc['b_enc']
      codes.append(torch.topk(pre, tc['k'], dim=-1).indices.cpu())
    a, b_ = codes
    inter = (a.unsqueeze(-1) == b_.unsqueeze(-2)).any(-1).sum(-1).float()
    jac = (inter / (2 * tc['k'] - inter)).mean()
    res[f'L{l}'] = dict(
      x_rel_err=round(float(rel), 5), code_jaccard=round(float(jac), 4)
    )
  res['pass'] = res['top1_agree'] >= 0.99 and all(
    res[f'L{l}']['code_jaccard'] >= 0.9 for l in layers
  )
  return res


def tiny(model_dir: Path | None):
  """A random 4-layer Gemma 4 with the real layer pattern (global last) and random transcoders,
  to check the hooks, mask and stream without the 5 GB checkpoint."""
  from transformers import AutoTokenizer, Gemma4ForCausalLM, Gemma4TextConfig

  torch.manual_seed(0)
  base = {}
  if model_dir:
    base = json.loads((model_dir / 'config.json').read_text()).get(
      'text_config', {}
    )
  cfg = dict(base)
  cfg.update(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=16,
    global_head_dim=32,
    hidden_size_per_layer_input=8,
    num_kv_shared_layers=2,
    layer_types=['sliding_attention', 'full_attention']
    + ['sliding_attention'] * 3
    + ['full_attention'],
    sliding_window=64,
  )
  cfg.pop('quantization', None)
  config = Gemma4TextConfig(**cfg)
  config._attn_implementation = 'eager'
  model = Gemma4ForCausalLM(config).eval()
  tok = AutoTokenizer.from_pretrained(str(model_dir)) if model_dir else None
  d, M = config.hidden_size, 32
  tcs = {
    l: dict(
      W_enc=torch.randn(d, M, dtype=torch.bfloat16),
      b_enc=torch.zeros(M),
      b_pre=torch.zeros(d),
      W_dec=torch.randn(M, d, dtype=torch.bfloat16),
      sx=1.0,
      sy=1.0,
      k=4,
    )
    for l in range(config.num_hidden_layers)
  }
  return model, tok, tcs


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--model', type=Path)
  ap.add_argument('--plt', type=Path)
  ap.add_argument('--tiny', action='store_true')
  ap.add_argument('--query', default='what does aaron think about attention')
  ap.add_argument(
    '--passages', type=Path, help='json list of {row, slug, text}'
  )
  ap.add_argument('--max-tokens', type=int, default=24)
  a = ap.parse_args()
  if a.tiny:
    model, tok, tcs = tiny(a.model)
    readout = [1, 5]
  else:
    model, tok = load_model(a.model)
    tcs = load_transcoders(a.plt, [7, 21, 35], 64)
    readout = [7, 21, 35]
  passages = (
    json.loads(a.passages.read_text())
    if a.passages
    else [
      dict(
        row=1,
        slug='a',
        text='attention is a soft dictionary lookup over keys.',
      ),
      dict(
        row=2, slug='b', text='the high park loop is 5.6 km with one climb.'
      ),
    ]
  )
  taps = Taps(model, tcs)
  req = dict(
    query=a.query, passages=passages, max_tokens=a.max_tokens, readout=readout
  )
  stops = stop_ids(a.model, tok)
  plain = list(
    read(model, tok, taps, req, labels={}, stops=stops, device='cpu')
  )
  for ev in plain[:3] + plain[-1:]:
    print(json.dumps(ev, ensure_ascii=False))
  if a.tiny:
    ids, spans = build_prompt(tok, a.query, passages, False)
    assert set(spans) == {1, 2}, spans
    toks = [e for e in plain if 'done' not in e]
    assert toks and all(e['feat'] for e in toks), 'readout taps did not fire'
    assert all(e['cites'] for e in toks), 'attention taps did not fire'
    muted = list(
      read(
        model,
        tok,
        taps,
        dict(req, mask=[dict(row=1, bias=-1e30)]),
        labels={},
        stops=stops,
        device='cpu',
      )
    )
    m1 = sum(
      m for e in muted if 'done' not in e for r, m in e['cites'] if r == 1
    )
    assert m1 == 0, f'masked passage still cited: {m1}'
    steered = list(
      read(
        model,
        tok,
        taps,
        dict(req, steer=[dict(layer=3, id=0, s=50.0)]),
        labels={},
        stops=stops,
        device='cpu',
      )
    )
    same = [e.get('t') for e in steered] == [e.get('t') for e in plain]
    assert not same, 'steering did not change the output'
    assert not taps.steer and taps.bias is None, 'taps stayed armed'
    print(
      '[tiny] taps, mask, steer and stream ok;',
      len(spans),
      'passages,',
      len(ids),
      'tokens',
    )
  taps.close()


if __name__ == '__main__':
  main()
