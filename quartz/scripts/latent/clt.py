# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "safetensors"]
# ///
"""Cross-layer transcoder (CLT) for Gemma 4 E4B, in torch, for a GPU (CUDA) or a Mac (MPS).

    uv run quartz/scripts/latent/clt.py --cache <acts.py cache> [--M 4096 -k 32 --span 4]
    # in process, from the vLLM capture plugin (docs: interp-contract.md §4):
    from clt import CLT, train;  train(CLT(...), batches, out, ...)

The math and file layout are fixed in /mnt/project-files/latent-circuits/interp-contract.md
(mirrored in docs/latent-interp-contract.md). Per source layer s, with x'_s = mlp_in_s / sx_s - b_pre_s:

    z_s   = TopK_k(ReLU(x'_s W_enc_s + b_enc_s))
    y'_t  = Σ_{s=t-S+1..t} z_s W_dec_s[:, t-s] + x'_t W_skip_t + b_dec_t      mlp_out_t = sy_t y'_t

A per-layer transcoder (transcoders.py) is the case S = 1. The CLT lets a feature that fires at
layer s write to the next S-1 layers too, so a computation that one MLP starts and the next
finishes is one feature, not two features plus an edge (Ameisen et al. 2025, "Circuit tracing").

Training, as in transcoders.py:
  - the first --fit-tokens only gather moments, then fix sx, sy (scalar RMS), b_pre (input mean)
    and W_skip (ridge of y on x), per layer, and seed the dictionary from those tokens;
  - the loss is the sum over target layers of SSE / the batch's variance (each layer's FVU,
    so a quiet early layer counts as much as a loud late one), plus AuxK on dead latents;
  - each feature's decoder (all its S_l targets together) is kept at unit norm.
--init-plt takes sx, sy, b_pre, W_skip, b_dec from a trained per-layer set instead of fitting.

Size at the pilot (M 4096, k 32, span 4, 42 layers): 2.1 B trainable parameters, 34 GB with grads
and Adam in fp32, ~7 GB of activations at batch 4096; one 80 GB GPU. A Mac holds a pilot only at
M 1024 or fewer layers.

`--score <plt dir>` prints a per-layer set's FVU on a cache instead of training: the gate for
serving mlx-trained weights on torch/vLLM activations (contract §6).

Output, <out>/ (clt/v1, contract §3): enc/L<ll>.safetensors {W_enc (d, M) bf16, b_enc, b_pre, sx},
dec/L<ll>.safetensors {W_dec (M, S_l, d) bf16}, skip/L<ll>.safetensors {W_skip bf16, b_dec, sy},
manifest.json, state.pt (fp32 weights for a restart; not published).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import torch
from clt_fit import Moments, from_plt, load_plt, seed_dictionary
from clt_io import read_cache, save, shuffled
from clt_model import CLT, Batch, device, log, loss_of, score


def train(
  m: CLT,
  batches: Iterable[Batch],
  out: Path,
  *,
  steps: int,
  lr: float = 2e-4,
  fit_tokens: int = 65536,
  dead_after: int = 2_000_000,
  save_every: int = 1000,
  info: dict | None = None,
) -> CLT:
  """Fit the moments (unless m is already fitted), then `steps` Adam steps. `batches` yields
  acts.stream()-shaped dicts already shuffled (see `shuffled`)."""
  dev = m.W_enc.device
  info = dict(info or {})
  moments = None if bool(m.fitted) else Moments(m.L, m.d, dev)
  opt = torch.optim.Adam(m.parameters(), lr=lr)
  since = torch.zeros(m.L, m.M, dtype=torch.long, device=dev)
  fvu_ema, tokens, step, t0 = None, 0, 0, time.time()
  amp = dev.type == 'cuda'
  for batch in batches:
    batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
    n = batch[0, 'in'].shape[0]
    tokens += n
    if moments is not None:
      moments.add(batch, m.L, m.M)
      if moments.n >= fit_tokens:
        moments.solve(m)
        moments = None
        log(
          f'fitted moments on {tokens} tokens; sx {m.sx.mean():.3g} sy {m.sy.mean():.3g}'
        )
      continue
    t = step
    warm = min(1.0, (t + 1) / max(1.0, min(500, 0.05 * steps)))
    decay = 1.0 if t < 0.8 * steps else max(0.02, (steps - t) / (0.2 * steps))
    for g in opt.param_groups:
      g['lr'] = lr * warm * decay
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
      loss, fvu, fired = loss_of(m, batch, since > dead_after)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    m.project_grads()
    opt.step()
    m.norm_decoders()
    since += n
    since[fired] = 0
    fvu_ema = fvu if fvu_ema is None else 0.98 * fvu_ema + 0.02 * fvu
    step += 1
    if step % 50 == 0 or step == steps:
      dead = (since > dead_after).float().mean().item()
      log(
        f'step {step}/{steps}, {tokens} tok, fvu mean {fvu_ema.mean():.3f}'
        f' (L00 {fvu_ema[0]:.3f}, L{m.L - 1} {fvu_ema[-1]:.3f}), dead {dead:.1%},'
        f' {tokens / (time.time() - t0):.0f} tok/s'
      )
    if step % save_every == 0 or step == steps:
      info.update(
        tokens=tokens,
        steps=step,
        fvu=[round(float(v), 4) for v in fvu_ema],
        dead=round((since > dead_after).float().mean().item(), 4),
      )
      save(m, out, info)
    if step >= steps:
      break
  return m


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument(
    '--cache',
    type=Path,
    required=True,
    help='an acts.py cache over all layers',
  )
  ap.add_argument('--out', type=Path, default=None)
  ap.add_argument('--M', type=int, default=4096)
  ap.add_argument('-k', type=int, default=32)
  ap.add_argument('--span', type=int, default=4)
  ap.add_argument('--steps', type=int, default=4000)
  ap.add_argument('--batch', type=int, default=4096)
  ap.add_argument(
    '--buffer', type=int, default=65536, help='tokens in the shuffle buffer'
  )
  ap.add_argument('--passes', type=int, default=3)
  ap.add_argument('--lr', type=float, default=2e-4)
  ap.add_argument(
    '--init-plt',
    type=Path,
    default=None,
    help='take sx, sy, b_pre, W_skip, b_dec from it',
  )
  ap.add_argument(
    '--engine',
    default=None,
    help='manifest engine; default from the cache meta',
  )
  ap.add_argument(
    '--score',
    type=Path,
    default=None,
    help='only print per-layer FVU of this PLT set on --cache',
  )
  ap.add_argument('--score-tokens', type=int, default=200_000)
  a = ap.parse_args()

  meta = json.loads((a.cache / 'meta.json').read_text())
  L, d = len(meta['layers']), meta['d']
  if a.score:
    m = from_plt(a.score, L).to(device())
    got = score(m, read_cache(a.cache, 1024), a.score_tokens)
    print(
      json.dumps(
        dict(plt=str(a.score), engine=meta.get('engine', 'mlx-4bit'), fvu=got)
      )
    )
    return
  assert meta['layers'] == list(range(L)), (
    'a CLT needs layers 0..L-1 contiguous'
  )
  out = a.out or Path('quartz/.quartz-cache/latent-clt') / (
    f'{meta["sidecar"][:8]}-w{a.M}-k{a.k}-s{a.span}'
  )
  dev = device()
  m = CLT(L, d, a.M, a.k, a.span)
  if (out / 'state.pt').exists():
    m.load_state_dict(torch.load(out / 'state.pt', map_location='cpu'))
    log(f'resumed from {out}')
  elif a.init_plt:
    load_plt(m, a.init_plt)
  m.to(dev)
  info = dict(
    model=meta.get('model'),
    revision=meta.get('revision'),
    sidecar=meta.get('sidecar'),
    engine=a.engine or meta.get('engine', 'mlx-4bit'),
  )
  data = shuffled(read_cache(a.cache, 1024, a.passes), a.buffer, a.batch)
  if bool(m.fitted) and not (out / 'state.pt').exists():
    seed_dictionary(m, [next(data) for _ in range(max(1, a.M // a.batch))])
  train(m, data, out, steps=a.steps, lr=a.lr, info=info)
  log('done', out)


if __name__ == '__main__':
  main()
