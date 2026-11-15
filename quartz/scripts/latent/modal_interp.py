"""Gemma 4 E4B interp on Modal through vLLM + the garden_vllm plugin, and the CLT harvest.

Not deployed, and nothing here has run: deploying and GPU runs wait on aarnphm's go ($30/month
total for Modal, scale to zero, never warm). Design, costs and the go/no-go: docs/latent-interp-vllm.md.

    uvx modal run quartz/scripts/latent/modal_gemma.py::fetch       # once: mlx snapshot + plt/v1 into the volume
    uvx modal run quartz/scripts/latent/modal_interp.py::convert    # once: bf16 HF checkpoint for vLLM and torch (~$0.05, CPU)
    uvx modal run quartz/scripts/latent/modal_interp.py::check      # the spike: vLLM vs torch codes, compile vs eager, tok/s
    uvx modal run quartz/scripts/latent/modal_interp.py::rows       # on the Mac: the sidecar's public rows into the volume
    uvx modal run quartz/scripts/latent/modal_interp.py::score      # contract §6.1: plt/v1 fvu on torch activations
    uvx modal run quartz/scripts/latent/modal_interp.py::train_clt  # the CLT pilot (H100, ~$3)
    uvx modal deploy quartz/scripts/latent/modal_interp.py          # /generate and /readout behind proxy auth

The reader (/read with cites and masks, and /graph) stays on modal_gemma.py: cites need attention
weights and masks need an additive bias, which vLLM's attention kernels do not expose, and graphs
need a backward pass.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from modal_interp_engine import (
  BF16,
  READOUT,
  VOL,
  _rows,
  _torch_engine,
  engine,
  garden_cfg,
  run,
  steps,
)
from modal_interp_images import HERE, torch_image, vllm_image


MODEL = 'mlx-community/gemma-4-e4b-it-4bit'
REVISION = '475b9088d29754a3379866cf5aeb6b41acd313c2'  # label.REVISION: what the transcoders read

app = modal.App('garden-interp')
volume = modal.Volume.from_name(
  'garden-gemma', create_if_missing=True
)  # shared with modal_gemma.py


@app.function(
  image=torch_image,
  volumes={VOL: volume},
  cpu=4,
  memory=40 * 1024,
  timeout=1800,
)
def convert():
  """The mlx 4-bit snapshot dequantised (gemma_torch.mlx_state) and saved as a plain HF bf16
  checkpoint, so vLLM and torch both load safetensors straight to the GPU instead of unpacking
  for 52 s. Same weights as the transcoders' model; sha256 per shard in convert.json."""
  import hashlib
  import shutil

  import gemma_torch as g
  import torch

  src = VOL / 'gemma'
  t0 = time.perf_counter()
  model, tok = g.load_model(src, dtype=torch.bfloat16, device='cpu')
  BF16.mkdir(exist_ok=True)
  model.save_pretrained(str(BF16), max_shard_size='4GB')
  tok.save_pretrained(str(BF16))
  for name in ('generation_config.json', 'chat_template.jinja'):
    if (src / name).exists():
      shutil.copy(src / name, BF16 / name)
  sha = {}
  for f in sorted(BF16.glob('*.safetensors')):
    h = hashlib.sha256()
    with f.open('rb') as fh:
      for block in iter(lambda: fh.read(1 << 24), b''):
        h.update(block)
    sha[f.name] = h.hexdigest()
  (BF16 / 'convert.json').write_text(
    json.dumps(
      dict(
        source=MODEL,
        revision=REVISION,
        recipe='gemma_torch.mlx_state: fp32 dequant, cast bf16',
        files=sha,
      ),
      indent=1,
    )
  )
  volume.commit()
  print(f'[convert] {time.perf_counter() - t0:.0f} s, {len(sha)} shards')


SAMPLE = [
  'attention is a soft dictionary lookup over keys and values, so a head is a learned retrieval.',
  'the high park loop is 5.6 km with one climb, and the descent on the west side is the fast part.',
  "Qu'est-ce que tu écris à la fin, quand le carnet est plein et que la lumière baisse ?",
  'a transcoder replaces an mlp with a sparse dictionary that reads its input and writes its output.',
]


@app.function(
  image=vllm_image,
  volumes={VOL: volume},
  gpu='L40S',
  memory=48 * 1024,
  timeout=1800,
)
def check(eager_too: bool = True):
  """The spike that decides the serving path (docs/latent-interp-vllm.md, go/no-go):

  1. the plugin's CPU tests, on this image's torch;
  2. teacher-forced readout codes, vLLM vs the torch engine on the same bf16 weights: top-4
     Jaccard per readout layer over every position (bar: >= 0.9, the torch-vs-mlx level);
  3. a steer changes greedy output, and an unsteered request in the same batch does not move;
  4. decode tok/s with CUDA graphs and with enforce_eager, and the cold-start parts."""
  import subprocess
  import sys

  import torch

  subprocess.run(
    [
      sys.executable,
      '-m',
      'pytest',
      '-q',
      '/root/garden_vllm/test_garden_vllm.py',
    ],
    check=True,
    cwd='/root',
  )
  from transformers import AutoTokenizer

  tok = AutoTokenizer.from_pretrained(str(BF16))
  rows = SAMPLE
  rf = VOL / 'rows'
  if rf.exists():
    files = sorted(rf.glob('*.jsonl'))
    if files:
      rows = [
        json.loads(x)['text'] for x in files[-1].read_text().splitlines()[:64]
      ]
  ids = [
    [tok.bos_token_id, *tok.encode(t, add_special_tokens=False)[:511]]
    for t in rows
  ]

  # torch engine reference: codes at every position, then free the GPU
  from garden_vllm import hf
  from garden_vllm.state import GardenConfig
  from transformers import Gemma4ForCausalLM

  t0 = time.perf_counter()
  ref_model = (
    Gemma4ForCausalLM
    .from_pretrained(
      str(BF16), dtype=torch.bfloat16, attn_implementation='eager'
    )
    .cuda()
    .eval()
  )
  print(f'[load] torch from bf16 safetensors {time.perf_counter() - t0:.1f} s')
  g = hf.install(
    ref_model, GardenConfig(**{**garden_cfg(steer_slots=0)}), n_max=4096
  )
  ref = []
  with torch.inference_mode():
    for p in ids:
      ref_model(input_ids=torch.tensor([p], device='cuda'), use_cache=False)
      ids_, vals_ = (
        g.ro_ids[:, : len(p)].cpu(),
        g.ro_vals[:, : len(p)].float().cpu(),
      )
      ref.append([
        [set(ids_[j, q][vals_[j, q] > 0].tolist()) for q in range(len(p))]
        for j in range(len(READOUT))
      ])
  del ref_model, g
  torch.cuda.empty_cache()

  report = {}
  for eager in [False, True] if eager_too else [False]:
    t0 = time.perf_counter()
    eng = engine(garden_cfg(), eager=eager, prefix_cache=False)
    boot = time.perf_counter() - t0
    got = run(
      eng,
      ids,
      [dict(readout='all')] * len(ids),
      1,
      [f'r{i}' for i in range(len(ids))],
    )
    jac = {l: [] for l in READOUT}
    for i, p in enumerate(ids):
      feats = got[f'r{i}']['readout']
      for pos in range(1, len(p)):
        for j, l in enumerate(READOUT):
          a = {f for ll, f, _ in feats.get(pos, []) if ll == l}
          b = ref[i][j][pos]
          jac[l].append(len(a & b) / max(1, len(a | b)))
    chat = tok.apply_chat_template(
      [{'role': 'user', 'content': 'say one sentence about gardens.'}],
      tokenize=True,
      add_generation_prompt=True,
      enable_thinking=False,
      return_dict=False,
    )
    t1 = time.perf_counter()
    plain = run(eng, [chat], [dict(readout='decode')], 96, ['plain'])['plain']
    dt = time.perf_counter() - t1
    both = run(
      eng,
      [chat, chat],
      [
        dict(readout='decode'),
        dict(readout='decode', steer=[dict(layer=21, id=0, s=4.0)]),
      ],
      96,
      ['same', 'steered'],
    )
    report['eager' if eager else 'graphs'] = dict(
      boot_s=round(boot, 1),
      code_jaccard={
        f'L{l}': round(sum(v) / len(v), 4) for l, v in jac.items()
      },
      decode_tps=round(len(plain['ids']) / dt, 1),
      unsteered_unchanged=both['same']['ids'] == plain['ids'],
      steer_changes_output=both['steered']['ids'] != plain['ids'],
      readout_every_token=all(
        pos in plain['readout']
        for pos in range(
          plain['prompt'] - 1, plain['prompt'] - 1 + len(plain['ids'])
        )
      ),
      text=tok.decode(plain['ids']),
    )
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
    del eng
    torch.cuda.empty_cache()
  return report


@app.local_entrypoint()
def rows(
  content: str = 'content', directory: str = 'embeddings', seed: int = 0
):
  """On the Mac, from ~/workspace/garden: the live sidecar's canonical note and pdf rows
  (protected, private and draft notes are already out of the store) as jsonl into the volume,
  for the harvest. Uses acts.corpus, so the rows and their shuffle are the mlx cache's."""
  import sys
  import tempfile

  sys.path.insert(0, str(HERE))
  import acts

  digest, rs = acts.corpus(Path(content), directory, seed)
  with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as f:
    for r, t in rs:
      f.write(json.dumps(dict(row=r, text=t), ensure_ascii=False) + '\n')
  with volume.batch_upload(force=True) as up:
    up.put_file(f.name, f'/rows/{digest[:8]}.jsonl')
  print(
    f'[rows] {len(rs)} rows of sidecar {digest[:8]} -> volume rows/{digest[:8]}.jsonl'
  )


@app.function(
  image=torch_image, volumes={VOL: volume}, gpu='L40S', timeout=3600
)
def score(tokens: int = 200_000, sidecar: str | None = None):
  """Contract §6.1: each plt/v1 layer's fvu on torch bf16-dequant activations against the fvu in
  its manifest (mlx). Any layer worse by more than 0.03 means plt/v2 on torch before Modal serves
  v1 readouts."""
  import clt
  from garden_vllm import hf

  model, tok, g = _torch_engine()
  name, rs = _rows(sidecar)
  batches = hf.stream(
    model, tok, g, rs, ctx=1024, batch_tokens=16384, keep=0.25, seed=1
  )
  m = clt.from_plt(VOL / 'plt').cuda()
  got = clt.score(m, batches, tokens)
  man = json.loads((VOL / 'plt' / 'manifest.json').read_text())
  want = [float(x['fvu']) for x in man.get('layers', [])] or [None] * len(got)
  worse = [
    l
    for l, (a, b) in enumerate(zip(got, want))
    if b is not None and a - b > 0.03
  ]
  print(
    json.dumps(
      dict(rows=name, fvu_torch=got, fvu_mlx=want, worse_than_003=worse)
    )
  )
  return worse


@app.function(
  image=torch_image,
  volumes={VOL: volume},
  gpu='H100',
  memory=64 * 1024,
  timeout=6 * 3600,
)
def train_clt(
  M: int = 4096,
  k: int = 32,
  span: int = 4,
  steps: int = 4500,
  keep: float = 0.25,
  sidecar: str | None = None,
  init_plt: bool = False,
):
  """The CLT pilot of the contract §6.3: harvest on the torch engine in this process (no shard
  on disk), shuffle, train, write clt/v1 into the volume. 18M tokens at batch 4096 is ~4,400
  steps. Waits on aarnphm's go, like every Modal run."""
  import clt
  from garden_vllm import hf

  model, tok, g = _torch_engine()
  name, rs = _rows(sidecar)
  L, d = model.config.num_hidden_layers, model.config.hidden_size

  def passes():
    for epoch in range(3):
      yield from hf.stream(
        model,
        tok,
        g,
        rs,
        ctx=1024,
        batch_tokens=16384,
        keep=keep,
        seed=7919 * epoch,
      )

  import torch

  out = VOL / 'clt' / f'{name}-w{M}-k{k}-s{span}'
  m = clt.CLT(L, d, M, k, span)
  data = clt.shuffled(passes(), capacity=100_000, size=4096)
  if (out / 'state.pt').exists():
    m.load_state_dict(
      torch.load(out / 'state.pt', map_location='cpu')
    )  # a restart resumes
    m.cuda()
  elif init_plt:  # contract §2: reuse plt/v1's sx, sy, b_pre, W_skip (fitted on mlx activations)
    clt.load_plt(m, VOL / 'plt')
    m.cuda()
    clt.seed_dictionary(m, [next(data) for _ in range(max(1, M // 4096))])
  else:  # default: fit the moments on this engine's own activations (train() does it, then seeds)
    m.cuda()
  clt.train(
    m,
    data,
    out,
    steps=steps,
    info=dict(
      engine='torch/bf16-dequant', model=MODEL, revision=REVISION, sidecar=name
    ),
  )
  volume.commit()


@app.cls(
  image=vllm_image,
  gpu='L40S',
  volumes={VOL: volume},
  cpu=2,
  memory=24 * 1024,
  scaledown_window=60,  # $30/mo: cold starts are fine, idle GPU time is the bill
  max_containers=1,
  timeout=600,
)
class Interp:
  @modal.enter()
  def boot(self):
    import threading

    from transformers import AutoTokenizer

    self.tok = AutoTokenizer.from_pretrained(str(BF16))
    self.eng = engine(
      garden_cfg(), prefix_cache=False
    )  # cached rows would get no readout
    self.lock = threading.Lock()

  @modal.asgi_app(requires_proxy_auth=True)
  def web(self):
    import asyncio
    import uuid

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def spec_of(req: dict) -> dict:
      steer = [
        dict(
          layer=int(s['layer']),
          id=int(s['id']),
          s=max(-8.0, min(8.0, float(s['s']))),
        )
        for s in (req.get('steer') or [])[:8]
      ]
      return dict(
        readout='decode' if req.get('readout', True) else False,
        steer=steer,
        steer_from=req.get('steer_from', 'all'),
      )

    @api.post('/generate')
    async def generate(request: Request):
      """{messages | prompt, max_tokens <= 256, readout, steer, steer_from} -> ndjson:
      {t, i, feat: [[layer, id, z]]} per token, then {done, tokens, decode_tps}."""
      req = await request.json()
      msgs = req.get('messages') or [
        {'role': 'user', 'content': str(req.get('prompt') or '')[:4000]}
      ]
      ids = self.tok.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
      )
      n = min(int(req.get('max_tokens') or 128), 256)
      tag = uuid.uuid4().hex
      spec = spec_of(req)

      def go():
        with self.lock:  # one engine loop; requests queue
          t = time.perf_counter()
          r = run(self.eng, [ids], [spec], n, [tag])[tag]
          return r, time.perf_counter() - t

      import queue
      import threading

      out: queue.SimpleQueue = queue.SimpleQueue()

      def loop():
        P, i = len(ids), 0
        try:
          with self.lock:  # one engine loop; requests queue behind it
            t = time.perf_counter()
            for _, new, feat, done in steps(self.eng, [ids], [spec], n, [tag]):
              for tid in new:
                out.put(
                  dict(
                    t=self.tok.decode([tid]), i=i, feat=feat.get(P - 1 + i, [])
                  )
                )
                i += 1
            dt = time.perf_counter() - t
          out.put(
            dict(
              done=True,
              tokens=dict(prompt=P, out=i),
              decode_tps=round(i / dt, 1) if dt else None,
            )
          )
        except Exception as e:  # the client sees why its stream ended
          out.put(dict(done=True, error=f'{type(e).__name__}: {e}'))

      threading.Thread(target=loop, daemon=True).start()

      async def stream():
        while True:
          ev = await asyncio.to_thread(out.get)
          yield json.dumps(ev, ensure_ascii=False) + '\n'
          if ev.get('done'):
            return

      return StreamingResponse(stream(), media_type='application/x-ndjson')

    @api.post('/readout')
    async def readout(request: Request):
      """{texts: [<= 64 strings]} -> [{tokens, feat: {pos: [[layer, id, z]]}}]: the bake's codes
      for every position, teacher-forced (prefix cache would skip rows, so texts are distinct)."""
      req = await request.json()
      texts = [str(t)[:8000] for t in (req.get('texts') or [])[:64]]
      if not texts:
        return JSONResponse(
          {'error': 'expected a list of texts'}, status_code=400
        )
      ids = [
        [
          self.tok.bos_token_id,
          *self.tok.encode(t, add_special_tokens=False)[:2047],
        ]
        for t in texts
      ]
      tags = [uuid.uuid4().hex for _ in ids]

      def go():
        with self.lock:
          return run(self.eng, ids, [dict(readout='all')] * len(ids), 1, tags)

      got = await asyncio.to_thread(go)
      return [
        dict(tokens=p, feat=got[t]['readout']) for p, t in zip(ids, tags)
      ]

    @api.get('/health')
    async def health():
      return {
        'model': MODEL,
        'revision': REVISION,
        'readout': list(READOUT),
        'engine': 'vllm-0.33/bf16-dequant',
      }

    return api
