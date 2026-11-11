"""Gemma 4 E4B with the garden's transcoder taps on Modal: the /read stream for Ctrl-O `?` mode.

Not deployed. Deploying publishes an endpoint and spends money, so it waits on aarnphm's go.
Design and costs: docs/latent-modal.md.

    uvx modal run quartz/scripts/latent/modal_gemma.py::fetch     # fill the volume once (~10 GB)
    uvx modal run quartz/scripts/latent/modal_gemma.py::check     # tiny taps, then a real /read with timings
    uvx modal deploy quartz/scripts/latent/modal_gemma.py         # the endpoint (needs the go)

The endpoint only answers requests that carry a Modal proxy auth token (Modal-Key /
Modal-Secret headers), which only the Cloudflare Worker holds (worker/gemma.ts). The browser never
talks to Modal.
"""

from __future__ import annotations

from pathlib import Path

import modal

MODEL = 'mlx-community/gemma-4-e4b-it-4bit'
REVISION = '475b9088d29754a3379866cf5aeb6b41acd313c2'  # label.REVISION: the transcoders' model
PLT = 'https://aarnphm.xyz/models/latent-plt/v1'  # published 2026-11-08, bf16, 42 x ~97 MB
READOUT = (7, 21, 35)
VOL = Path('/vol')

app = modal.App('garden-gemma')
volume = modal.Volume.from_name('garden-gemma', create_if_missing=True)
base = (
  modal.Image
  .debian_slim(python_version='3.12')
  .uv_pip_install(
    'torch==2.13.0',
    'transformers==5.10.1',
    'safetensors',
    'numpy',
    'huggingface_hub',
    'fastapi[standard]>=0.115',
  )
  .env({'TOKENIZERS_PARALLELISM': 'false'})
)
# local files go last: nothing can be layered on top of them
image = base.env({'HF_HUB_OFFLINE': '1'}).add_local_file(
  Path(__file__).with_name('gemma_torch.py'), '/root/gemma_torch.py'
)


@app.function(image=base, volumes={VOL: volume}, timeout=3600)
def fetch():
  """The pinned mlx snapshot from the Hub, and the published transcoders and labels from R2."""
  import json
  import urllib.error
  import urllib.request

  from huggingface_hub import snapshot_download

  snapshot_download(
    MODEL,
    revision=REVISION,
    local_dir=VOL / 'gemma',
    allow_patterns=['*.json', '*.safetensors', '*.jinja'],
  )
  plt = VOL / 'plt'
  plt.mkdir(exist_ok=True)

  def get(name: str) -> bytes:
    req = urllib.request.Request(
      f'{PLT}/{name}', headers={'User-Agent': 'garden-gemma-fetch/1'}
    )
    try:
      return urllib.request.urlopen(req, timeout=300).read()
    except (
      urllib.error.HTTPError
    ) as e:  # HTTPError does not pickle back to the client
      raise RuntimeError(f'{name}: HTTP {e.code}') from None

  def pull(name: str) -> None:
    """Stream to a dot file, resuming with Range after a cut connection."""
    part, done = plt / f'.{name}', plt / name
    for attempt in range(8):
      have = part.stat().st_size if part.exists() else 0
      headers = {'User-Agent': 'garden-gemma-fetch/1'}
      if have:
        headers['Range'] = f'bytes={have}-'
      try:
        with urllib.request.urlopen(
          urllib.request.Request(f'{PLT}/{name}', headers=headers), timeout=300
        ) as r:
          if have and r.status != 206:
            have = 0  # the server ignored Range: start over
          with part.open('ab' if have else 'wb') as f:
            while chunk := r.read(1 << 22):
              f.write(chunk)
        part.rename(done)
        return
      except urllib.error.HTTPError as e:
        raise RuntimeError(f'{name}: HTTP {e.code}') from None
      except Exception as e:  # noqa: BLE001 - IncompleteRead, resets
        print(
          f'[fetch] {name}: {type(e).__name__}, retry {attempt + 1}',
          flush=True,
        )
    raise RuntimeError(f'{name}: gave up after 8 tries')

  manifest = json.loads(get('manifest.json'))
  (plt / 'manifest.json').write_text(json.dumps(manifest))
  for name in [f'L{l:02d}.safetensors' for l in range(42)] + ['labels.json']:
    if not (plt / name).exists():
      pull(name)
      volume.commit()  # a later run resumes from here
      print(f'[fetch] {name}', flush=True)
  volume.commit()


@app.cls(
  image=image,
  gpu='L40S',
  volumes={VOL: volume},
  cpu=2,
  memory=40
  * 1024,  # host copy of the dequantised weights (~16 GB) for the memory snapshot
  scaledown_window=60,  # $30/mo budget: cold starts are fine, idle GPU time is the bill
  max_containers=1,  # the cost ceiling is one GPU; a queue beats a second container
  timeout=300,
  enable_memory_snapshot=True,
)
class Gemma:
  @modal.enter(snap=True)
  def load(self):
    """Before the snapshot: dequantise into host memory, so a cold start skips the 5 GB read
    and the group unpacking and only pays the copy to the GPU."""
    import json

    import gemma_torch as g

    self.g = g
    self.model, self.tok = g.load_model(VOL / 'gemma', device='cpu')
    manifest = json.loads((VOL / 'plt' / 'manifest.json').read_text())
    self.tcs = g.load_transcoders(
      VOL / 'plt', READOUT, int(manifest['arch']['k'])
    )
    self.labels = json.loads((VOL / 'plt' / 'labels.json').read_text())
    self.stops = g.stop_ids(VOL / 'gemma', self.tok)

  @modal.enter(snap=False)
  def to_gpu(self):
    import torch

    self.device = 'cuda'
    self.model.to(self.device)
    for tc in self.tcs.values():
      for k, v in tc.items():
        if isinstance(v, torch.Tensor):
          tc[k] = v.to(self.device)
    self.taps = self.g.Taps(self.model, self.tcs)

  @modal.asgi_app(requires_proxy_auth=True)
  def web(self):
    import json
    import threading

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    lock = (
      threading.Lock()
    )  # one generation at a time; the taps are per-model state

    @api.post('/read')
    async def read(request: Request):
      req = await request.json()
      if not isinstance(req, dict) or not str(req.get('query') or '').strip():
        return JSONResponse(
          {'error': 'want {query, passages}'}, status_code=400
        )

      def stream():
        with lock:
          for ev in self.g.read(
            self.model,
            self.tok,
            self.taps,
            req,
            labels=self.labels,
            stops=self.stops,
            device=self.device,
          ):
            yield json.dumps(ev, ensure_ascii=False) + '\n'

      return StreamingResponse(stream(), media_type='application/x-ndjson')

    @api.post('/graph')
    async def graph():
      return JSONResponse(
        {
          'error': 'live circuits are not on this endpoint yet: attribute.py is mlx-only'
        },
        status_code=501,
      )

    @api.get('/health')
    async def health():
      return {'model': MODEL, 'revision': REVISION, 'readout': list(READOUT)}

    return api


@app.function(
  image=image, volumes={VOL: volume}, gpu='L40S', memory=48 * 1024, timeout=900
)
def check(
  query: str = 'what does aaron think about attention', dtype: str = 'bfloat16'
):
  """On the real GPU: the tiny-model checks, then one real /read with timings."""
  import json
  import subprocess
  import sys

  subprocess.run(
    [
      sys.executable,
      '/root/gemma_torch.py',
      '--tiny',
      '--model',
      str(VOL / 'gemma'),
    ],
    check=True,
  )
  import time

  import gemma_torch as g
  import torch

  t0 = time.perf_counter()
  model, tok = g.load_model(
    VOL / 'gemma', dtype=getattr(torch, dtype), device='cpu'
  )
  print(f'[dtype] {dtype}')
  t1 = time.perf_counter()
  model.to('cuda')
  torch.cuda.synchronize()
  t2 = time.perf_counter()
  tcs = g.load_transcoders(VOL / 'plt', READOUT, 64, device='cuda')
  print(
    f'[load] dequant to host {t1 - t0:.1f} s, host to gpu {t2 - t1:.1f} s, transcoders {time.perf_counter() - t2:.1f} s, gpu {torch.cuda.memory_allocated() / 2**30:.1f} GiB'
  )
  taps = g.Taps(model, tcs)
  passages = [
    dict(
      row=1,
      slug='a',
      text='attention is a soft dictionary lookup over keys and values.',
    ),
    dict(row=2, slug='b', text='the high park loop is 5.6 km with one climb.'),
  ]
  req = dict(query=query, passages=passages, max_tokens=80)
  stops = g.stop_ids(VOL / 'gemma', tok)
  for run in ('cold', 'warm', 'warm'):
    t = time.perf_counter()
    text = ''
    for ev in g.read(
      model, tok, taps, req, labels={}, stops=stops, device='cuda'
    ):
      text += ev.get('t', '')
      if ev.get('done'):
        print(
          f'[{run}]',
          round(time.perf_counter() - t, 2),
          's',
          json.dumps(ev, ensure_ascii=False),
        )
  print('[text]', text)
  print(f'[peak] gpu {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB')


@app.local_entrypoint()
def bench(gpu: str = 'L40S', dtype: str = 'bfloat16'):
  """The check on another GPU or dtype: `modal run ...::bench --gpu L4 --dtype float32`."""
  check.with_options(gpu=gpu).remote(dtype=dtype)
