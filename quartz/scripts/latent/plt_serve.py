# /// script
# requires-python = ">=3.11"
# ///
"""Serve the Gemma bake to /embeddings on localhost (the selection tab reads it from here first).

    .venv/bin/python quartz/scripts/latent/plt_serve.py [--port 8093]

  GET /bake/<sidecar8>/meta.json, rows/<row>.bin, index/L<ll>.bin, gloss/L<ll>.json,
      graphs/<row>.json, pooled.bin
      from quartz/.quartz-cache/latent-bake/<sidecar8>/ (plt_bake.py)
  GET /bake/<sidecar8>/labels.json
      the transcoders' labels (the plt folder named in the bake's meta.json)
  POST /graph {h8, row, text, end}
      the attribution graph of the token after a reader selection (plt_graphs.live_graph). A
      running plt_graphs.py answers it between passages through <bake>/live/, so Gemma and the
      transcoders sit in memory once; with none running, this process loads them itself and
      lets them go after 10 idle minutes. 503 while a graph run is paused on battery or the bake
      holds the memory.

Public pages read the same files from /models/latent-plt/bake/<sidecar8>/ once published. Only
loopback requests are answered, with CORS open to the dev page.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BAKE = Path('quartz/.quartz-cache/latent-bake')
SAFE = re.compile(
  r'^/bake/([0-9a-f]{8})/((?:rows|index|gloss|graphs)/[A-Za-z0-9_.-]+|[a-z]+\.(?:json|bin))$'
)
TYPES = {'.json': 'application/json', '.bin': 'application/octet-stream'}
WAIT = 180  # s for a graph run to answer: one passage (~15 s) then the request (~12 s)
IDLE = 600  # s before an in-process model is let go
CTX, NODES = 32, 200


def pids(pattern: str) -> list[tuple[int, str]]:
  """(pid, ps state) of the processes whose command line matches"""
  got = subprocess.run(
    ['pgrep', '-f', pattern], capture_output=True, text=True
  ).stdout.split()
  out = []
  for p in got:
    st = subprocess.run(
      ['ps', '-o', 'state=', '-p', p], capture_output=True, text=True
    ).stdout
    if st.strip():
      out.append((int(p), st.strip()))
  return out


class Local:
  """Gemma and the transcoders in this process, for when no graph run is up."""

  def __init__(self) -> None:
    self.lock = threading.Lock()
    self.kw: dict | None = None
    self.used = 0.0
    threading.Thread(target=self._reaper, daemon=True).start()

  def _reaper(self) -> None:
    while True:
      time.sleep(60)
      with self.lock:
        if self.kw and time.time() - self.used > IDLE:
          self.kw = None
          import mlx.core as mx

          mx.clear_cache()
          print('[plt-serve] model released', flush=True)

  def graph(self, plt: str, req: dict) -> dict:
    with self.lock:
      if self.kw is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import mlx.core as mx
        from mlx_lm import load

        import acts
        from attribute import load_plt, unembed_mean
        from label import MODEL, REVISION, resolve

        mx.set_cache_limit(2 << 30)
        model, tok = load(resolve(MODEL, REVISION))
        lm = acts.text_model(model)
        t = getattr(tok, '_tokenizer', tok)
        tcs = load_plt(Path(plt), len(lm.model.layers))
        self.kw = dict(
          model=model,
          lm=lm,
          t=t,
          tok=tok,
          tcs=tcs,
          mean=unembed_mean(lm),
          plt=plt,
        )
        print('[plt-serve] model loaded', flush=True)
      from plt_graphs import live_graph

      self.used = time.time()
      return live_graph(**self.kw, req=req, ctx=CTX, nodes=NODES)


LOCAL = Local()


def live(h8: str, req: dict) -> tuple[int, dict]:
  root = BAKE / h8
  meta = json.loads((root / 'meta.json').read_text())
  runs = pids('plt_graphs.py')
  if runs:
    if all(state.startswith('T') for _, state in runs):
      return 503, dict(error='the graph run is paused on battery')
    box = root / 'live'
    box.mkdir(exist_ok=True)
    rid = uuid.uuid4().hex[:12]
    (box / f'.req-{rid}.json').write_text(json.dumps(req, ensure_ascii=False))
    (box / f'.req-{rid}.json').rename(box / f'req-{rid}.json')
    done = box / f'{rid}.json'
    t0 = time.time()
    while time.time() - t0 < WAIT:
      if done.exists():
        body = json.loads(done.read_text())
        done.unlink(missing_ok=True)
        return (500 if 'error' in body else 200), body
      time.sleep(0.5)
    (box / f'req-{rid}.json').unlink(missing_ok=True)
    return 504, dict(error='the graph run did not answer')
  if pids('plt_bake.py'):
    return 503, dict(error='the bake is running')
  try:
    return 200, LOCAL.graph(meta['plt'], req)
  except Exception as e:  # noqa: BLE001 - the reader shows it
    return 500, dict(error=f'{type(e).__name__}: {e}')


class Handler(BaseHTTPRequestHandler):
  def log_message(self, fmt: str, *args) -> None:  # quiet
    pass

  def _cors(self) -> None:
    self.send_header(
      'Access-Control-Allow-Origin', self.headers.get('Origin') or '*'
    )
    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    self.send_header('Vary', 'Origin')

  def do_OPTIONS(self) -> None:
    self.send_response(204)
    self._cors()
    self.end_headers()

  def do_POST(self) -> None:
    if (
      self.client_address[0] not in ('127.0.0.1', '::1')
      or self.path != '/graph'
    ):
      self.send_error(403 if self.path == '/graph' else 404)
      return
    try:
      req = json.loads(
        self.rfile.read(int(self.headers.get('Content-Length') or 0))
      )
      h8 = str(req['h8'])
      assert (
        re.fullmatch(r'[0-9a-f]{8}', h8)
        and (BAKE / h8 / 'meta.json').is_file()
      )
      req = dict(
        row=int(req['row']), text=str(req['text']), end=int(req['end'])
      )
    except Exception:  # noqa: BLE001
      code, body = 400, dict(error='want {h8, row, text, end}')
    else:
      code, body = live(h8, req)
    raw = json.dumps(body, ensure_ascii=False).encode()
    self.send_response(code)
    self._cors()
    self.send_header('Content-Type', 'application/json')
    self.send_header('Content-Length', str(len(raw)))
    self.end_headers()
    self.wfile.write(raw)

  def do_GET(self) -> None:
    if self.client_address[0] not in ('127.0.0.1', '::1'):
      self.send_error(403)
      return
    m = SAFE.match(self.path.split('?', 1)[0])
    if not m:
      self.send_error(404)
      return
    root = BAKE / m.group(1)
    rel = m.group(2)
    if rel == 'labels.json':
      meta = json.loads((root / 'meta.json').read_text())
      path = Path(meta['plt']) / 'labels.json'
    else:
      path = root / rel
    if not path.is_file():
      self.send_response(404)
      self._cors()
      self.send_header('Content-Length', '0')
      self.end_headers()
      return
    body = path.read_bytes()
    self.send_response(200)
    self._cors()
    self.send_header(
      'Content-Type', TYPES.get(path.suffix, 'application/octet-stream')
    )
    self.send_header('Content-Length', str(len(body)))
    self.send_header('Cache-Control', 'no-cache')
    self.end_headers()
    self.wfile.write(body)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--port', type=int, default=8093)
  a = ap.parse_args()
  print(f'[plt-serve] http://localhost:{a.port}/bake/ from {BAKE}', flush=True)
  ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()


if __name__ == '__main__':
  main()
