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

Public pages read the same files from /models/latent-plt/bake/<sidecar8>/ once published. Only
loopback requests are answered, with CORS open to the dev page.
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BAKE = Path('quartz/.quartz-cache/latent-bake')
SAFE = re.compile(
  r'^/bake/([0-9a-f]{8})/((?:rows|index|gloss|graphs)/[A-Za-z0-9_.-]+|[a-z]+\.(?:json|bin))$'
)
TYPES = {'.json': 'application/json', '.bin': 'application/octet-stream'}


class Handler(BaseHTTPRequestHandler):
  def log_message(self, fmt: str, *args) -> None:  # quiet
    pass

  def _cors(self) -> None:
    self.send_header(
      'Access-Control-Allow-Origin', self.headers.get('Origin') or '*'
    )
    self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
    self.send_header('Vary', 'Origin')

  def do_OPTIONS(self) -> None:
    self.send_response(204)
    self._cors()
    self.end_headers()

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
