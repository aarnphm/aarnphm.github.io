"""Own local model servers and resume PDF extraction through final publication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import portalocker
import pymupdf

from build import public_pdfs
from cache import canonical
from extract import OCR_MODEL, OCR_REVISION

EMBED_REVISION = '1a4ffddb7905d3f63486748deabe091a01fb6201'
SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[2]


def stop(process: subprocess.Popen[bytes]) -> None:
  if process.poll() is not None:
    return
  try:
    os.killpg(process.pid, signal.SIGTERM)
    process.wait(timeout=20)
  except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=5)
  except ProcessLookupError:
    pass


class Runner:
  def __init__(self, args: argparse.Namespace) -> None:
    self.args = args
    self.children: list[subprocess.Popen[bytes]] = []
    self.status = args.run_dir / 'status.json'
    self.state: dict[str, object] = (
      json.loads(self.status.read_text()) if self.status.exists() else {}
    )
    self.batches = int(str(self.state.get('batches', 0)))
    self.uv = shutil.which('uv')
    if self.uv is None:
      raise RuntimeError('uv is required to run the isolated MLX servers')

  def update(self, **values: object) -> None:
    self.state.update(values)
    self.state.update(
      updatedAt=datetime.now(timezone.utc).isoformat(), pid=os.getpid()
    )
    temporary = self.status.with_suffix('.tmp')
    temporary.write_bytes(canonical(self.state))
    temporary.replace(self.status)

  def start(self, command: list[str], log: Path) -> subprocess.Popen[bytes]:
    with log.open('ab') as output:
      process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, 'PYTHONUNBUFFERED': '1'},
      )
    self.children.append(process)
    return process

  def server(
    self, script: Path, port: int, identity: dict[str, str], *extra: str
  ) -> None:
    with socket.socket() as probe:
      if probe.connect_ex(('127.0.0.1', port)) == 0:
        raise RuntimeError(
          f'Port {port} is occupied; its process is untouched'
        )
    process = self.start(
      [str(self.uv), 'run', str(script), '--port', str(port), *extra],
      self.args.run_dir / f'server-{port}.log',
    )
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
      if process.poll() is not None:
        raise RuntimeError(
          f'Model server on port {port} exited during startup'
        )
      try:
        with urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
          health = json.load(response)
        if any(health.get(key) != value for key, value in identity.items()):
          raise ValueError(f'Unexpected model identity on port {port}')
        self.update(**{f'server{port}': {'pid': process.pid, **health}})
        return
      except (URLError, TimeoutError):
        time.sleep(2)
    raise TimeoutError(f'Model server on port {port} did not become ready')

  def command(self, name: str, command: list[str]) -> dict[str, object]:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    log = self.args.run_dir / f'{stamp}-{name}.log'
    self.update(phase=name, command=command, log=str(log), error=None)
    process = self.start(command, log)
    try:
      while process.poll() is None:
        if time.time() - log.stat().st_mtime > 900:
          raise TimeoutError(f'{name} made no logged progress for 15 minutes')
        self.update()
        time.sleep(5)
      if process.returncode:
        raise RuntimeError(
          f'{name} exited {process.returncode}; inspect {log}'
        )
    finally:
      stop(process)
      self.children.remove(process)
    for line in reversed(log.read_text().splitlines()):
      try:
        report = json.loads(line)
      except json.JSONDecodeError:
        continue
      if isinstance(report, dict):
        return report
    raise ValueError(f'{name} produced no completion report: {log}')

  def coverage(self) -> dict[str, int]:
    inventory = public_pdfs(self.args.public_index, self.args.content)
    sources = {
      slug: source
      for slug, source in inventory.items()
      if slug.startswith(self.args.prefix)
    }
    if not sources:
      raise ValueError(f'No public PDFs match {self.args.prefix}')
    manifest = self.args.out / 'manifest.json'
    documents = (
      json.loads(manifest.read_text())['documents']
      if manifest.exists()
      else {}
    )
    total, indexed, complete = 0, 0, 0
    for slug, source in sources.items():
      with pymupdf.open(source.source) as pdf:
        count = len(pdf)
      if self.args.first_pages:
        count = min(count, self.args.first_pages)
      pages = documents.get(slug, {}).get('pages', [])
      available = sum(number <= count for number in pages)
      total += count
      indexed += available
      complete += available == count
    return {
      'documents': len(sources),
      'totalPages': total,
      'indexedPages': indexed,
      'completeDocuments': complete,
    }

  def run(self) -> None:
    args = self.args
    self.update(
      phase='starting',
      prefix=args.prefix,
      batchPages=args.batch_pages,
      firstPages=args.first_pages,
      coverage=self.coverage(),
      error=None,
    )
    self.start(
      ['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())],
      args.run_dir / 'caffeinate.log',
    )
    self.server(
      SCRIPTS / 'mlx-server.py',
      args.ocr_port,
      {'model': OCR_MODEL, 'revision': OCR_REVISION},
    )
    self.server(
      ROOT / 'quartz/embed_server.py',
      args.embed_port,
      {
        'repo': 'mlx-community/embeddinggemma-2-bf16',
        'revision': EMBED_REVISION,
      },
      '--revision',
      EMBED_REVISION,
    )
    common = [
      '--cache',
      str(args.cache),
      '--content',
      str(args.content),
      '--public-index',
      str(args.public_index),
    ]
    while True:
      if shutil.disk_usage(args.cache).free < 10 * (1 << 30):
        raise RuntimeError(
          'Less than 10 GiB free; cached progress is preserved'
        )
      batch = self.command(
        'extracting',
        [
          sys.executable,
          str(SCRIPTS / 'batch.py'),
          *common,
          '--prefix',
          args.prefix,
          '--first-pages',
          str(args.first_pages),
          '--max-new-pages',
          str(args.batch_pages),
          '--port',
          str(args.ocr_port),
        ],
      )
      if type(batch.get('completedPages')) is not int:
        raise ValueError('Extraction omitted its completed-page count')
      self.update(lastBatch=batch)
      published = self.command(
        'publishing',
        [
          sys.executable,
          str(SCRIPTS / 'build.py'),
          *common,
          '--out',
          str(args.out),
          '--port',
          str(args.embed_port),
        ],
      )
      self.batches += 1
      coverage = self.coverage()
      self.update(
        batches=self.batches, publication=published, coverage=coverage
      )
      print(json.dumps(self.state), flush=True)
      if batch['completedPages'] == 0:
        if coverage['indexedPages'] != coverage['totalPages']:
          raise RuntimeError(
            'Extraction stopped before full scope publication'
          )
        self.update(
          phase='complete', completedAt=datetime.now(timezone.utc).isoformat()
        )
        return

  def close(self) -> None:
    for process in reversed(self.children):
      stop(process)


def interrupted(signum: int, _frame: object) -> None:
  raise InterruptedError(f'Runner received signal {signum}; caches preserved')


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--prefix', default='thoughts/papers/')
  parser.add_argument('--batch-pages', type=int, default=32)
  parser.add_argument('--first-pages', type=int, default=0)
  parser.add_argument('--ocr-port', type=int, default=8001)
  parser.add_argument('--embed-port', type=int, default=8003)
  parser.add_argument(
    '--public-index', type=Path, default=Path('public/static/searchIndex.json')
  )
  parser.add_argument('--content', type=Path, default=Path('content'))
  parser.add_argument(
    '--cache', type=Path, default=Path('quartz/.quartz-cache/pdf-search')
  )
  parser.add_argument(
    '--out', type=Path, default=Path('content/embeddings/pdf')
  )
  parser.add_argument(
    '--run-dir',
    type=Path,
    default=Path('quartz/.quartz-cache/pdf-search/background-papers'),
  )
  args = parser.parse_args()
  if (
    args.batch_pages < 1
    or args.first_pages < 0
    or args.ocr_port == args.embed_port
  ):
    parser.error(
      'Use a positive batch size, nonnegative page limit and distinct ports'
    )
  os.chdir(ROOT)
  args.run_dir.mkdir(parents=True, exist_ok=True)
  signal.signal(signal.SIGTERM, interrupted)
  signal.signal(signal.SIGINT, interrupted)
  with portalocker.Lock(args.cache / '.runner.lock', timeout=0):
    runner = Runner(args)
    try:
      runner.run()
    except Exception as error:
      runner.update(phase='error', error=str(error))
      raise
    finally:
      runner.close()


if __name__ == '__main__':
  main()
