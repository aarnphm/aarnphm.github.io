"""Own local model servers and resume PDF extraction through final publication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
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
from cache import canonical, ExtractionSpec, PageCache
from contract import OCR_MODELS

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
    (args.run_dir / 'pid').write_text(str(os.getpid()))
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
        cwd=self.args.root,
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
  ) -> subprocess.Popen[bytes] | None:
    def health() -> dict[str, object]:
      if script.name == 'mlx-server.py':
        from extract import server_identity

        value = server_identity(port)
      else:
        with urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as response:
          value = json.load(response)
        if value.get('status') != 'ok':
          raise ValueError(f'Embedding server on port {port} is not ready')
      if any(value.get(key) != expected for key, expected in identity.items()):
        raise ValueError(
          f'Unexpected model identity on port {port}; its process is untouched'
        )
      return value

    with socket.socket() as probe:
      occupied = probe.connect_ex(('127.0.0.1', port)) == 0
    if occupied:
      self.update(**{
        f'server{port}': {**health(), 'pid': None, 'owned': False}
      })
      return None
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
        self.update(**{
          f'server{port}': {**health(), 'pid': process.pid, 'owned': True}
        })
        return process
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
      and (not self.args.slug or slug in self.args.slug)
    }
    if not sources:
      raise ValueError(f'No public PDFs match {self.args.prefix}')
    manifest = self.args.out / 'manifest.json'
    documents = (
      json.loads(manifest.read_text())['documents']
      if manifest.exists()
      else {}
    )
    total, indexed, complete, cached = 0, 0, 0, 0
    recipe_path = self.args.cache / 'recipe.json'
    spec = (
      ExtractionSpec(**json.loads(recipe_path.read_text()))
      if recipe_path.exists()
      else None
    )
    for slug, source in sources.items():
      with pymupdf.open(source.source) as pdf:
        count = len(pdf)
      if self.args.first_pages:
        count = min(count, self.args.first_pages)
      with source.source.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
      document = documents.get(slug, {})
      pages = (
        document.get('pages', [])
        if document.get('sourceSha256') == sha
        else []
      )
      available = len(set(pages) & set(range(1, count + 1)))
      if spec:
        cache = PageCache(self.args.cache, sha, spec)
        cached += sum(
          cache.read(number) is not None for number in range(1, count + 1)
        )
      total += count
      indexed += available
      complete += available == count
    return {
      'documents': len(sources),
      'totalPages': total,
      'indexedPages': indexed,
      'completeDocuments': complete,
      'cachedPages': cached,
    }

  def run(self) -> None:
    args = self.args
    coverage = self.coverage()
    self.update(
      phase='starting',
      prefix=args.prefix,
      batchPages=args.batch_pages,
      firstPages=args.first_pages,
      ocrConcurrency=args.ocr_concurrency,
      pipelinePages=args.pipeline_pages,
      ocrPrecision=args.ocr_precision,
      coverage=coverage,
      slugs=args.slug,
      publicIndex=str(args.public_index),
      error=None,
    )
    if coverage['indexedPages'] == coverage['totalPages']:
      self.update(
        phase='complete', completedAt=datetime.now(timezone.utc).isoformat()
      )
      return
    self.start(
      ['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())],
      args.run_dir / 'caffeinate.log',
    )
    if args.embed_port is None:
      args.embed_port = 8003
      try:
        with urlopen('http://127.0.0.1:8000/health', timeout=2) as response:
          health = json.load(response)
        if (
          health.get('repo') == 'mlx-community/embeddinggemma-2-bf16'
          and health.get('revision') == EMBED_REVISION
          and health.get('status') == 'ok'
        ):
          args.embed_port = 8000
      except (OSError, ValueError):
        pass
    ocr_ready = False
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
      if coverage['cachedPages'] < coverage['totalPages']:
        if not ocr_ready:
          model, revision = OCR_MODELS[args.ocr_precision]
          self.server(
            SCRIPTS / 'mlx-server.py',
            args.ocr_port,
            {'model': model, 'revision': revision},
            '--max-num-seqs',
            str(args.ocr_concurrency),
            '--precision',
            args.ocr_precision,
          )
          ocr_ready = True
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
            '--concurrency',
            str(args.ocr_concurrency),
            '--pipeline-pages',
            str(args.pipeline_pages),
            *[value for slug in args.slug for value in ('--slug', slug)],
          ],
        )
      else:
        batch = {'completedPages': 0}
      if type(batch.get('completedPages')) is not int:
        raise ValueError('Extraction omitted its completed-page count')
      self.update(lastBatch=batch)
      embedding = self.server(
        ROOT / 'quartz/embed_server.py',
        args.embed_port,
        {
          'repo': 'mlx-community/embeddinggemma-2-bf16',
          'revision': EMBED_REVISION,
        },
        '--revision',
        EMBED_REVISION,
      )
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
      if embedding is not None:
        stop(embedding)
        self.children.remove(embedding)
      self.batches += 1
      coverage = self.coverage()
      self.update(
        batches=self.batches, publication=published, coverage=coverage
      )
      print(json.dumps(self.state), flush=True)
      if coverage['indexedPages'] == coverage['totalPages']:
        self.update(
          phase='complete', completedAt=datetime.now(timezone.utc).isoformat()
        )
        return
      if batch['completedPages'] == 0:
        raise RuntimeError('Extraction stopped before full scope publication')

  def close(self) -> None:
    for process in reversed(self.children):
      stop(process)


def interrupted(signum: int, _frame: object) -> None:
  raise InterruptedError(f'Runner received signal {signum}; caches preserved')


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root', type=Path, default=ROOT)
  parser.add_argument('--slug', action='append', default=[])
  parser.add_argument('--wait-for-lock', action='store_true')
  parser.add_argument('--prefix', default='thoughts/papers/')
  parser.add_argument('--batch-pages', type=int, default=64)
  parser.add_argument('--first-pages', type=int, default=0)
  parser.add_argument('--ocr-port', type=int, default=8001)
  parser.add_argument('--ocr-concurrency', type=int, default=8)
  parser.add_argument('--pipeline-pages', type=int, default=8)
  parser.add_argument('--ocr-precision', choices=OCR_MODELS, default='8bit')
  parser.add_argument('--embed-port', type=int)
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
    or not 1 <= args.ocr_concurrency <= 16
    or not 1 <= args.pipeline_pages <= 16
  ):
    parser.error(
      'Use a positive batch size, nonnegative page limit, distinct ports, and concurrency/lookahead 1-16'
    )
  args.root = args.root.resolve()
  os.chdir(args.root)
  args.run_dir.mkdir(parents=True, exist_ok=True)
  signal.signal(signal.SIGTERM, interrupted)
  signal.signal(signal.SIGINT, interrupted)
  flags = (
    portalocker.LOCK_EX
    if args.wait_for_lock
    else portalocker.LOCK_EX | portalocker.LOCK_NB
  )
  with portalocker.Lock(
    args.cache / '.runner.lock',
    timeout=None if args.wait_for_lock else 0,
    flags=flags,
  ):
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
