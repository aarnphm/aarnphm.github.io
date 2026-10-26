"""Stream read-only observations of the PDF queue and embedding publications."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from inputs import Inventory, JsonFiles, extraction_spec, mapping, number, text
from logs import Logs
from models import Models


def iso(timestamp: float) -> str:
  return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def age(value: object) -> float | None:
  if not isinstance(value, str):
    return None
  try:
    return max(0, time.time() - datetime.fromisoformat(value).timestamp())
  except ValueError:
    return None


def alive(pid: int) -> bool:
  if pid <= 0:
    return False
  try:
    os.kill(pid, 0)
    return True
  except (ProcessLookupError, PermissionError):
    return False


@dataclass
class Document:
  slug: str
  title: str
  totalPages: int
  ocrPages: int
  indexedPages: int
  remainingPages: int
  invalidPages: int
  lastPage: int | None
  lastProgressAt: str | None
  status: str


class Observer:
  def __init__(self, root: Path) -> None:
    self.root = root
    self.cache = root / 'quartz/.quartz-cache/pdf-search'
    self.run = self.cache / 'background-papers'
    self.files = JsonFiles()
    self.inventory = Inventory(root)
    self.logs = Logs()
    self.models = Models()
    self.starts: dict[int, float] = {}
    self.queue_updated: str | None = None

  def started_at(self, pid: int) -> float | None:
    if pid in self.starts:
      return self.starts[pid]
    if not alive(pid):
      return None
    # Bound timing logs to the current runner's process lifetime.
    try:
      result = subprocess.run(
        ['ps', '-p', str(pid), '-o', 'lstart='],
        capture_output=True,
        text=True,
        timeout=2,
        env={**os.environ, 'TZ': 'UTC', 'LC_ALL': 'C'},
      )
    except (OSError, subprocess.TimeoutExpired):
      return None
    try:
      started = (
        datetime
        .strptime(result.stdout.strip(), '%a %b %d %H:%M:%S %Y')
        .replace(tzinfo=timezone.utc)
        .timestamp()
      )
    except ValueError:
      return None
    self.starts[pid] = started
    return started

  def indexes(self) -> list[dict[str, object]]:
    result = []
    for name, directory in [
      ('notes', ''),
      ('images', 'images'),
      ('pdf', 'pdf'),
    ]:
      source = self.root / 'content/embeddings' / directory / 'manifest.json'
      generated = self.files.read(source)
      served = self.files.read(
        self.root / 'public/embeddings' / directory / 'manifest.json'
      )
      shards = mapping(generated.get('vectors')).get('shards')
      result.append({
        'name': name,
        'rows': generated.get('rows'),
        'servedRows': served.get('rows'),
        'shards': len(shards) if isinstance(shards, list) else None,
        'generation': generated.get('generation'),
        'updatedAt': iso(source.stat().st_mtime) if source.exists() else None,
      })
    bases = sorted(
      (self.root / 'content/embeddings/bases').glob('*/manifest.json')
    )
    if bases:
      manifests = [self.files.read(path) for path in bases]
      result.append({
        'name': 'bases',
        'rows': sum(number(m.get('rows')) for m in manifests),
        'servedRows': sum(
          number(
            self.files.read(
              self.root
              / 'public/embeddings/bases'
              / p.parent.name
              / 'manifest.json'
            ).get('rows')
          )
          for p in bases
        ),
        'shards': sum(
          len(s)
          for m in manifests
          if isinstance(s := mapping(m.get('vectors')).get('shards'), list)
        ),
        'generation': f'{len(bases)} bases',
        'updatedAt': iso(max(p.stat().st_mtime for p in bases)),
      })
    return result

  def snapshot(self) -> dict[str, object]:
    self.files.errors.clear()
    state = self.files.read(self.run / 'status.json')
    queue = self.files.read(self.cache / 'queue.json')
    queue_jobs = [mapping(job) for job in queue.get('jobs', [])]
    if queue.get('updatedAt') != self.queue_updated:
      self.inventory.updated = 0
      self.queue_updated = text(queue.get('updatedAt'))
    prefix = text(state.get('prefix'), 'thoughts/papers/')
    index = self.root / 'public/static/searchIndex.json'
    command = state.get('command')
    if isinstance(command, list) and '--public-index' in command:
      position = command.index('--public-index') + 1
      if position < len(command) and isinstance(command[position], str):
        index = self.root / command[position]
    self.inventory.refresh(index, prefix, number(state.get('firstPages')))
    spec = extraction_spec(self.files.read(self.cache / 'recipe.json'))
    manifest = self.files.read(
      self.root / 'content/embeddings/pdf/manifest.json'
    )
    served = self.files.read(self.root / 'public/embeddings/pdf/manifest.json')
    published = mapping(manifest.get('documents'))
    served_docs = mapping(served.get('documents'))
    phase = text(state.get('phase'), 'idle')
    runner_age = age(state.get('updatedAt'))
    running = alive(number(state.get('pid')))
    healthy = running and runner_age is not None and runner_age <= 20
    active_log = (
      self.logs.read(self.root / text(state.get('log')))
      if state.get('log')
      else None
    )
    documents = []
    remaining_work: set[tuple[str, int]] = set()
    extracted: dict[str, set[int]] = {}
    served_pages = 0
    current = None
    batch_pages = []
    for source in self.inventory.sources:
      observation = self.inventory.pages(source, self.cache, spec)
      cached = observation.numbers
      remaining_work.update(
        (source.sha256, page)
        for page in range(1, source.total + 1)
        if page not in cached
      )
      if observation.key:
        extracted.setdefault(observation.key, set()).update(
          observation.extracted
        )
      entry = mapping(published.get(source.slug))
      title = text(entry.get('title'), observation.title or source.title)
      indexed = (
        entry.get('pages', [])
        if entry.get('sourceSha256') == source.sha256
        else []
      )
      indexed_count = (
        len({n for n in indexed if type(n) is int and 1 <= n <= source.total})
        if isinstance(indexed, list)
        else 0
      )
      local = mapping(served_docs.get(source.slug))
      local_pages = (
        local.get('pages', [])
        if local.get('sourceSha256') == source.sha256
        else []
      )
      served_pages += (
        len({
          n for n in local_pages if type(n) is int and 1 <= n <= source.total
        })
        if isinstance(local_pages, list)
        else 0
      )
      status = (
        'complete'
        if indexed_count == source.total
        else 'ready'
        if len(cached) > indexed_count
        else 'partial'
        if indexed_count
        else 'queued'
      )
      if active_log and phase in ('extracting', 'error'):
        pending = [
          page
          for page in active_log.planned.get(source.slug, [])
          if page not in cached
        ]
        active = []
        for page in active_log.planned.get(source.slug, []):
          identity = (str(source.path.resolve()), page)
          failure = active_log.failures.get(identity)
          admitted = (
            identity in active_log.admitted and not active_log.stream_closed
          )
          if not active_log.admitted and pending:
            admitted = page == pending[0]
          page_state = (
            'done'
            if page in cached
            else 'failed'
            if failure
            else ('active' if healthy else 'paused')
            if admitted
            else 'queued'
          )
          batch_pages.append({
            'slug': source.slug,
            'page': page,
            'state': page_state,
          })
          if failure:
            self.files.errors.append(f'{source.slug}, page {page}: {failure}')
          if page_state in ('active', 'paused'):
            active.append(page)
        if active:
          status = 'active' if healthy else 'paused'
          if current is None:
            current = {
              'slug': source.slug,
              'title': title,
              'page': active[0],
              'pages': active,
              'totalPages': source.total,
            }
      documents.append(
        Document(
          source.slug,
          title,
          source.total,
          len(cached),
          indexed_count,
          source.total - len(cached),
          observation.invalid,
          max(cached) if cached else None,
          iso(observation.updated_at)
          if observation.updated_at is not None
          else None,
          status,
        )
      )
    total = sum(d.totalPages for d in documents)
    cached_count = sum(d.ocrPages for d in documents)
    indexed_count = sum(d.indexedPages for d in documents)
    complete = sum(d.status == 'complete' for d in documents)
    started_at = self.started_at(number(state.get('pid')))
    history, timings = self.logs.history(self.run, extracted, started_at)
    minimum_samples = 5
    mean = statistics.fmean(t.seconds for t in timings) if timings else None
    estimated = mean if len(timings) >= minimum_samples else None
    remaining = max(0, total - cached_count)
    model_servers = [
      self.models.observe(
        key.removeprefix('server'), value, alive(number(value.get('pid')))
      )
      for key, value in state.items()
      if key.startswith('server') and isinstance(value, dict)
    ]
    return {
      'observedAt': iso(time.time()),
      'prefix': prefix,
      'runner': {
        'pid': state.get('pid'),
        'alive': running,
        'phase': phase,
        'stale': not healthy and phase not in ('complete', 'error', 'idle'),
        'updatedAt': state.get('updatedAt'),
        'ageSeconds': runner_age,
        'error': state.get('error'),
        'batches': number(state.get('batches')),
        'batchPages': number(state.get('batchPages'), 32),
        'batchCompleted': sum(p['state'] == 'done' for p in batch_pages)
        if batch_pages
        else None,
        'batchPlanned': len(batch_pages),
        'batchActive': sum(p['state'] == 'active' for p in batch_pages),
        'batchQueued': sum(p['state'] == 'queued' for p in batch_pages),
        'pages': batch_pages,
        'log': Path(text(state.get('log'))).name,
        'servers': model_servers,
      },
      'queue': {
        'jobs': queue_jobs,
        'pending': sum(
          job.get('state') in ('queued', 'active') for job in queue_jobs
        ),
        'failed': sum(job.get('state') == 'error' for job in queue_jobs),
        'updatedAt': queue.get('updatedAt'),
      },
      'current': current,
      'totals': {
        'documents': len(documents),
        'completeDocuments': complete,
        'remainingDocuments': len(documents) - complete,
        'totalPages': total,
        'ocrPages': cached_count,
        'indexedPages': indexed_count,
        'servedPages': served_pages,
        'remainingOcrPages': remaining,
        'remainingIndexPages': max(0, total - indexed_count),
        'awaitingPublicationPages': max(0, cached_count - indexed_count),
        'vectors': manifest.get('rows'),
        'invalidPages': sum(d.invalidPages for d in documents),
      },
      'rate': {
        'samples': len(timings),
        'minimumSamples': minimum_samples,
        'sampleDocuments': len({t.key for t in timings}),
        'runStartedAt': iso(started_at) if started_at is not None else None,
        'recipe': spec.recipe_sha256 if spec else None,
        'remainingUniquePages': len(remaining_work),
        'secondsPerPage': mean,
        'pagesPerHour': 3600 / estimated if estimated else None,
        'remainingOcrSeconds': (
          len(remaining_work) * estimated if estimated else None
        )
        if remaining_work
        else 0,
      },
      'documents': [asdict(document) for document in documents],
      'indexes': self.indexes(),
      'history': history,
      'issues': list(self.files.errors),
    }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--root', type=Path, default=Path(__file__).resolve().parents[2]
  )
  parser.add_argument('--interval', type=float, default=3)
  parser.add_argument('--once', action='store_true')
  args = parser.parse_args()
  root = args.root.resolve()
  os.chdir(root)
  observer = Observer(root)
  while True:
    try:
      frame = {'type': 'snapshot', 'data': observer.snapshot()}
    except Exception as error:
      frame = {
        'type': 'error',
        'message': str(error),
        'observedAt': iso(time.time()),
      }
    print(json.dumps(frame, allow_nan=False), flush=True)
    if args.once:
      return
    time.sleep(max(0.1, args.interval))


if __name__ == '__main__':
  main()
