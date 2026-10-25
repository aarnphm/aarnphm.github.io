"""Persist public PDF requests independently of clients and inference processes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal

import portalocker
import pymupdf

from build import PdfSource, public_pdfs
from cache import canonical


@dataclass
class Job:
  slug: str
  sourceSha256: str
  title: str
  totalPages: int
  queuedAt: str
  state: Literal['queued', 'active', 'complete', 'error'] = 'queued'
  error: str | None = None


def source_hash(source: Path) -> str:
  with source.open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def published_pages(manifest: Path, slug: str, sha: str, total: int) -> bool:
  if not manifest.exists():
    return False
  document = (
    json.loads(manifest.read_text()).get('documents', {}).get(slug, {})
  )
  return document.get('sourceSha256') == sha and set(
    range(1, total + 1)
  ).issubset(document.get('pages', []))


class PendingPdfs:
  def __init__(self, root: Path) -> None:
    self.root = root.resolve()
    self.cache = self.root / 'quartz/.quartz-cache/pdf-search'
    self.path = self.cache / 'queue.json'
    self.index = self.root / 'public/static/searchIndex.json'
    self.manifest = self.root / 'content/embeddings/pdf/manifest.json'
    self.cache.mkdir(parents=True, exist_ok=True)

  def lock(self) -> portalocker.Lock:
    return portalocker.Lock(self.cache / '.queue.lock', timeout=10)

  def read(self) -> list[Job]:
    if not self.path.exists():
      return []
    data = json.loads(self.path.read_text())
    if data.get('version') != 1 or not isinstance(data.get('jobs'), list):
      raise ValueError('Unsupported PDF queue format')
    return [Job(**job) for job in data['jobs']]

  def write(self, jobs: list[Job]) -> None:
    temporary = self.path.with_suffix('.tmp')
    temporary.write_bytes(
      canonical({
        'version': 1,
        'updatedAt': datetime.now(timezone.utc).isoformat(),
        'jobs': [asdict(job) for job in jobs],
      })
    )
    temporary.replace(self.path)

  def inventory(self) -> dict[str, PdfSource]:
    return {
      slug: source
      for slug, source in public_pdfs(
        self.index, self.root / 'content'
      ).items()
      if slug.startswith('thoughts/papers/')
    }

  def enqueue(self, slugs: list[str]) -> dict[str, object]:
    inventory = self.inventory()
    selected = list(dict.fromkeys(slugs)) if slugs else sorted(inventory)
    unknown = set(selected) - inventory.keys()
    if unknown:
      raise ValueError(
        f'PDF is absent from the public paper inventory: {sorted(unknown)}'
      )
    candidates: list[Job] = []
    for slug in selected:
      source = inventory[slug]
      with pymupdf.open(source.source) as pdf:
        count = len(pdf)
      if count < 1:
        raise ValueError(f'PDF has no pages: {slug}')
      candidates.append(
        Job(
          slug,
          source_hash(source.source),
          source.title,
          count,
          datetime.now(timezone.utc).isoformat(),
        )
      )
    queued = 0
    with self.lock():
      jobs = {job.slug: job for job in self.read()}
      for candidate in candidates:
        old = jobs.get(candidate.slug)
        if published_pages(
          self.manifest,
          candidate.slug,
          candidate.sourceSha256,
          candidate.totalPages,
        ):
          if old:
            candidate.state = 'complete'
            jobs[candidate.slug] = candidate
          continue
        if (
          old
          and old.sourceSha256 == candidate.sourceSha256
          and old.state in ('queued', 'active')
        ):
          continue
        jobs[candidate.slug] = candidate
        queued += 1
      self.write(list(jobs.values()))
      pending = sum(job.state in ('queued', 'active') for job in jobs.values())
    return {
      'queued': queued,
      'pending': pending,
      'checked': len(candidates),
      'queue': str(self.path),
    }

  def recover(self) -> None:
    with self.lock():
      jobs = self.read()
      for job in jobs:
        if job.state == 'active':
          job.state = 'queued'
      self.write(jobs)

  def claim(self, worker_lock: portalocker.Lock) -> list[Job]:
    with self.lock():
      jobs = self.read()
      selected = [job for job in jobs if job.state == 'queued']
      if not selected:
        # Release ownership inside the enqueue transaction, so a concurrent client
        # cannot lose its wake-up between an empty check and worker exit.
        worker_lock.release()
        return []
      for job in selected:
        job.state = 'active'
        job.error = None
      self.write(jobs)
      return selected

  def finish(self, claimed: list[Job], error: str | None) -> None:
    inventory = self.inventory()
    with self.lock():
      jobs = self.read()
      selected = {(job.slug, job.sourceSha256) for job in claimed}
      for job in jobs:
        if (job.slug, job.sourceSha256) not in selected:
          continue
        source = inventory.get(job.slug)
        if (
          source is not None
          and source_hash(source.source) == job.sourceSha256
          and published_pages(
            self.manifest, job.slug, job.sourceSha256, job.totalPages
          )
        ):
          job.state, job.error = 'complete', None
        else:
          job.state, job.error = (
            'error',
            error or 'Publication did not verify the queued source and pages',
          )
      self.write(jobs)
