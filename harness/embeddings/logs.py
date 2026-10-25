"""Reduce the runner's append-only batch logs to timings and publication history."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from inputs import mapping, number, text


@dataclass(frozen=True)
class PageTiming:
  key: str
  page: int
  seconds: float


@dataclass
class BatchLog:
  signature: tuple[int, int]
  planned: dict[str, list[int]] = field(default_factory=dict)
  admitted: set[tuple[str, int]] = field(default_factory=set)
  failures: dict[tuple[str, int], str] = field(default_factory=dict)
  stream_closed: bool = False
  timings: list[PageTiming] = field(default_factory=list)
  publication: dict[str, object] = field(default_factory=dict)


class Logs:
  def __init__(self) -> None:
    self.saved: dict[Path, BatchLog] = {}

  def read(self, path: Path) -> BatchLog:
    try:
      stat = path.stat()
    except FileNotFoundError:
      return BatchLog((0, 0))
    signature = (stat.st_mtime_ns, stat.st_size)
    if path in self.saved and self.saved[path].signature == signature:
      return self.saved[path]
    result = BatchLog(signature)
    key = ''
    for line in path.read_text(errors='replace').splitlines():
      if not line.startswith('{'):
        continue
      try:
        event = mapping(json.loads(line))
      except ValueError:
        continue
      if isinstance(event.get('slug'), str) and isinstance(
        event.get('newPages'), list
      ):
        result.planned[text(event['slug'])] = [
          n for n in event['newPages'] if type(n) is int
        ]
        key = ''
      if isinstance(event.get('source'), str):
        key = text(event.get('key'))
        identity = (
          str(Path(event['source']).resolve()),
          number(event.get('page')),
        )
        if event.get('event') == 'page-admitted':
          result.admitted.add(identity)
        elif event.get('event') == 'page-rejected':
          result.failures[identity] = text(event.get('reason'), 'OCR failed')
      if event.get('event') == 'stream-closed':
        result.stream_closed = True
      if type(event.get('page')) is int and type(event.get('seconds')) in (
        int,
        float,
      ):
        seconds = float(event['seconds'])
        if key and math.isfinite(seconds) and seconds > 0:
          result.timings.append(PageTiming(key, event['page'], seconds))
      if type(event.get('newVectors')) is int and isinstance(
        event.get('generation'), str
      ):
        result.publication = event
    self.saved[path] = result
    return result

  def history(
    self,
    directory: Path,
    extracted: dict[str, set[int]],
    started_at: float | None,
  ) -> tuple[list[dict[str, object]], list[PageTiming]]:
    history = []
    timings: dict[tuple[str, int], PageTiming] = {}
    for path in sorted(directory.glob('*-extracting.log')):
      if started_at is None:
        break
      try:
        started = (
          datetime
          .strptime(path.name, '%Y%m%dT%H%M%S%f-extracting.log')
          .replace(tzinfo=timezone.utc)
          .timestamp()
        )
      except ValueError:
        continue
      if started < started_at:
        continue
      for timing in self.read(path).timings:
        if timing.page not in extracted.get(timing.key, set()):
          continue
        identity = (timing.key, timing.page)
        timings.pop(identity, None)
        timings[identity] = timing
    for path in sorted(directory.glob('*-publishing.log')):
      parsed = self.read(path)
      if not parsed.publication:
        continue
      event = parsed.publication
      coverage = mapping(event.get('coverage'))
      history.append({
        'at': datetime.fromtimestamp(
          parsed.signature[0] / 1e9, timezone.utc
        ).isoformat(),
        'generation': text(event.get('generation')),
        'newVectors': number(event.get('newVectors')),
        'cachedVectors': number(event.get('cachedVectors')),
        'rows': number(event.get('passages')),
        'pages': number(coverage.get('indexedPages')),
        'documents': number(coverage.get('indexedDocuments')),
      })
    return history[-12:], list(timings.values())[-50:]
