"""Fill missing timeline dates from source notes, then tracked file additions.

A note's explicit published/date/created field describes its timeline date. A bare
catalogue `year` describes the work, and is deliberately not substituted for the
note's date. Images inherit dates only from an explicit poster owner, never from
an arbitrary page that happens to embed them. Existing embedding dates win.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import yaml


class TimelineRow(TypedDict, total=False):
  doc: str
  store: str
  date: str | None
  prec: str | None
  dateSource: str | None


class Stamp(TypedDict):
  date: str
  prec: str
  dateSource: str


def slug(path: str) -> str:
  return '/'.join(
    re
    .sub(r'\s', '-', part)
    .replace('&', '-and-')
    .replace('%', '-percent')
    .replace('?', '')
    .replace('#', '')
    for part in path.split('/')
  ).rstrip('/')


def stamp(value: object, source: str) -> Stamp | None:
  match = re.match(r'^(\d{4})-(\d{2})-(\d{2})(?:$|[ T])', str(value or ''))
  if not match:
    return None
  try:
    date = dt.date(*map(int, match.groups()))
  except ValueError:
    return None
  # The v1 track stores unsigned days since 1900. Do not silently wrap an older date.
  if (
    not dt.date(1900, 1, 1) <= date <= dt.date.today() + dt.timedelta(days=366)
  ):
    return None
  return {'date': date.isoformat(), 'prec': 'day', 'dateSource': source}


def git_additions(content: Path, wanted: set[Path]) -> dict[Path, Stamp]:
  if not wanted:
    return {}
  repo = Path(
    subprocess.check_output(
      ['git', '-C', str(content), 'rev-parse', '--show-toplevel'], text=True
    ).strip()
  )
  # Use commit timestamps: imported history can retain an older author timestamp.
  # Read history once. Follow rename chains backwards so a rename does not become
  # a new Garden addition date. NUL fields preserve spaces, Unicode and tabs.
  result = subprocess.run(
    [
      'git',
      '-C',
      str(repo),
      'log',
      '--format=%x00DATE%x00%cI',
      '--name-status',
      '-z',
      '--find-renames',
      '--diff-filter=AR',
      '--',
      content.relative_to(repo).as_posix(),
    ],
    capture_output=True,
    check=True,
    timeout=120,
  )
  fields = result.stdout.decode('utf-8').split('\0')
  aliases: dict[str, set[Path]] = defaultdict(set)
  for path in wanted:
    aliases[path.relative_to(repo).as_posix()].add(path)
  dates: dict[Path, Stamp] = {}
  current: Stamp | None = None
  i = 0
  while i < len(fields):
    field = fields[i].strip('\n')
    i += 1
    if field == 'DATE' and i < len(fields):
      current = stamp(fields[i][:10], 'git-added')
      i += 1
    elif field == 'A' and i < len(fields):
      path = fields[i]
      i += 1
      if current:
        for original in aliases.get(path, set()):
          if (
            original not in dates or current['date'] < dates[original]['date']
          ):
            dates[original] = current
    elif field.startswith('R') and field[1:].isdigit() and i + 1 < len(fields):
      old, new = fields[i : i + 2]
      i += 2
      aliases[old].update(aliases.get(new, set()))
  return dates


def resolve_rows(content: Path, rows: list[TimelineRow]) -> None:
  content = content.resolve()
  for row in rows:
    origin = row.get('dateSource') or ''
    if origin == 'git-added' or origin.startswith((
      'frontmatter:',
      'poster-owner:',
    )):
      row['date'] = None
      row['prec'] = None
      row['dateSource'] = None
  sources: dict[str, Path] = {}
  notes: dict[str, Stamp] = {}
  owners: dict[str, list[tuple[str, Stamp]]] = defaultdict(list)
  for path in sorted(content.rglob('*')):
    if not path.is_file() or 'embeddings' in path.relative_to(content).parts:
      continue
    relative = path.relative_to(content)
    key = slug(
      relative.with_suffix('').as_posix()
      if path.suffix in ('.md', '.ipynb')
      else relative.as_posix()
    )
    sources.setdefault(key, path)
    if path.suffix != '.md':
      continue
    text = path.read_text(encoding='utf-8', errors='replace')
    header = re.match(r'\A---\s*\n(.*?)\n---(?:\n|$)', text, re.S)
    if not header:
      continue
    metadata = yaml.safe_load(header[1])
    if not isinstance(metadata, dict):
      continue
    dated = next(
      (
        found
        for field in ('published', 'date', 'created')
        if (found := stamp(metadata.get(field), f'frontmatter:{field}'))
      ),
      None,
    )
    if not dated:
      continue
    notes[key] = dated
    posters = metadata.get('posters', [])
    if isinstance(posters, str):
      posters = [posters]
    if isinstance(posters, list):
      for poster in posters:
        if not isinstance(poster, str):
          continue
        target = (
          poster
          .removeprefix('[[')
          .split('|')[0]
          .removesuffix(']]')
          .split('#')[0]
        )
        owners[slug(target.lstrip('/'))].append((key, dated))
  resolved: dict[str, Stamp] = {}
  pending: set[Path] = set()
  for row in rows:
    if row.get('date'):
      if not row.get('dateSource'):
        row['dateSource'] = 'embedding-source'
      continue
    key = row['doc']
    dated = notes.get(key)
    if (
      dated is None
      and row['store'] == 'image'
      and len(owners.get(key, [])) == 1
    ):
      owner, source = owners[key][0]
      dated = {
        **source,
        'dateSource': f'poster-owner:{owner}:{source["dateSource"]}',
      }
    if dated:
      resolved[key] = dated
    elif key in sources:
      pending.add(sources[key])
  additions = git_additions(content, pending)
  for row in rows:
    if row.get('date'):
      continue
    dated = resolved.get(row['doc']) or additions.get(sources.get(row['doc']))
    if dated:
      row.update(dated)
