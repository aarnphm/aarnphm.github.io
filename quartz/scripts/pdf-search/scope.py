"""Read PDF collection scope from the authored embedding index."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import re

import yaml

if TYPE_CHECKING:
  from build import PdfSource


def collection_metadata(content: Path) -> dict:
  config = content / 'embeddings/index.md'
  match = re.match(
    r'\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)', config.read_text(), re.DOTALL
  )
  if match is None:
    raise ValueError(f'{config}: expected YAML frontmatter')
  metadata = yaml.safe_load(match[1])
  if not isinstance(metadata, dict):
    raise ValueError(f'{config}: expected frontmatter mapping')
  return metadata


def collection_exclusions(content: Path) -> frozenset[str]:
  if not (content / 'embeddings/index.md').exists():
    return frozenset()
  entries = collection_metadata(content).get('pdfExclude', [])
  if not isinstance(entries, list) or not all(
    isinstance(entry, str)
    and entry.endswith('.pdf')
    and not entry.startswith('/')
    and all(part not in ('', '.', '..') for part in entry.split('/'))
    for entry in entries
  ):
    raise ValueError('pdfExclude must contain content-relative PDF paths')
  return frozenset(entries)


def collection_prefixes(content: Path) -> tuple[str, ...]:
  config = content / 'embeddings/index.md'
  entries = collection_metadata(content).get('embeddings')
  if not isinstance(entries, list):
    raise ValueError(
      f'{config}: embeddings must be a list of directory wikilinks'
    )
  prefixes = []
  for entry in entries:
    link = (
      re.fullmatch(r'\[\[([^|#\[\]]+)(?:\|[^\[\]]*)?\]\]', entry)
      if isinstance(entry, str)
      else None
    )
    if link is None:
      raise ValueError(f'{config}: invalid directory wikilink {entry!r}')
    target = link[1].strip().rstrip('/')
    if (
      not target
      or target.startswith('/')
      or '\\' in target
      or any(part in ('', '.', '..') for part in target.split('/'))
      or not (content / target).resolve().is_relative_to(content.resolve())
    ):
      raise ValueError(
        f'{config}: directory must stay inside content: {entry!r}'
      )
    prefixes.append(target + '/')
  return tuple(dict.fromkeys(prefixes))


def queued_pdfs(index: Path, content: Path) -> dict[str, PdfSource]:
  from build import public_pdfs

  prefixes = collection_prefixes(content)
  return {
    slug: source
    for slug, source in public_pdfs(index, content).items()
    if slug.startswith(prefixes)
  }
