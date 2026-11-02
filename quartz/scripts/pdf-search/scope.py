"""Read PDF collection scope from the authored embedding index."""

from pathlib import Path
import re

import yaml

from build import PdfSource, public_pdfs


def collection_prefixes(content: Path) -> tuple[str, ...]:
  config = content / 'embeddings/index.md'
  match = re.match(
    r'\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)', config.read_text(), re.DOTALL
  )
  if match is None:
    raise ValueError(f'{config}: expected YAML frontmatter')
  metadata = yaml.safe_load(match[1])
  entries = metadata.get('embeddings') if isinstance(metadata, dict) else None
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
  prefixes = collection_prefixes(content)
  return {
    slug: source
    for slug, source in public_pdfs(index, content).items()
    if slug.startswith(prefixes)
  }
