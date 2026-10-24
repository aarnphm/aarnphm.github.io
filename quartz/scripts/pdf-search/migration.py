"""Activate a server recipe while retaining verified earlier OCR provenance."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import portalocker

from cache import ExtractionSpec, PageCache, canonical


def activate_recipe(
  root: Path,
  spec: ExtractionSpec,
  *,
  previous_fingerprint: str,
  previous_model_revision: str | None = None,
) -> None:
  root.mkdir(parents=True, exist_ok=True)
  target = root / 'recipe.json'
  with portalocker.Lock(root / '.recipe.lock', timeout=1):
    if target.exists():
      previous = ExtractionSpec(**json.loads(target.read_text()))
      if previous == spec:
        return
      preserved = (
        'layout_revision',
        'sdk_version',
        'dpi',
        'schema',
        'figure_policy',
      )
      if (
        previous.recipe_sha256 != previous_fingerprint
        or previous.model_revision
        != (previous_model_revision or spec.model_revision)
        or any(
          getattr(previous, field) != getattr(spec, field)
          for field in preserved
        )
      ):
        raise ValueError(
          'Unapproved extraction recipe change; use a separate cache'
        )
      for path in sorted(root.glob('*/document.json')):
        document = json.loads(path.read_text())
        if document['recipe'] != asdict(previous):
          continue
        source = Path(document['source'])
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        if sha != document['sourceSha256']:
          raise ValueError(f'Source changed before cache migration: {source}')
        old = PageCache(root, sha, previous)
        new = PageCache(root, sha, spec)
        if old.directory != path.parent or old.key != document['key']:
          raise ValueError(f'Invalid earlier cache identity: {path}')
        numbers = sorted({
          *document['pages'],
          *(
            int(page.stem)
            for page in old.directory.glob('*.json')
            if page.stem.isdecimal()
          ),
        })
        for number in numbers:
          page = old.read(number)
          if page is None:
            raise ValueError(f'Corrupt earlier OCR page: {old.path(number)}')
          new.write(
            page,
            origin={
              'key': old.key,
              'recipe': asdict(previous),
              'sha256': json.loads(old.path(number).read_text())['sha256'],
            },
          )
        new.directory.mkdir(parents=True, exist_ok=True)
        document.update(recipe=asdict(spec), key=new.key, pages=numbers)
        temporary = new.directory / 'document.tmp'
        temporary.write_bytes(canonical(document))
        temporary.replace(new.directory / 'document.json')
    temporary = root / 'recipe.tmp'
    temporary.write_bytes(canonical(asdict(spec)))
    temporary.replace(target)
