#!/usr/bin/env python3
"""Archive verified Instagram exports; preview by default, mutate with --apply."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


IMAGES = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.avif', '.heic', '.svg'}
VIDEOS = {'.mp4', '.mov', '.webm', '.m4v'}
AUDIO = {'.m4a', '.mp3', '.wav', '.ogg', '.aac'}
MAX_IMAGE_BYTES = 25 * 1024 * 1024


def digest(path: Path) -> str:
  with path.open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode(value: bytes) -> str:
  return base64.urlsafe_b64encode(value).decode().rstrip('=')


def plan(
  repo: Path, sources: list[Path]
) -> tuple[dict, list[tuple[Path, Path, str]], bytes]:
  images = repo / 'content/thoughts/images'
  archive = images / 'instagram'
  manifest = archive / 'attribution.json'
  before = manifest.read_bytes() if manifest.exists() else b''
  attribution = json.loads(before) if before else {}
  copies: list[tuple[Path, Path, str]] = []
  planned: dict[Path, str] = {}
  for source in sources:
    reports = [
      source / name
      for name in ['verification.json', 'manifest.json']
      if (source / name).is_file()
    ]
    if not reports:
      raise ValueError(f'{source}: no verification.json or manifest.json')
    records = [json.loads(p.read_text()) for p in reports]
    for record in records:
      if record.get('status') not in (None, 'passed'):
        raise ValueError(f'{source}: capture status is {record["status"]}')
    metadata = {
      key: value
      for record in reversed(records)
      for key, value in record.items()
    }
    account = metadata.get('account', metadata.get('creator', ''))
    authors = [
      name.strip().removeprefix('@')
      for name in re.split(r',\s*| and | \u00d7 ', account)
      if name.strip()
    ]
    if not authors or any(
      not re.fullmatch(r'[A-Za-z0-9_.]+', name) for name in authors
    ):
      raise ValueError(f'{source}: missing or invalid author metadata')
    url = metadata.get(
      'source_url', metadata.get('source_reel', metadata.get('source', ''))
    )
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.netloc not in {
      'www.instagram.com',
      'instagram.com',
    }:
      raise ValueError(f'{source}: invalid Instagram source URL')
    if not re.fullmatch(r'/(?:p|reel|reels)/[A-Za-z0-9_-]+/', parsed.path):
      raise ValueError(f'{source}: source URL must identify a post or reel')
    url = parsed._replace(query='', fragment='').geturl()
    outputs = metadata.get(
      'outputs', metadata.get('media', metadata.get('files', []))
    )
    by_name = {
      Path(item.get('file', item.get('path', ''))).name: item
      for item in outputs
    }
    if not by_name:
      raise ValueError(f'{source}: report has no output files')
    for name in by_name:
      if not (source / name).is_file():
        raise ValueError(f'{source}: reported output is missing: {name}')
    for file in sorted(source.iterdir()):
      if not file.is_file() or file.is_symlink():
        continue
      extension = file.suffix.lower()
      media = extension in IMAGES | VIDEOS | AUDIO
      if media and file.name not in by_name:
        continue
      if not media and extension not in {'.json', '.txt', '.zip'}:
        continue
      sha = digest(file)
      output = by_name.get(file.name, {})
      if output.get('sha256') and output['sha256'] != sha:
        raise ValueError(f'{file}: SHA-256 differs from capture report')
      if extension in IMAGES and file.stat().st_size >= MAX_IMAGE_BYTES:
        raise ValueError(f'{file}: image must be below 25 MiB')
      if media:
        with file.open('rb') as stream:
          if stream.read(60).startswith(
            b'version https://git-lfs.github.com/spec/v1\n'
          ):
            raise ValueError(f'{file}: unresolved Git LFS pointer')
      target = (images if extension in IMAGES else archive) / (
        encode(bytes.fromhex(sha)) + extension
      )
      entry = None
      if media:
        match = re.match(r'^(\d+)', file.stem)
        entry = {
          'authors': authors,
          'source_url': url,
          'kind': 'image'
          if extension in IMAGES
          else 'video'
          if extension in VIDEOS
          else 'audio',
          'position': output.get(
            'position', output.get('slide', int(match[1]) if match else None)
          ),
          'sha256': sha,
        }
        link = '[[' + target.relative_to(repo / 'content').as_posix() + ']]'
        if link in attribution and attribution[link] != entry:
          suffix = encode(
            hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).digest()
          )[:8]
          target = target.with_stem(target.stem + '-' + suffix)
          link = '[[' + target.relative_to(repo / 'content').as_posix() + ']]'
        if link in attribution and attribution[link] != entry:
          raise ValueError(f'{file}: attribution collision')
        attribution[link] = entry
      if target.exists() and digest(target) != sha:
        raise ValueError(f'{target}: existing file has different bytes')
      if target in planned and planned[target] != sha:
        raise ValueError(f'{target}: planned filename collision')
      planned[target] = sha
      copies.append((file, target, sha))
  return dict(sorted(attribution.items())), copies, before


def apply(
  repo: Path,
  attribution: dict,
  copies: list[tuple[Path, Path, str]],
  before: bytes,
  move: bool,
) -> None:
  manifest = repo / 'content/thoughts/images/instagram/attribution.json'
  if (manifest.read_bytes() if manifest.exists() else b'') != before:
    raise ValueError(
      'Attribution changed during planning; retry from the current map'
    )
  for source, destination, sha in copies:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
      created = False
      try:
        with destination.open('xb') as output, source.open('rb') as stream:
          created = True
          shutil.copyfileobj(stream, output)
      except FileExistsError:
        pass
      except OSError:
        if created:
          destination.unlink(missing_ok=True)
        raise
    if digest(destination) != sha:
      raise ValueError(
        f'{destination}: copied bytes failed SHA-256 verification'
      )
  if (manifest.read_bytes() if manifest.exists() else b'') != before:
    raise ValueError(
      'Attribution changed during copying; files retained, map not overwritten'
    )
  manifest.parent.mkdir(parents=True, exist_ok=True)
  fd, name = tempfile.mkstemp(prefix='.attribution-', dir=manifest.parent)
  try:
    with os.fdopen(fd, 'w') as output:
      output.write(
        json.dumps(attribution, ensure_ascii=False, indent=2) + '\n'
      )
    os.replace(name, manifest)
  finally:
    Path(name).unlink(missing_ok=True)
  if move:
    for source, destination, sha in copies:
      if source.resolve() == destination.resolve():
        continue
      if digest(source) != sha:
        raise ValueError(
          f'{source}: changed during archiving; source retained'
        )
      source.unlink()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source', action='append', required=True, type=Path)
  parser.add_argument(
    '--repo', type=Path, default=Path(__file__).resolve().parents[4]
  )
  parser.add_argument('--apply', action='store_true')
  parser.add_argument(
    '--move',
    action='store_true',
    help='Remove verified source files after saving the attribution map',
  )
  args = parser.parse_args()
  if args.move and not args.apply:
    parser.error('--move requires --apply')
  repo = args.repo.resolve()
  sources = list(dict.fromkeys(p.resolve() for p in args.source))
  if any(not p.is_dir() for p in sources):
    parser.error('Every --source must be an existing download directory')
  if args.apply:
    cache = repo / '.quartz-cache'
    cache.mkdir(exist_ok=True)
    with (cache / 'instagram-archive.lock').open('a') as lock:
      fcntl.flock(lock, fcntl.LOCK_EX)
      attribution, copies, before = plan(repo, sources)
      apply(repo, attribution, copies, before, args.move)
  else:
    attribution, copies, before = plan(repo, sources)
  print(
    json.dumps({
      'status': 'applied' if args.apply else 'preview',
      'input_files': len(copies),
      'new_files': sum(not target.exists() for _, target, _ in copies)
      if not args.apply
      else None,
      'attribution_entries': len(attribution),
      'moved_sources': args.move,
    })
  )


if __name__ == '__main__':
  try:
    main()
  except (OSError, ValueError, KeyError, TypeError) as error:
    raise SystemExit(str(error)) from error
