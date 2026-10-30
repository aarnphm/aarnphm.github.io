"""Push generated files to the FONTS KV, skipping keys the site already serves unchanged."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .faces import FETCH_HEADERS, KV_NAMESPACE, ROOT


def live_sha(url: str) -> str | None:
  req = urllib.request.Request(url, headers=FETCH_HEADERS)
  try:
    with urllib.request.urlopen(req, timeout=60) as r:
      return hashlib.sha256(r.read()).hexdigest()
  except (urllib.error.URLError, TimeoutError):
    return None


def put(key: str, path: Path) -> bool:
  cmd = [
    'pnpm',
    'exec',
    'wrangler',
    'kv',
    'key',
    'put',
    key,
    '--path',
    str(path),
    '--namespace-id',
    KV_NAMESPACE,
    '--remote',
  ]
  # wrangler occasionally exits 1 with no output on a transient API error; one key failing
  # must not leave the rest unpublished, so retry and report at the end.
  for attempt in range(3):
    if subprocess.run(cmd, cwd=ROOT).returncode == 0:
      return True
    print(f'  retry {attempt + 1} for {key}', file=sys.stderr)
  return False


def publish(uploads: list[tuple[str, Path]], source_url: str) -> int:
  """Upload every file whose content the site does not already serve. Returns the failure count.

  The check fetches the versioned URL the stylesheet uses, so a stale edge-cached copy under
  another query string cannot mask a missing key: the worker only caches 200 responses, and a
  hash URL first answers 200 once its key holds that content.
  """
  failed: list[str] = []
  unchanged = 0
  for key, path in uploads:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if live_sha(f'{source_url.rstrip("/")}/{key}?v={digest[:8]}') == digest:
      unchanged += 1
      continue
    print(f'kv put {key}')
    if not put(key, path):
      failed.append(key)
  print(
    f'KV: {len(uploads) - unchanged - len(failed)} uploaded, {unchanged} unchanged, {len(failed)} failed'
  )
  if failed:
    print('failed: ' + ', '.join(failed), file=sys.stderr)
  return len(failed)
