"""Shared constants and helpers for the token budget harness.

LIMIT, TARGET and BYTES_PER_TOKEN mirror quartz/scripts/check-token-budget.mjs; the harness reads
the hook's --json report, which carries them, and checks that they agree.
"""

import math
import re
import subprocess

LIMIT = 5000
TARGET = 4000
BYTES_PER_TOKEN = 4

_EXT = re.compile(r'\.(?:[cm]?[jt]sx?|s?css|sass)$')


def tokens(n_bytes: int) -> int:
  return math.ceil(n_bytes / BYTES_PER_TOKEN)


def text_tokens(text: str) -> int:
  return tokens(len(text.encode()))


def strip_ext(p: str) -> str:
  return _EXT.sub('', p)


def git(*args: str) -> str:
  return subprocess.run(
    ['git', *args], check=True, capture_output=True, text=True
  ).stdout


def index_blob(path: str) -> str | None:
  r = subprocess.run(
    ['git', 'rev-parse', f':{path}'], capture_output=True, text=True
  )
  return r.stdout.strip() if r.returncode == 0 else None


def slug(path: str) -> str:
  return re.sub(r'[^\w.-]+', '-', path)


def check_report(report: dict) -> None:
  """Refuses a hook report whose budget constants differ from the harness's."""
  theirs = (
    report.get('limit'),
    report.get('target'),
    report.get('bytesPerToken', BYTES_PER_TOKEN),
  )
  if theirs != (LIMIT, TARGET, BYTES_PER_TOKEN):
    raise SystemExit(
      f'report budget {theirs} differs from harness {(LIMIT, TARGET, BYTES_PER_TOKEN)}'
    )


def cuts(blocks: list[dict]) -> list[dict]:
  """Consecutive blocks packed up to TARGET tokens (same rule as the hook's cut plan)."""
  parts: list[dict] = []
  for b in blocks:
    last = parts[-1] if parts else None
    if last and not last['over'] and last['tokens'] + b['tokens'] <= TARGET:
      last['end'] = b['end']
      last['tokens'] += b['tokens']
      last['blocks'].append(b['id'])
    else:
      parts.append({
        'start': b['start'],
        'end': b['end'],
        'tokens': b['tokens'],
        'label': b['label'],
        'blocks': [b['id']],
        'over': b['tokens'] > TARGET,
      })
  return parts
