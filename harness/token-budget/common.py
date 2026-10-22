"""Shared constants and helpers for the token budget harness.

check.py (the pre-commit hook) and the refactor stages share LIMIT, TARGET and ENCODING. A saved
check.py --json report carries the values it was made with; check_report refuses a stale one.
"""

import functools
import os
import re
import subprocess
from pathlib import Path

LIMIT = 5000
TARGET = 4000
# The tokenizer of o1, o3, o4-mini, GPT-4o and GPT-4.1 (tiktoken.encoding_for_model). Every token
# decodes to at least one byte, so a file of at most LIMIT bytes is under budget without encoding.
ENCODING = 'o200k_base'

# tiktoken downloads the 3.6 MB rank file once; its default cache is under $TMPDIR, which macOS
# clears, and a pre-commit hook should not need the network.
os.environ.setdefault(
  'TIKTOKEN_CACHE_DIR', str(Path.home() / '.cache' / 'tiktoken')
)

_EXT = re.compile(r'\.(?:[cm]?[jt]sx?|s?css|sass)$')


@functools.cache
def _encoder():
  import tiktoken

  return tiktoken.get_encoding(ENCODING)


def text_tokens(text: str) -> int:
  # encode_ordinary: a file may contain special-token text such as <|endoftext|>.
  return len(_encoder().encode_ordinary(text))


def texts_tokens(texts: list[str]) -> list[int]:
  return [
    len(t) for t in _encoder().encode_ordinary_batch(texts, num_threads=8)
  ]


def strip_ext(p: str) -> str:
  return _EXT.sub('', p)


def module_dir(p: str) -> str:
  # The private folder drops the extension and a `.inline` build marker: popover.inline.ts -> popover/.
  return re.sub(r'\.inline$', '', strip_ext(p))


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
  theirs = (report.get('limit'), report.get('target'), report.get('encoding'))
  if theirs != (LIMIT, TARGET, ENCODING):
    raise SystemExit(
      f'report budget {theirs} differs from harness {(LIMIT, TARGET, ENCODING)}; '
      'run check.py --json again'
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
