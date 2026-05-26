#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["tiktoken"]
# ///

import json
import sys
import urllib.request
from pathlib import Path

import tiktoken


def fetch_content(source: str) -> str:
  if source.startswith('http://') or source.startswith('https://'):
    with urllib.request.urlopen(source) as resp:
      return resp.read().decode('utf-8')
  return Path(source).read_text(encoding='utf-8')


def main():
  if len(sys.argv) < 2:
    print(f'usage: {sys.argv[0]} <path-or-url>', file=sys.stderr)
    sys.exit(1)

  source = sys.argv[1]
  content = fetch_content(source)

  enc = tiktoken.get_encoding('o200k_base')
  tokens = enc.encode(content, disallowed_special=())

  words = content.split()
  chars = len(content)
  chars_no_ws = len(
    content.replace(' ', '').replace('\n', '').replace('\t', '')
  )

  stats = {
    'source': source,
    'characters': chars,
    'characters_no_whitespace': chars_no_ws,
    'words': len(words),
    'tokens': len(tokens),
    'tokens_per_word': round(len(tokens) / len(words), 3) if words else 0,
    'chars_per_token': round(chars / len(tokens), 3) if tokens else 0,
  }

  out_path = Path(__file__).parent.parent.parent / 'txts' / 'stats.json'
  out_path.write_text(json.dumps(stats, indent=2) + '\n')
  print(json.dumps(stats, indent=2))


if __name__ == '__main__':
  main()
