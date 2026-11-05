# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pyyaml", "mlx-lm", "mlx-vlm", "pillow"]
# ///
"""Label the sparse features of the `layout: latent` sidecar with a local Gemma on mlx.

    uv run quartz/scripts/latent/label.py [--only text] [--limit 50] [--force]
    uv run quartz/scripts/latent/label.py --setup   # fetch the pinned Gemma into the HF cache

The model is pinned to one snapshot (MODEL @ REVISION) and loaded from the Hugging Face cache by
path, so a deploy never touches the network once it is cached; the first run downloads it (~3 GB).
LATENT_LABEL_MODEL points at another repo id or a local directory.

Each feature is shown its strongest rows (Bills et al. 2023; Prism) and asked for a short label
and a one-sentence description; image features are shown the images themselves. Matryoshka
levels are labelled coarse to fine, and a child sees its parent's label, so it names what it adds.

Every label is then scored by detection (Paulo et al. 2024): a second prompt gets the label, four
held-out rows that fire the feature and four of the same family that do not, and says which fit;
the score is the balanced accuracy, 0.5 for a label that says nothing.

Labels go to content/embeddings/latent/labels.json, `{dictionary: {id: {label, description,
score, rows, model, at}}}`, keyed by feature id (warm starts keep ids). `rows` is the shown set;
a later run relabels a feature only when its top rows moved (Jaccard < 0.5). The labels are
folded into the current sidecar's features.json as they land.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import Texts, apply_labels, load_canon, load_side  # noqa: E402

MODEL = os.environ.get(
  'LATENT_LABEL_MODEL', 'mlx-community/gemma-4-e4b-it-4bit'
)
# the snapshot the labels were made with; a new one is a deliberate bump (and relabels everything,
# since a label records the model that wrote it)
REVISION = '475b9088d29754a3379866cf5aeb6b41acd313c2'
SHOWN = 10  # rows in the labelling prompt
HELD = 4  # held-out firing rows in the scoring prompt (and as many that do not fire)
CLIP = 160  # characters per row; prefill (~450 tokens/s on an M1 Max) is most of the cost
MIN_FIRES = 3
JACCARD = 0.5
BATCH = (
  4  # prompts per mlx-lm batch_generate call: ~1.5x one at a time; 16 thrashes
)

ASK = """You are naming one feature of a sparse autoencoder trained on text embeddings of a personal \
research garden: notes on machine learning, philosophy, mathematics and literature, plus the \
papers and books they cite. Below are the {what} that activate this feature most strongly, \
strongest first.

{parent}Find what they share (a topic, an author, a kind of argument, a format, a language) and \
name it as precisely as the {what} allow.

{rows}

Reply with JSON only: {{"label": "<2 to 6 words, lowercase except proper nouns>", \
"description": "<at most 20 words: what makes a {one} fire this feature>"}}"""

SCORE = """A feature of a text-embedding model is described as:
label: {label}
description: {description}

For each numbered {what} below, answer 1 if the feature would fire on it, else 0.

{rows}

Reply with JSON only: a list of {n} numbers, 0 or 1, in order."""


def log(*args: object) -> None:
  print('[label]', *args, flush=True)


def squash(text: str) -> str:
  return re.sub(r'\s+', ' ', text).strip()


def loads_lax(s: str):
  """json.loads, then again with TeX backslashes (\\mathbb, \\{) escaped: maths features answer in
  LaTeX, which is not valid JSON."""
  for t in (s, re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)):
    try:
      return json.loads(t, strict=False)
    except ValueError:
      continue
  return None


def first_json(text: str):
  text = re.sub(r'<\|channel\>.*?<channel\|>', '', text, flags=re.S)
  text = re.sub(r'```(?:json)?', '', text)
  # greedy first: TeX braces inside a description end the lazy match early
  for pat in (r'\{.*\}', r'\{.*?\}', r'\[.*?\]'):
    m = re.search(pat, text, flags=re.S)
    if m and (got := loads_lax(m.group(0))) is not None:
      return got
  # cut off at max_tokens: keep the label, and as much of the description as arrived
  m = re.search(r'"label"\s*:\s*"((?:[^"\\]|\\.)+)"', text)
  if not m:
    return None
  d = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)', text)
  return {
    'label': m.group(1),
    'description': d.group(1).rstrip('\\') + '…' if d else '',
  }


def resolve(model: str, revision: str | None) -> str:
  """Local path of the pinned snapshot: the HF cache when it is there (no network), else one
  download into it."""
  if Path(model).expanduser().is_dir():
    return str(Path(model).expanduser())
  from huggingface_hub import snapshot_download

  rev = revision if model == 'mlx-community/gemma-4-e4b-it-4bit' else None
  try:
    return snapshot_download(model, revision=rev, local_files_only=True)
  except Exception:
    log(
      f'{model} {(rev or "main")[:7]} is not cached; downloading it once (~3 GB)'
    )
    return snapshot_download(model, revision=rev)


class Gemma:
  """Text through mlx-lm, images through mlx-vlm; each loads on first use."""

  def __init__(self, path: str):
    self.path = path
    self.lm = None
    self.vlm = None

  def text(self, prompts: list[str], max_tokens: int = 90) -> list[str]:
    """Batched: one batch_generate per BATCH prompts, several times the throughput of one
    prompt at a time on the same GPU."""
    from mlx_lm import batch_generate, load

    if self.lm is None:
      self.lm = load(self.path)
    model, tok = self.lm
    out: list[str] = []
    for a in range(0, len(prompts), BATCH):
      ids = [
        tok.apply_chat_template(
          [{'role': 'user', 'content': p}],
          add_generation_prompt=True,
          enable_thinking=False,
        )
        for p in prompts[a : a + BATCH]
      ]
      # transformers 5 returns an encoding where 4 returned the ids
      ids = [
        list(x['input_ids']) if hasattr(x, 'keys') else list(x) for x in ids
      ]
      out.extend(batch_generate(model, tok, ids, max_tokens=max_tokens).texts)
    return out

  def images(
    self, prompt: str, images: list[str], max_tokens: int = 120
  ) -> str:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    if self.vlm is None:
      self.vlm = load(self.path)
    model, processor = self.vlm
    p = apply_chat_template(
      processor, model.config, prompt, num_images=len(images)
    )
    out = generate(
      model, processor, p, images, max_tokens=max_tokens, verbose=False
    )
    return getattr(out, 'text', out)


def thumbs(paths: list[Path], tmp: Path) -> list[str]:
  """512 px copies: posters and photos are megapixels the model cannot use."""
  from PIL import Image

  tmp.mkdir(parents=True, exist_ok=True)
  out = []
  for p in paths:
    dest = tmp / (re.sub(r'[^\w.-]', '_', p.as_posix())[-80:] + '.jpg')
    if not dest.exists():
      try:
        im = Image.open(p).convert('RGB')
        im.thumbnail((512, 512))
        im.save(dest, quality=85)
      except OSError:
        continue
    out.append(str(dest))
  return out


def load_codes(dest: Path, rows: int, k: int, block: int) -> np.ndarray:
  ids = np.empty((rows, k), np.uint16)
  for b, a in enumerate(range(0, rows, block)):
    n = min(block, rows - a)
    raw = (dest / 'codes' / f'{b:05d}.bin').read_bytes()
    ids[a : a + n] = np.frombuffer(raw[: n * k * 2], '<u2').reshape(n, k)
  return ids


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--content', default='content')
  ap.add_argument('--directory', default='embeddings')
  ap.add_argument('--model', default=MODEL)
  ap.add_argument(
    '--only', default='', help='comma-separated dictionary names'
  )
  ap.add_argument(
    '--limit', type=int, default=0, help='label at most this many'
  )
  ap.add_argument('--force', action='store_true', help='relabel everything')
  ap.add_argument(
    '--since',
    default='',
    help='with --force: keep labels written at or after this UTC time (resume an interrupted run)',
  )
  ap.add_argument('--no-score', action='store_true')
  ap.add_argument(
    '--setup', action='store_true', help='cache the pinned model and exit'
  )
  args = ap.parse_args()
  path = resolve(args.model, REVISION)
  if args.setup:
    log(f'{args.model} ready at {path}')
    return

  content = Path(args.content).resolve()
  root = content / args.directory / 'latent'
  side = root / json.loads((root / 'index.json').read_text())['hash']
  labels_path = root / 'labels.json'
  store = json.loads(labels_path.read_text()) if labels_path.exists() else {}
  index, docs, _, doc_of, kinds = load_side(side)
  unique = load_canon(side, len(doc_of)) == np.arange(len(doc_of))
  texts = Texts(side, docs, index['part'], kinds)
  only = {n for n in args.only.split(',') if n}
  gemma = Gemma(path)
  rng = np.random.default_rng(0)
  done = 0
  t0 = time.time()

  def key(r: int) -> str:
    d = docs[int(doc_of[r])]
    return f'{d["id"]}#{r - d["r0"]}'

  def show(r: int) -> str:
    d = docs[int(doc_of[r])]
    body = squash(texts.text(r))
    if d['s'] in ('image', 'base') or not body:
      return d['t']
    return f'[{d["t"][:60]}] {body[:CLIP]}'

  def save() -> None:
    labels_path.write_text(
      json.dumps(store, ensure_ascii=False, indent=1, sort_keys=True) + '\n'
    )

  # the small dictionaries first: an interrupted run still leaves them whole
  for spec in sorted(index.get('features') or [], key=lambda d: d['features']):
    name = spec['name']
    if only and name not in only:
      continue
    dest = side / 'feat' / name
    table = json.loads((dest / 'features.json').read_text())
    meta = json.loads((dest / 'meta.json').read_text())
    ids = load_codes(dest, meta['rows'], meta['k'], meta['codeBlock'])
    # negatives for detection scoring come from canonical rows: a copy of a top row is no negative
    own = np.flatnonzero(np.isin(kinds, spec['kinds']) & unique)
    have = store.setdefault(name, {})
    pictures = name == 'image'
    what, one = (
      (('images', 'image') if pictures else ('entries', 'entry'))
      if name in ('image', 'base')
      else ('passages', 'passage')
    )
    todo = []
    for row in sorted(table, key=lambda r: (r.get('level', 0), r['f'])):
      if row['n'] < MIN_FIRES or not row['top']:
        # too rare to label; a record left from an earlier dictionary describes other rows (and
        # can name rows that have since left the store), so it goes. Hand labels stay.
        if isinstance(have.get(str(row['f'])), dict):
          del have[str(row['f'])]
          log(f'{name} {row["f"]}: fires {row["n"]}x, stale label dropped')
        continue
      shown = [r for r, _ in row['top'][:SHOWN]]
      keys = [key(r) for r in shown]
      prev = have.get(str(row['f']))
      if isinstance(prev, str):
        continue  # written by hand
      # a resumed --force run: what this run already labelled stays
      if (
        args.since
        and isinstance(prev, dict)
        and str(prev.get('at') or '') >= args.since
      ):
        continue
      if not args.force and isinstance(prev, dict):
        a, b = set(prev.get('rows') or []), set(keys)
        if (
          prev.get('model') == args.model
          and len(a & b) / max(1, len(a | b)) >= JACCARD
        ):
          continue
      todo.append((row, shown, keys))
    if args.limit:
      todo = todo[: max(0, args.limit - done)]
    log(f'{name}: {len(todo)} features to label')
    levels = sorted({row.get('level', 0) for row, _, _ in todo})
    for level in levels:
      batch = [t for t in todo if t[0].get('level', 0) == level]
      for a in range(0, len(batch), BATCH * 8):
        chunk = batch[a : a + BATCH * 8]
        prompts, images = [], []
        for row, shown, _ in chunk:
          parent = ''
          p = row.get('parent', -1)
          up = have.get(str(p)) if p is not None and p >= 0 else None
          up_label = up.get('label') if isinstance(up, dict) else up
          if isinstance(up_label, str) and up_label:
            parent = f'Its parent feature, one level coarser, is labelled "{up_label}". Name what this one adds.\n\n'
          if pictures:
            files = [content / docs[int(doc_of[r])]['id'] for r in shown[:6]]
            images.append(
              thumbs(
                [q for q in files if q.exists()],
                Path('quartz/.quartz-cache/latent-thumbs'),
              )
            )
            names = '\n'.join(
              f'{j + 1}. {show(r)}' for j, r in enumerate(shown[:6])
            )
            body = f'Images attached, with their file names:\n{names}'
          else:
            body = '\n'.join(
              f'{j + 1}. {show(r)}' for j, r in enumerate(shown)
            )
          prompts.append(
            ASK.format(what=what, one=one, parent=parent, rows=body)
          )
        if pictures:
          raws = [
            gemma.images(p, im) if im else gemma.text([p])[0]
            for p, im in zip(prompts, images)
          ]
        else:
          raws = gemma.text(prompts)
        recs: list[tuple[dict, dict]] = []
        for (row, shown, keys), raw in zip(chunk, raws):
          got = first_json(raw)
          if (
            not isinstance(got, dict)
            or not str(got.get('label') or '').strip()
          ):
            log(f'{name} {row["f"]}: unparsed {raw[:120]!r}')
            continue
          rec = dict(
            label=squash(str(got['label']))[:60],
            description=squash(str(got.get('description') or ''))[:240],
            rows=keys,
            model=args.model,
            at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
          )
          have[str(row['f'])] = rec
          recs.append((row, rec))
          done += 1
        # detection scoring, batched the same way
        mixes, asks = [], []
        for row, rec in recs:
          held = [r for r, _ in row['top'][SHOWN:]][:HELD]
          if args.no_score or pictures or len(held) < 2:
            continue
          pool = own[~np.any(ids[own] == row['f'], axis=1)]
          neg = [
            int(r) for r in rng.choice(pool, size=len(held), replace=False)
          ]
          mix = [(r, 1) for r in held] + [(r, 0) for r in neg]
          rng.shuffle(mix)
          rows = '\n'.join(
            f'{j + 1}. {show(r)}' for j, (r, _) in enumerate(mix)
          )
          mixes.append((rec, mix))
          asks.append(
            SCORE.format(
              label=rec['label'],
              description=rec['description'],
              what=one,
              rows=rows,
              n=len(mix),
            )
          )
        for (rec, mix), raw in zip(
          mixes, gemma.text(asks, max_tokens=40) if asks else []
        ):
          ans = first_json(raw)
          if isinstance(ans, list) and len(ans) == len(mix):
            y = np.array([t for _, t in mix])
            got = np.array([
              1 if str(v).strip() in ('1', 'True', 'true') else 0 for v in ans
            ])
            rec['score'] = round(
              float(0.5 * (got[y == 1].mean() + (1 - got[y == 0]).mean())), 2
            )
        save()
        apply_labels(table, have)
        (dest / 'features.json').write_text(
          json.dumps(table, ensure_ascii=False, separators=(',', ':')),
          encoding='utf-8',
        )
        rate = (time.time() - t0) / max(1, done)
        last = recs[-1][1] if recs else {}
        log(
          f'{name} L{level}: {min(a + len(chunk), len(batch))}/{len(batch)}, {rate:.1f} s each, "{last.get("label")}" ({last.get("score")})'
        )
    save()
    apply_labels(table, have)
    (dest / 'features.json').write_text(
      json.dumps(table, ensure_ascii=False, separators=(',', ':')),
      encoding='utf-8',
    )
    scored = [
      v['score'] for v in have.values() if isinstance(v, dict) and 'score' in v
    ]
    if scored:
      log(
        f'{name}: detection score mean {np.mean(scored):.2f} over {len(scored)}'
      )
  log(f'done: {done} labelled, {time.time() - t0:.0f}s')


if __name__ == '__main__':
  main()
