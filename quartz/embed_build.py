# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain-text-splitters",
#     "numpy",
#     "openai",
#     "tqdm",
#     "sentence-transformers",
#     "tiktoken",
# ]
# ///

from __future__ import annotations

import os, json, argparse, hashlib, math, random, logging, re, html, fnmatch, subprocess

from pathlib import Path
from functools import lru_cache
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken, numpy as np

from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)
DEFAULT_VLLM_URL = os.environ.get('VLLM_URL') or os.environ.get('VLLM_EMBED_URL') or 'http://127.0.0.1:8000/v1'
NOTEBOOK_TEXT_MIMES = ('text/markdown', 'text/plain')
NOTEBOOK_HTML_TAG = re.compile(r'<[^>]+>')
NOTEBOOK_HEADING = re.compile(r'^#{1,6}\s+(.+?)\s*#*\s*$')
QUOTED_STRING = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`", re.S)


def configure_logging() -> None:
  logging.basicConfig(level=logging.INFO, format='%(message)s')
  for noisy_logger in ('httpx', 'httpcore', 'openai'):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def resolve_vllm_base_url(url: str) -> str:
  trimmed = url.rstrip('/')
  if trimmed.endswith('/v1/embeddings'):
    trimmed = trimmed[: -len('/embeddings')]
  elif trimmed.endswith('/embeddings'):
    trimmed = trimmed[: trimmed.rfind('/')]

  if not trimmed.endswith('/v1'):
    trimmed = f'{trimmed}/v1'

  return trimmed


def load_jsonl(fp: str) -> Iterable[dict]:
  with open(fp, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      yield json.loads(line)


def as_text(value) -> str:
  if isinstance(value, str):
    return value
  if isinstance(value, list):
    return ''.join(as_text(item) for item in value)
  if value is None:
    return ''
  return json.dumps(value, ensure_ascii=False)


def normalize_notebook_text(value: str) -> str:
  lines = [line.rstrip() for line in value.replace('\r\n', '\n').replace('\r', '\n').split('\n')]
  cleaned: list[str] = []
  previous_blank = False
  for line in lines:
    blank = not line.strip()
    if blank and previous_blank:
      continue
    cleaned.append(line)
    previous_blank = blank
  return '\n'.join(cleaned).strip()


def notebook_output_text(output: dict) -> str:
  output_type = output.get('output_type')
  if output_type == 'stream':
    return as_text(output.get('text'))
  if output_type == 'error':
    traceback = output.get('traceback')
    if isinstance(traceback, list) and traceback:
      return '\n'.join(as_text(line) for line in traceback)
    return normalize_notebook_text(f'{as_text(output.get("ename"))}: {as_text(output.get("evalue"))}')
  if output_type not in {'display_data', 'execute_result'}:
    return ''

  data = output.get('data')
  if not isinstance(data, dict):
    return ''
  for mime in NOTEBOOK_TEXT_MIMES:
    text = as_text(data.get(mime))
    if text.strip():
      return text
  html_text = as_text(data.get('text/html'))
  if html_text.strip():
    return html.unescape(NOTEBOOK_HTML_TAG.sub(' ', html_text))
  return ''


def notebook_language(metadata: dict) -> str:
  language_info = metadata.get('language_info')
  kernelspec = metadata.get('kernelspec')
  candidates = []
  if isinstance(language_info, dict):
    candidates.append(language_info.get('name'))
  if isinstance(kernelspec, dict):
    candidates.extend([kernelspec.get('language'), kernelspec.get('name')])
  for value in candidates:
    if isinstance(value, str) and value.strip():
      return re.sub(r'[^A-Za-z0-9_+#.-]', '', value) or 'python'
  return 'python'


def markdown_heading_title(source: str) -> str | None:
  for line in source.splitlines():
    match = NOTEBOOK_HEADING.match(line)
    if not match:
      continue
    title = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', match.group(1))
    title = title.replace('*', '').replace('_', '').replace('`', '').strip()
    if title:
      return title
  return None


def notebook_title(doc: dict, fp: Path) -> str:
  cells = doc.get('cells')
  if isinstance(cells, list):
    for raw_cell in cells:
      if not isinstance(raw_cell, dict) or raw_cell.get('cell_type') != 'markdown':
        continue
      title = markdown_heading_title(as_text(raw_cell.get('source')))
      if title:
        return title
  return fp.stem


def notebook_text(doc: dict) -> str:
  cells = doc.get('cells')
  if not isinstance(cells, list):
    return ''
  language = notebook_language(doc.get('metadata') if isinstance(doc.get('metadata'), dict) else {})
  parts: list[str] = []
  for raw_cell in cells:
    if not isinstance(raw_cell, dict):
      continue
    cell_type = raw_cell.get('cell_type')
    source = normalize_notebook_text(as_text(raw_cell.get('source')))
    if source:
      if cell_type == 'code':
        parts.append(f'```{language}\n{source}\n```')
      else:
        parts.append(source)
    outputs = raw_cell.get('outputs')
    if cell_type == 'code' and isinstance(outputs, list):
      for output in outputs:
        if not isinstance(output, dict):
          continue
        text = normalize_notebook_text(notebook_output_text(output))
        if text:
          parts.append(text)
  return normalize_notebook_text('\n\n'.join(parts))


def sluggify(value: str) -> str:
  return '/'.join(
    segment
    .replace(' ', '-')
    .replace('\t', '-')
    .replace('\n', '-')
    .replace('\r', '-')
    .replace('\f', '-')
    .replace('&', '-and-')
    .replace('%', '-percent')
    .replace('?', '')
    .replace('#', '')
    for segment in value.split('/')
  ).rstrip('/')


def slugify_file_path(fp: str, exclude_ext: bool = False) -> str:
  fp = fp.strip('/')
  ext = Path(fp).suffix
  without_ext = fp[: -len(ext)] if ext else fp
  if exclude_ext or ext in {'.md', '.html', '.base', ''}:
    ext = ''
  slug = sluggify(without_ext)
  if slug.endswith('_index'):
    slug = f'{slug[: -len("_index")]}index'
  return f'{slug}{ext}'


def ts_array_body(source: str, key: str) -> str | None:
  key_index = source.find(f'{key}:')
  if key_index < 0:
    return None
  start = source.find('[', key_index)
  if start < 0:
    return None
  depth = 0
  quote = ''
  escaped = False
  for index in range(start, len(source)):
    char = source[index]
    if quote:
      if escaped:
        escaped = False
      elif char == '\\':
        escaped = True
      elif char == quote:
        quote = ''
      continue
    if char in ('"', "'", '`'):
      quote = char
      continue
    if char == '[':
      depth += 1
      continue
    if char == ']':
      depth -= 1
      if depth == 0:
        return source[start + 1 : index]
  return None


def unquote_ts_string(value: str) -> str:
  quote = value[0]
  inner = value[1:-1]
  if quote == '`':
    return inner.replace('\\`', '`').replace('\\\\', '\\')
  return bytes(inner, 'utf-8').decode('unicode_escape')


def quartz_ignore_patterns(config_path: Path) -> list[str]:
  if not config_path.exists():
    return []
  try:
    source = config_path.read_text(encoding='utf-8')
  except OSError as exc:
    logger.warning('Could not read Quartz config %s: %s', config_path, exc)
    return []
  body = ts_array_body(source, 'ignorePatterns')
  if body is None:
    return []
  return [unquote_ts_string(match.group(0)) for match in QUOTED_STRING.finditer(body)]


def content_relative_path(fp: Path, content_dir: Path) -> str | None:
  content_root = content_dir.resolve()
  try:
    return fp.resolve().relative_to(content_root).as_posix()
  except ValueError:
    return None


def content_path(candidate: str, content_dir: Path) -> tuple[Path, str] | None:
  content_root = content_dir.resolve()
  fp = Path(candidate)
  if not fp.is_absolute():
    fp = content_root.parent / fp if candidate.startswith('content/') else content_root / fp
  rel = content_relative_path(fp, content_root)
  if rel is None:
    return None
  return fp, rel


def content_index_file_records(content_index: Path, content_dir: Path, suffix: str | None = None) -> list[dict]:
  if not content_index.exists():
    return []
  try:
    raw = json.loads(content_index.read_text(encoding='utf-8'))
  except json.JSONDecodeError as exc:
    logger.warning('Could not parse notebook content index %s: %s', content_index, exc)
    return []
  if not isinstance(raw, dict):
    return []

  records: list[dict] = []
  for slug, value in raw.items():
    if not isinstance(slug, str) or not isinstance(value, dict):
      continue
    file_path = value.get('filePath')
    file_name = value.get('fileName')
    candidate = file_path if isinstance(file_path, str) else file_name
    if not isinstance(candidate, str):
      continue
    if suffix is not None and not candidate.endswith(suffix):
      continue
    resolved = content_path(candidate, content_dir)
    if resolved is None:
      continue
    fp, rel = resolved
    title = value.get('title')
    records.append({
      'slug': slug,
      'title': title if isinstance(title, str) and title else fp.stem,
      'path': fp,
      'relative_path': rel,
    })
  return records


def notebook_records_from_content_index(content_index: Path, content_dir: Path) -> list[dict]:
  return content_index_file_records(content_index, content_dir, '.ipynb')


def notebook_records_from_content_dir(content_dir: Path) -> list[dict]:
  if not content_dir.exists():
    return []
  records: list[dict] = []
  for fp in sorted(content_dir.rglob('*.ipynb')):
    rel = fp.relative_to(content_dir).as_posix()
    records.append({'slug': slugify_file_path(rel, True), 'title': fp.stem, 'path': fp, 'relative_path': rel})
  return records


def git_ignored_paths(paths: list[str], content_dir: Path) -> set[str]:
  if not paths:
    return set()
  try:
    result = subprocess.run(
      ['git', 'check-ignore', '--stdin', '-z', '--no-index'],
      input=('\0'.join(paths) + '\0').encode(),
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
      cwd=content_dir,
      check=False,
    )
  except OSError:
    return set()
  if result.returncode not in (0, 1):
    return set()
  return {path for path in result.stdout.decode().split('\0') if path}


def ignore_pattern_matches(path: str, pattern: str) -> bool:
  normalized = pattern.strip().strip('/')
  if not normalized:
    return False
  if normalized.startswith('!'):
    return False
  if normalized.startswith('content/') and ignore_pattern_matches(path, normalized[len('content/') :]):
    return True
  if fnmatch.fnmatch(path, normalized):
    return True
  if fnmatch.fnmatch(f'/{path}', normalized):
    return True
  if '/' not in normalized:
    parts = path.split('/')
    return normalized in parts or any(fnmatch.fnmatch(part, normalized) for part in parts)
  return path.startswith(f'{normalized}/') or fnmatch.fnmatch(path, f'{normalized}/**')


def filter_notebook_records(records: list[dict], content_dir: Path, ignore_patterns: list[str]) -> list[dict]:
  ignored = git_ignored_paths([record['relative_path'] for record in records], content_dir)
  return [
    record
    for record in records
    if record['relative_path'] not in ignored
    and not any(ignore_pattern_matches(record['relative_path'], pattern) for pattern in ignore_patterns)
  ]


def ignored_slugs_from_content_index(content_index: Path, content_dir: Path, ignore_patterns: list[str]) -> set[str]:
  records = content_index_file_records(content_index, content_dir)
  ignored = git_ignored_paths([record['relative_path'] for record in records], content_dir)
  return {
    record['slug']
    for record in records
    if record['relative_path'] in ignored
    or any(ignore_pattern_matches(record['relative_path'], pattern) for pattern in ignore_patterns)
  }


def load_notebook_documents(content_index: Path, content_dir: Path, ignore_patterns: list[str]) -> list[dict]:
  content_dir = content_dir.resolve()
  records = notebook_records_from_content_index(content_index, content_dir)
  if not records:
    records = notebook_records_from_content_dir(content_dir)
  records = filter_notebook_records(records, content_dir, ignore_patterns)
  docs: list[dict] = []
  failed = 0
  seen: set[str] = set()
  for record in tqdm(records, desc='Loading notebooks', unit='notebook'):
    slug = record['slug']
    if slug in seen:
      continue
    seen.add(slug)
    fp = record['path']
    try:
      notebook = json.loads(fp.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
      failed += 1
      logger.warning('Skipping notebook %s: %s', fp, exc)
      continue
    if not isinstance(notebook, dict):
      failed += 1
      logger.warning('Skipping notebook %s: notebook root is not an object', fp)
      continue
    text = notebook_text(notebook)
    if not text:
      continue
    title = notebook_title(notebook, fp)
    docs.append({'slug': slug, 'title': title or record['title'], 'text': text})
  if docs or failed:
    print(f'Loaded {len(docs)} notebook document(s) for embeddings ({failed} skipped)')
  return docs


def merge_documents(recs: list[dict], additions: list[dict]) -> list[dict]:
  merged = list(recs)
  index_by_slug = {rec.get('slug'): i for i, rec in enumerate(merged)}
  replaced = 0
  for addition in tqdm(additions, desc='Merging notebook docs', unit='doc', disable=len(additions) == 0):
    idx = index_by_slug.get(addition['slug'])
    if idx is None:
      index_by_slug[addition['slug']] = len(merged)
      merged.append(addition)
      continue
    merged[idx] = {**merged[idx], **addition}
    replaced += 1
  if additions:
    print(f'Merged {len(additions)} notebook document(s) ({replaced} replaced, {len(additions) - replaced} added)')
  return merged


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
  norms = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
  norms[norms == 0] = 1.0
  return x / norms


@lru_cache(maxsize=1)
def get_tiktoken_encoder():
  return tiktoken.get_encoding('o200k_base')


def count_tokens(text: str) -> int:
  encoder = get_tiktoken_encoder()
  return len(encoder.encode(text))


def get_text_splitter(chunk_size: int, overlap: int):
  encoder = get_tiktoken_encoder()
  return RecursiveCharacterTextSplitter(
    chunk_size=chunk_size * 4,
    chunk_overlap=overlap * 4,
    separators=['\n\n', '\n', '. ', ' ', ''],
    length_function=lambda t: len(encoder.encode(t)),
    is_separator_regex=False,
  )


def chunk_document(
  doc: dict,
  chunk_size: int = 512,
  overlap_tokens: int = 128,
  min_chunk_size: int = 100,
  model_id: str = '',
  model_max_tokens: int = 512,
) -> list[dict]:
  text = doc['text']
  token_count = count_tokens(text)

  prefix_overhead = get_prefix_overhead(model_id)
  effective_model_limit = model_max_tokens - prefix_overhead
  actual_chunk_size = min(chunk_size, effective_model_limit)

  if token_count <= actual_chunk_size:
    return [
      {
        'slug': doc['slug'],
        'title': doc.get('title', doc['slug']),
        'text': text,
        'chunk_id': 0,
        'parent_slug': doc['slug'],
        'is_chunked': False,
      }
    ]

  splitter = get_text_splitter(actual_chunk_size, overlap_tokens)
  raw_chunks = splitter.split_text(text)

  valid_chunks = [c for c in raw_chunks if count_tokens(c) >= min_chunk_size]

  return [
    {
      'slug': f'{doc["slug"]}#chunk{i}',
      'title': doc.get('title', doc['slug']),
      'text': chunk,
      'chunk_id': i,
      'parent_slug': doc['slug'],
      'is_chunked': True,
    }
    for i, chunk in enumerate(valid_chunks)
  ]


def write_shards(vectors: np.ndarray, shard_size: int, dtype: str, out_dir: Path) -> list[dict]:
  out_dir.mkdir(parents=True, exist_ok=True)
  rows, dims = vectors.shape
  shards_meta: list[dict] = []
  np_dtype = np.float16 if dtype == 'fp16' else np.float32
  bytes_per_value = np.dtype(np_dtype).itemsize
  row_offset = 0
  shard_starts = range(0, rows, shard_size)
  for si, start in enumerate(tqdm(shard_starts, total=len(shard_starts), desc='Writing vector shards', unit='shard')):
    end = min(start + shard_size, rows)
    shard = vectors[start:end]
    bin_path = out_dir / f'vectors-{si:03d}.bin'
    payload = shard.astype(np_dtype, copy=False).tobytes(order='C')
    digest = hashlib.sha256(payload).hexdigest()
    with open(bin_path, 'wb') as f:
      f.write(payload)
    shard_rows = int(shard.shape[0])
    shards_meta.append({
      'path': f'/embeddings/{bin_path.name}',
      'rows': shard_rows,
      'rowOffset': row_offset,
      'byteLength': len(payload),
      'sha256': digest,
      'byteStride': dims * bytes_per_value,
    })
    row_offset += shard_rows
  return shards_meta


def write_hnsw_graph(levels: list[list[list[int]]], rows: int, out_path: Path) -> tuple[list[dict], str]:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  offset = 0
  meta: list[dict] = []
  digest = hashlib.sha256()
  with open(out_path, 'wb') as f:
    for lvl in tqdm(levels, desc='Writing HNSW graph', unit='level'):
      indptr = np.zeros(rows + 1, dtype=np.uint32)
      edge_accum: list[int] = []
      for idx in range(rows):
        neighbors = lvl[idx] if idx < len(lvl) else []
        indptr[idx + 1] = indptr[idx] + len(neighbors)
        edge_accum.extend(neighbors)
      indptr_bytes = indptr.tobytes(order='C')
      indptr_offset = offset
      f.write(indptr_bytes)
      digest.update(indptr_bytes)
      offset += len(indptr_bytes)

      if edge_accum:
        indices = np.asarray(edge_accum, dtype=np.uint32)
        indices_bytes = indices.tobytes(order='C')
      else:
        indices = np.zeros(0, dtype=np.uint32)
        indices_bytes = indices.tobytes(order='C')
      indices_offset = offset
      f.write(indices_bytes)
      digest.update(indices_bytes)
      offset += len(indices_bytes)

      meta.append({
        'level': len(meta),
        'indptr': {'offset': indptr_offset, 'elements': int(indptr.shape[0]), 'byteLength': len(indptr_bytes)},
        'indices': {'offset': indices_offset, 'elements': int(indices.shape[0]), 'byteLength': len(indices_bytes)},
      })
  return meta, digest.hexdigest()


def get_prefix_overhead(model_id: str) -> int:
  model_lower = model_id.lower()
  if 'embeddinggemma' in model_lower:
    return count_tokens('title: none | text: ')
  elif 'e5' in model_lower:
    return count_tokens('passage: ')
  return 0


def validate_token_limits(texts: list[str], max_tokens: int, model_id: str) -> None:
  prefix_overhead = get_prefix_overhead(model_id)
  effective_max = max_tokens - prefix_overhead
  over_limit = []
  for i, text in enumerate(texts):
    tokens = count_tokens(text)
    if tokens > effective_max:
      over_limit.append((i, tokens, effective_max))
  if over_limit:
    logger.error(
      'ERROR: %d/%d chunks exceed token limit after prefix. First few: %s. '
      'This indicates chunking failed - chunk_size (%d) should prevent this.',
      len(over_limit),
      len(texts),
      over_limit[:3],
      effective_max,
    )
    raise ValueError(
      f'{len(over_limit)} chunks exceed {max_tokens} token limit (effective: {effective_max} after prefix)'
    )


def embed_vllm(
  texts: list[str], model_id: str, vllm_url: str, batch_size: int = 64, concurrency: int = 8, max_tokens: int = 512
) -> np.ndarray:
  base_url = resolve_vllm_base_url(vllm_url)
  api_key = os.environ.get('VLLM_API_KEY') or os.environ.get('OPENAI_API_KEY') or 'not-set'
  client = OpenAI(base_url=base_url, api_key=api_key, timeout=300)

  validate_token_limits(texts, max_tokens, model_id)

  model_lower = model_id.lower()
  if 'e5' in model_lower:
    prefixed = [f'passage: {t}' for t in texts]
  elif 'embeddinggemma' in model_lower:
    prefixed = [f'title: none | text: {t}' for t in texts]
  else:
    prefixed = texts

  print(
    f'Embedding {len(prefixed)} texts with vLLM (model={model_id}, batch_size={batch_size}, concurrency={concurrency})'
  )

  batches = []
  for i in range(0, len(prefixed), batch_size):
    batch = prefixed[i : i + batch_size]
    batches.append((i, batch))

  def send_batch(batch_info: tuple[int, list[str]]) -> tuple[int, list[np.ndarray]]:
    idx, batch = batch_info
    response = client.embeddings.create(model=model_id, input=batch)
    embeddings = [np.asarray(item.embedding, dtype=np.float32) for item in response.data]
    return (idx, embeddings)

  results: dict[int, list[np.ndarray]] = {}
  if len(batches) == 1:
    idx, embeddings = send_batch(batches[0])
    results[idx] = embeddings
  else:
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
      futures = {executor.submit(send_batch, batch_info): batch_info[0] for batch_info in batches}
      for future in tqdm(as_completed(futures), total=len(futures), desc='Embedding batches', unit='batch'):
        idx, embeddings = future.result()
        results[idx] = embeddings

  out: list[np.ndarray] = []
  for i in sorted(results.keys()):
    out.extend(results[i])

  return np.stack(out, axis=0)


def embed_hf(texts: list[str], model_id: str, device: str, max_tokens: int = 512) -> np.ndarray:
  from sentence_transformers import SentenceTransformer

  model = SentenceTransformer(model_id, device=device)

  validate_token_limits(texts, max_tokens, model_id)

  model_lower = model_id.lower()
  if 'e5' in model_lower:
    prefixed = [f'passage: {t}' for t in texts]
  elif 'embeddinggemma' in model_lower:
    prefixed = [f'title: none | text: {t}' for t in texts]
  else:
    prefixed = texts

  vecs = model.encode(
    prefixed, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
  )
  return vecs.astype(np.float32, copy=False)


def main():
  configure_logging()
  ap = argparse.ArgumentParser()
  ap.add_argument('--jsonl', default='public/embeddings-text.jsonl')
  ap.add_argument('--content-dir', default='content')
  ap.add_argument('--content-index')
  ap.add_argument('--quartz-config', default='quartz.config.ts')
  ap.add_argument('--ignore-pattern', action='append', default=[])
  ap.add_argument('--no-notebooks', action='store_true')
  ap.add_argument('--model', default=os.environ.get('SEM_MODEL', 'intfloat/multilingual-e5-large'))
  ap.add_argument('--dims', type=int, default=int(os.environ.get('SEM_DIMS', '1024')))
  ap.add_argument('--dtype', choices=['fp16', 'fp32'], default=os.environ.get('SEM_DTYPE', 'fp32'))
  ap.add_argument('--shard-size', type=int, default=int(os.environ.get('SEM_SHARD', '1024')))
  ap.add_argument('--out', default='public/embeddings')
  ap.add_argument('--use-vllm', action='store_true', default=bool(os.environ.get('USE_VLLM', '')))
  ap.add_argument(
    '--vllm-url',
    default=DEFAULT_VLLM_URL,
    help='Base URL for the vLLM OpenAI-compatible server (accepts either /v1 or /v1/embeddings)',
  )
  ap.add_argument('--chunk-size', type=int, default=512, help='Max tokens per chunk')
  ap.add_argument('--chunk-overlap', type=int, default=128, help='Overlap tokens between chunks')
  ap.add_argument('--no-chunking', action='store_true', help='Disable chunking (embed full docs)')
  ap.add_argument(
    '--max-tokens',
    type=int,
    default=512,
    help="Model's maximum context length in tokens (e5-large: 512, embeddinggemma: 8192)",
  )
  ap.add_argument(
    '--concurrency',
    type=int,
    default=int(os.environ.get('VLLM_CONCURRENCY', '8')),
    help='Number of concurrent requests to vLLM (default: 8)',
  )
  ap.add_argument(
    '--batch-size',
    type=int,
    default=int(os.environ.get('VLLM_BATCH_SIZE', '64')),
    help='Batch size for vLLM requests (default: 64)',
  )
  args = ap.parse_args()

  jsonl_path = Path(args.jsonl)
  content_dir = Path(args.content_dir)
  content_index = (
    Path(args.content_index) if args.content_index else jsonl_path.parent / 'static' / 'contentIndex.json'
  )
  ignore_patterns = [*quartz_ignore_patterns(Path(args.quartz_config)), *args.ignore_pattern]
  recs = list(load_jsonl(args.jsonl))
  ignored_slugs = ignored_slugs_from_content_index(content_index, content_dir, ignore_patterns)
  if ignored_slugs:
    before_filter = len(recs)
    recs = [rec for rec in recs if rec.get('slug') not in ignored_slugs]
    print(f'Filtered {before_filter - len(recs)} ignored document(s) from embeddings input')
  if not args.no_notebooks:
    recs = merge_documents(recs, load_notebook_documents(content_index, content_dir, ignore_patterns))
  if not recs:
    print(f'No input found in {args.jsonl}; run the site build first to emit JSONL.')
    return

  # Filter out are.na (423 chunks, structural outlier)
  recs = [r for r in recs if r['slug'] != 'are.na']
  print(f'Filtered to {len(recs)} documents (excluding are.na)')

  if args.no_chunking:
    chunks = recs
    chunk_metadata = {}
    print(f'Chunking disabled. Processing {len(chunks)} full documents')
  else:
    chunks = []
    chunk_metadata = {}
    for rec in tqdm(recs, desc='Chunking documents', unit='doc'):
      doc_chunks = chunk_document(
        rec,
        chunk_size=args.chunk_size,
        overlap_tokens=args.chunk_overlap,
        model_id=args.model,
        model_max_tokens=args.max_tokens,
      )
      chunks.extend(doc_chunks)
      for chunk in doc_chunks:
        if chunk['is_chunked']:
          chunk_metadata[chunk['slug']] = {'parentSlug': chunk['parent_slug'], 'chunkId': chunk['chunk_id']}
    chunked_count = sum(1 for c in chunks if c.get('is_chunked', False))
    prefix_overhead = get_prefix_overhead(args.model)
    effective_limit = args.max_tokens - prefix_overhead
    actual_chunk_size = min(args.chunk_size, effective_limit)
    print(
      f'Chunked {len(recs)} documents into {len(chunks)} chunks '
      f'({chunked_count} chunked, {len(chunks) - chunked_count} unchanged)'
    )
    print(
      f'  Chunk size: {actual_chunk_size} tokens (requested: {args.chunk_size}, '
      f'model limit: {args.max_tokens}, prefix overhead: {prefix_overhead}), '
      f'overlap: {args.chunk_overlap} tokens'
    )

  ids = [c['slug'] for c in chunks]
  titles = [c.get('title', c['slug']) for c in chunks]
  texts = [c['text'] for c in chunks]

  if args.use_vllm:
    vecs = embed_vllm(
      texts,
      args.model,
      args.vllm_url,
      batch_size=args.batch_size,
      concurrency=args.concurrency,
      max_tokens=args.max_tokens,
    )
  else:
    device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    vecs = embed_hf(texts, args.model, device, max_tokens=args.max_tokens)

  if vecs.shape[1] != args.dims:
    if vecs.shape[1] > args.dims:
      vecs = vecs[:, : args.dims]
    else:
      vecs = np.pad(vecs, ((0, 0), (0, args.dims - vecs.shape[1])))
  vecs = l2_normalize_rows(vecs.astype(np.float32, copy=False))

  out_dir = Path(args.out)
  shards = write_shards(vecs, args.shard_size, args.dtype, out_dir)

  def hnsw_build(data: np.ndarray, M: int = 16, efC: int = 200, seed: int = 0) -> dict:
    rng = random.Random(seed)
    N, _D = data.shape

    node_levels = []
    for _ in range(N):
      lvl = 0
      while rng.random() < 1 / math.e:
        lvl += 1
      node_levels.append(lvl)
    max_level = max(node_levels) if N > 0 else 0
    levels: list[list[list[int]]] = [[[] for _ in range(N)] for _ in range(max_level + 1)]

    def sim(i: int, j: int) -> float:
      return float((data[i] * data[j]).sum())

    entry = 0 if N > 0 else -1
    entry_level = node_levels[entry] if entry >= 0 else -1

    def search_layer(q: int, ep: int, ef: int, L: int) -> list[int]:
      if ep < 0:
        return []
      visited = set()
      cand: list[tuple[float, int]] = []
      top: list[tuple[float, int]] = []

      def push(node: int):
        if node in visited:
          return
        visited.add(node)
        cand.append((sim(q, node), node))

      push(ep)
      while cand:
        cand.sort(reverse=True)
        s, v = cand.pop(0)
        if len(top) >= ef and s <= top[-1][0]:
          break
        top.append((s, v))
        for u in levels[L][v]:
          push(u)
      top.sort(reverse=True)
      return [n for _, n in top]

    for i in tqdm(range(N), desc='Building HNSW graph', unit='vector'):
      if i == 0:
        continue
      lvl = node_levels[i]
      ep = entry
      for L in range(max_level, lvl, -1):
        c = search_layer(i, ep, 1, L)
        if c:
          ep = c[0]
      for L in range(min(max_level, lvl), -1, -1):
        W = search_layer(i, ep, efC, L)
        neigh = sorted(((sim(i, j), j) for j in W if j != i), reverse=True)[:M]
        for _, e in neigh:
          if e not in levels[L][i]:
            levels[L][i].append(e)
          if i not in levels[L][e]:
            levels[L][e].append(i)
      if lvl > entry_level:
        entry = i
        entry_level = lvl

    for L in tqdm(range(len(levels)), desc='Pruning HNSW graph', unit='level'):
      for i in range(N):
        if len(levels[L][i]) > M:
          nb = levels[L][i]
          nb = sorted(nb, key=lambda j: sim(i, j), reverse=True)[:M]
          levels[L][i] = nb
        else:
          unique = list(dict.fromkeys(levels[L][i]))
          unique.sort(key=lambda j: sim(i, j), reverse=True)
          levels[L][i] = unique[:M]

    return {'M': M, 'efConstruction': efC, 'entryPoint': entry, 'maxLevel': max_level, 'levels': levels}

  hnsw = hnsw_build(vecs, M=16, efC=200)
  hnsw_meta, hnsw_sha = write_hnsw_graph(hnsw['levels'], int(vecs.shape[0]), out_dir / 'hnsw.bin')

  manifest = {
    'version': 2,
    'model': args.model,
    'dims': args.dims,
    'dtype': args.dtype,
    'normalized': True,
    'rows': int(vecs.shape[0]),
    'shardSizeRows': args.shard_size,
    'vectors': {'dtype': args.dtype, 'rows': int(vecs.shape[0]), 'dims': args.dims, 'shards': shards},
    'ids': ids,
    'titles': titles,
    'chunkMetadata': chunk_metadata,
    'hnsw': {
      'M': hnsw['M'],
      'efConstruction': hnsw['efConstruction'],
      'entryPoint': hnsw['entryPoint'],
      'maxLevel': hnsw['maxLevel'],
      'graph': {'path': '/embeddings/hnsw.bin', 'sha256': hnsw_sha, 'levels': hnsw_meta},
    },
  }
  (out_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
  print(f'Wrote {len(shards)} vector shard(s), HNSW graph, and manifest to {out_dir}')


if __name__ == '__main__':
  main()
