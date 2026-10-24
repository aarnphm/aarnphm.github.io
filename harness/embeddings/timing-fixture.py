"""Write real PDF/cache/log inputs for the remaining-work regression replay."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pymupdf

sys.path.insert(
  0, str(Path(__file__).resolve().parents[2] / 'quartz/scripts/pdf-search')
)
from cache import ExtractionSpec, ExtractedPage, PageCache, TextRegion


root = Path(sys.argv[1])
count = int(sys.argv[2])
pid = int(sys.argv[3])
cache_root = root / 'quartz/.quartz-cache/pdf-search'
run = cache_root / 'background-papers'
content = root / 'content/thoughts/papers'
index = root / 'public/static/searchIndex.json'
for folder in (run, content, index.parent):
  folder.mkdir(parents=True, exist_ok=True)
spec = ExtractionSpec('ocr', 'layout', 'sdk', 200, 'current-recipe')
(cache_root / 'recipe.json').write_text(json.dumps(asdict(spec)))
(root / 'quartz.config.ts').write_text('export default {ignorePatterns: []}')
for name, total in [('alpha', 7), ('beta', 2)]:
  path = content / f'{name}.pdf'
  if not path.exists():
    with pymupdf.open() as pdf:
      for number in range(total):
        page = pdf.new_page()
        page.insert_text((72, 72), f'{name} {number + 1}')
      pdf.save(path)
shutil.copyfile(content / 'beta.pdf', content / 'beta-copy.pdf')
index.write_text(
  json.dumps({
    f'thoughts/papers/{p.name}': {
      'filePath': str(p),
      'fileName': p.name,
      'title': p.stem,
    }
    for p in content.glob('*.pdf')
  })
)
source = content / 'alpha.pdf'
cache = PageCache(
  cache_root, hashlib.sha256(source.read_bytes()).hexdigest(), spec
)
for number in range(1, count + 2):
  cache.write(
    ExtractedPage(number, '', (TextRegion(0, 'text', 'paragraph', None),), 0),
    origin={'recipe': 'previous'} if number == 1 else None,
  )
header = {'source': str(source), 'key': cache.key}
old = [
  {'slug': 'thoughts/papers/alpha.pdf', 'newPages': [2]},
  header,
  {'page': 2, 'seconds': 999},
]
(run / '20000101T000000000000-extracting.log').write_text(
  '\n'.join(map(json.dumps, old))
)
log = run / (
  datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '-extracting.log'
)
events = [
  {'slug': 'thoughts/papers/alpha.pdf', 'newPages': list(range(2, 8))},
  {'source': str(source), 'key': 'previous-recipe-key'},
  {'page': 2, 'seconds': 999},
  header,
  {'page': 1, 'seconds': 999},
]
events.extend(
  {'page': i + 2, 'seconds': seconds}
  for i, seconds in enumerate([10, 10, 10, 10, 110, 50][:count])
)
log.write_text('\n'.join(map(json.dumps, events)) + '\n')
(run / 'status.json').write_text(
  json.dumps({
    'pid': pid,
    'phase': 'extracting',
    'updatedAt': datetime.now(timezone.utc).isoformat(),
    'prefix': 'thoughts/papers/',
    'firstPages': 0,
    'log': str(log),
  })
)
if '--finish' in sys.argv:
  beta = content / 'beta.pdf'
  beta_cache = PageCache(
    cache_root, hashlib.sha256(beta.read_bytes()).hexdigest(), spec
  )
  for number in (1, 2):
    beta_cache.write(ExtractedPage(number, '', (), 0))
