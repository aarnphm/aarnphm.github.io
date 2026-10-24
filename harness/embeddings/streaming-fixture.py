"""Record a real out-of-order page stream for the observer replay."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root, pid = Path(sys.argv[1]), sys.argv[2]
subprocess.run(
  [
    sys.executable,
    str(Path(__file__).with_name('timing-fixture.py')),
    str(root),
    '0',
    pid,
  ],
  check=True,
)
sys.path.insert(
  0, str(Path(__file__).resolve().parents[2] / 'quartz/scripts/pdf-search')
)
from cache import ExtractionSpec, ExtractedPage, PageCache

cache_root = root / 'quartz/.quartz-cache/pdf-search'
run = cache_root / 'background-papers'
spec = ExtractionSpec(**json.loads((cache_root / 'recipe.json').read_text()))
events = []
sources = {}
for name, planned in [('alpha', [2, 3, 4, 5, 6, 7]), ('beta', [1, 2])]:
  source = root / f'content/thoughts/papers/{name}.pdf'
  cache = PageCache(
    cache_root, hashlib.sha256(source.read_bytes()).hexdigest(), spec
  )
  sources[name] = (source, cache)
  events.append({'slug': f'thoughts/papers/{name}.pdf', 'newPages': planned})
for name, number in [
  ('alpha', 2),
  ('alpha', 3),
  ('alpha', 4),
  ('beta', 1),
  ('alpha', 5),
  ('beta', 2),
]:
  source, _ = sources[name]
  events.append({
    'event': 'page-admitted',
    'source': str(source),
    'page': number,
  })
for name, number in [('alpha', 5), ('beta', 2), ('alpha', 3), ('alpha', 3)]:
  source, cache = sources[name]
  cache.write(ExtractedPage(number, '', (), 0))
  events.append({
    'source': str(source),
    'key': cache.key,
    'page': number,
    'seconds': 10,
  })
if '--failed' in sys.argv:
  events.append({
    'event': 'page-rejected',
    'source': str(sources['beta'][0]),
    'page': 1,
    'reason': 'truncated region',
  })
state = json.loads((run / 'status.json').read_text())
log = Path(state['log'])
log.write_text('\n'.join(map(json.dumps, events)) + '\n')
state.update(updatedAt=datetime.now(timezone.utc).isoformat(), batchPages=64)
(run / 'status.json').write_text(json.dumps(state))
