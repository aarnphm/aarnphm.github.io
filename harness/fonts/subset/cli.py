"""Argument parsing and the generation loop."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

from fontTools.ttLib import TTFont
from . import __doc__ as PACKAGE_DOC
from .build import (
  SCSS_HEADER,
  fetch_source,
  scss_rule,
  subset_font,
  unicode_range,
  verify,
)
from .census import census, gpos_partners, source_chars
from .publish import publish
from .faces import (
  CACHE,
  CORE_PAGE_SHARE,
  FACES,
  KEEP_FEATURES,
  OUT_KV,
  PRELOAD,
  ROOT,
  STYLES,
)


def main() -> int:
  ap = argparse.ArgumentParser(
    description=PACKAGE_DOC,
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  ap.add_argument(
    '--public',
    default=str(ROOT / 'public'),
    help='rendered site root for the character census',
  )
  ap.add_argument(
    '--source-url',
    default='https://aarnphm.xyz/fonts/',
    help='where KV originals are fetched from (cached under quartz/.quartz-cache/fonts/src)',
  )
  ap.add_argument(
    '--no-upload',
    action='store_true',
    help='write the outputs locally without pushing them to the FONTS KV',
  )
  ap.add_argument(
    '--census-json',
    help='reuse a census written by a previous run instead of rescanning',
  )
  args = ap.parse_args()

  OUT_KV.mkdir(parents=True, exist_ok=True)
  CACHE.mkdir(parents=True, exist_ok=True)

  if args.census_json:
    c = json.loads(Path(args.census_json).read_text())
    npages, counts, page_share = c['pages'], c['counts'], c['share']
  else:
    public = Path(args.public)
    if not public.is_dir():
      print(
        f'no rendered site at {public}; build first or pass --public',
        file=sys.stderr,
      )
      return 2
    npages, counter, pages = census(public)
    counts = {ch: n for ch, n in counter.items()}
    page_share = {ch: len(pages[ch]) / npages for ch in counter}
    (CACHE / 'census.json').write_text(
      json.dumps(
        {'pages': npages, 'counts': counts, 'share': page_share},
        ensure_ascii=False,
      )
    )
  src_chars = source_chars(ROOT)
  print(
    f'census: {npages} pages, {len(counts)} distinct characters; {len(src_chars)} non-ASCII characters in source strings'
  )

  used = {ord(ch) for ch in counts} | {ord(ch) for ch in src_chars}
  blocks = (
    set(range(0x20, 0x7F))
    | set(range(0xA0, 0x100))
    | set(range(0x100, 0x180))
    | set(range(0x2000, 0x2070))
  )
  universe = (
    (used | blocks) - set(range(0, 0x20)) - {0x7F} - set(range(0x80, 0xA0))
  )
  core_seed = (
    set(range(0x20, 0x7F))
    | {0xA0}
    | {
      ord(ch)
      for ch, s in page_share.items()
      if s >= CORE_PAGE_SHARE and ord(ch) >= 0x20
    }
  )

  manifest = {
    'generatedFrom': {
      'pages': npages,
      'corePageShare': CORE_PAGE_SHARE,
      'keepFeatures': KEEP_FEATURES,
    },
    'faces': [],
  }
  scss = [SCSS_HEADER]
  preload = []
  uploads: list[tuple[str, Path]] = []
  total_before = total_after = 0
  for face in FACES:
    if not face.enabled:
      manifest['faces'].append({
        'family': face.family,
        'source': face.source,
        'weight': face.weight,
        'style': face.style,
        'enabled': False,
        'note': face.note,
      })
      continue
    src = fetch_source(face, args.source_url)
    full = TTFont(src)
    cmap = full.getBestCmap()
    # Used characters the font lacks but shapes by canonical decomposition (Vietnamese
    # ố = o + ◌̂ + ◌́) need their parts in core, where their base letters already live.
    parts = {
      ord(ch)
      for cp in used
      if cp not in cmap
      for ch in unicodedata.normalize('NFD', chr(cp))
      if ord(ch) in cmap
    }
    keep = {cp for cp in universe | parts if cp in cmap}
    # Core closure over GPOS relations with glyph names, mapped back to code points.
    rel = gpos_partners(full)
    g2cp: dict[str, set[int]] = collections.defaultdict(set)
    for cp, g in cmap.items():
      if cp in keep:
        g2cp[g].add(cp)
    seed_glyphs = {cmap[cp] for cp in core_seed | parts if cp in cmap}
    pulled = {
      g
      for g in g2cp
      if g not in seed_glyphs and rel.get(g) and rel[g] & seed_glyphs
    }
    core_glyphs = seed_glyphs | pulled
    core = {cp for g in core_glyphs for cp in g2cp.get(g, ())}
    ext = keep - core
    entry = {
      'family': face.family,
      'source': face.source,
      'origin': face.origin,
      'weight': face.weight,
      'style': face.style,
      'enabled': True,
      'sourceBytes': src.stat().st_size,
      'sourceSha256': hashlib.sha256(src.read_bytes()).hexdigest()[:16],
      'sourceGlyphs': full['maxp'].numGlyphs,
      'sourceChars': len(cmap),
      'pin': face.pin or None,
      'rename': face.rename,
      'corePulledByGpos': sorted(pulled),
      'files': {},
    }
    total_before += src.stat().st_size
    core_cmap: set[int] = set()
    for part, cps in (('core', core), ('ext', ext)):
      if not cps:
        continue
      filename = f'{face.stem}.{part}.woff2'
      out = OUT_KV / filename
      # CoreText rasterizes a face's glyphs differently depending on which other glyphs the
      # file holds: an ext-only file rendered its characters 3-6% lighter (300-600 visible px
      # per test page) while transplanting every metadata table changed nothing, and the full
      # glyph set with the core characters unmapped rendered pixel-identical. So the ext file
      # carries every glyph of both parts and maps only the ext characters.
      font = subset_font(src, cps if part == 'core' else core | ext, face)
      # The subsetter keeps every cmap entry whose glyph survived closure, so a file can
      # carry characters beyond the plan. Those stay (the range must list what the file
      # holds), except entries already served by core, which are pruned so the two ranges
      # never overlap.
      if part == 'ext':
        for table in font['cmap'].tables:
          for cp in [cp for cp in table.cmap if cp in core_cmap]:
            del table.cmap[cp]
      font.save(out)
      actual = set(TTFont(out).getBestCmap())
      extras = actual - cps
      if part == 'core':
        core_cmap = actual
      else:
        assert not (actual & core_cmap), (
          f'{filename}: overlaps core {sorted(actual & core_cmap)[:5]}'
        )
      info = verify(out, actual, face)
      info['sha256'] = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
      info['extraChars'] = sorted(extras)
      # A single-file font also renders characters missing from its cmap when the shaper can
      # decompose them (NFD) into glyphs it has, e.g. Vietnamese ố → o + ◌̂ + ◌́ with mark
      # positioning. unicode-range would hide those from the file, so the range also lists
      # every used character whose decomposition this file covers.
      composed = {
        cp
        for cp in used
        if cp >= 0x80
        and cp not in cmap
        and cp not in core_cmap
        and (nfd := unicodedata.normalize('NFD', chr(cp))) != chr(cp)
        and all(ord(ch) in actual for ch in nfd)
      }
      info['composedChars'] = sorted(composed)
      if part == 'core':
        core_cmap = actual | composed
      rng = unicode_range(actual | composed)
      info['unicodeRange'] = rng
      info['file'] = filename
      entry['files'][part] = info
      # The worker serves /fonts/ with a one-year immutable cache, so a regenerated file under
      # the same name would never reach a returning browser without a content hash in the URL.
      versioned = f'{filename}?v={info["sha256"][:8]}'
      scss.append(scss_rule(face, versioned, '/fonts/', rng))
      total_after += info['bytes']
      uploads.append((filename, out))
      if part == 'core' and face.stem in PRELOAD:
        preload.append({'href': '/fonts/' + versioned, 'type': 'font/woff2'})
    manifest['faces'].append(entry)
    c, e = entry['files'].get('core', {}), entry['files'].get('ext', {})
    print(
      f'{face.source:<42} {entry["sourceBytes"]:>7} → core {c.get("bytes", 0):>6} ({c.get("chars", 0):>3} ch, {c.get("glyphs", 0):>3} g) + ext {e.get("bytes", 0):>6} ({e.get("chars", 0):>3} ch, {e.get("glyphs", 0):>3} g)  pulled={len(pulled)}'
    )

  (STYLES / 'fonts.scss').write_text(''.join(scss))
  (STYLES / 'fonts.preload.json').write_text(
    json.dumps(preload, indent=2) + '\n'
  )
  (STYLES / 'fonts.manifest.json').write_text(
    json.dumps(manifest, indent=1, ensure_ascii=False) + '\n'
  )
  print(
    f'\nsource bytes {total_before} → output bytes {total_after} across enabled faces'
  )
  print(
    f'wrote {STYLES / "fonts.scss"}, fonts.preload.json, fonts.manifest.json'
  )

  if args.no_upload:
    print(
      f'{len(uploads)} files written under {OUT_KV}, not uploaded (--no-upload)'
    )
    return 0
  return 1 if publish(uploads, args.source_url) else 0
