"""Fetch, rename, subset and verify one face; unicode-range and SCSS rule text."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from .faces import FETCH_HEADERS, KEEP_FEATURES, SRC_CACHE, STATIC, Face


def fetch_source(face: Face, source_url: str) -> Path:
  if face.origin == 'static':
    return STATIC / face.source
  SRC_CACHE.mkdir(parents=True, exist_ok=True)
  dst = SRC_CACHE / face.source
  if not dst.exists():
    url = source_url.rstrip('/') + '/' + face.source
    print(f'fetch {url}')
    req = urllib.request.Request(url, headers=FETCH_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
      dst.write_bytes(r.read())
  return dst


def rename_font(font: TTFont, new_family: str, face: Face) -> None:
  name = font['name']
  style = name.getDebugName(17) or name.getDebugName(2) or 'Regular'
  compact = re.sub(r'[^A-Za-z0-9]', '', new_family)
  ps_style = re.sub(r'[^A-Za-z0-9]', '', style)
  for rec in list(name.names):
    if rec.nameID in (1, 16):
      rec.string = new_family
    elif rec.nameID == 4:
      rec.string = f'{new_family} {style}'.strip()
    elif rec.nameID == 6:
      rec.string = f'{compact}-{ps_style}'
    elif rec.nameID == 3:
      rec.string = f'{compact}-{ps_style}-garden-subset'
    elif rec.nameID in (21, 22):
      name.names.remove(rec)
  orig = face.family
  name.setName(
    f'Subset of {orig} for aarnphm.xyz; renamed under the OFL Reserved Font Name clause.',
    10,
    3,
    1,
    0x409,
  )


def subset_font(src: Path, keep: set[int], face: Face) -> TTFont:
  # recalcTimestamp would stamp head.modified with the wall clock, so every rerun would
  # rewrite each woff2 by a few dozen bytes with identical content.
  font = TTFont(src, recalcTimestamp=False)
  if face.pin and 'fvar' in font:
    defaults = {a.axisTag: a.defaultValue for a in font['fvar'].axes}
    # A static instance rounds interpolated coordinates to integers; the browser interpolates in
    # floats, so only a pin at the default location reproduces today's pixels.
    assert all(defaults.get(k) == v for k, v in face.pin.items()), (
      f'{face.source}: pin {face.pin} is not the axis default {defaults}; keep it variable'
    )
    instancer.instantiateVariableFont(
      font, face.pin, inplace=True, optimize=True, updateFontNames=False
    )
  opts = subset.Options()
  opts.layout_features = list(KEEP_FEATURES) + list(face.features)
  opts.layout_scripts = ['*']
  # TrueType instructions do not change CoreText output (measured), so they go. CFF hints
  # (BlueValues, stems) do: Source Serif 4 moved by ~14k px per test page without them.
  opts.hinting = 'CFF ' in font  # fontTools: True keeps hints
  # Keep post-table glyph names (format 2.0). With format 3.0 CoreText rasterizes
  # every glyph of a TrueType face at a different sub-pixel offset (measured:
  # ~10k px per 1100x600 page of 16 px text). Costs ~1 KB per face after brotli.
  opts.glyph_names = True
  opts.desubroutinize = True
  opts.name_IDs = [0, 1, 2, 3, 4, 5, 6, 10, 13, 14, 16, 17]
  opts.name_languages = ['*']
  opts.name_legacy = False
  opts.notdef_outline = False
  opts.recalc_bounds = True
  opts.recalc_timestamp = False
  opts.prune_unicode_ranges = True
  opts.drop_tables += ['gasp', 'DSIG', 'meta']
  if 'fvar' not in font:
    opts.drop_tables += ['STAT', 'MVAR', 'HVAR', 'VVAR']
  opts.flavor = 'woff2'
  s = subset.Subsetter(options=opts)
  s.populate(unicodes=sorted(keep))
  s.subset(font)
  if face.rename:
    rename_font(font, face.rename, face)
  font.flavor = 'woff2'
  return font


def unicode_range(cps: set[int]) -> str:
  cps_sorted = sorted(cps)
  parts: list[str] = []
  i = 0
  while i < len(cps_sorted):
    j = i
    while j + 1 < len(cps_sorted) and cps_sorted[j + 1] == cps_sorted[j] + 1:
      j += 1
    a, b = cps_sorted[i], cps_sorted[j]
    parts.append(f'U+{a:X}' if a == b else f'U+{a:X}-{b:X}')
    i = j + 1
  return ', '.join(parts)


def verify(path: Path, expected: set[int], face: Face) -> dict:
  f = TTFont(path)
  cmap = set(f.getBestCmap())
  assert cmap == expected, (
    f'{path.name}: cmap {len(cmap)} != expected {len(expected)}: extra={sorted(cmap - expected)[:10]} missing={sorted(expected - cmap)[:10]}'
  )
  assert f.flavor == 'woff2', path.name
  assert ('fvar' in f) == (
    not face.pin and 'fvar' in TTFont(fetch_source(face, None))
  ), f'{path.name}: variable state'
  for t in ('fpgm', 'prep', 'cvt ', 'hdmx', 'VDMX', 'LTSH', 'gasp'):
    assert t not in f, f'{path.name} kept hinting table {t}'
  if 'glyf' in f:
    g = f['glyf']
    for gn in f.getGlyphOrder():
      gl = g[gn]
      if hasattr(gl, 'program') and gl.program and gl.program.getBytecode():
        raise AssertionError(f'{path.name}: glyph {gn} kept instructions')
  feats = set()
  for tbl in ('GSUB', 'GPOS'):
    if tbl in f and f[tbl].table.FeatureList:
      feats.update(
        fr.FeatureTag for fr in f[tbl].table.FeatureList.FeatureRecord
      )
  allowed = set(KEEP_FEATURES) | set(face.features)
  assert feats <= allowed, (
    f'{path.name}: unexpected features {feats - allowed}'
  )
  fam = f['name'].getDebugName(1)
  if face.rename:
    assert fam == face.rename, f'{path.name}: internal family {fam!r}'
  return {
    'bytes': path.stat().st_size,
    'glyphs': f['maxp'].numGlyphs,
    'chars': len(cmap),
    'features': sorted(feats),
    'internalFamily': fam,
    'tables': sorted(f.keys()),
  }


# ---------------------------------------------------------------------------------------------
# Generated files
# ---------------------------------------------------------------------------------------------
SCSS_HEADER = """// GENERATED by harness/fonts/subset.py (pnpm fonts:subset). Do not edit by hand.
// Each face is split into a preloaded core file (ASCII, frequent punctuation, and every glyph
// that kerns with or attaches to those) and an extended file the browser fetches only when a
// page uses one of its characters. unicode-range lists exactly the characters each file holds.
// Every file is served from the FONTS KV at /fonts/ (?v= is the file's content hash, since the
// worker caches fonts for a year); originals live in quartz/static/ or the same KV.
@use './mixin.scss' as *;

"""


def scss_rule(face: Face, filename: str, path: str, rng: str) -> str:
  return f"@include font-face(\n  '{face.family}',\n  '{filename}',\n  'woff2',\n  {face.style},\n  {face.weight},\n  '{path}',\n  '{rng}'\n);\n"
