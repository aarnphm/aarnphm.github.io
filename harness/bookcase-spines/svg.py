"""Constrained pure SVG validation and reproducible native raster exports."""

from __future__ import annotations

from dataclasses import asdict
import io
import math
from pathlib import Path
import re
import struct
import subprocess
import xml.etree.ElementTree as ET

from files import atomic, dumps, pin, sha
from process import execute
from schema import Candidate, Config, Image, Limits

ELEMENTS = set(
  'svg g defs path rect circle ellipse line polyline polygon linearGradient radialGradient stop clipPath'.split()
)
ATTRIBUTES = set(
  'viewBox width height x y x1 y1 x2 y2 cx cy r rx ry d points fill stroke stroke-width stroke-linecap stroke-linejoin stroke-miterlimit fill-rule clip-rule opacity fill-opacity stroke-opacity id transform clip-path gradientUnits gradientTransform offset stop-color stop-opacity clipPathUnits fx fy fr'.split()
)
NUMBERS = re.compile(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?')
NAMESPACE = '{http://www.w3.org/2000/svg}'


def inspect_svg(svg: str, limits: Limits) -> tuple[float, float]:
  if len(svg.encode()) > limits.maxSvgBytes:
    raise ValueError('SVG byte limit exceeded')
  if '<!' in svg or '<?' in svg:
    raise ValueError(
      'SVG declarations, entities and processing instructions are forbidden'
    )
  depth = count = segments = 0
  root = None
  try:
    for event, element in ET.iterparse(
      io.StringIO(svg), events=('start', 'end')
    ):
      if root is None:
        root = element
      depth += 1 if event == 'start' else -1
      count += event == 'start'
      if depth > limits.maxDepth or count > 20000:
        raise ValueError('SVG nesting or element limit exceeded')
  except ET.ParseError as error:
    raise ValueError(f'Malformed SVG: {error}') from error
  if root is None or root.tag != NAMESPACE + 'svg':
    raise ValueError('Expected an SVG namespace root')
  ids: dict[str, str] = {}
  edges: dict[str, set[str]] = {}
  references: list[tuple[str, str]] = []

  def numeric(value: str) -> int:
    values = NUMBERS.findall(value)
    if not values or any(
      not math.isfinite(float(item)) or abs(float(item)) > limits.maxCoordinate
      for item in values
    ):
      raise ValueError('Invalid or excessive SVG coordinate')
    return len(values)

  def visit(node: ET.Element, parents: tuple[str, ...]) -> None:
    nonlocal segments
    tag = node.tag.removeprefix(NAMESPACE)
    if tag not in ELEMENTS or (tag == 'svg' and node is not root):
      raise ValueError(f'Unsupported SVG element: {tag}')
    if (node.text or '').strip() or (node.tail or '').strip():
      raise ValueError('SVG text content is forbidden')
    identity = node.get('id')
    ancestors = parents
    if identity:
      if not re.fullmatch(r'[a-zA-Z_][\w.-]*', identity) or identity in ids:
        raise ValueError('Invalid or duplicate SVG ID')
      ids[identity] = tag
      ancestors += (identity,)
    for name, value in node.attrib.items():
      if name not in ATTRIBUTES:
        raise ValueError(f'Unsupported SVG attribute: {name}')
      if re.search(r'[;&]|data:|https?:|javascript:', value, re.I):
        raise ValueError('External or encoded SVG resource')
      if name == 'id':
        continue
      if 'url' in value.lower() or name == 'clip-path':
        match = re.fullmatch(r'url\(#([a-zA-Z_][\w.-]*)\)', value)
        if not match or name not in ['fill', 'stroke', 'clip-path']:
          raise ValueError(
            'Only internal paint and clip references are allowed'
          )
        target = match[1]
        references.append((
          target,
          'clipPath' if name == 'clip-path' else 'gradient',
        ))
        for ancestor in ancestors:
          edges.setdefault(ancestor, set()).add(target)
      elif name in ['fill', 'stroke', 'stop-color']:
        if not re.fullmatch(
          r'(?:none|[a-z]+|#[\da-f]{3,8}|rgba?\([\d.,%\s]+\))', value, re.I
        ) or value.lower() in ['currentcolor', 'inherit']:
          raise ValueError('Invalid SVG paint')
      elif name in [
        'fill-rule',
        'clip-rule',
        'stroke-linecap',
        'stroke-linejoin',
        'gradientUnits',
        'clipPathUnits',
      ]:
        if value not in [
          'nonzero',
          'evenodd',
          'butt',
          'round',
          'square',
          'miter',
          'bevel',
          'userSpaceOnUse',
          'objectBoundingBox',
        ]:
          raise ValueError('Invalid SVG enumeration')
      elif name == 'd':
        if not re.fullmatch(r'[MmZzLlHhVvCcSsQqTtAa\d.eE+,\s-]+', value):
          raise ValueError('Invalid SVG path')
        segments += numeric(value)
      elif name in ['transform', 'gradientTransform']:
        if not re.fullmatch(
          r'(?:(?:matrix|translate|scale|rotate|skewX|skewY)\([\d.eE+,\s-]+\)\s*)+',
          value,
        ):
          raise ValueError('Invalid SVG transform')
        numeric(value)
      else:
        if not re.fullmatch(r'[\d.eE+,\s%-]+', value):
          raise ValueError('Invalid numeric SVG attribute')
        size = numeric(value)
        if name == 'points':
          segments += size
      if segments > limits.maxSegments:
        raise ValueError('SVG segment limit exceeded')
    for child in node:
      visit(child, ancestors)

  visit(root, ())
  for identity, kind in references:
    if ids.get(identity) not in (
      ['linearGradient', 'radialGradient']
      if kind == 'gradient'
      else ['clipPath']
    ):
      raise ValueError('Missing or mismatched SVG resource')
  active: set[str] = set()
  done: set[str] = set()

  def cycle(identity: str) -> None:
    if identity in active:
      raise ValueError('Cyclic SVG resource')
    if identity in done:
      return
    active.add(identity)
    for target in edges.get(identity, set()):
      cycle(target)
    active.remove(identity)
    done.add(identity)

  for identity in ids:
    cycle(identity)
  box = [
    float(item) for item in re.split(r'[ ,]+', root.get('viewBox', '').strip())
  ]
  if (
    len(box) != 4
    or any(not math.isfinite(value) for value in box)
    or box[2] <= 0
    or box[3] <= 0
  ):
    raise ValueError('SVG needs a positive finite viewBox')
  return box[2], box[3]


def dimensions(file: Path) -> tuple[int, int]:
  """Read dimensions only; native codecs validate complete pixels during rendering."""
  payload = file.read_bytes()
  if (
    payload.startswith(b'\x89PNG\r\n\x1a\n')
    and payload[12:16] == b'IHDR'
    and len(payload) >= 24
  ):
    width, height = struct.unpack('>II', payload[16:24])
  elif payload[:4] == b'RIFF' and payload[8:12] == b'WEBP':
    offset = 12
    width = height = 0
    while offset + 8 <= len(payload):
      kind = payload[offset : offset + 4]
      size = int.from_bytes(payload[offset + 4 : offset + 8], 'little')
      chunk = payload[offset + 8 : offset + 8 + size]
      if kind == b'VP8X' and len(chunk) >= 10:
        width, height = (
          1 + int.from_bytes(chunk[4:7], 'little'),
          1 + int.from_bytes(chunk[7:10], 'little'),
        )
        break
      if kind == b'VP8L' and len(chunk) >= 5 and chunk[0] == 47:
        bits = int.from_bytes(chunk[1:5], 'little')
        width, height = 1 + (bits & 16383), 1 + ((bits >> 14) & 16383)
        break
      if (
        kind == b'VP8 ' and len(chunk) >= 10 and chunk[3:6] == b'\x9d\x01\x2a'
      ):
        width, height = (
          value & 16383 for value in struct.unpack('<HH', chunk[6:10])
        )
        break
      offset += 8 + size + size % 2
  else:
    raise ValueError('Reference must be PNG or WebP')
  if width <= 0 or height <= 0 or width * height > 2097152:
    raise ValueError('Invalid or excessive raster dimensions')
  return width, height


def native(argv: list[str], directory: Path, config: Config) -> None:
  result = execute(
    argv, directory, timeout=config.limits.maxRasterMs / 1000, max_bytes=65536
  )
  if result.exitCode != 0 or result.error:
    raise ValueError(
      result.error or Path(result.stderrFile).read_text()[-2000:]
    )


def rasterize(
  svg: str,
  reference: Path,
  directory: Path,
  config: Config,
  provenance: list[str],
) -> Candidate:
  shape = inspect_svg(svg, config.limits)
  width, height = dimensions(reference)
  if abs(shape[0] / shape[1] - width / height) > 0.000001:
    raise ValueError('SVG viewBox must preserve the reference aspect ratio')
  directory.mkdir(parents=True)
  master = directory / 'master.svg'
  atomic(master, svg)
  if reference.read_bytes().startswith(b'RIFF'):
    native(
      ['dwebp', str(reference), '-o', str(directory / 'reference.png')],
      directory / 'decode-reference',
      config,
    )
  else:
    native(
      [
        'cwebp',
        '-quiet',
        '-lossless',
        str(reference),
        '-o',
        str(directory / 'reference.webp'),
      ],
      directory / 'decode-reference',
      config,
    )
  images: list[Image] = []
  for name, tier in [
    ('shelf', config.exports.shelf),
    ('detail', config.exports.detail),
  ]:
    scale = min(1, tier.width / width, tier.height / height)
    size = (
      max(1, math.floor(width * scale + 0.5)),
      max(1, math.floor(height * scale + 0.5)),
    )
    png, webp = directory / f'{name}.png', directory / f'{name}.webp'
    native(
      [
        'rsvg-convert',
        '--width',
        str(size[0]),
        '--height',
        str(size[1]),
        '--output',
        str(png),
        str(master),
      ],
      directory / f'render-{name}',
      config,
    )
    native(
      ['cwebp', '-quiet', '-q', str(tier.quality), str(png), '-o', str(webp)],
      directory / f'encode-{name}',
      config,
    )
    decoded = directory / f'{name}-decoded.png'
    native(
      ['dwebp', str(webp), '-o', str(decoded)],
      directory / f'decode-{name}',
      config,
    )
    if dimensions(decoded) != size or dimensions(webp) != size:
      raise ValueError('Raster dimensions differ from their recipe')
    final = directory / f'{name}-{sha(webp.read_bytes())[:16]}.webp'
    webp.rename(final)
    pinned = pin(final)
    images.append(
      Image(pinned.file, pinned.sha256, *size, final.stat().st_size)
    )
  versions = {
    tool: subprocess.check_output(
      [tool, '--version' if tool == 'rsvg-convert' else '-version'],
      text=True,
      timeout=5,
    ).strip()
    for tool in ['rsvg-convert', 'cwebp', 'dwebp']
  }
  recipe = sha(
    dumps({'exports': asdict(config.exports), 'versions': versions})
  )
  return Candidate(pin(master), images[0], images[1], provenance, recipe)
