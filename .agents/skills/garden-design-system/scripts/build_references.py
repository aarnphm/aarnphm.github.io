#!/usr/bin/env python3
"""Rebuild references/ from a downloaded copy of the Garden Design System artifact.

Usage: python3 build_references.py <download-dir> [skill-dir]

<download-dir> holds project/README.md, project/*.md, project/tokens.json and
project/components/<Name>/README.md, as an Artifact `read` with `paths` saves them.
The skill dir defaults to this script's parent. Generated files are overwritten whole;
SKILL.md is hand-written and never touched.
"""

import json
import re
import shutil
import sys
from pathlib import Path

SECTIONS = {
  'README.md': 'brand-book.md',
  'layout.md': 'layout.md',
  'tool-surfaces.md': 'tool-surfaces.md',
  'arena.md': 'arena.md',
  'triathlon.md': 'triathlon.md',
  'courses.md': 'courses.md',
  'home.md': 'home.md',
  'math.md': 'math.md',
  'review.md': 'review.md',
  'focus.md': 'focus.md',
}


def cell(value) -> str:
  return str(value).replace('|', '\\|').replace('\n', ' ')


def color_value(value, themes) -> list[str]:
  if isinstance(value, str):
    return [value] + [''] * (len(themes) - 1)
  first = value.get(themes[0]['id'], '')
  return [value.get(theme['id'], first) for theme in themes]


def tokens_md(tokens: dict) -> str:
  out = [
    '# Tokens',
    '',
    'Generated from the design system `tokens.json`. Names are the system names. In the site, the',
    'theme colours are CSS custom properties (`--light`, `--dark`, `--secondary`) set from',
    '`theme.colors` in `quartz.config.ts`; spacing, radii, z-index and transitions live in',
    '`quartz/styles/tokens/`; surface prefixes (`--arena-*`, `--course-*`, `--tri-*`) live in each',
    "surface's SCSS. `surface`, `selected`, `line` and `fig-frame` are resolved hex here; the code",
    'writes them with `color-mix()`. `{name}` is an alias. Usage notes give contrast on paper.',
    '',
  ]
  color = tokens['color']
  themes = color['themes']
  out += [
    '## Color',
    '',
    '| name | ' + ' | '.join(t['name'] for t in themes) + ' | usage |',
  ]
  out += ['| --- ' * (len(themes) + 2) + '|']
  for token in color['tokens']:
    values = color_value(token['value'], themes)
    out.append(
      f'| `{token["name"]}` | '
      + ' | '.join(f'`{cell(v)}`' if v else '' for v in values)
      + f' | {cell(token.get("usage", ""))} |'
    )
  type_ = tokens.get('type', {})
  if type_.get('families'):
    out += ['', '## Type families', '', '| family | stack |', '| --- | --- |']
    for name, stack in type_['families'].items():
      out.append(f'| `{name}` | `{cell(stack)}` |')
  for group in type_.get('groups', []):
    out += [
      '',
      f'## Type: {group["name"]}',
      '',
      '| style | family | size | line height | weight | usage |',
    ]
    out.append('| --- | --- | --- | --- | --- | --- |')
    for style in group.get('styles', []):
      out.append(
        f'| `{style["name"]}` | {cell(style.get("family", group.get("family", "")))} | '
        f'{cell(style.get("fontSize", ""))} | {cell(style.get("lineHeight", ""))} | '
        f'{cell(style.get("fontWeight", ""))} | {cell(style.get("usage", ""))} |'
      )
  for family, data in tokens.items():
    if family in (
      'name',
      'version',
      'meta',
      'color',
      'type',
    ) or not isinstance(data, dict):
      continue
    out += [
      '',
      f'## {family}',
      '',
      '| name | value | usage |',
      '| --- | --- | --- |',
    ]
    for token in data.get('tokens', []):
      out.append(
        f'| `{token["name"]}` | `{cell(token["value"])}` | {cell(token.get("usage", ""))} |'
      )
  return '\n'.join(out) + '\n'


def summary(text: str) -> str:
  body = re.sub(r'^#.*\n+', '', text.strip())
  first = re.split(
    r'(?<=[.!?])\s', body.split('\n\n')[0].replace('\n', ' '), maxsplit=1
  )[0]
  return cell(first)


def main() -> None:
  if len(sys.argv) < 2:
    sys.exit(__doc__)
  project = Path(sys.argv[1]) / 'project'
  skill = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else Path(__file__).resolve().parent.parent
  )
  refs = skill / 'references'
  if refs.exists():
    shutil.rmtree(refs)
  (refs / 'components').mkdir(parents=True)

  for source, target in SECTIONS.items():
    path = project / source
    if path.exists():
      (refs / target).write_text(path.read_text())
    else:
      print(f'missing section: {source}', file=sys.stderr)
  extra = sorted(
    p.name for p in project.glob('*.md') if p.name not in SECTIONS
  )
  for name in extra:
    (refs / name).write_text((project / name).read_text())

  (refs / 'tokens.md').write_text(
    tokens_md(json.loads((project / 'tokens.json').read_text()))
  )

  rows = []
  for readme in sorted((project / 'components').glob('*/README.md')):
    name = readme.parent.name
    text = readme.read_text()
    (refs / 'components' / f'{name}.md').write_text(
      text if text.startswith('#') else f'# {name}\n\n{text}'
    )
    rows.append(f'| [{name}]({name}.md) | {summary(text)} |')
  index = [
    '# Components',
    '',
    '| component | summary |',
    '| --- | --- |',
    *rows,
  ]
  (refs / 'components' / 'index.md').write_text('\n'.join(index) + '\n')

  imagery = project / 'assets' / 'Imagery' / 'README.md'
  if imagery.exists():
    (refs / 'imagery.md').write_text(imagery.read_text())
  print(
    f'{len(SECTIONS) + len(extra)} sections, {len(rows)} components -> {refs}'
  )


if __name__ == '__main__':
  main()
