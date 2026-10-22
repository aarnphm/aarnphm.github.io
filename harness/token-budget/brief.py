# ruff: file-ignore[ambiguous-unicode-character-string] -- the multiplication sign in `refs` is brief text the workers read.
"""Stage 1: the brief one worker reads to propose a module boundary for one target."""

import json

from common import LIMIT, TARGET, module_dir

CHURN_DAYS = 180

PRINCIPLES = """\
Use these terms exactly (mattpocock/skills codebase-design): **module**, **interface**, **depth**, **seam**, **adapter**, **leverage**, **locality**. Do not write "component", "service", "API" or "boundary".

- **Interface**: everything a caller must know: types, invariants, ordering constraints, error modes, required configuration, performance characteristics.
- **Depth** is leverage at the interface: behaviour a caller or test can exercise per unit of interface it must learn. Never measure it as a ratio of implementation lines to interface lines.
- Depth is a property of the interface. A deep module can be made of many small private files. Splitting the file for the token budget is internal structure; it must not widen the interface.
- **Deletion test**: if deleting a module makes complexity vanish, it was a pass-through. If complexity reappears across callers, it earns its interface. Apply it to every export you keep.
- **The interface is the test surface.** Tests cross the same seam as callers. Tests that import internals test past the interface.
- **One adapter means a hypothetical seam; two adapters means a real one.** Do not add a port unless you can name two adapters (usually production and test).
- Dependency categories (DEEPENING.md): in-process, local-substitutable, remote but owned, true external. The category decides how the module is tested across its seam."""


def table(rows: list[list], head: list[str]) -> str:
  lines = [f'| {" | ".join(head)} |', f'| {" | ".join("---" for _ in head)} |']
  lines += [f'| {" | ".join(str(c) for c in r)} |' for r in rows]
  return '\n'.join(lines)


def code(names: list[str]) -> str:
  return ', '.join(f'`{n}`' for n in names)


def schema(g: dict) -> str:
  example = {
    'target': g['path'],
    'blob': g['blob'],
    'worker': '<your worker id>',
    'constraint': '<your constraint id>',
    'interface': {
      'entries': [g['path']],
      'exports': ['<names the entry keeps exporting>'],
      'invariants': '<what callers must keep true>',
      'ordering': '<ordering constraints, or none>',
      'error_modes': '<how it fails>',
    },
    'files': [
      {
        'path': '<file>',
        'role': 'entry | internal',
        'blocks': ['b0'],
        'concept': '<the one concept this file holds>',
      }
    ],
    'hidden_exports': [
      {
        'symbol': '<export no longer public>',
        'deletion_test': '<why callers do not need it>',
      }
    ],
    'importer_changes': [
      {
        'importer': '<file>',
        'symbol': '<name>',
        'to': '<entry file>',
        'replacement': '<exported name the importer uses instead, when symbol is hidden>',
      }
    ],
    'seams': [
      {
        'at': '<where>',
        'category': 'in-process | local-substitutable | remote-owned | true-external',
        'adapters': ['<adapter 1>', '<adapter 2>'],
      }
    ],
    'tests': [{'path': '<test file>', 'through': '<entry file>'}],
    'risks': ['<what could break>'],
  }
  return json.dumps(example, indent=2)


def brief(g: dict) -> str:
  style = g['kind'] == 'style'
  iface = g['interface']
  exports = table(
    [
      [
        f'`{e["name"]}`',
        'yes' if e['production'] else '',
        'yes' if e['tests'] else '',
      ]
      for e in iface['exports']
    ],
    ['export', 'production importers', 'test importers'],
  )
  users = table(
    [
      [
        f'`{u["path"]}`',
        'test' if u['test'] else 'production',
        code(u['names']),
        ' '.join(
          n
          for n, on in (
            ('type-only', u['typeOnly']),
            ('dynamic', u['dynamic']),
            (
              f'**over budget ({u.get("overBudget")} tokens): do not edit**',
              u.get('overBudget'),
            ),
          )
          if on
        ),
      ]
      for u in g['importers']
    ],
    ['importer', 'kind', 'names', 'notes'],
  )

  def refs(b: dict) -> str:
    return ' '.join(f'{i}×{n}' if n > 1 else i for i, n in b['refs'].items())

  def imports(b: dict) -> str:
    if b['kind'] == 'import':
      return f'from `{b["from"]}`'
    return '; '.join(
      f'{src}: {", ".join(ns)}' for src, ns in b['imports'].items()
    )

  blocks = table(
    [
      [
        b['id'],
        f'{b["start"]}-{b["end"]}',
        b['tokens'],
        b['kind'] + (' (exported)' if b.get('exported') else ''),
        code(b['names']) or '`{}`'.format(b['label'].replace('|', '\\|')),
        refs(b),
        imports(b),
      ]
      for b in g['blocks']
    ],
    ['id', 'lines', 'tokens', 'kind', 'names', 'refs', 'imports used'],
  )
  heavy = (
    '\nThe pre-commit hook checks every staged file, so an importer marked over budget cannot'
    ' change in this pass. Keep every name such an importer uses exported from the entry, and do'
    ' not list it in `importer_changes` or `tests`.\n'
    if any(u.get('overBudget') for u in g['importers'])
    else ''
  )
  order = (
    ' Each file holds one contiguous run of blocks. The entry `@use`s its partials before its own'
    " rules, so the partials' blocks come first in source order and the entry keeps only the"
    ' blocks after them.'
    if style
    else ''
  )
  return f"""# Boundary proposal: {g['path']}

Target: `{g['path']}`, blob `{g['blob']}`, {g['tokens']} tokens (limit {LIMIT}, aim for at most {TARGET} per file), {g['churn']} commits in the last {CHURN_DAYS} days.

## Vocabulary and principles

{PRINCIPLES}

## Module shape

- `{g['path']}` stays the **entry**: it exports the interface, and importers keep this path.
- Implementation moves to the private folder `{module_dir(g['path'])}/` (kebab-case file names). Code outside the module imports only the entry. Files in the private folder import each other freely. Tests import the entry. No import cycles.
- More than one entry file is allowed when callers use distinct parts. No barrel files.
{'- This is a stylesheet: keep the block order (the cascade depends on it). Partials are `@use`d from the entry in the original order.' + chr(10) if style else ''}
## Interface as callers use it today

{exports}

Exported with no importer (candidates to hide): {code(iface['unused']) or 'none'}.
Used only by tests (tests past the interface?): {code(iface['testOnly']) or 'none'}.

{users}
{heavy}
## Blocks

`refs` lists the blocks this block reads (×n = occurrences). A seam belongs where these references are sparse. Read source ranges with `sed -n '<start>,<end>p' {g['path']}` only when the label is not enough; find a name's uses with `rg -nw <name>`{'' if style else f" or `ast-grep run -p '<pattern>' -l ts {g['path']}`"}.

{blocks}

## Output

Write one JSON file, the path given in your assignment, with this shape. Every block ID except `import` blocks appears in exactly one file; each new file gets the imports it uses, so import blocks need not be listed. `blob` must be `{g['blob']}`.{order} List a seam only where a port has two or more adapters; an in-process call with no port is not a seam.

```json
{schema(g)}
```

Stopping rule: read what you need, write the JSON file, then stop. Do not edit any other file. Do not redesign code outside this target.
"""
