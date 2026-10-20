"""Blocks of a target file and the references between them.

TypeScript and JavaScript parse with tree-sitter (ast-grep-py): one block per top-level statement,
and the inner statements of any block over TARGET (ids like b40.3). Stylesheets reuse the hook's
line blocks. A block owns the comments above it, from the end of the previous statement.
"""

import bisect
import re
from collections import Counter

from ast_grep_py import SgNode, SgRoot

from common import TARGET, text_tokens

DECLARATIONS = {
  'function_declaration',
  'generator_function_declaration',
  'class_declaration',
  'abstract_class_declaration',
  'interface_declaration',
  'type_alias_declaration',
  'enum_declaration',
  'internal_module',
  'module',
}
FUNCTIONS = {
  'arrow_function',
  'function_expression',
  'function',
  'function_declaration',
  'generator_function_declaration',
  'method_definition',
}
# Identifier kinds that name a binding or a reference; property names have their own kinds.
IDENTS = (
  'identifier',
  'type_identifier',
  'shorthand_property_identifier',
  'shorthand_property_identifier_pattern',
)


def language(path: str) -> str:
  return 'tsx' if re.search(r'\.[jt]sx$', path) else 'typescript'


def string_value(node: SgNode | None) -> str | None:
  return node.text()[1:-1] if node else None


def statements(node: SgNode) -> list[SgNode]:
  return [c for c in node.named_children() if c.kind() != 'comment']


def binding_names(node: SgNode | None) -> list[str]:
  """Names a binding declares: `a`, `{ a, b: c }`, `[d, ...e]`."""
  if node is None:
    return []
  if node.kind() == 'identifier':
    return [node.text()]
  out: list[str] = []
  for c in node.named_children():
    k = c.kind()
    if k in ('identifier', 'shorthand_property_identifier_pattern'):
      out.append(c.text())
    elif k == 'pair_pattern':
      out += binding_names(c.field('value'))
    elif k in (
      'rest_pattern',
      'object_assignment_pattern',
      'assignment_pattern',
    ):
      out += binding_names(c.field('left') or c.named_children()[0])
    elif k in ('object_pattern', 'array_pattern'):
      out += binding_names(c)
  return out


def unwrap(st: SgNode) -> SgNode:
  """`export <declaration>` and `declare <declaration>` are the declaration with a modifier."""
  if st.kind() == 'export_statement' and st.field('declaration'):
    return unwrap(st.field('declaration'))
  if st.kind() == 'ambient_declaration':
    inner = [c for c in st.named_children() if c.kind() != 'statement_block']
    return unwrap(inner[0]) if inner else st
  return st


def import_names(st: SgNode) -> list[str]:
  clause = next(
    (c for c in st.named_children() if c.kind() == 'import_clause'), None
  )
  names: list[str] = []
  for c in clause.named_children() if clause else []:
    if c.kind() == 'identifier':
      names.append(c.text())
    elif c.kind() == 'namespace_import':
      names += [
        i.text() for i in c.named_children() if i.kind() == 'identifier'
      ]
    elif c.kind() == 'named_imports':
      for s in c.named_children():
        if s.kind() == 'import_specifier':
          names.append((s.field('alias') or s.field('name')).text())
  return names


def statement_info(st: SgNode) -> dict:
  kind = st.kind()
  if kind == 'import_statement':
    return {
      'kind': 'import',
      'names': import_names(st),
      'from': string_value(st.field('source')),
      'exported': False,
    }
  if kind == 'export_statement' and not st.field('declaration'):
    clause = next(
      (c for c in st.named_children() if c.kind() == 'export_clause'), None
    )
    specs = (
      [s for s in clause.named_children() if s.kind() == 'export_specifier']
      if clause
      else []
    )
    source = string_value(st.field('source'))
    if source:
      names = [(s.field('alias') or s.field('name')).text() for s in specs]
      return {
        'kind': 'reexport',
        'names': names or ['*'],
        'from': source,
        'exported': True,
      }
    if clause:
      return {
        'kind': 'export-list',
        'names': [],
        'locals': [s.field('name').text() for s in specs],
      }
    return {'kind': 'export-default', 'names': [], 'exported': True}
  exported = kind == 'export_statement'
  is_default = exported and any(c.kind() == 'default' for c in st.children())
  decl = unwrap(st)
  names: list[str] = []
  if decl.kind() in ('lexical_declaration', 'variable_declaration'):
    for d in decl.named_children():
      if d.kind() == 'variable_declarator':
        names += binding_names(d.field('name'))
  elif decl.kind() in DECLARATIONS and decl.field('name'):
    names = [decl.field('name').text()]
  elif decl.kind() == 'ambient_declaration' and decl.text().startswith(
    'declare global'
  ):
    names = ['global']
  return {
    'kind': 'declaration' if names else 'statement',
    'names': names,
    'exported': exported,
    'isDefault': is_default,
  }


def identifiers_used(node: SgNode) -> Counter:
  """Identifiers a node reads or binds, without property names and `A.B`'s right side."""
  found = []
  for kind in IDENTS:
    for i in node.find_all(kind=kind):
      p = i.parent()
      if p and p.kind() == 'nested_type_identifier' and p.field('name') == i:
        continue
      found.append((i.range().start.index, i.text()))
  # Source order, so derived lists (imports used) read in the order the code uses them.
  return Counter(t for _, t in sorted(found))


def body_of(st: SgNode) -> list[SgNode] | None:
  """Statements inside a block worth splitting: a function body (declaration,
  `const f = () => {}`, or a callback such as `document.addEventListener('nav', () => {})`),
  or class members."""
  st = unwrap(st)
  k = st.kind()
  if k in ('class_declaration', 'abstract_class_declaration'):
    return statements(st.field('body'))
  fn = None
  if k in FUNCTIONS:
    fn = st
  elif k == 'lexical_declaration':
    decls = [
      d for d in st.named_children() if d.kind() == 'variable_declarator'
    ]
    if len(decls) == 1:
      fn = decls[0].field('value')
  elif k == 'expression_statement':
    e = st.named_children()[0]
    if e.kind() == 'await_expression':
      e = e.named_children()[0]
    if e.kind() == 'call_expression' and e.field('arguments'):
      fn = next(
        (
          a
          for a in e.field('arguments').named_children()
          if a.kind() in FUNCTIONS
        ),
        None,
      )
  body = fn.field('body') if fn and fn.kind() in FUNCTIONS else None
  return (
    statements(body) if body and body.kind() == 'statement_block' else None
  )


def script_blocks(path: str, text: str) -> list[dict]:
  """Top-level statements, and the inner statements of any block over TARGET, in source order.
  A block with children keeps only its shell (signature, braces) in `tokens`."""
  root = SgRoot(text, language(path)).root()
  newlines = [i for i, ch in enumerate(text) if ch == '\n']

  def line_at(pos: int) -> int:
    return bisect.bisect_right(newlines, pos - 1) + 1

  out: list[dict] = []

  def walk(
    sts: list[SgNode],
    prefix: str,
    parent: str | None,
    open_end: int,
    close: bool,
  ):
    for i, st in enumerate(sts):
      if parent is None and i == 0:
        start = 0
      else:
        start = sts[i - 1].range().end.index if i else open_end
      # Comments above the next statement belong to it, so a block ends where its statement does;
      # the last top-level block runs to the end of the file.
      last = i + 1 == len(sts)
      end = len(text) if last and not close else st.range().end.index
      info = statement_info(st)
      decl = unwrap(st)
      name = decl.field('name')
      if not info['names'] and name and name.text() != 'constructor':
        info = {**info, 'kind': 'member', 'names': [name.text()]}
      full = text_tokens(text[start:end])
      used = identifiers_used(st)
      block = {
        'id': f'{prefix}{i}',
        'parent': parent,
        'start': line_at(
          0
          if start == 0
          else start + (1 if text[start : start + 1] == '\n' else 0)
        ),
        'end': line_at(max(start, end - 1)),
        'tokens': full,
        'fullTokens': full,
        'label': st.text().split('\n')[0].strip()[:64],
        **info,
        'used': used,
        'fullUsed': Counter(used),
        'children': [],
      }
      out.append(block)
      inner = body_of(st) if full > TARGET else None
      if not inner or len(inner) < 2:
        continue
      before = len(out)
      brace = inner[0].prev()
      walk(
        inner, f'{block["id"]}.', block['id'], brace.range().end.index, True
      )
      kids = [c for c in out[before:] if c['parent'] == block['id']]
      block['children'] = [c['id'] for c in kids]
      block['tokens'] = max(1, full - sum(c['fullTokens'] for c in kids))
      for c in kids:
        used.subtract(c['fullUsed'])
      for n in [n for n, count in used.items() if count <= 0]:
        del used[n]

  walk(statements(root), 'b', None, 0, False)
  # `export { a, b }` marks the local declarations as exported.
  owner = {
    n: b
    for b in out
    if not b.get('parent') and b['kind'] != 'import'
    for n in b['names']
  }
  for b in out:
    if b['kind'] == 'export-list':
      for n in b['locals']:
        if n in owner:
          owner[n]['exported'] = True
  return out


SCSS_DECL = re.compile(
  r'^(\$[\w-]+)\s*:|^@(?:mixin|function) ([\w-]+)|^(%[\w-]+)', re.M
)


def style_blocks(text: str, line_blocks: list[dict]) -> list[dict]:
  """The hook's line blocks, with the `$variables`, mixins and placeholders each declares and uses."""
  lines = text.split('\n')
  out = []
  for b in line_blocks:
    body = '\n'.join(lines[b['start'] - 1 : b['end']])
    names = [
      m.group(1) or m.group(2) or m.group(3) for m in SCSS_DECL.finditer(body)
    ]
    used: Counter = Counter()
    used.update(f'${m}' for m in re.findall(r'\$([\w-]+)', body))
    used.update(re.findall(r'@include ([\w-]+)', body))
    used.update(f'%{m}' for m in re.findall(r'@extend %([\w-]+)', body))
    used.update(re.findall(r'\b([\w-]+)\(', body))
    use = re.search(
      r"""^@(?:use|forward|import) ['"]([^'"]+)['"]""", body, re.M
    )
    # A comment-only block emits no CSS, so it may sit anywhere in the cascade.
    code = re.sub(
      r'^\s*//.*$', '', re.sub(r'/\*.*?\*/', '', body, flags=re.S), flags=re.M
    )
    if not code.strip():
      used = Counter()
    kind = 'import' if use else 'declaration' if names else 'rule'
    out.append({
      **b,
      'kind': kind if code.strip() else 'comment',
      'names': names,
      'from': use.group(1) if use else None,
      'exported': False,
      'used': used,
    })
  return out


def link(bs: list[dict]) -> list[dict]:
  """References between blocks, weighted by occurrences, and the imported names each block uses.
  Names resolve lexically: a child sees its siblings, then its ancestors' siblings, then the file."""
  import_of = {
    n: b['from']
    for b in bs
    if b['kind'] == 'import' and not b.get('parent')
    for n in b['names']
  }
  by_id = {b['id']: b for b in bs}
  scopes: dict[str, dict[str, str]] = {}

  def scope_for(parent: str | None) -> dict[str, str]:
    key = parent or ''
    if key not in scopes:
      scope = dict(scope_for(by_id[parent].get('parent'))) if parent else {}
      for b in bs:
        if b.get('parent') == parent and b['kind'] != 'import':
          for n in b['names']:
            scope[n] = b['id']
      scopes[key] = scope
    return scopes[key]

  out = []
  for b in bs:
    scope = scope_for(b.get('parent'))
    refs: dict[str, int] = {}
    imports: dict[str, list[str]] = {}
    for n, count in (b.get('used') or {}).items():
      target = scope.get(n)
      if target and target != b['id'] and n not in b['names']:
        refs[target] = refs.get(target, 0) + count
      source = import_of.get(n)
      if source and b['kind'] != 'import':
        names = imports.setdefault(source, [])
        if n not in names:
          names.append(n)
    drop = ('used', 'fullUsed', 'locals')
    rest = {
      k: v
      for k, v in b.items()
      if k not in drop and not (k == 'parent' and v is None)
    }
    out.append({**rest, 'refs': refs, 'imports': imports})
  return out
