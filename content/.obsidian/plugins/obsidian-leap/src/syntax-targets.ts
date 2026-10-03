import type { Text } from '@codemirror/state'
import type { EditorView } from '@codemirror/view'
import { DocInput, ensureSyntaxTree, syntaxTree } from '@codemirror/language'
import { GFM, parser } from '@lezer/markdown'
import type { SyntaxTarget } from './types'

const markdownParser = parser.configure(GFM)
type MarkdownTree = ReturnType<typeof markdownParser.parse>

interface SelectionNode {
  from: number
  to: number
  name: string
  parent: SelectionNode | null
  type: { isTop: boolean }
}

interface SelectionTree {
  length: number
  resolveInner(position: number, side: -1 | 0 | 1): SelectionNode
}

interface CachedParse {
  parse: ReturnType<typeof markdownParser.startParse>
  tree: MarkdownTree | null
}

const parses = new WeakMap<Text, CachedParse>()

function markdownTree(doc: Text, deadline: number): MarkdownTree | null {
  let cached = parses.get(doc)
  if (!cached) {
    cached = { parse: markdownParser.startParse(new DocInput(doc)), tree: null }
    parses.set(doc, cached)
  }

  while (!cached.tree && performance.now() < deadline) cached.tree = cached.parse.advance()
  return cached.tree
}

function ancestors(
  tree: SelectionTree,
  cursor: number,
  length: number,
  native: boolean,
): SyntaxTarget[] {
  const targets: SyntaxTarget[] = []
  let embeddedLanguage = false

  for (
    let node: SelectionNode | null = tree.resolveInner(cursor, cursor === length ? -1 : 1);
    node;
    node = node.parent
  ) {
    if (node.from >= node.to || node.from < 0 || node.to > length) continue
    if (native) {
      if (
        !node.parent ||
        node.name.startsWith('HyperMD-') ||
        node.name.includes('hmd-') ||
        node.name.includes('formatting')
      ) {
        continue
      }
      embeddedLanguage ||= node.type.isTop
    } else {
      if (!node.parent && tree.length !== length) continue
      if (node.name.endsWith('Mark')) continue
    }
    targets.push({ from: node.from, to: node.to, name: node.name })
  }

  return native && (!embeddedLanguage || targets.length < 2) ? [] : targets
}

export function syntaxTargets(
  view: EditorView,
  position: number,
  linewise: boolean,
): SyntaxTarget[] {
  const { state } = view
  const { doc } = state
  const cursor = Math.max(0, Math.min(position, doc.length))
  const deadline = performance.now() + 50
  const semanticTree = markdownTree(doc, deadline)
  const nativeTree =
    ensureSyntaxTree(state, cursor, Math.max(0, deadline - performance.now())) ?? syntaxTree(state)
  const semantic = semanticTree ? ancestors(semanticTree, cursor, doc.length, false) : []
  const native = ancestors(nativeTree, cursor, doc.length, true)
  const leaf = semantic[0]
  const nested = leaf
    ? native.filter(target => target.from >= leaf.from && target.to <= leaf.to)
    : native
  const candidates = [...nested, ...semantic]
  const targets: SyntaxTarget[] = []
  const ranges = new Map<string, number>()

  for (const candidate of candidates) {
    let { from, to } = candidate
    if (linewise) {
      const firstLine = doc.lineAt(from)
      // Node ends are exclusive, including when they coincide with a new line's start.
      const lastLine = doc.lineAt(to - 1)
      if (firstLine.number === lastLine.number) continue
      from = firstLine.from
      to = lastLine.to
    }

    const inner = targets[targets.length - 1]
    if (inner && (from > inner.from || to < inner.to)) continue

    const target = { from, to, name: candidate.name }
    const key = `${from}:${to}`
    const existing = ranges.get(key)
    if (existing !== undefined) {
      if (linewise) targets[existing] = target
      continue
    }
    ranges.set(key, targets.length)
    targets.push(target)
  }

  return targets
}
