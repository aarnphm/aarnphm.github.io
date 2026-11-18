import type { Completion, CompletionResult } from '@codemirror/autocomplete'
import type { EditorState } from '@codemirror/state'
import { syntaxTree } from '@codemirror/language'
import type { ContentDetails } from '../../../plugins/emitters/contentIndex'
import { isFullSlug, getFileExtension } from '../../../util/path'
import {
  createReferenceCatalog,
  type ReferenceCatalog,
  type ReferenceCatalogInput,
} from '../../../util/references/catalog'
import { isEscapedAt } from '../../../util/references/scan'
import { referenceAtCursor } from '../../../util/references/syntax'
import { fuzzyMatch } from './fuzzy'

export interface ReferenceCompletionContext {
  field: 'target' | 'heading' | 'block' | 'heading-search' | 'block-search'
  start: number
  query: string
  target: string
  hasClosingBracket: boolean
}

function protectedNode(state: EditorState, position: number): boolean {
  let node = syntaxTree(state).resolveInner(position, -1)
  for (;;) {
    if (
      /^(?:InlineCode|FencedCode|CodeBlock|HTMLBlock|CommentBlock|Comment|InlineMath|BlockMath)$/.test(
        node.name,
      )
    )
      return true
    if (!node.parent) return false
    node = node.parent
  }
}

function protectedMarkers(state: EditorState, source: string, position: number): boolean {
  let comment = false
  let math = false
  for (let index = 0; index < position; index++) {
    if (source[index] === '%' && source[index + 1] === '%' && !isEscapedAt(source, index)) {
      if (comment || !protectedNode(state, index + 1)) comment = !comment
      index++
    } else if (
      !comment &&
      source[index] === '$' &&
      !isEscapedAt(source, index) &&
      !protectedNode(state, index + 1)
    ) {
      math = !math
      if (source[index + 1] === '$') index++
    }
  }
  return comment || math
}

export function referenceCompletionContext(
  state: EditorState,
  position: number,
): ReferenceCompletionContext | null {
  const line = state.doc.lineAt(position)
  const prefix = line.text.slice(0, position - line.from)
  const cursor = referenceAtCursor(prefix, prefix.length, {
    protected: protectedNode(state, position),
  })
  if (
    cursor.kind !== 'incomplete' ||
    cursor.field === 'settings' ||
    cursor.field === 'label' ||
    cursor.field === 'close'
  )
    return null
  if (protectedMarkers(state, state.doc.sliceString(0, position), position)) return null
  const opening = cursor.raw.indexOf('[[')
  const body = cursor.raw.slice(opening + 2)
  const bodyStart = line.from + cursor.start + opening + 2
  const field = cursor.field
  const hash = body.lastIndexOf('#')
  const offset =
    field === 'heading-search' || field === 'block-search'
      ? 0
      : field === 'heading'
        ? hash + 1
        : field === 'block'
          ? hash + 2
          : 0
  return {
    field,
    start: bodyStart + offset,
    query:
      field === 'heading-search' || field === 'block-search' ? body.slice(2) : body.slice(offset),
    target: hash >= 0 ? body.slice(0, body.indexOf('#')) : body,
    hasClosingBracket: line.text.slice(position - line.from).startsWith(']]'),
  }
}

const completionCatalogs = new WeakMap<object, ReferenceCatalog>()

function completionCatalog(data: Readonly<Record<string, ContentDetails>>): ReferenceCatalog {
  const cached = completionCatalogs.get(data)
  if (cached) return cached
  const inputs: ReferenceCatalogInput[] = []
  for (const [slug, item] of Object.entries(data)) {
    if (item.protected || !isFullSlug(slug)) continue
    const filename = item.fileName || slug
    const source =
      (item.filePath?.endsWith('.md') && !filename.endsWith('.md')) || !getFileExtension(filename)
        ? `${filename}.md`
        : filename
    inputs.push({ path: source, slug, aliases: item.aliases })
  }
  const catalog = createReferenceCatalog(inputs)
  completionCatalogs.set(data, catalog)
  return catalog
}

export function fragmentCompletions(
  context: ReferenceCompletionContext | null,
  data: Readonly<Record<string, ContentDetails>>,
  origin: string,
  baseUrl = '',
): CompletionResult | null {
  if (!context || context.field === 'target') return null
  const global = context.field === 'heading-search' || context.field === 'block-search'
  const block = context.field === 'block' || context.field === 'block-search'
  const target = global ? undefined : completionCatalog(data).resolve(context.target, origin)
  if (!global && target?.kind !== 'found') return null
  const options: Completion[] = []
  const suffix = context.hasClosingBracket ? '' : ']]'
  for (const [slug, item] of Object.entries(data)) {
    if (item.protected || !item.referenceTargets) continue
    if (!global && (target?.kind !== 'found' || target.entry.slug !== slug)) continue
    const targets = block
      ? item.referenceTargets.blocks.map(text => ({ text, anchor: text }))
      : item.referenceTargets.headings
    for (const target of targets) {
      const match = context.query ? fuzzyMatch(context.query, target.text) : { score: 0 }
      if (!match) continue
      const selector = block ? target.text : target.text.replace(/([\\#|[\]])/g, '\\$1')
      options.push({
        label: target.text,
        detail: `${item.title} (${slug})`,
        type: block ? 'variable' : 'text',
        boost: match.score,
        apply: `${global ? `${baseUrl}${slug}#${block ? '^' : ''}` : ''}${selector}${suffix}`,
      })
    }
  }
  options.sort((left, right) => (right.boost ?? 0) - (left.boost ?? 0))
  return options.length
    ? { from: context.start, options: options.slice(0, 50), filter: false }
    : null
}
