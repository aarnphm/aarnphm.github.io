import type { Blockquote, BlockContent, Code, DefinitionContent, Nodes } from 'mdast'
import type { Point } from 'unist'
import { fromMarkdown } from 'mdast-util-from-markdown'
import type { ReferencePart, ReferenceSpan, ReferenceSyntax } from '../../../util/references/model'
import {
  wikilink,
  wikilinkFromMarkdown,
  isWikilink,
} from '../../../extensions/micromark-extension-ofm-wikilinks'

function shiftReference(reference: ReferenceSyntax, delta: number): ReferenceSyntax {
  const span = (value: ReferenceSpan): ReferenceSpan =>
    Object.freeze({ start: value.start + delta, end: value.end + delta })
  const part = (value: ReferencePart): ReferencePart =>
    Object.freeze({ ...value, span: span(value.span) })
  return Object.freeze({
    ...reference,
    span: span(reference.span),
    target: part(reference.target),
    fragments: Object.freeze(reference.fragments.map(part)),
    ...(reference.settings ? { settings: part(reference.settings) } : {}),
    ...(reference.label ? { label: part(reference.label) } : {}),
    ...(reference.labelSeparator ? { labelSeparator: part(reference.labelSeparator) } : {}),
  })
}

/** Fenced content drops container prefixes and normalizes newlines before parsing. */
export function quoteBlock(code: Code, source: string): Blockquote {
  const tree = fromMarkdown(code.value, {
    extensions: [wikilink()],
    mdastExtensions: [wikilinkFromMarkdown()],
  })
  const sourceLines = source.split(/\r\n|\r|\n/)
  const lineOffsets = [
    0,
    ...Array.from(source.matchAll(/\r\n|\r|\n/g), match => match.index + match[0].length),
  ]
  const contentLines = code.value.split('\n')
  const point = (local: Point): Point | undefined => {
    const fenceLine = code.position?.start.line
    if (fenceLine === undefined) return undefined
    const index = fenceLine + local.line - 1
    const raw = sourceLines[index]
    const content = contentLines[local.line - 1]
    if (raw === undefined || content === undefined || !raw.endsWith(content)) return undefined
    const prefix = raw.length - content.length
    return {
      line: index + 1,
      column: prefix + local.column,
      offset: lineOffsets[index] + prefix + local.column - 1,
    }
  }
  function rebase(node: Nodes): void {
    const start = node.position && point(node.position.start)
    const end = node.position && point(node.position.end)
    if (isWikilink(node) && node.data?.reference) {
      const reference = node.data.reference
      if (
        start?.offset !== undefined &&
        end?.offset !== undefined &&
        source.slice(start.offset, end.offset) === reference.raw
      ) {
        node.data.reference = shiftReference(reference, start.offset - reference.span.start)
      } else {
        node.data.reference = Object.freeze({ ...reference, source: 'generated' })
        delete node.position
      }
    }
    if (node.position) node.position = start && end ? { start, end } : undefined
    if ('children' in node) node.children.forEach(rebase)
  }
  rebase(tree)
  const children = tree.children.filter((node): node is BlockContent | DefinitionContent =>
    [
      'blockquote',
      'code',
      'heading',
      'html',
      'list',
      'paragraph',
      'table',
      'thematicBreak',
      'definition',
      'footnoteDefinition',
    ].includes(node.type),
  )
  if (children.length !== tree.children.length)
    throw new Error('Expected quoted Markdown to contain block content')
  return {
    type: 'blockquote',
    data: { hProperties: { className: ['quotes'] } },
    children,
    position: code.position,
  }
}
