import type { BlockContent, DefinitionContent, Root } from 'mdast'
import type { Point } from 'unist'
import type { ReferencePart } from '../../../util/references/model'
import { isWikilink } from '../../../extensions/micromark-extension-ofm-wikilinks/node'
import { visitReferenceNodes } from '../../../util/references/markdown'

/** Definition continuations lose indentation during their independent Markdown parse. */
export function sidenoteBlockContent(
  tree: Root,
  fragment: string,
  source: string,
  startLine: number | undefined,
): (BlockContent | DefinitionContent)[] {
  const lines = source.split(/\r\n|\r|\n/)
  const offsets = [
    0,
    ...Array.from(source.matchAll(/\r\n|\r|\n/g), match => match.index + match[0].length),
  ]
  const contentLines = fragment.split('\n')
  const point = (local: Point): Point | undefined => {
    if (startLine === undefined) return undefined
    const index = startLine + local.line - 2
    const raw = lines[index]
    const content = contentLines[local.line - 1]
    if (raw === undefined || content === undefined || !raw.endsWith(content)) return undefined
    const prefix = raw.length - content.length
    return {
      line: index + 1,
      column: prefix + local.column,
      offset: offsets[index] + prefix + local.column - 1,
    }
  }
  visitReferenceNodes(tree, node => {
    const start = node.position && point(node.position.start)
    const end = node.position && point(node.position.end)
    if (isWikilink(node) && node.data?.reference) {
      const reference = node.data.reference
      if (
        start?.offset !== undefined &&
        end?.offset !== undefined &&
        source.slice(start.offset, end.offset) === reference.raw
      ) {
        const delta = start.offset - reference.span.start
        const span = (value: { readonly start: number; readonly end: number }) =>
          Object.freeze({ start: value.start + delta, end: value.end + delta })
        const part = (value: ReferencePart): ReferencePart =>
          Object.freeze({ ...value, span: span(value.span) })
        node.data.reference = Object.freeze({
          ...reference,
          span: span(reference.span),
          target: part(reference.target),
          fragments: Object.freeze(reference.fragments.map(part)),
          ...(reference.settings ? { settings: part(reference.settings) } : {}),
          ...(reference.label ? { label: part(reference.label) } : {}),
          ...(reference.labelSeparator ? { labelSeparator: part(reference.labelSeparator) } : {}),
        })
      } else {
        node.data.reference = Object.freeze({ ...reference, source: 'generated' })
        delete node.position
      }
    }
    if (node.position) node.position = start && end ? { start, end } : undefined
  })
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
      'sidenoteDefinition',
    ].includes(node.type),
  )
  if (children.length !== tree.children.length)
    throw new Error('Expected sidenote Markdown to contain block content')
  return children
}
