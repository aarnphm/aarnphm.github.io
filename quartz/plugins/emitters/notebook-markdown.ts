import type { Root } from 'mdast'
import type { Point } from 'unist'
import type { QuartzMdProcessor } from '../../processors/markdown-processors'
import type { ReferencePart, ReferenceSpan, ReferenceSyntax } from '../../util/references/model'
import { isWikilink } from '../../extensions/micromark-extension-ofm-wikilinks/node'
import { visitReferenceNodes } from '../../util/references/markdown'
import { annotateNotebookSource, type PublishedCodeBlock } from '../transformers/code-search-blocks'

function shiftReference(reference: ReferenceSyntax, offset: number): ReferenceSyntax {
  const span = (value: ReferenceSpan): ReferenceSpan => ({
    start: value.start + offset,
    end: value.end + offset,
  })
  const part = (value: ReferencePart): ReferencePart => ({ ...value, span: span(value.span) })
  return {
    ...reference,
    span: span(reference.span),
    target: part(reference.target),
    fragments: reference.fragments.map(part),
    ...(reference.settings ? { settings: part(reference.settings) } : {}),
    ...(reference.label ? { label: part(reference.label) } : {}),
    ...(reference.labelSeparator ? { labelSeparator: part(reference.labelSeparator) } : {}),
  }
}

/** Independent chunks keep notebook HTML wrappers from swallowing Markdown cell content. */
export function parseNotebookMarkdownChunks(
  processor: QuartzMdProcessor,
  chunks: readonly string[],
  sourceRecords: ReadonlyMap<number, PublishedCodeBlock[]>,
): { ast: Root; source: string } {
  const children: Root['children'] = []
  let offset = 0
  let lines = 0
  for (const [index, chunk] of chunks.entries()) {
    const tree = processor.parse(chunk)
    const records = sourceRecords.get(index)
    if (records) annotateNotebookSource(tree, records)
    const point = (value: Point): Point => ({
      ...value,
      line: value.line + lines,
      ...(value.offset !== undefined ? { offset: value.offset + offset } : {}),
    })
    visitReferenceNodes(tree, node => {
      if (node.position)
        node.position = { start: point(node.position.start), end: point(node.position.end) }
      if (isWikilink(node) && node.data?.reference)
        node.data.reference = shiftReference(node.data.reference, offset)
    })
    children.push(...tree.children)
    offset += chunk.length + 2
    lines += (chunk.match(/\r\n|\r|\n/g)?.length ?? 0) + 2
  }
  return { ast: { type: 'root', children }, source: chunks.join('\n\n') }
}
