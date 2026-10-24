import type { Code, Root } from 'mdast'
import { createHash } from 'node:crypto'
import { visit, SKIP } from 'unist-util-visit'
import type { Cell } from '../../util/notebook/types'
import { notebookCellDisplayLanguage } from '../../util/notebook/code-language'

export interface PublishedCodeBlock {
  text: string
  originalLanguage: string
  anchor: string
  startLine: number
  kind: 'block' | 'cell'
  sourcePath?: string
}

export function codeSourceId(text: string): string {
  return createHash('sha256').update(text).digest('hex')
}

export function codeBlockRecords(
  tree: Root,
  source: string,
  prefix = 'code',
): PublishedCodeBlock[] {
  const counts = new Map<string, number>()
  const records: PublishedCodeBlock[] = []
  const lines = source.split(/\r\n|\n|\r/)
  visit(tree, 'code', node => {
    const data = node.data
    if (data?.githubCodeTransclude) return
    const transcluded = data && 'codeTranscludeTarget' in data
    const pathMatch = transcluded ? node.meta?.match(/(?:^|\s)path="([^"]+)"/) : undefined
    if (!node.position && !pathMatch) return
    const hash = codeSourceId(node.value).slice(0, 16)
    const occurrence = (counts.get(hash) ?? 0) + 1
    counts.set(hash, occurrence)
    const start = node.position?.start
    const firstLine = start ? (lines[start.line - 1] ?? '').slice(start.column - 1) : ''
    const fenced = /^(?:`{3,}|~{3,})/.test(firstLine)
    const record: PublishedCodeBlock = {
      text: node.value,
      originalLanguage: node.lang ?? 'text',
      anchor: `${prefix}-${hash}-${occurrence}`,
      startLine: pathMatch ? 1 : (start?.line ?? 1) + Number(fenced),
      kind: 'block',
      ...(pathMatch ? { sourcePath: pathMatch[1] } : {}),
    }
    node.data ??= {}
    node.data.codeSearchRecord = record
    records.push(record)
  })
  return records
}

export function notebookCellRecords(
  cell: Cell,
  parse: (source: string) => Root,
): PublishedCodeBlock[] {
  if (cell.cellType === 'code')
    return cell.source.trim()
      ? [
          {
            text: cell.source,
            originalLanguage: notebookCellDisplayLanguage(
              cell.metadata,
              cell.source,
              cell.language,
            ),
            anchor: cell.id,
            startLine: 1,
            kind: 'cell',
          },
        ]
      : []
  if (cell.cellType !== 'markdown') return []
  return codeBlockRecords(parse(cell.source), cell.source, `${cell.id}-code`)
}

/** Called before notebook transforms, only for source chunks, never output chunks. */
export function anchorCodeBlocks(tree: Root): void {
  visit(tree, 'code', (node, index, parent) => {
    const record = node.data?.codeSearchRecord
    if (
      !record ||
      record.kind !== 'block' ||
      index === undefined ||
      !parent ||
      node.data?.codeSearchAnchored
    )
      return
    // A sibling boundary survives renderers that replace a code node or its metadata.
    parent.children.splice(index, 0, {
      type: 'paragraph',
      children: [],
      data: { hName: 'span', hProperties: { id: record.anchor, 'data-code-search-anchor': '' } },
    })
    node.data ??= {}
    node.data.codeSearchAnchored = true
    return [SKIP, index + 2]
  })
}

export function annotateNotebookSource(tree: Root, records: readonly PublishedCodeBlock[]): void {
  let index = 0
  visit(tree, 'code', (node: Code) => {
    const record = records[index++]
    if (!record) return
    node.data ??= {}
    node.data.codeSearchRecord = record
  })
  anchorCodeBlocks(tree)
}

declare module 'mdast' {
  interface CodeData {
    codeSearchRecord?: PublishedCodeBlock
    codeSearchAnchored?: boolean
  }
}
declare module 'vfile' {
  interface DataMap {
    codeSearchBlocks?: PublishedCodeBlock[]
  }
}
