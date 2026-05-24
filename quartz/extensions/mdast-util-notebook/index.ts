import type { BlockContent, Code, Root as MdRoot } from 'mdast'
import remarkParse from 'remark-parse'
import { unified } from 'unified'
import type {
  CellId,
  CodeCell,
  MarkdownCell,
  NotebookDoc,
  NotebookId,
  Output,
  RawCell,
} from '../../util/notebook/types'
import type { NotebookCell, NotebookOutput, NotebookOutputData } from './types'

export type { NotebookCell, NotebookOutput, NotebookCellData, NotebookOutputData } from './types'

const markdownParser = unified().use(remarkParse)

function parseMarkdownToBlocks(source: string): BlockContent[] {
  if (!source.trim()) return []
  const tree = markdownParser.parse(source) as MdRoot
  return tree.children.filter((child): child is BlockContent => child.type !== 'yaml')
}

function outputData(output: Output, cellId: CellId, outputIndex: number): NotebookOutputData {
  if (output.kind === 'stream') {
    return { cellId, kind: 'stream', outputIndex, text: output.text, name: output.name }
  }
  if (output.kind === 'error') {
    return {
      cellId,
      kind: 'error',
      outputIndex,
      ename: output.ename,
      evalue: output.evalue,
      traceback: output.traceback,
    }
  }
  if (output.kind === 'display_data') {
    return {
      cellId,
      kind: 'display_data',
      outputIndex,
      data: output.data,
      metadata: output.metadata,
    }
  }
  return {
    cellId,
    kind: 'execute_result',
    outputIndex,
    data: output.data,
    metadata: output.metadata,
    executionCount: output.executionCount,
  }
}

function outputNode(output: Output, cellId: CellId, outputIndex: number): NotebookOutput {
  return {
    type: 'notebookOutput',
    data: { notebookOutput: outputData(output, cellId, outputIndex) },
    children: [],
  }
}

function codeCellToMdast(cell: CodeCell, notebookId: NotebookId): NotebookCell {
  const codeNode: Code = { type: 'code', lang: cell.language, value: cell.source }
  const children: (BlockContent | NotebookOutput)[] = []
  if (cell.source.trim()) children.push(codeNode)
  cell.outputs.forEach((output, index) => {
    children.push(outputNode(output, cell.id, index))
  })
  return {
    type: 'notebookCell',
    data: {
      notebookCell: {
        cellId: cell.id,
        notebookId,
        language: cell.language,
        executionCount: cell.executionCount,
        cellType: 'code',
        metadata: cell.metadata,
      },
    },
    children,
  }
}

function markdownCellToMdast(cell: MarkdownCell, notebookId: NotebookId): NotebookCell {
  return {
    type: 'notebookCell',
    data: {
      notebookCell: {
        cellId: cell.id,
        notebookId,
        language: 'markdown',
        executionCount: null,
        cellType: 'markdown',
        metadata: cell.metadata,
      },
    },
    children: parseMarkdownToBlocks(cell.source),
  }
}

function rawCellToMdast(cell: RawCell, notebookId: NotebookId): NotebookCell {
  return {
    type: 'notebookCell',
    data: {
      notebookCell: {
        cellId: cell.id,
        notebookId,
        language: cell.mimeType ?? 'text',
        executionCount: null,
        cellType: 'raw',
        metadata: cell.metadata,
      },
    },
    children: [{ type: 'code', lang: cell.mimeType ?? null, value: cell.source }],
  }
}

export function notebookToMdast(doc: NotebookDoc): MdRoot {
  const children: NotebookCell[] = doc.cells.map(cell => {
    if (cell.cellType === 'code') return codeCellToMdast(cell, doc.id)
    if (cell.cellType === 'markdown') return markdownCellToMdast(cell, doc.id)
    return rawCellToMdast(cell, doc.id)
  })
  return { type: 'root', children: children as unknown as MdRoot['children'] }
}
