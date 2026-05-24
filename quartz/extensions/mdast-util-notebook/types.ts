import type { BlockContent, Node, PhrasingContent } from 'mdast'
import type { CellId, MimeBundle, NotebookId, Output } from '../../util/notebook/types'

export interface NotebookCellData {
  cellId: CellId
  notebookId: NotebookId
  language: string
  executionCount: number | null
  cellType: 'code' | 'markdown' | 'raw'
  metadata: Readonly<Record<string, unknown>>
}

export interface NotebookCell extends Node {
  type: 'notebookCell'
  data: { notebookCell: NotebookCellData }
  children: (BlockContent | NotebookOutput)[]
}

export interface NotebookOutputData {
  cellId: CellId
  kind: Output['kind']
  outputIndex: number
  data?: MimeBundle
  text?: string
  name?: 'stdout' | 'stderr'
  ename?: string
  evalue?: string
  traceback?: readonly string[]
  executionCount?: number | null
  metadata?: Readonly<Record<string, unknown>>
}

export interface NotebookOutput extends Node {
  type: 'notebookOutput'
  data: { notebookOutput: NotebookOutputData }
  children: PhrasingContent[]
}

declare module 'mdast' {
  interface Data {
    notebookCell?: NotebookCellData
    notebookOutput?: NotebookOutputData
  }

  interface StaticBlockContentMap {
    notebookCell: NotebookCell
    notebookOutput: NotebookOutput
  }
}
