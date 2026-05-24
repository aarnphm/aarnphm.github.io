import type { CellId, NotebookId } from './types'

export function notebookId(sourcePath: string): NotebookId {
  let hash = 2166136261
  for (let i = 0; i < sourcePath.length; i += 1) {
    hash ^= sourcePath.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return `notebook-runtime-${(hash >>> 0).toString(36)}` as NotebookId
}

export function codeCellId(codeIndex: number): CellId {
  return `cell-${codeIndex}` as CellId
}

export function markdownCellId(globalIndex: number): CellId {
  return `notebook-md-${globalIndex}` as CellId
}

export function rawCellId(globalIndex: number): CellId {
  return `notebook-raw-${globalIndex}` as CellId
}
