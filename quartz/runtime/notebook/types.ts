export type NotebookRuntimeCell = {
  id: string
  source: string
  language: string
  executionIndex: number | null
}

export type NotebookRuntimeConfig = {
  enabled?: boolean
  indexUrl?: string
  sourcePath?: string
  importableModules?: string[]
}

export type NotebookRuntimeData = {
  id: string
  sourcePath: string
  language: string
  indexUrl: string
  cells: NotebookRuntimeCell[]
  toolbar?: boolean
  debug?: boolean
  vimMode?: boolean
  importableModules?: string[]
}

export type NotebookRuntimeStreamOutput = { type: 'stream'; name: string; text: string }

export type NotebookRuntimeErrorOutput = {
  type: 'error'
  ename: string
  evalue: string
  traceback: string
  debug?: NotebookRuntimeDebugOutput
}

export type NotebookRuntimeDebugOutput = {
  phase: string
  cellId?: string
  errorName?: string
  errorMessage?: string
  stack?: string
}

export type NotebookRuntimeTextOutput = { type: 'text'; text: string }
export type NotebookRuntimeJsonOutput = { type: 'json'; text: string }
export type NotebookRuntimeHtmlOutput = { type: 'html'; html: string }
export type NotebookRuntimeSuccessOutput = { type: 'success' }

export type NotebookRuntimeOutput =
  | NotebookRuntimeStreamOutput
  | NotebookRuntimeErrorOutput
  | NotebookRuntimeTextOutput
  | NotebookRuntimeJsonOutput
  | NotebookRuntimeHtmlOutput
  | NotebookRuntimeSuccessOutput

export const notebookSuccessOutputLabel = 'exit 0'

export function notebookRuntimeLocalSourceKey(sourcePath: string, cellId: string): string {
  return `quartz:notebook-source:${encodeURIComponent(sourcePath)}:${encodeURIComponent(cellId)}`
}
