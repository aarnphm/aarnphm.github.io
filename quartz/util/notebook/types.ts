import type { UnknownRecord } from '../type-guards'

export type NotebookId = string & { readonly __brand: 'NotebookId' }
export type CellId = string & { readonly __brand: 'CellId' }

export type MimeBundle = Readonly<Partial<Record<string, string | readonly string[]>>>

export type StreamOutput = {
  readonly kind: 'stream'
  readonly name: 'stdout' | 'stderr'
  readonly text: string
}

export type ErrorOutput = {
  readonly kind: 'error'
  readonly ename: string
  readonly evalue: string
  readonly traceback: readonly string[]
}

export type DisplayOutput = {
  readonly kind: 'display_data'
  readonly data: MimeBundle
  readonly metadata: Readonly<Record<string, unknown>>
}

export type ExecuteResult = {
  readonly kind: 'execute_result'
  readonly data: MimeBundle
  readonly metadata: Readonly<Record<string, unknown>>
  readonly executionCount: number | null
}

export type Output = StreamOutput | ErrorOutput | DisplayOutput | ExecuteResult

export type CodeCell = {
  readonly cellType: 'code'
  readonly id: CellId
  readonly language: string
  readonly source: string
  readonly outputs: readonly Output[]
  readonly executionCount: number | null
  readonly metadata: Readonly<Record<string, unknown>>
}

export type MarkdownCell = {
  readonly cellType: 'markdown'
  readonly id: CellId
  readonly source: string
  readonly attachments: Readonly<Record<string, MimeBundle>>
  readonly metadata: Readonly<Record<string, unknown>>
}

export type RawCell = {
  readonly cellType: 'raw'
  readonly id: CellId
  readonly source: string
  readonly mimeType: string | null
  readonly metadata: Readonly<Record<string, unknown>>
}

export type Cell = CodeCell | MarkdownCell | RawCell

export type NotebookDoc = {
  readonly id: NotebookId
  readonly sourcePath: string
  readonly nbformat: number
  readonly nbformatMinor: number
  readonly language: string
  readonly cells: readonly Cell[]
  readonly metadata: Readonly<Record<string, unknown>>
  readonly raw: UnknownRecord
}

export type NotebookParseError = {
  readonly kind: 'NotebookParseError'
  readonly path: string
  readonly reason: string
  readonly cause?: unknown
}

export function isNotebookParseError(value: unknown): value is NotebookParseError {
  return (
    typeof value === 'object' &&
    value !== null &&
    (value as { kind?: unknown }).kind === 'NotebookParseError'
  )
}
