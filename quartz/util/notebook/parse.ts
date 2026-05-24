import type {
  Cell,
  CodeCell,
  MarkdownCell,
  MimeBundle,
  NotebookDoc,
  NotebookParseError,
  Output,
  RawCell,
} from './types'
import { isRecord, type UnknownRecord } from '../type-guards'
import { codeCellId, markdownCellId, notebookId, rawCellId } from './identity'

function asText(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(asText).join('')
  if (value === undefined || value === null) return ''
  return JSON.stringify(value, null, 2)
}

function asTextList(value: unknown): readonly string[] {
  if (Array.isArray(value)) return value.map(asText)
  const text = asText(value)
  return text.length > 0 ? [text] : []
}

function asNumberOrNull(value: unknown): number | null {
  return typeof value === 'number' ? value : null
}

function asRecord(value: unknown): Readonly<Record<string, unknown>> {
  return isRecord(value) ? (value as Record<string, unknown>) : {}
}

function asMimeBundle(value: unknown): MimeBundle {
  if (!isRecord(value)) return {}
  const result: Record<string, string | readonly string[]> = {}
  for (const [key, raw] of Object.entries(value)) {
    if (typeof raw === 'string') {
      result[key] = raw
    } else if (Array.isArray(raw) && raw.every(item => typeof item === 'string')) {
      result[key] = raw as readonly string[]
    } else if (raw !== undefined && raw !== null) {
      result[key] = asText(raw)
    }
  }
  return result
}

function languageFromMetadata(metadata: UnknownRecord): string {
  const languageInfo = isRecord(metadata.language_info) ? metadata.language_info : {}
  const name = typeof languageInfo.name === 'string' ? languageInfo.name : 'python'
  return name.replace(/[^A-Za-z0-9_+#.-]/g, '') || 'python'
}

export function parseOutput(raw: unknown): Output | undefined {
  if (!isRecord(raw)) return undefined
  const outputType = typeof raw.output_type === 'string' ? raw.output_type : ''
  if (outputType === 'stream') {
    const rawName = typeof raw.name === 'string' ? raw.name.trim() : ''
    const name: 'stdout' | 'stderr' = rawName === 'stderr' ? 'stderr' : 'stdout'
    return { kind: 'stream', name, text: asText(raw.text) }
  }
  if (outputType === 'error') {
    return {
      kind: 'error',
      ename: asText(raw.ename),
      evalue: asText(raw.evalue),
      traceback: asTextList(raw.traceback),
    }
  }
  if (outputType === 'display_data') {
    return { kind: 'display_data', data: asMimeBundle(raw.data), metadata: asRecord(raw.metadata) }
  }
  if (outputType === 'execute_result') {
    return {
      kind: 'execute_result',
      data: asMimeBundle(raw.data),
      metadata: asRecord(raw.metadata),
      executionCount: asNumberOrNull(raw.execution_count),
    }
  }
  return undefined
}

type ParseCellContext = { language: string; codeIndex: number; globalIndex: number }

function parseCodeCell(raw: UnknownRecord, ctx: ParseCellContext): CodeCell {
  const outputs: Output[] = []
  if (Array.isArray(raw.outputs)) {
    for (const item of raw.outputs) {
      const parsed = parseOutput(item)
      if (parsed) outputs.push(parsed)
    }
  }
  return {
    cellType: 'code',
    id: codeCellId(ctx.codeIndex),
    language: ctx.language,
    source: asText(raw.source),
    outputs,
    executionCount: asNumberOrNull(raw.execution_count),
    metadata: asRecord(raw.metadata),
  }
}

function parseMarkdownCell(raw: UnknownRecord, ctx: ParseCellContext): MarkdownCell {
  const attachments: Record<string, MimeBundle> = {}
  if (isRecord(raw.attachments)) {
    for (const [key, value] of Object.entries(raw.attachments)) {
      attachments[key] = asMimeBundle(value)
    }
  }
  return {
    cellType: 'markdown',
    id: markdownCellId(ctx.globalIndex),
    source: asText(raw.source),
    attachments,
    metadata: asRecord(raw.metadata),
  }
}

function parseRawCell(raw: UnknownRecord, ctx: ParseCellContext): RawCell {
  const format = typeof raw.format === 'string' ? raw.format : null
  return {
    cellType: 'raw',
    id: rawCellId(ctx.globalIndex),
    source: asText(raw.source),
    mimeType: format,
    metadata: asRecord(raw.metadata),
  }
}

function parseCell(raw: UnknownRecord, ctx: ParseCellContext): Cell | undefined {
  const cellType = typeof raw.cell_type === 'string' ? raw.cell_type : ''
  if (cellType === 'code') return parseCodeCell(raw, ctx)
  if (cellType === 'markdown') return parseMarkdownCell(raw, ctx)
  if (cellType === 'raw') return parseRawCell(raw, ctx)
  return undefined
}

export function parseNotebookDoc(
  raw: string,
  sourcePath: string,
): NotebookDoc | NotebookParseError {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch (cause) {
    return { kind: 'NotebookParseError', path: sourcePath, reason: 'invalid JSON', cause }
  }
  if (!isRecord(parsed)) {
    return {
      kind: 'NotebookParseError',
      path: sourcePath,
      reason: 'notebook root is not an object',
    }
  }
  if (!Array.isArray(parsed.cells)) {
    return { kind: 'NotebookParseError', path: sourcePath, reason: 'notebook has no cells array' }
  }
  const metadata = asRecord(parsed.metadata)
  const language = languageFromMetadata(metadata as UnknownRecord)
  const cells: Cell[] = []
  let codeIndex = 0
  let globalIndex = 0
  for (const item of parsed.cells) {
    if (!isRecord(item)) continue
    globalIndex += 1
    if (typeof item.cell_type === 'string' && item.cell_type === 'code') codeIndex += 1
    const cell = parseCell(item, { language, codeIndex, globalIndex })
    if (cell) cells.push(cell)
  }
  const nbformat = typeof parsed.nbformat === 'number' ? parsed.nbformat : 4
  const nbformatMinor = typeof parsed.nbformat_minor === 'number' ? parsed.nbformat_minor : 0
  return {
    id: notebookId(sourcePath),
    sourcePath,
    nbformat,
    nbformatMinor,
    language,
    cells,
    metadata,
    raw: parsed as UnknownRecord,
  }
}
