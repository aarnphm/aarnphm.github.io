import path from 'path'
import { escapeHTML } from './escape'

type JsonRecord = Record<string, unknown>

type NotebookOutput = JsonRecord & {
  output_type?: unknown
  data?: unknown
  text?: unknown
  name?: unknown
  ename?: unknown
  evalue?: unknown
  traceback?: unknown
}

type NotebookCell = JsonRecord & { cell_type?: unknown; source?: unknown; outputs?: unknown }

export type NotebookDocument = JsonRecord & { cells: NotebookCell[]; metadata?: JsonRecord }

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function asText(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(asText).join('')
  if (value === undefined || value === null) return ''
  return JSON.stringify(value, null, 2)
}

function asTextList(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(asText)
  const text = asText(value)
  return text.length > 0 ? [text] : []
}

function maxBacktickRun(value: string): number {
  return Math.max(0, ...Array.from(value.matchAll(/`+/g), match => match[0].length))
}

function fenced(value: string, language = 'text'): string {
  const fence = '`'.repeat(Math.max(3, maxBacktickRun(value) + 1))
  return `${fence}${language}\n${value.replace(/\s+$/, '')}\n${fence}`
}

function languageName(notebook: NotebookDocument): string {
  const metadata = isRecord(notebook.metadata) ? notebook.metadata : {}
  const languageInfo = isRecord(metadata.language_info) ? metadata.language_info : {}
  const name = typeof languageInfo.name === 'string' ? languageInfo.name : 'python'
  return name.replace(/[^A-Za-z0-9_+#.-]/g, '') || 'python'
}

function titleFromMarkdown(source: string): string | undefined {
  for (const line of source.split(/\r?\n/)) {
    const match = line.match(/^#{1,6}\s+(.+?)\s*#*\s*$/)
    if (match) {
      return match[1]
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        .replace(/[*_`]/g, '')
        .trim()
    }
  }
}

export function parseNotebook(raw: string, sourcePath: string): NotebookDocument {
  const parsed: unknown = JSON.parse(raw)
  if (!isRecord(parsed) || !Array.isArray(parsed.cells)) {
    throw new Error(`${sourcePath} is not a valid notebook`)
  }

  return { ...parsed, cells: parsed.cells.filter(isRecord) }
}

export function notebookTitle(notebook: NotebookDocument, sourcePath: string): string {
  for (const cell of notebook.cells) {
    if (cell.cell_type === 'markdown') {
      const title = titleFromMarkdown(asText(cell.source))
      if (title) return title
    }
  }

  return path.basename(sourcePath, path.extname(sourcePath))
}

function frontmatter(title: string): string {
  return ['---', `title: ${JSON.stringify(title)}`, 'cssclasses:', '  - notebook-page', '---'].join(
    '\n',
  )
}

function mimeBundleOutput(data: unknown): string[] {
  if (!isRecord(data)) return []

  const html = asText(data['text/html'])
  if (html.trim()) return [`<div class="notebook-output notebook-output-html">\n${html}\n</div>`]

  const markdown = asText(data['text/markdown'])
  if (markdown.trim()) return [markdown]

  const latex = asText(data['text/latex'])
  if (latex.trim()) return [latex.trim().startsWith('$') ? latex : `$$\n${latex.trim()}\n$$`]

  const svg = asText(data['image/svg+xml'])
  if (svg.trim()) return [`<div class="notebook-output notebook-output-svg">\n${svg}\n</div>`]

  for (const mime of ['image/png', 'image/jpeg', 'image/gif']) {
    const image = asText(data[mime])
    if (image.trim()) {
      return [
        `<p class="notebook-output notebook-output-image"><img src="data:${mime};base64,${escapeHTML(
          image.replace(/\s/g, ''),
        )}" alt="notebook output" /></p>`,
      ]
    }
  }

  const text = asText(data['text/plain'])
  return text.trim() ? [fenced(text, 'text')] : []
}

function notebookOutput(output: NotebookOutput): string[] {
  const outputType = typeof output.output_type === 'string' ? output.output_type : ''

  if (outputType === 'stream') {
    const text = asText(output.text)
    return text.trim() ? [fenced(text, 'text')] : []
  }

  if (outputType === 'error') {
    const traceback = asTextList(output.traceback).join('\n')
    const header = [asText(output.ename), asText(output.evalue)].filter(Boolean).join(': ')
    const text = traceback || header
    return text.trim() ? [fenced(text, 'pytb')] : []
  }

  if (outputType === 'display_data' || outputType === 'execute_result') {
    return mimeBundleOutput(output.data)
  }

  return []
}

function notebookCell(cell: NotebookCell, language: string): string[] {
  const cellType = typeof cell.cell_type === 'string' ? cell.cell_type : ''
  const source = asText(cell.source)

  if (cellType === 'markdown') {
    return source.trim() ? [source.trim()] : []
  }

  if (cellType !== 'code') {
    return []
  }

  const parts = source.trim() ? [fenced(source, language)] : []
  const outputs = Array.isArray(cell.outputs) ? cell.outputs.filter(isRecord) : []
  for (const output of outputs) {
    parts.push(...notebookOutput(output))
  }
  return parts
}

export function notebookToMarkdown(notebook: NotebookDocument, sourcePath: string): string {
  const language = languageName(notebook)
  const title = notebookTitle(notebook, sourcePath)
  const chunks = [frontmatter(title)]

  for (const cell of notebook.cells) {
    chunks.push(...notebookCell(cell, language))
  }

  return `${chunks.filter(chunk => chunk.trim()).join('\n\n')}\n`
}
