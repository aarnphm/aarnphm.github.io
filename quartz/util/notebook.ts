import katex from 'katex'
import path from 'path'
import { customMacros, katexOptions } from '../cfg'
import { escapeHTML } from './escape'
import {
  defaultNotebookPyodideIndexUrl,
  notebookCellActions,
  notebookCellControls,
  notebookCellFrameOpen,
  notebookCellRuntimeOutput,
  notebookRuntimeControls,
  notebookRuntimeDataScript,
  notebookRuntimeId,
  notebookSourceEditor,
  renderNotebookRuntimeOutput,
  type NotebookRuntimeCell,
  type NotebookRuntimeConfig,
  type NotebookRuntimeData,
} from './notebook-runtime'
import { isRecord, type UnknownRecord } from './type-guards'

type JsonRecord = UnknownRecord

type NotebookOutput = JsonRecord & {
  output_type?: unknown
  data?: unknown
  text?: unknown
  name?: unknown
  ename?: unknown
  evalue?: unknown
  traceback?: unknown
}

type NotebookCell = JsonRecord & {
  cell_type?: unknown
  source?: unknown
  attachments?: unknown
  outputs?: unknown
  execution_count?: unknown
}

export type NotebookDocument = JsonRecord & { cells: NotebookCell[]; metadata?: JsonRecord }

type NotebookMarkdownOptions = { runtime?: false | NotebookRuntimeConfig }

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

function classToken(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, '-') || 'output'
}

const ansiPattern = new RegExp(`${String.fromCharCode(27)}\\[[0-?]*[ -/]*[@-~]`, 'g')

function stripAnsi(value: string): string {
  return value.replace(ansiPattern, '')
}

function htmlAttributes(attrs: Record<string, string>): string {
  return Object.entries(attrs)
    .map(([key, value]) => ` ${key}="${escapeHTML(value)}"`)
    .join('')
}

function outputPre(
  classNames: string[],
  value: string,
  attrs: Record<string, string> = {},
): string {
  return `<pre class="${classNames.join(' ')}"${htmlAttributes(attrs)}><samp>${escapeHTML(
    value.replace(/\s+$/, ''),
  )}</samp></pre>`
}

function latexSource(value: string): { source: string; displayMode: boolean } {
  const trimmed = value.trim()
  const blockMath = trimmed.match(/^\$\$([\s\S]*)\$\$$/)
  if (blockMath) return { source: blockMath[1].trim(), displayMode: true }
  const bracketMath = trimmed.match(/^\\\[([\s\S]*)\\\]$/)
  if (bracketMath) return { source: bracketMath[1].trim(), displayMode: true }
  const inlineMath = trimmed.match(/^\$([\s\S]*)\$$/)
  if (inlineMath) return { source: inlineMath[1].trim(), displayMode: false }
  return { source: trimmed, displayMode: true }
}

function outputLatex(value: string): string | undefined {
  if (!value.trim()) return undefined
  const latex = latexSource(value)
  try {
    return `<div class="notebook-output notebook-output-latex" data-output-name="result">${katex.renderToString(
      latex.source,
      {
        output: 'htmlAndMathml',
        macros: customMacros,
        ...katexOptions,
        displayMode: latex.displayMode,
      },
    )}</div>`
  } catch {
    return outputPre(['notebook-output', 'notebook-output-text'], value, {
      'data-output-name': 'result',
    })
  }
}

const ipythonDisplayObjectPattern = /^<IPython\.core\.display\.[A-Za-z0-9_]+ object>$/
const attachmentMimeTypes = ['image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/svg+xml']
const htmlImageSrcPattern = /(<img\b[^>]*\bsrc\s*=\s*)(?:"([^"]*)"|'([^']*)'|([^\s>]+))/gi
const markdownImagePattern = /(!\[[^\]]*\]\()(\s*)([^)\s]+)([^)]*\))/g
const standaloneHtmlImageLinePattern = /^\s*<img\b[^>\n]*>(?:\s*<\/img>)?\s*$/i
const markdownFenceLinePattern = /^( {0,3})(`{3,}|~{3,})(.*)$/
const htmlDivOpenTagPattern = /<div\b[^>]*>/gi
const htmlDivCloseTagPattern = /<\/div\s*>/gi
const htmlFloatStylePattern =
  /\bstyle\s*=\s*(?:"[^"]*\bfloat\s*:\s*(?:left|right)[^"]*"|'[^']*\bfloat\s*:\s*(?:left|right)[^']*'|[^\s>]*\bfloat\s*:\s*(?:left|right)[^\s>]*)/i

function languageName(notebook: NotebookDocument): string {
  const metadata = isRecord(notebook.metadata) ? notebook.metadata : {}
  const languageInfo = isRecord(metadata.language_info) ? metadata.language_info : {}
  const name = typeof languageInfo.name === 'string' ? languageInfo.name : 'python'
  return name.replace(/[^A-Za-z0-9_+#.-]/g, '') || 'python'
}

function notebookRuntimeLanguage(notebook: NotebookDocument): string {
  return languageName(notebook).toLowerCase()
}

export function notebookSupportsRuntime(notebook: NotebookDocument): boolean {
  return notebookRuntimeLanguage(notebook).startsWith('python')
}

function markdownHeadingTitle(line: string): string | undefined {
  const match = line.match(/^#{1,6}\s+(.+?)\s*#*\s*$/)
  if (!match) return undefined
  const title = match[1]
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*_`]/g, '')
    .trim()
  return title.length > 0 ? title : undefined
}

function titleFromMarkdown(source: string): string | undefined {
  for (const line of source.split(/\r?\n/)) {
    const title = markdownHeadingTitle(line)
    if (title) return title
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
  return [
    '---',
    `title: ${JSON.stringify(title)}`,
    'cssclasses:',
    '  - notebook-page',
    'collapseHeadings: false',
    '---',
  ].join('\n')
}

export function notebookRuntimeData(
  notebook: NotebookDocument,
  sourcePath: string,
  runtime: NotebookRuntimeConfig,
): NotebookRuntimeData | undefined {
  if (runtime.enabled === false || !notebookSupportsRuntime(notebook)) return undefined
  const language = languageName(notebook)
  const cells: NotebookRuntimeCell[] = []
  for (const cell of notebook.cells) {
    if (cell.cell_type !== 'code') continue
    const index = cells.length + 1
    cells.push({
      id: `cell-${index}`,
      source: asText(cell.source),
      language,
      executionIndex: typeof cell.execution_count === 'number' ? cell.execution_count : null,
    })
  }
  if (cells.length === 0) return undefined
  return {
    id: notebookRuntimeId(runtime.sourcePath ?? sourcePath),
    sourcePath: runtime.sourcePath ?? sourcePath,
    language,
    pyodideIndexUrl: runtime.pyodideIndexUrl ?? defaultNotebookPyodideIndexUrl,
    cells,
  }
}

function notebookMarkdownCellBoundary(): string {
  return '<div class="notebook-markdown-cell-boundary" aria-hidden="true"></div>'
}

function htmlFloatDivOpen(line: string): boolean {
  for (const match of line.matchAll(htmlDivOpenTagPattern)) {
    if (htmlFloatStylePattern.test(match[0])) return true
  }
  return false
}

function nextNonEmptyLine(lines: string[], start: number): string | undefined {
  for (let index = start; index < lines.length; index += 1) {
    const line = lines[index]
    if (line.trim()) return line
  }
}

function clearRawMarkdownFloatDivs(source: string): string {
  const lines = source.split(/\r?\n/)
  const result: string[] = []
  const divStack: boolean[] = []

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    for (const match of line.matchAll(htmlDivOpenTagPattern)) {
      divStack.push(divStack.includes(true) || htmlFloatStylePattern.test(match[0]))
    }

    result.push(line)

    let closedFloatingDiv = false
    for (const _match of line.matchAll(htmlDivCloseTagPattern)) {
      const wasFloatingDiv = divStack.pop() === true
      closedFloatingDiv ||= wasFloatingDiv && !divStack.includes(true)
    }

    const next = nextNonEmptyLine(lines, index + 1)
    if (closedFloatingDiv && next !== undefined && !htmlFloatDivOpen(next)) {
      result.push('', notebookMarkdownCellBoundary(), '')
    }
  }

  return result.join('\n')
}

function mimeBundleOutput(data: unknown): string[] {
  if (!isRecord(data)) return []

  const html = asText(data['text/html'])
  if (html.trim()) return [`<div class="notebook-output notebook-output-html">\n${html}\n</div>`]

  const markdown = asText(data['text/markdown'])
  if (markdown.trim()) return [markdown]

  const latex = asText(data['text/latex'])
  const renderedLatex = outputLatex(latex)
  if (renderedLatex) return [renderedLatex]

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

  const json = data['application/json']
  if (json !== undefined) {
    const text = asText(json)
    if (text.trim()) {
      return [
        outputPre(['notebook-output', 'notebook-output-text', 'notebook-output-json'], text, {
          'data-output-name': 'result',
        }),
      ]
    }
  }

  const text = asText(data['text/plain'])
  if (!text.trim() || ipythonDisplayObjectPattern.test(text.trim())) return []

  return [
    outputPre(['notebook-output', 'notebook-output-text'], text, { 'data-output-name': 'result' }),
  ]
}

function notebookOutput(output: NotebookOutput): string[] {
  const outputType = typeof output.output_type === 'string' ? output.output_type : ''

  if (outputType === 'stream') {
    const text = asText(output.text)
    const name =
      typeof output.name === 'string' && output.name.trim() ? output.name.trim() : 'stdout'
    return text.trim()
      ? [
          outputPre(
            [
              'notebook-output',
              'notebook-output-stream',
              `notebook-output-stream-${classToken(name)}`,
            ],
            text,
            { 'data-output-name': name },
          ),
        ]
      : []
  }

  if (outputType === 'error') {
    const traceback = stripAnsi(asTextList(output.traceback).join('\n'))
    const header = [asText(output.ename), asText(output.evalue)].filter(Boolean).join(': ')
    const text = traceback || header
    return text.trim()
      ? [
          outputPre(['notebook-output', 'notebook-output-error'], text, {
            'data-output-name': 'error',
          }),
        ]
      : []
  }

  if (outputType === 'display_data' || outputType === 'execute_result') {
    return mimeBundleOutput(output.data)
  }

  return []
}

function stripFirstTitleHeading(
  source: string,
  title: string,
): { source: string; removed: boolean } {
  const lines = source.split(/\r?\n/)
  const result: string[] = []
  let removed = false
  for (const line of lines) {
    if (!removed && markdownHeadingTitle(line) === title) {
      removed = true
      continue
    }
    result.push(line)
  }
  return { source: result.join('\n'), removed }
}

function notebookCell(
  cell: NotebookCell,
  language: string,
  sourcePath: string,
  runtimeCell?: NotebookRuntimeCell,
  titleHeading?: string,
): { chunks: string[]; titleHeadingRemoved: boolean } {
  const cellType = typeof cell.cell_type === 'string' ? cell.cell_type : ''
  const source = asText(cell.source)

  if (cellType === 'markdown') {
    let resolved = resolveMarkdownAttachments(source, cell.attachments)
    resolved = resolveNotebookImagePaths(resolved, sourcePath)
    resolved = separateStandaloneHtmlImageLines(resolved)
    resolved = closeDanglingMarkdownFence(resolved)
    resolved = clearRawMarkdownFloatDivs(resolved)
    let titleHeadingRemoved = false
    if (titleHeading) {
      const stripped = stripFirstTitleHeading(resolved, titleHeading)
      resolved = stripped.source
      titleHeadingRemoved = stripped.removed
    }
    return {
      chunks: resolved.trim() ? [resolved.trim(), notebookMarkdownCellBoundary()] : [],
      titleHeadingRemoved,
    }
  }

  if (cellType !== 'code') {
    return { chunks: [], titleHeadingRemoved: false }
  }

  const parts = runtimeCell
    ? [
        notebookCellFrameOpen(runtimeCell.id),
        ...notebookCellControls(runtimeCell),
        notebookCellActions(runtimeCell),
        notebookSourceEditor(runtimeCell.id),
      ]
    : []
  if (source.trim()) parts.push(fenced(source, language))
  const outputs = Array.isArray(cell.outputs) ? cell.outputs.filter(isRecord) : []
  let renderedOutputCount = 0
  for (const output of outputs) {
    const renderedOutputs = notebookOutput(output)
    renderedOutputCount += renderedOutputs.length
    parts.push(...renderedOutputs)
  }
  if (runtimeCell && renderedOutputCount === 0) {
    parts.push(renderNotebookRuntimeOutput({ type: 'success' }))
  }
  if (runtimeCell) {
    parts.push(notebookCellRuntimeOutput(runtimeCell.id))
    parts.push('</div>')
  }
  return { chunks: parts, titleHeadingRemoved: false }
}

function attachmentDataUrl(attachment: unknown): string | undefined {
  if (!isRecord(attachment)) return undefined
  for (const mime of attachmentMimeTypes) {
    const value = asText(attachment[mime])
    const data = mime === 'image/svg+xml' ? encodeURIComponent(value) : value.replace(/\s/g, '')
    if (!data) continue
    return mime === 'image/svg+xml' ? `data:${mime},${data}` : `data:${mime};base64,${data}`
  }
}

function resolveMarkdownAttachments(source: string, attachments: unknown): string {
  if (!isRecord(attachments) || !source.includes('attachment:')) return source
  return source.replace(/attachment:([^\s"'<>)]*)/g, (match, rawName: string) => {
    let name = rawName
    try {
      name = decodeURIComponent(rawName)
    } catch {}
    const dataUrl = attachmentDataUrl(attachments[name])
    return dataUrl ?? match
  })
}

function isNotebookRelativeAssetUrl(value: string): boolean {
  const trimmed = value.trim()
  if (!trimmed || trimmed.startsWith('#') || trimmed.startsWith('/')) return false
  if (/^[A-Za-z][A-Za-z0-9+.-]*:/.test(trimmed)) return false
  return true
}

function resolveNotebookImageUrl(value: string, sourcePath: string): string {
  if (!isNotebookRelativeAssetUrl(value)) return value

  const match = value.match(/^([^?#]*)([?#].*)?$/)
  const pathname = match?.[1] ?? value
  const suffix = match?.[2] ?? ''
  const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(sourcePath), pathname))
  if (resolved.startsWith('../')) return value
  return `${resolved}${suffix}`
}

function resolveNotebookImagePaths(source: string, sourcePath: string): string {
  const withHtmlImages = source.replace(
    htmlImageSrcPattern,
    (match, prefix, double, single, bare) => {
      const value = double ?? single ?? bare
      if (typeof value !== 'string') return match
      const resolved = resolveNotebookImageUrl(value, sourcePath)
      if (double !== undefined) return `${prefix}"${resolved}"`
      if (single !== undefined) return `${prefix}'${resolved}'`
      return `${prefix}${resolved}`
    },
  )

  return withHtmlImages.replace(markdownImagePattern, (match, prefix, spacing, target, suffix) => {
    const bracketed = target.startsWith('<') && target.endsWith('>')
    const value = bracketed ? target.slice(1, -1) : target
    const resolved = resolveNotebookImageUrl(value, sourcePath)
    if (resolved === value) return match
    return `${prefix}${spacing}${bracketed ? `<${resolved}>` : resolved}${suffix}`
  })
}

function separateStandaloneHtmlImageLines(source: string): string {
  const lines = source.split(/\r?\n/)
  const result: string[] = []
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    result.push(line)
    const next = lines[index + 1]
    if (standaloneHtmlImageLinePattern.test(line) && next !== undefined && next.trim().length > 0) {
      result.push('')
    }
  }
  return result.join('\n')
}

function closeDanglingMarkdownFence(source: string): string {
  let open: { marker: string; length: number } | undefined
  for (const line of source.split(/\r?\n/)) {
    const match = line.match(markdownFenceLinePattern)
    if (!match) continue
    const fence = match[2]
    const marker = fence[0]
    if (open) {
      if (marker === open.marker && fence.length >= open.length && match[3].trim() === '') {
        open = undefined
      }
      continue
    }
    open = { marker, length: fence.length }
  }
  if (!open) return source
  return `${source}${source.endsWith('\n') ? '' : '\n'}${open.marker.repeat(open.length)}`
}

export function notebookToMarkdown(
  notebook: NotebookDocument,
  sourcePath: string,
  options: NotebookMarkdownOptions = {},
): string {
  return `${notebookToMarkdownChunks(notebook, sourcePath, options).join('\n\n')}\n`
}

export function notebookToMarkdownChunks(
  notebook: NotebookDocument,
  sourcePath: string,
  options: NotebookMarkdownOptions = {},
): string[] {
  const language = languageName(notebook)
  const title = notebookTitle(notebook, sourcePath)
  let titleHeading: string | undefined = title
  const chunks = [frontmatter(title)]
  const runtime =
    options.runtime === false || options.runtime === undefined
      ? undefined
      : notebookRuntimeData(notebook, sourcePath, options.runtime)
  const runtimeCells = new Map(runtime?.cells.map(cell => [cell.id, cell]) ?? [])
  let runtimeIndex = 0

  if (runtime) {
    chunks.push(...notebookRuntimeControls(runtime))
    chunks.push(notebookRuntimeDataScript(runtime))
  }

  for (const cell of notebook.cells) {
    const runtimeCell =
      cell.cell_type === 'code' ? runtimeCells.get(`cell-${(runtimeIndex += 1)}`) : undefined
    const rendered = notebookCell(cell, language, sourcePath, runtimeCell, titleHeading)
    chunks.push(...rendered.chunks)
    if (rendered.titleHeadingRemoved) {
      titleHeading = undefined
    }
  }

  return chunks.filter(chunk => chunk.trim())
}
