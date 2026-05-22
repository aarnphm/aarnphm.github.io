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

type NotebookCell = JsonRecord & {
  cell_type?: unknown
  source?: unknown
  attachments?: unknown
  outputs?: unknown
  execution_count?: unknown
}

export type NotebookDocument = JsonRecord & { cells: NotebookCell[]; metadata?: JsonRecord }

export type NotebookRuntimeConfig = {
  enabled?: boolean
  pyodideIndexUrl?: string
  sourcePath?: string
}

export type NotebookRuntimeCell = {
  id: string
  source: string
  language: string
  executionIndex: number | null
}

export type NotebookRuntimeData = {
  id: string
  sourcePath: string
  language: string
  pyodideIndexUrl: string
  cells: NotebookRuntimeCell[]
}

type NotebookMarkdownOptions = { runtime?: false | NotebookRuntimeConfig }

const defaultPyodideIndexUrl = 'https://cdn.jsdelivr.net/pyodide/v0.29.4/full/'

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

const ipythonDisplayObjectPattern = /^<IPython\.core\.display\.[A-Za-z0-9_]+ object>$/
const attachmentMimeTypes = ['image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/svg+xml']
const htmlImageSrcPattern = /(<img\b[^>]*\bsrc\s*=\s*)(?:"([^"]*)"|'([^']*)'|([^\s>]+))/gi
const markdownImagePattern = /(!\[[^\]]*\]\()(\s*)([^)\s]+)([^)]*\))/g
const standaloneHtmlImageLinePattern = /^\s*<img\b[^>\n]*>(?:\s*<\/img>)?\s*$/i

function languageName(notebook: NotebookDocument): string {
  const metadata = isRecord(notebook.metadata) ? notebook.metadata : {}
  const languageInfo = isRecord(metadata.language_info) ? metadata.language_info : {}
  const name = typeof languageInfo.name === 'string' ? languageInfo.name : 'python'
  return name.replace(/[^A-Za-z0-9_+#.-]/g, '') || 'python'
}

function notebookRuntimeId(sourcePath: string): string {
  let hash = 2166136261
  for (let i = 0; i < sourcePath.length; i += 1) {
    hash ^= sourcePath.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return `notebook-runtime-${(hash >>> 0).toString(36)}`
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

function runtimeJson(value: NotebookRuntimeData): string {
  return JSON.stringify(value)
    .replace(/</g, '\\u003c')
    .replace(/>/g, '\\u003e')
    .replace(/&/g, '\\u0026')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029')
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
    pyodideIndexUrl: runtime.pyodideIndexUrl ?? defaultPyodideIndexUrl,
    cells,
  }
}

function notebookRuntimeControls(data: NotebookRuntimeData): string[] {
  return [
    [
      `<div class="notebook-runtime" data-notebook-runtime="${escapeHTML(data.id)}">`,
      '<div class="notebook-runtime-toolbar" data-notebook-runtime-toolbar role="toolbar" aria-label="Notebook runtime">',
      '<button type="button" data-notebook-run-all>Run all</button>',
      '<button type="button" data-notebook-stop disabled>Stop</button>',
      '<button type="button" data-notebook-reset>Reset runtime</button>',
      '<button type="button" data-notebook-debug aria-pressed="false">Debug</button>',
      '<button type="button" data-notebook-vim-mode aria-pressed="false">Vim</button>',
      '<span class="notebook-runtime-status" data-notebook-status aria-live="polite">idle</span>',
      '</div>',
      '</div>',
      `<script type="application/json" data-notebook-runtime-data>${runtimeJson(data)}</script>`,
    ].join('\n'),
  ]
}

type NotebookIcon = 'run' | 'edit' | 'save' | 'revert'

const notebookIconSvg: Record<NotebookIcon, string> = {
  run: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 5.5v13l10-6.5z"/></svg>',
  edit: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m4 16.5-.5 4 4-.5L19 8.5 15.5 5z"/><path d="m14 6.5 3.5 3.5"/></svg>',
  save: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M5 4h11l3 3v13H5z"/><path d="M8 4v6h8V4"/><path d="M8 20v-6h8v6"/></svg>',
  revert:
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/></svg>',
}

function notebookExecutionLabel(count: number | null): string {
  return count === null ? 'In [ ]:' : `In [${count}]:`
}

function notebookIconButton(
  cellId: string,
  icon: NotebookIcon,
  dataAttribute: string,
  label: string,
  hidden = false,
): string {
  const escapedId = escapeHTML(cellId)
  const escapedLabel = escapeHTML(label)
  return `<button type="button" class="notebook-icon-button" ${dataAttribute}="${escapedId}" aria-label="${escapedLabel}" title="${escapedLabel}"${
    hidden ? ' hidden' : ''
  }>${notebookIconSvg[icon]}</button>`
}

function notebookCellControls(cell: NotebookRuntimeCell): string[] {
  const cellId = cell.id
  const escaped = escapeHTML(cellId)
  return [
    `<div class="notebook-runtime-cell" data-notebook-cell="${escaped}" data-notebook-execution-count="${cell.executionIndex ?? ''}">`,
    `<span class="notebook-execution-prompt" data-notebook-execution-label="${escaped}" aria-live="polite">${escapeHTML(
      notebookExecutionLabel(cell.executionIndex),
    )}</span>`,
    '</div>',
  ]
}

function notebookCellFrameOpen(cellId: string): string {
  const escaped = escapeHTML(cellId)
  return `<div class="notebook-code-cell" data-notebook-cell-frame="${escaped}">`
}

function notebookCellActions(cell: NotebookRuntimeCell): string {
  const escaped = escapeHTML(cell.id)
  return [
    `<div class="notebook-cell-actions" data-notebook-cell-actions="${escaped}">`,
    notebookIconButton(cell.id, 'run', 'data-notebook-run-cell', `Run ${cell.id}`),
    notebookIconButton(cell.id, 'edit', 'data-notebook-edit-cell', `Edit ${cell.id}`),
    notebookIconButton(cell.id, 'save', 'data-notebook-save-cell', `Save ${cell.id} locally`, true),
    notebookIconButton(
      cell.id,
      'revert',
      'data-notebook-revert-cell',
      `Revert ${cell.id} local edit`,
      true,
    ),
    `<span class="notebook-local-source-status" data-notebook-local-source-status="${escaped}" hidden></span>`,
    '</div>',
  ].join('\n')
}

function notebookSourceEditor(cellId: string): string {
  return `<div class="notebook-source-editor" data-notebook-source-editor="${escapeHTML(
    cellId,
  )}" hidden></div>`
}

function notebookCellRuntimeOutput(cellId: string): string {
  return `<div class="notebook-runtime-output" data-notebook-output="${escapeHTML(cellId)}" hidden></div>`
}

function notebookMarkdownCellBoundary(): string {
  return '<div class="notebook-markdown-cell-boundary" aria-hidden="true"></div>'
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
  for (const output of outputs) {
    parts.push(...notebookOutput(output))
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

export function notebookToMarkdown(
  notebook: NotebookDocument,
  sourcePath: string,
  options: NotebookMarkdownOptions = {},
): string {
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

  return `${chunks.filter(chunk => chunk.trim()).join('\n\n')}\n`
}
