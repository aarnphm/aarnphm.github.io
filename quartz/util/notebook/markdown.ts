import path from 'path'
import type {
  NotebookRuntimeCell,
  NotebookRuntimeConfig,
  NotebookRuntimeData,
} from '../../runtime/notebook/types'
import '../../runtime/notebook/registry'
import type { Cell, CodeCell, MarkdownCell, NotebookDoc, Output } from './types'
import { backendFor } from '../../runtime/notebook/backend'
import { isRecord } from '../type-guards'
import {
  notebookCellActions,
  notebookCellControls,
  notebookCellFrameOpen,
  notebookCellRuntimeOutput,
  notebookRuntimeControls,
  notebookRuntimeDataScript,
  notebookSourceEditor,
  notebookStaticCellActions,
} from './cell-html'
import { codeCellId, notebookId } from './identity'
import { renderOutputHtml, renderSuccessMarkerHtml } from './render/output-to-hast'

export type NotebookMarkdownOptions = { runtime?: false | NotebookRuntimeConfig }

const writefileMagicPattern = /^%%(?:writefile|file)\b(.*)$/i
const cellMagicPattern = /^%%([A-Za-z_][A-Za-z0-9_-]*)\b/
const captureCellMagics = new Set(['capture'])
const languageCellMagics = new Map<string, string>([
  ['bash', 'bash'],
  ['script', 'bash'],
  ['sh', 'bash'],
  ['javascript', 'javascript'],
  ['js', 'javascript'],
  ['html', 'html'],
  ['sql', 'sql'],
])
const runtimeCellMagics = new Map<string, string>([
  ['javascript', 'javascript'],
  ['js', 'javascript'],
])
const filenameLanguageExtensions = new Map<string, string>([
  ['bash', 'bash'],
  ['c', 'c'],
  ['cc', 'cpp'],
  ['cpp', 'cpp'],
  ['csv', 'csv'],
  ['cxx', 'cpp'],
  ['go', 'go'],
  ['h', 'c'],
  ['hpp', 'cpp'],
  ['html', 'html'],
  ['java', 'java'],
  ['js', 'javascript'],
  ['json', 'json'],
  ['jsx', 'jsx'],
  ['l', 'c'],
  ['md', 'markdown'],
  ['ml', 'ocaml'],
  ['mli', 'ocaml'],
  ['py', 'python'],
  ['rs', 'rust'],
  ['sh', 'bash'],
  ['ts', 'typescript'],
  ['tsx', 'tsx'],
  ['txt', 'text'],
  ['wat', 'wasm'],
  ['zig', 'zig'],
])

const htmlImageSrcPattern = /(<img\b[^>]*\bsrc\s*=\s*)(?:"([^"]*)"|'([^']*)'|([^\s>]+))/gi
const markdownImagePattern = /(!\[[^\]]*\]\()(\s*)([^)\s]+)([^)]*\))/g
const standaloneHtmlImageLinePattern = /^\s*<img\b[^>\n]*>(?:\s*<\/img>)?\s*$/i
const htmlTagPattern = /<\/?([A-Za-z][A-Za-z0-9:-]*)\b[^>]*>/g
const markdownFenceLinePattern = /^( {0,3})(`{3,}|~{3,})(.*)$/
const htmlDivOpenTagPattern = /<div\b[^>]*>/gi
const htmlDivCloseTagPattern = /<\/div\s*>/gi
const htmlFloatStylePattern =
  /\bstyle\s*=\s*(?:"[^"]*\bfloat\s*:\s*(?:left|right)[^"]*"|'[^']*\bfloat\s*:\s*(?:left|right)[^']*'|[^\s>]*\bfloat\s*:\s*(?:left|right)[^\s>]*)/i

const attachmentMimeTypes = ['image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/svg+xml']
const htmlVoidTags = new Set([
  'area',
  'base',
  'br',
  'col',
  'embed',
  'hr',
  'img',
  'input',
  'link',
  'meta',
  'param',
  'source',
  'track',
  'wbr',
])

function maxBacktickRun(value: string): number {
  return Math.max(0, ...Array.from(value.matchAll(/`+/g), match => match[0].length))
}

function fenced(value: string, language = 'text'): string {
  const fence = '`'.repeat(Math.max(3, maxBacktickRun(value) + 1))
  return `${fence}${language}\n${value.replace(/\s+$/, '')}\n${fence}`
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

export function notebookTitle(doc: NotebookDoc, sourcePath: string): string {
  for (const cell of doc.cells) {
    if (cell.cellType === 'markdown') {
      const title = titleFromMarkdown(cell.source)
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

function attachmentDataUrl(attachment: unknown): string | undefined {
  if (!isRecord(attachment)) return undefined
  for (const mime of attachmentMimeTypes) {
    const raw = attachment[mime]
    const value = typeof raw === 'string' ? raw : Array.isArray(raw) ? raw.join('') : ''
    const data = mime === 'image/svg+xml' ? encodeURIComponent(value) : value.replace(/\s/g, '')
    if (!data) continue
    return mime === 'image/svg+xml' ? `data:${mime},${data}` : `data:${mime};base64,${data}`
  }
}

function resolveMarkdownAttachments(
  source: string,
  attachments: Readonly<Record<string, unknown>>,
): string {
  if (!source.includes('attachment:')) return source
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

function updateHtmlContainerStack(stack: string[], line: string): void {
  for (const match of line.matchAll(htmlTagPattern)) {
    const raw = match[0]
    const tag = match[1].toLowerCase()
    if (htmlVoidTags.has(tag) || /\/\s*>$/.test(raw)) continue
    if (raw.startsWith('</')) {
      const index = stack.lastIndexOf(tag)
      if (index >= 0) stack.splice(index)
      continue
    }
    stack.push(tag)
  }
}

function separateStandaloneHtmlImageLines(source: string): string {
  const lines = source.split(/\r?\n/)
  const result: string[] = []
  const htmlContainerStack: string[] = []
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    const insideHtmlContainer = htmlContainerStack.length > 0
    result.push(line)
    const next = lines[index + 1]
    if (
      !insideHtmlContainer &&
      standaloneHtmlImageLinePattern.test(line) &&
      next !== undefined &&
      next.trim().length > 0
    ) {
      result.push('')
    }
    updateHtmlContainerStack(htmlContainerStack, line)
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

function notebookMarkdownCellBoundary(): string {
  return '<div class="notebook-markdown-cell-boundary" aria-hidden="true"></div>'
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

function notebookCellMetadataLanguage(
  metadata: Readonly<Record<string, unknown>>,
): string | undefined {
  const vscode = isRecord(metadata.vscode) ? metadata.vscode : {}
  const language = metadata.language ?? metadata.languageId ?? vscode.languageId
  return typeof language === 'string' && language.trim() ? language.trim() : undefined
}

function filenameLanguage(filename: string): string | undefined {
  const normalized = filename.replace(/\\/g, '/').split('/').filter(Boolean).at(-1)?.toLowerCase()
  const extension = normalized?.match(/\.([a-z0-9+#-]+)$/)?.[1]
  return extension ? filenameLanguageExtensions.get(extension) : undefined
}

function writefileLanguage(line: string): string | undefined {
  const match = line.trim().match(writefileMagicPattern)
  if (!match) return undefined
  for (const part of match[1].trim().split(/\s+/)) {
    if (!part || part.startsWith('-')) continue
    return filenameLanguage(part.replace(/^["']|["']$/g, ''))
  }
}

function cellMagicLanguage(source: string): string | undefined {
  for (const line of source.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const writefile = writefileLanguage(trimmed)
    if (writefile) return writefile
    const magic = trimmed.match(cellMagicPattern)?.[1]?.toLowerCase()
    if (magic === undefined) return undefined
    const language = languageCellMagics.get(magic)
    if (language) return language
    if (!captureCellMagics.has(magic)) return undefined
  }
}

function cellMagicRuntimeLanguage(source: string): string | undefined {
  for (const line of source.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    const magic = trimmed.match(cellMagicPattern)?.[1]?.toLowerCase()
    if (magic === undefined) return undefined
    const language = runtimeCellMagics.get(magic)
    if (language) return language
    if (!captureCellMagics.has(magic)) return undefined
  }
}

function notebookCellDisplayLanguage(
  metadata: Readonly<Record<string, unknown>>,
  source: string,
  language: string,
): string {
  return cellMagicLanguage(source) ?? notebookCellMetadataLanguage(metadata) ?? language
}

function notebookCellRuntimeBackend(cell: CodeCell, notebookLanguage: string) {
  const runtimeLanguage = cellMagicRuntimeLanguage(cell.source)
  if (runtimeLanguage !== undefined) {
    const backend = backendFor(runtimeLanguage)
    if (backend) return backend
  }
  return backendFor(notebookLanguage)
}

export function notebookSupportsRuntime(doc: NotebookDoc): boolean {
  return backendFor(doc.language) !== undefined
}

export function notebookRuntimeData(
  doc: NotebookDoc,
  sourcePath: string,
  runtime: NotebookRuntimeConfig,
): NotebookRuntimeData | undefined {
  if (runtime.enabled === false) return undefined
  const backend = backendFor(doc.language)
  if (!backend) return undefined
  const cells: NotebookRuntimeCell[] = []
  let codeIndex = 0
  for (const cell of doc.cells) {
    if (cell.cellType !== 'code') continue
    codeIndex += 1
    const cellBackend = notebookCellRuntimeBackend(cell, doc.language)
    if (!cellBackend) continue
    const displayLanguage = notebookCellDisplayLanguage(cell.metadata, cell.source, doc.language)
    const displayLanguageField = displayLanguage === cellBackend.name ? {} : { displayLanguage }
    cells.push({
      id: codeCellId(codeIndex),
      source: cell.source,
      language: cellBackend.name,
      ...displayLanguageField,
      executionIndex: cell.executionCount,
    })
  }
  if (cells.length === 0) return undefined
  const targetSourcePath = runtime.sourcePath ?? sourcePath
  const indexUrl = runtime.indexUrl ?? backend.defaultIndexUrl
  const data: NotebookRuntimeData = {
    id: notebookId(targetSourcePath),
    sourcePath: targetSourcePath,
    language: backend.name,
    cells,
  }
  if (indexUrl !== undefined) data.indexUrl = indexUrl
  if (runtime.importableModules !== undefined) {
    data.importableModules = [...new Set(runtime.importableModules)].sort()
  }
  return data
}

function outputToString(output: Output, raw?: Readonly<Record<string, unknown>>): string[] {
  // Pass through `text/markdown` if it's the only payload — the markdown processor handles it directly.
  if (output.kind === 'display_data' || output.kind === 'execute_result') {
    const data = output.data
    const markdown = data['text/markdown']
    const markdownText =
      typeof markdown === 'string' ? markdown : Array.isArray(markdown) ? markdown.join('') : ''
    if (markdownText.trim()) {
      const otherKeys = Object.keys(data).filter(key => key !== 'text/markdown')
      const otherHasContent = otherKeys.some(key => {
        const value = data[key]
        const text = typeof value === 'string' ? value : Array.isArray(value) ? value.join('') : ''
        return text.trim().length > 0
      })
      if (!otherHasContent) {
        // Reparse from raw if present (preserves original whitespace); fall back to bundled value.
        if (raw && typeof raw['text/markdown'] === 'string') {
          return [raw['text/markdown'] as string]
        }
        return [markdownText]
      }
    }
  }
  const html = renderOutputHtml(output)
  return html ? [html] : []
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

function renderMarkdownCell(
  cell: MarkdownCell,
  sourcePath: string,
  titleHeading: string | undefined,
): { chunks: string[]; titleHeadingRemoved: boolean } {
  let resolved = cell.source
  if (Object.keys(cell.attachments).length > 0) {
    resolved = resolveMarkdownAttachments(resolved, cell.attachments)
  }
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

function rawOutputDataFor(
  doc: NotebookDoc,
  cellIndex: number,
  outputIndex: number,
): Readonly<Record<string, unknown>> | undefined {
  const rawCells = Array.isArray(doc.raw.cells) ? doc.raw.cells : []
  const rawCell = rawCells[cellIndex]
  if (!isRecord(rawCell)) return undefined
  const outputs = Array.isArray(rawCell.outputs) ? rawCell.outputs : []
  const rawOutput = outputs[outputIndex]
  if (!isRecord(rawOutput) || !isRecord(rawOutput.data)) return undefined
  return rawOutput.data as Readonly<Record<string, unknown>>
}

function renderCodeCell(
  cell: CodeCell,
  doc: NotebookDoc,
  cellIndex: number,
  language: string,
  runtimeCell: NotebookRuntimeCell | undefined,
): string[] {
  const displayLanguage = notebookCellDisplayLanguage(cell.metadata, cell.source, language)
  const parts = runtimeCell
    ? [
        notebookCellFrameOpen(runtimeCell.id, displayLanguage),
        ...notebookCellControls(runtimeCell),
        notebookCellActions(runtimeCell),
        notebookSourceEditor(runtimeCell.id),
      ]
    : [
        notebookCellFrameOpen(cell.id, displayLanguage),
        notebookStaticCellActions(cell.id, displayLanguage),
      ]
  if (cell.source.trim()) parts.push(fenced(cell.source, displayLanguage))
  let renderedOutputCount = 0
  cell.outputs.forEach((output, outputIndex) => {
    const rawData = rawOutputDataFor(doc, cellIndex, outputIndex)
    const rendered = outputToString(output, rawData)
    renderedOutputCount += rendered.length
    parts.push(...rendered)
  })
  if (runtimeCell && renderedOutputCount === 0) parts.push(renderSuccessMarkerHtml())
  if (runtimeCell) parts.push(notebookCellRuntimeOutput(runtimeCell.id))
  parts.push('</div>')
  return parts
}

function renderCell(
  cell: Cell,
  doc: NotebookDoc,
  cellIndex: number,
  language: string,
  sourcePath: string,
  runtimeCell: NotebookRuntimeCell | undefined,
  titleHeading: string | undefined,
): { chunks: string[]; titleHeadingRemoved: boolean } {
  if (cell.cellType === 'markdown') return renderMarkdownCell(cell, sourcePath, titleHeading)
  if (cell.cellType === 'code') {
    return {
      chunks: renderCodeCell(cell, doc, cellIndex, language, runtimeCell),
      titleHeadingRemoved: false,
    }
  }
  return { chunks: [], titleHeadingRemoved: false }
}

export function notebookToMarkdownChunks(
  doc: NotebookDoc,
  sourcePath: string,
  options: NotebookMarkdownOptions = {},
): string[] {
  const title = notebookTitle(doc, sourcePath)
  let titleHeading: string | undefined = title
  const chunks = [frontmatter(title)]
  const runtime =
    options.runtime === false || options.runtime === undefined
      ? undefined
      : notebookRuntimeData(doc, sourcePath, options.runtime)
  const runtimeCells = new Map(runtime?.cells.map(cell => [cell.id, cell]) ?? [])
  if (runtime) {
    chunks.push(...notebookRuntimeControls(runtime))
    chunks.push(notebookRuntimeDataScript(runtime))
  }
  let codeIndex = 0
  doc.cells.forEach((cell, cellIndex) => {
    const matchingRuntimeCell =
      cell.cellType === 'code' ? runtimeCells.get(codeCellId((codeIndex += 1))) : undefined
    const rendered = renderCell(
      cell,
      doc,
      cellIndex,
      doc.language,
      sourcePath,
      matchingRuntimeCell,
      titleHeading,
    )
    chunks.push(...rendered.chunks)
    if (rendered.titleHeadingRemoved) titleHeading = undefined
  })
  return chunks.filter(chunk => chunk.trim())
}

export function notebookToMarkdown(
  doc: NotebookDoc,
  sourcePath: string,
  options: NotebookMarkdownOptions = {},
): string {
  return `${notebookToMarkdownChunks(doc, sourcePath, options).join('\n\n')}\n`
}
