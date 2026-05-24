import { escapeHTML } from './escape'
import { isRecord } from './type-guards'

export type NotebookRuntimeCell = {
  id: string
  source: string
  language: string
  executionIndex: number | null
}

export type NotebookRuntimeConfig = {
  enabled?: boolean
  pyodideIndexUrl?: string
  sourcePath?: string
}

export type NotebookRuntimeData = {
  id: string
  sourcePath: string
  language: string
  pyodideIndexUrl: string
  cells: NotebookRuntimeCell[]
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

export type NotebookRuntimeOutput =
  | NotebookRuntimeStreamOutput
  | NotebookRuntimeErrorOutput
  | NotebookRuntimeTextOutput
  | NotebookRuntimeJsonOutput
  | NotebookRuntimeHtmlOutput

export const defaultNotebookPyodideIndexUrl = 'https://cdn.jsdelivr.net/pyodide/v0.29.4/full/'

export function notebookRuntimeLocalSourceKey(sourcePath: string, cellId: string): string {
  return `quartz:notebook-source:${encodeURIComponent(sourcePath)}:${encodeURIComponent(cellId)}`
}

export function notebookRuntimeId(sourcePath: string): string {
  let hash = 2166136261
  for (let i = 0; i < sourcePath.length; i += 1) {
    hash ^= sourcePath.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return `notebook-runtime-${(hash >>> 0).toString(36)}`
}

export function notebookRuntimeJson(value: NotebookRuntimeData): string {
  return JSON.stringify(value)
    .replace(/</g, '\\u003c')
    .replace(/>/g, '\\u003e')
    .replace(/&/g, '\\u0026')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029')
}

export function notebookRuntimeControls(data: NotebookRuntimeData): string[] {
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
    ].join('\n'),
  ]
}

export function notebookRuntimeDataScript(data: NotebookRuntimeData): string {
  return `<script type="application/json" data-notebook-runtime-data>${notebookRuntimeJson(data)}</script>`
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

export function notebookCellControls(cell: NotebookRuntimeCell): string[] {
  const cellId = cell.id
  const escaped = escapeHTML(cellId)
  return [
    `<div class="notebook-runtime-cell" data-notebook-cell="${escaped}" data-notebook-execution-count="${cell.executionIndex ?? ''}"><span class="notebook-execution-prompt" data-notebook-execution-label="${escaped}" aria-live="polite">${escapeHTML(
      notebookExecutionLabel(cell.executionIndex),
    )}</span></div>`,
  ]
}

export function notebookCellFrameOpen(cellId: string): string {
  const escaped = escapeHTML(cellId)
  return `<div class="notebook-code-cell" data-notebook-cell-frame="${escaped}">`
}

export function notebookCellActions(cell: NotebookRuntimeCell): string {
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

export function notebookSourceEditor(cellId: string): string {
  return `<div class="notebook-source-editor" data-notebook-source-editor="${escapeHTML(
    cellId,
  )}" hidden></div>`
}

export function notebookCellRuntimeOutput(cellId: string): string {
  return `<div class="notebook-runtime-output" data-notebook-output="${escapeHTML(cellId)}" hidden></div>`
}

const browserRuntimeExtensionDirectives = new Set(['autoreload', 'nb_mypy'])
const wat2wasmFeatureOptions = new Set([
  '--enable-annotations',
  '--enable-bulk-memory',
  '--enable-code-metadata',
  '--enable-exceptions',
  '--enable-extended-const',
  '--enable-function-references',
  '--enable-gc',
  '--enable-memory64',
  '--enable-multi-memory',
  '--enable-multi-value',
  '--enable-mutable-globals',
  '--enable-reference-types',
  '--enable-relaxed-simd',
  '--enable-saturating-float-to-int',
  '--enable-sign-extension',
  '--enable-simd',
  '--enable-tail-call',
  '--enable-threads',
])
const notebookLsOptionChars = new Set(['1', 'a', 'h', 'l'])

function notebookShellWords(value: string): string[] {
  const words: string[] = []
  let word = ''
  let quote = ''
  let escaped = false
  for (const char of value) {
    if (escaped) {
      word += char
      escaped = false
      continue
    }
    if (char === '\\') {
      escaped = true
      continue
    }
    if (quote.length > 0) {
      if (char === quote) {
        quote = ''
      } else {
        word += char
      }
      continue
    }
    if (char === '"' || char === "'") {
      quote = char
      continue
    }
    if (/\s/.test(char)) {
      if (word.length > 0) {
        words.push(word)
        word = ''
      }
      continue
    }
    word += char
  }
  if (escaped) word += '\\'
  if (word.length > 0) words.push(word)
  return words
}

function notebookSandboxPathReason(
  command: string,
  path: string,
  allowDot = false,
): string | undefined {
  const normalized = path.trim().replace(/\\/g, '/')
  if (normalized.length === 0 || normalized.includes('\0')) {
    return `${command} path is unavailable in the browser runtime sandbox`
  }
  if (normalized.startsWith('/')) {
    return `${command} path ${path} is outside the browser runtime sandbox`
  }
  const parts = normalized.split('/')
  if (parts.some(part => part === '..')) {
    return `${command} path ${path} is outside the browser runtime sandbox`
  }
  if (!allowDot && (normalized === '.' || parts.every(part => part === '' || part === '.'))) {
    return `${command} path ${path} is unavailable in the browser runtime sandbox`
  }
}

function sourceText(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(sourceText).join('')
  if (value === undefined || value === null) return ''
  return String(value)
}

export function notebookRuntimeImportCandidates(source: string): string[] {
  const names = new Set<string>()
  for (const line of source.split(/\r?\n/)) {
    const withoutComment = line.replace(/#.*/, '')
    for (const importMatch of withoutComment.matchAll(/(?:^|[;:])\s*import\s+([^;]+)/g)) {
      for (const part of importMatch[1].split(',')) {
        const name = part
          .trim()
          .split(/\s+|\./)[0]
          ?.replace(/\W+$/, '')
        if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) names.add(name)
      }
    }
    for (const fromMatch of withoutComment.matchAll(
      /(?:^|[;:])\s*from\s+([A-Za-z_][A-Za-z0-9_]*)\b/g,
    )) {
      names.add(fromMatch[1])
    }
  }
  names.delete('import_ipynb')
  names.delete('jax')
  names.delete('nbimporter')
  names.delete('torch')
  return [...names]
}

export function notebookRuntimeModuleSource(raw: string, sourcePath: string): string {
  const parsed: unknown = JSON.parse(raw)
  if (!isRecord(parsed) || !Array.isArray(parsed.cells)) {
    throw new Error(`${sourcePath} is not a valid notebook`)
  }

  const cells = parsed.cells.filter(isRecord)
  return cells
    .filter(cell => cell.cell_type === 'code')
    .map(cell => sourceText(cell.source).trim())
    .filter(source => source.length > 0)
    .join('\n\n')
}

function classToken(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, '-') || 'output'
}

function preOutput(classes: string[], label: string, value: string): string {
  return `<pre class="${classes.join(' ')}" data-output-name="${escapeHTML(label)}"><samp>${escapeHTML(
    value.replace(/\s+$/, ''),
  )}</samp></pre>`
}

function debugOutputText(debug: NotebookRuntimeDebugOutput): string {
  return [
    ['phase', debug.phase],
    ['cell', debug.cellId],
    ['error', debug.errorName],
    ['message', debug.errorMessage],
    ['stack', debug.stack],
  ]
    .filter((entry): entry is [string, string] => entry[1] !== undefined && entry[1].length > 0)
    .map(([key, value]) => `${key}: ${value}`)
    .join('\n')
}

function notebookExtensionDirective(line: string): string | undefined {
  const match = line.match(/^%?(?:load_ext|reload_ext)\s+([A-Za-z_][A-Za-z0-9_.]*)\s*$/)
  return match?.[1]?.toLowerCase()
}

function browserRuntimeDirectiveReason(line: string): string | undefined {
  const extension = notebookExtensionDirective(line)
  if (extension !== undefined) {
    return browserRuntimeExtensionDirectives.has(extension)
      ? undefined
      : `IPython extension ${extension} is unavailable in the browser runtime`
  }
  if (/^%autoreload(?:\s|$)/.test(line)) return undefined
  if (/^%matplotlib(?:\s|$)/.test(line)) return undefined
}

function notebookWriteFileDirectiveReason(source: string): { handled: boolean; reason?: string } {
  const first = source.split(/\r?\n/, 1)[0]?.trim() ?? ''
  if (!first.startsWith('%%writefile')) return { handled: false }
  const words = first
    .replace(/^%%writefile\b/, '')
    .trim()
    .split(/\s+/)
    .filter(Boolean)
  let filename = ''
  for (const word of words) {
    if (word === '-a') continue
    if (word.startsWith('-')) {
      return {
        handled: true,
        reason: `%%writefile option ${word} is unavailable in the browser runtime`,
      }
    }
    if (filename) return { handled: true, reason: '%%writefile accepts one file name' }
    filename = word
  }
  if (!filename) return { handled: true, reason: '%%writefile requires a file name' }
  return { handled: true, reason: notebookSandboxPathReason('%%writefile', filename) }
}

function notebookShellDirectiveReason(line: string): { handled: boolean; reason?: string } {
  if (!line.startsWith('!')) return { handled: false }
  const command = line.slice(1).trim()
  if (/^cat(?:\s|$)/.test(command)) {
    const words = notebookShellWords(command)
    if (words.length === 1) return { handled: true, reason: 'cat requires a file' }
    for (let index = 1; index < words.length; index += 1) {
      const word = words[index]
      if (word.startsWith('-')) {
        return { handled: true, reason: 'cat options are unavailable in the browser runtime' }
      }
      const pathReason = notebookSandboxPathReason('cat', word)
      if (pathReason) return { handled: true, reason: pathReason }
    }
    return { handled: true }
  }
  if (/^ls(?:\s|$)/.test(command)) {
    const words = notebookShellWords(command)
    for (let index = 1; index < words.length; index += 1) {
      const word = words[index]
      if (word.startsWith('-')) {
        if (
          word === '-' ||
          Array.from(word.slice(1)).some(char => !notebookLsOptionChars.has(char))
        ) {
          return {
            handled: true,
            reason: `ls option ${word} is unavailable in the browser runtime`,
          }
        }
        continue
      }
      const pathReason = notebookSandboxPathReason('ls', word, true)
      if (pathReason) return { handled: true, reason: pathReason }
    }
    return { handled: true }
  }
  if (/^(?:wat2wasm|wasm2wat)(?:\s|$)/.test(command)) {
    const name = command.split(/\s+/, 1)[0] ?? 'wabt'
    const words = notebookShellWords(command)
    let input = ''
    for (let index = 1; index < words.length; index += 1) {
      const word = words[index]
      if (word === '-o' || word === '--output') {
        index += 1
        if (!words[index]) return { handled: true, reason: `${word} requires a file name` }
        const pathReason = notebookSandboxPathReason(name, words[index])
        if (pathReason) return { handled: true, reason: pathReason }
        continue
      }
      if (word.startsWith('-o') && word.length > 2) {
        const pathReason = notebookSandboxPathReason(name, word.slice(2))
        if (pathReason) return { handled: true, reason: pathReason }
        continue
      }
      if (word.startsWith('--output=')) {
        const pathReason = notebookSandboxPathReason(name, word.slice('--output='.length))
        if (pathReason) return { handled: true, reason: pathReason }
        continue
      }
      if (word.startsWith('--enable-')) {
        if (wat2wasmFeatureOptions.has(word)) continue
        return {
          handled: true,
          reason: `${name} option ${word} is unavailable in the browser runtime`,
        }
      }
      if (word.startsWith('--disable-')) {
        const enabled = `--enable-${word.slice('--disable-'.length)}`
        if (wat2wasmFeatureOptions.has(enabled)) continue
        return {
          handled: true,
          reason: `${name} option ${word} is unavailable in the browser runtime`,
        }
      }
      if (name === 'wasm2wat' && (word === '--fold-exprs' || word === '--inline-exports')) continue
      if (word.startsWith('-')) {
        return {
          handled: true,
          reason: `${name} option ${word} is unavailable in the browser runtime`,
        }
      }
      if (input) return { handled: true, reason: `${name} accepts one input file` }
      const pathReason = notebookSandboxPathReason(name, word)
      if (pathReason) return { handled: true, reason: pathReason }
      input = word
    }
    return input ? { handled: true } : { handled: true, reason: `${name} requires an input file` }
  }
  return { handled: true, reason: 'shell escapes are unavailable in the browser runtime' }
}

export function renderNotebookRuntimeOutput(
  output: NotebookRuntimeOutput,
  options: { debug?: boolean } = {},
): string {
  if (output.type === 'stream') {
    return preOutput(
      [
        'notebook-output',
        'notebook-output-stream',
        `notebook-output-stream-${classToken(output.name)}`,
      ],
      output.name,
      output.text,
    )
  }
  if (output.type === 'error') {
    const text = output.traceback || [output.ename, output.evalue].filter(Boolean).join(': ')
    const rendered = [preOutput(['notebook-output', 'notebook-output-error'], 'error', text)]
    if (options.debug === true && output.debug) {
      rendered.push(
        preOutput(
          ['notebook-output', 'notebook-output-debug'],
          'debug',
          debugOutputText(output.debug),
        ),
      )
    }
    return rendered.join('\n')
  }
  if (output.type === 'html') {
    return `<div class="notebook-output notebook-output-html" data-output-name="display">${output.html}</div>`
  }
  if (output.type === 'json') {
    return preOutput(
      ['notebook-output', 'notebook-output-text', 'notebook-output-json'],
      'json',
      output.text,
    )
  }
  return preOutput(['notebook-output', 'notebook-output-text'], 'result', output.text)
}

export function unsupportedNotebookRuntimeReason(source: string): string | undefined {
  const writeFile = notebookWriteFileDirectiveReason(source)
  if (writeFile.handled) return writeFile.reason
  for (const line of source.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (trimmed.length === 0) continue
    if (/^%pip\s+install(?:\s|$)/.test(trimmed)) continue
    if (/^%timeit(?:\s|$)/.test(trimmed)) continue
    if (/^%time(?:\s|$)/.test(trimmed)) continue
    if (/^!(?:pip|uv\s+pip|python3?\s+-m\s+pip)\s+install(?:\s|$)/.test(trimmed)) continue
    const directiveReason = browserRuntimeDirectiveReason(trimmed)
    if (directiveReason !== undefined) return directiveReason
    if (notebookExtensionDirective(trimmed) !== undefined) continue
    if (/^%autoreload(?:\s|$)/.test(trimmed)) continue
    if (/^%matplotlib(?:\s|$)/.test(trimmed)) continue
    const shell = notebookShellDirectiveReason(trimmed)
    if (shell.handled) return shell.reason
    if (trimmed.startsWith('%%')) return 'cell magics are unavailable in the browser runtime'
    if (trimmed.startsWith('%')) return 'IPython magics are unavailable in the browser runtime'
    if (trimmed.startsWith('!')) return 'shell escapes are unavailable in the browser runtime'
  }
}
