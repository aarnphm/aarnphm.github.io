import { escapeHTML } from './escape'

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

export type NotebookRuntimeHtmlOutput = { type: 'html'; html: string }

export type NotebookRuntimeOutput =
  | NotebookRuntimeStreamOutput
  | NotebookRuntimeErrorOutput
  | NotebookRuntimeTextOutput
  | NotebookRuntimeHtmlOutput

export function notebookRuntimeLocalSourceKey(sourcePath: string, cellId: string): string {
  return `quartz:notebook-source:${encodeURIComponent(sourcePath)}:${encodeURIComponent(cellId)}`
}

const browserRuntimeExtensionDirectives = new Set(['autoreload', 'nb_mypy'])

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
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
  return preOutput(['notebook-output', 'notebook-output-text'], 'result', output.text)
}

export function unsupportedNotebookRuntimeReason(source: string): string | undefined {
  for (const line of source.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (trimmed.length === 0) continue
    if (/^%pip\s+install(?:\s|$)/.test(trimmed)) continue
    if (/^%timeit(?:\s|$)/.test(trimmed)) continue
    if (/^!(?:pip|uv\s+pip|python3?\s+-m\s+pip)\s+install(?:\s|$)/.test(trimmed)) continue
    const directiveReason = browserRuntimeDirectiveReason(trimmed)
    if (directiveReason !== undefined) return directiveReason
    if (notebookExtensionDirective(trimmed) !== undefined) continue
    if (/^%autoreload(?:\s|$)/.test(trimmed)) continue
    if (/^%matplotlib(?:\s|$)/.test(trimmed)) continue
    if (trimmed.startsWith('%%')) return 'cell magics are unavailable in the browser runtime'
    if (trimmed.startsWith('%')) return 'IPython magics are unavailable in the browser runtime'
    if (trimmed.startsWith('!')) return 'shell escapes are unavailable in the browser runtime'
  }
}
