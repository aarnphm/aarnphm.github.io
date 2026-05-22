import type { NotebookRuntimeOutput } from '../../util/notebook-runtime'
import {
  notebookRuntimeImportCandidates,
  notebookRuntimeLocalSourceKey,
  notebookRuntimeModuleSource,
  renderNotebookRuntimeOutput,
  unsupportedNotebookRuntimeReason,
} from '../../util/notebook-runtime'
import { MarkdownEditor } from './markdown-editor'

type RuntimeCell = { id: string; source: string; language: string; executionIndex: number | null }

type RuntimeModule = { name: string; sourcePath: string; source: string }

type SourceControls = {
  frame: HTMLElement
  editor: MarkdownEditor
  editorHost: HTMLElement
  figure: HTMLElement | undefined
  status: HTMLElement
  editButton: HTMLButtonElement
  saveButton: HTMLButtonElement
  revertButton: HTMLButtonElement
}

type RuntimePayload = {
  id: string
  sourcePath: string
  language: string
  pyodideIndexUrl: string
  cells: RuntimeCell[]
}

type AssetResult = {
  runtimeId: string
  assetId: string
  ok: boolean
  status: number
  statusText: string
  contentType: string
  bytes?: ArrayBuffer
  error?: string
}

type FrameMessage =
  | { type: 'ready'; runtimeId: string }
  | { type: 'output'; runtimeId: string; cellId: string; output: NotebookRuntimeOutput }
  | { type: 'done'; runtimeId: string; cellId: string }
  | { type: 'asset'; runtimeId: string; cellId: string; assetId: string; url: string }

type CellWaiter = { resolve: () => void; reject: (error: Error) => void }

type NotebookIcon = 'run' | 'edit' | 'save' | 'revert'

const notebookIconSvg: Record<NotebookIcon, string> = {
  run: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 5.5v13l10-6.5z"/></svg>',
  edit: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m4 16.5-.5 4 4-.5L19 8.5 15.5 5z"/><path d="m14 6.5 3.5 3.5"/></svg>',
  save: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M5 4h11l3 3v13H5z"/><path d="M8 4v6h8V4"/><path d="M8 20v-6h8v6"/></svg>',
  revert:
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/></svg>',
}

function setNotebookIconButton(button: HTMLButtonElement, icon: NotebookIcon, label: string) {
  button.classList.add('notebook-icon-button')
  button.setAttribute('aria-label', label)
  button.title = label
  button.textContent = ''
  button.insertAdjacentHTML('afterbegin', notebookIconSvg[icon])
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function readString(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key]
  return typeof value === 'string' ? value : undefined
}

function readRuntimeOutput(value: unknown): NotebookRuntimeOutput | undefined {
  if (!isRecord(value)) return undefined
  const type = readString(value, 'type')
  if (type === 'stream') {
    const name = readString(value, 'name')
    const text = readString(value, 'text')
    if (name !== undefined && text !== undefined) return { type, name, text }
  }
  if (type === 'error') {
    const ename = readString(value, 'ename')
    const evalue = readString(value, 'evalue')
    const traceback = readString(value, 'traceback')
    if (ename !== undefined && evalue !== undefined && traceback !== undefined) {
      return { type, ename, evalue, traceback }
    }
  }
  if (type === 'text') {
    const text = readString(value, 'text')
    if (text !== undefined) return { type, text }
  }
  if (type === 'html') {
    const html = readString(value, 'html')
    if (html !== undefined) return { type, html }
  }
}

function readFrameMessage(value: unknown): FrameMessage | undefined {
  if (!isRecord(value) || value.source !== 'quartz-notebook-runtime') return undefined
  const type = readString(value, 'type')
  const runtimeId = readString(value, 'runtimeId')
  if (!type || !runtimeId) return undefined
  if (type === 'ready') return { type, runtimeId }
  if (type === 'done') {
    const cellId = readString(value, 'cellId')
    if (cellId) return { type, runtimeId, cellId }
  }
  if (type === 'output') {
    const cellId = readString(value, 'cellId')
    const output = readRuntimeOutput(value.output)
    if (cellId && output) return { type, runtimeId, cellId, output }
  }
  if (type === 'asset') {
    const cellId = readString(value, 'cellId')
    const assetId = readString(value, 'assetId')
    const url = readString(value, 'url')
    if (cellId && assetId && url) return { type, runtimeId, cellId, assetId, url }
  }
}

function readRuntimeCell(value: unknown): RuntimeCell | undefined {
  if (!isRecord(value)) return undefined
  const id = readString(value, 'id')
  const source = readString(value, 'source')
  const language = readString(value, 'language')
  const executionIndex = value.executionIndex
  if (!id || source === undefined || !language) return undefined
  if (typeof executionIndex === 'number' || executionIndex === null) {
    return { id, source, language, executionIndex }
  }
}

function readRuntimePayload(value: unknown): RuntimePayload | undefined {
  if (!isRecord(value)) return undefined
  const id = readString(value, 'id')
  const sourcePath = readString(value, 'sourcePath')
  const language = readString(value, 'language')
  const pyodideIndexUrl = readString(value, 'pyodideIndexUrl')
  if (!id || !sourcePath || !language || !pyodideIndexUrl || !Array.isArray(value.cells)) {
    return undefined
  }
  const cells = value.cells.map(readRuntimeCell)
  if (cells.some(cell => cell === undefined)) return undefined
  return {
    id,
    sourcePath,
    language,
    pyodideIndexUrl,
    cells: cells.filter(cell => cell !== undefined),
  }
}

function parseRuntimeJson(text: string): unknown | undefined {
  try {
    return JSON.parse(text)
  } catch {}

  const textarea = document.createElement('textarea')
  textarea.innerHTML = text
  if (textarea.value === text) return undefined

  try {
    return JSON.parse(textarea.value)
  } catch {
    return undefined
  }
}

function sanitizeHtml(html: string): string {
  const template = document.createElement('template')
  template.innerHTML = html
  const blocked = template.content.querySelectorAll(
    'script, iframe, object, embed, link, meta, base',
  )
  blocked.forEach(node => node.remove())
  const elements = template.content.querySelectorAll('*')
  elements.forEach(element => {
    const attributes = Array.from(element.attributes)
    for (const attr of attributes) {
      const name = attr.name.toLowerCase()
      const value = attr.value.trim().toLowerCase()
      if (name.startsWith('on')) element.removeAttribute(attr.name)
      if (
        (name === 'href' || name === 'src' || name === 'xlink:href') &&
        value.startsWith('javascript:')
      ) {
        element.removeAttribute(attr.name)
      }
    }
  })
  return template.innerHTML
}

function frameSource(): string {
  const closeScript = `${String.fromCharCode(60)}/script>`
  const source = String.raw`
<!doctype html>
<html>
<head><meta charset="utf-8"></head>
<body>
<script>
const source = 'quartz-notebook-runtime'
let runtimeId = ''
let pyodide
let currentCellId = ''
let assetSequence = 0
const pendingAssets = new Map()
const streamBuffers = new Map()
const stdoutDecoder = new TextDecoder()
const stderrDecoder = new TextDecoder()
function post(message, transfer) {
  parent.postMessage({ source, runtimeId, ...message }, '*', transfer || [])
}
function textOf(value) {
  if (value === undefined || value === null) return ''
  if (typeof value === 'string') return value
  try {
    return String(value)
  } catch {
    return Object.prototype.toString.call(value)
  }
}
function emitOutput(output) {
  if (!currentCellId) return
  emitOutputForCell(currentCellId, output)
}
function emitOutputForCell(cellId, output) {
  if (!cellId) return
  post({ type: 'output', cellId, output })
}
function appendStream(payload) {
  appendStreamForCell(currentCellId, payload)
}
function appendStreamForCell(cellId, payload) {
  if (!payload) return
  emitOutputForCell(cellId, { type: 'stream', name: textOf(payload.name || 'stdout'), text: textOf(payload.text) })
}
function bufferStreamForCell(cellId, name, text) {
  if (!cellId || !text) return
  const key = cellId + '\u0000' + name
  streamBuffers.set(key, (streamBuffers.get(key) || '') + text)
}
function bufferStreamBytesForCell(cellId, name, bytes, decoder) {
  if (!bytes) return 0
  const text = decoder.decode(bytes, { stream: true })
  bufferStreamForCell(cellId, name, text)
  return bytes.length
}
function flushStreamDecoderForCell(cellId, name, decoder) {
  bufferStreamForCell(cellId, name, decoder.decode())
}
function flushStreamsForCell(cellId) {
  flushStreamDecoderForCell(cellId, 'stdout', stdoutDecoder)
  flushStreamDecoderForCell(cellId, 'stderr', stderrDecoder)
  for (const [key, text] of [...streamBuffers.entries()]) {
    const [owner, name] = key.split('\u0000')
    if (owner !== cellId) continue
    streamBuffers.delete(key)
    if (text) emitOutputForCell(cellId, { type: 'stream', name, text })
  }
}
function emitError(error) {
  const text = textOf(error)
  emitOutput({ type: 'error', ename: error && error.name ? textOf(error.name) : 'Error', evalue: text, traceback: text })
}
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script')
    script.src = src
    script.onload = () => resolve()
    script.onerror = () => reject(new Error('failed to load ' + src))
    document.head.appendChild(script)
  })
}
function sanitizeDisplayElement(element) {
  element.querySelectorAll('script, iframe, object, embed, link, meta, base').forEach(node => node.remove())
  element.querySelectorAll('*').forEach(node => {
    for (const attr of [...node.attributes]) {
      const name = attr.name.toLowerCase()
      const value = attr.value.trim().toLowerCase()
      if (name.startsWith('on')) node.removeAttribute(attr.name)
      if ((name === 'href' || name === 'src' || name === 'xlink:href') && value.startsWith('javascript:')) {
        node.removeAttribute(attr.name)
      }
    }
  })
}
function sandboxFetch(input, cellId) {
  const url = typeof input === 'string' ? input : input && input.url
  if (typeof url !== 'string') return Promise.reject(new Error('unsupported fetch input'))
  const assetId = 'asset-' + ++assetSequence
  post({ type: 'asset', cellId, assetId, url })
  return new Promise((resolve, reject) => {
    pendingAssets.set(assetId, { resolve, reject, cellId })
  }).catch(error => {
    emitOutputForCell(cellId, {
      type: 'error',
      ename: error && error.name ? textOf(error.name) : 'AssetError',
      evalue: textOf(error),
      traceback: textOf(error),
    })
    throw error
  })
}
async function executeDisplayJavascript(code) {
  const cellId = currentCellId
  const element = document.createElement('div')
  const append = payload => appendStreamForCell(cellId, payload)
  const fetchAsset = input => sandboxFetch(input, cellId)
  const context = { append_stream: append }
  const runner = new Function('element', 'append_stream', 'fetch', 'return (async function () {' + code + '\n}).call(this)')
  await runner.call(context, element, append, fetchAsset)
  sanitizeDisplayElement(element)
  if (element.innerHTML.trim()) emitOutputForCell(cellId, { type: 'html', html: element.innerHTML })
}
function handleDisplayPayload(serialized) {
  let data
  try {
    data = JSON.parse(serialized)
  } catch (error) {
    emitError(error)
    return
  }
  if (data['text/html']) {
    emitOutput({ type: 'html', html: textOf(data['text/html']) })
  } else if (data['application/javascript']) {
    executeDisplayJavascript(textOf(data['application/javascript'])).catch(emitError)
  } else if (data['text/plain']) {
    emitOutput({ type: 'text', text: textOf(data['text/plain']) })
  }
}
function unsupportedReason(code) {
  for (const line of code.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed) continue
    if (trimmed.startsWith('%%')) return 'cell magics are unavailable in the browser runtime'
    if (trimmed.startsWith('%')) return 'IPython magics are unavailable in the browser runtime'
    if (trimmed.startsWith('!')) return 'shell escapes are unavailable in the browser runtime'
  }
}
async function ensurePyodide(indexURL) {
  if (pyodide) return pyodide
  await loadScript(indexURL.replace(/\/?$/, '/') + 'pyodide.js')
  if (typeof loadPyodide !== 'function') throw new Error('loadPyodide was not installed')
  pyodide = await loadPyodide({ indexURL })
  pyodide.setStdout({
    write: bytes => bufferStreamBytesForCell(currentCellId, 'stdout', bytes, stdoutDecoder),
  })
  pyodide.setStderr({
    write: bytes => bufferStreamBytesForCell(currentCellId, 'stderr', bytes, stderrDecoder),
  })
  globalThis.quartz_notebook_display = handleDisplayPayload
  pyodide.runPython([
    'import ast',
    'import json',
    'import sys',
    'import types',
    'import importlib.abc',
    'import importlib.util',
    'import js',
    '',
    '_quartz_notebook_modules = {}',
    '',
    'class _QuartzNotebookLoader(importlib.abc.Loader):',
    '    def create_module(self, spec):',
    '        return None',
    '    def exec_module(self, module):',
    '        source, filename = _quartz_notebook_modules[module.__name__]',
    '        module.__file__ = filename',
    '        exec(compile(source, filename, "exec"), module.__dict__)',
    '',
    'class _QuartzNotebookFinder(importlib.abc.MetaPathFinder):',
    '    def find_spec(self, fullname, path=None, target=None):',
    '        if fullname in _quartz_notebook_modules:',
    '            return importlib.util.spec_from_loader(fullname, _QuartzNotebookLoader(), origin=_quartz_notebook_modules[fullname][1])',
    '        return None',
    '',
    'def __quartz_register_notebook_module(name, source, filename):',
    '    _quartz_notebook_modules[name] = (source, filename)',
    '',
    'if not any(isinstance(finder, _QuartzNotebookFinder) for finder in sys.meta_path):',
    '    sys.meta_path.insert(0, _QuartzNotebookFinder())',
    '',
    'class _QuartzDisplayObject:',
    '    def __init__(self, mime, data):',
    '        self.mime = mime',
    '        self.data = data',
    '    def _repr_mimebundle_(self, include=None, exclude=None):',
    '        return ({self.mime: self.data, "text/plain": repr(self)}, {})',
    '',
    'class Javascript(_QuartzDisplayObject):',
    '    def __init__(self, data):',
    '        super().__init__("application/javascript", data)',
    '    def __repr__(self):',
    '        return "<IPython.core.display.Javascript object>"',
    '',
    'class HTML(_QuartzDisplayObject):',
    '    def __init__(self, data):',
    '        super().__init__("text/html", data)',
    '    def __repr__(self):',
    '        return "<IPython.core.display.HTML object>"',
    '',
    'def display(*objects):',
    '    for obj in objects:',
    '        if hasattr(obj, "_repr_mimebundle_"):',
    '            data, _metadata = obj._repr_mimebundle_()',
    '            js.quartz_notebook_display(json.dumps(data))',
    '        else:',
    '            js.quartz_notebook_display(json.dumps({"text/plain": str(obj)}))',
    '',
    'display_module = types.ModuleType("IPython.display")',
    'display_module.display = display',
    'display_module.Javascript = Javascript',
    'display_module.HTML = HTML',
    'ipython_module = types.ModuleType("IPython")',
    'ipython_module.display = display_module',
    'sys.modules["IPython"] = ipython_module',
    'sys.modules["IPython.display"] = display_module',
    'sys.modules["import_ipynb"] = types.ModuleType("import_ipynb")',
    'nbimporter_module = types.ModuleType("nbimporter")',
    'nbimporter_module.options = {"only_defs": False}',
    'sys.modules["nbimporter"] = nbimporter_module',
    '',
    'def __quartz_run_cell(source):',
    '    tree = ast.parse(source, mode="exec")',
    '    if len(tree.body) > 0 and isinstance(tree.body[-1], ast.Expr):',
    '        expr = ast.Expression(tree.body.pop().value)',
    '        ast.fix_missing_locations(tree)',
    '        ast.fix_missing_locations(expr)',
    '        exec(compile(tree, "<notebook-cell>", "exec"), globals())',
    '        value = eval(compile(expr, "<notebook-cell>", "eval"), globals())',
    '        if value is not None:',
    '            display(value)',
    '    else:',
    '        exec(compile(tree, "<notebook-cell>", "exec"), globals())',
  ].join('\n'))
  return pyodide
}
async function runCell(message) {
  currentCellId = message.cellId
  const reason = unsupportedReason(message.code)
  if (reason) {
    emitOutput({ type: 'error', ename: 'UnsupportedRuntimeFeature', evalue: reason, traceback: reason })
    post({ type: 'done', cellId: message.cellId })
    currentCellId = ''
    return
  }
  try {
    const runtime = await ensurePyodide(message.pyodideIndexUrl)
    if (Array.isArray(message.modules)) {
      const register = runtime.globals.get('__quartz_register_notebook_module')
      if (!register || typeof register !== 'function') throw new Error('runtime notebook importer is unavailable')
      for (const module of message.modules) {
        if (!module || typeof module.name !== 'string' || typeof module.source !== 'string' || typeof module.sourcePath !== 'string') continue
        register(module.name, module.source, module.sourcePath)
      }
      if (register && 'destroy' in register && typeof register.destroy === 'function') {
        register.destroy()
      }
    }
    const runner = runtime.globals.get('__quartz_run_cell')
    if (!runner || typeof runner !== 'function') throw new Error('runtime cell runner is unavailable')
    const result = await runner(message.code)
    if (result && typeof result === 'object' && 'destroy' in result && typeof result.destroy === 'function') {
      result.destroy()
    }
  } catch (error) {
    flushStreamsForCell(message.cellId)
    emitError(error)
  } finally {
    flushStreamsForCell(message.cellId)
    post({ type: 'done', cellId: message.cellId })
    currentCellId = ''
  }
}
window.addEventListener('message', event => {
  const message = event.data
  if (!message || message.source !== source) return
  if (message.type === 'init') {
    runtimeId = message.runtimeId
    post({ type: 'ready' })
  } else if (message.type === 'run' && message.runtimeId === runtimeId) {
    runCell(message)
  } else if (message.type === 'asset-result' && message.runtimeId === runtimeId) {
    const pending = pendingAssets.get(message.assetId)
    if (!pending) return
    pendingAssets.delete(message.assetId)
    if (!message.ok) {
      pending.reject(new Error(message.error || 'failed to fetch notebook asset'))
      return
    }
    pending.resolve(new Response(message.bytes, {
      status: message.status,
      statusText: message.statusText,
      headers: { 'content-type': message.contentType || 'application/octet-stream' },
    }))
  }
})
${closeScript}
</body>
</html>`
  return source
}

class NotebookRuntime {
  private payload: RuntimePayload
  private root: HTMLElement
  private iframe: HTMLIFrameElement | undefined
  private ready: Promise<void> | undefined
  private readyResolve: (() => void) | undefined
  private waiters = new Map<string, CellWaiter>()
  private moduleCache = new Map<string, RuntimeModule | null>()
  private moduleFetches = new Map<string, Promise<RuntimeModule | null>>()
  private savedOutputs = new Map<string, HTMLElement[]>()
  private sourceControls = new Map<string, SourceControls>()
  private executionCounter: number
  private running = false
  private stopped = false

  constructor(root: HTMLElement, payload: RuntimePayload) {
    this.root = root
    this.payload = payload
    this.executionCounter = Math.max(0, ...payload.cells.map(cell => cell.executionIndex ?? 0))
  }

  mount() {
    this.decorateCells()
    const runAll = this.root.querySelector<HTMLButtonElement>('[data-notebook-run-all]')
    const stop = this.root.querySelector<HTMLButtonElement>('[data-notebook-stop]')
    const reset = this.root.querySelector<HTMLButtonElement>('[data-notebook-reset]')
    runAll?.addEventListener('click', this.runAll)
    stop?.addEventListener('click', this.stop)
    reset?.addEventListener('click', this.reset)
    window.addEventListener('message', this.onMessage)
    window.addCleanup(() => {
      runAll?.removeEventListener('click', this.runAll)
      stop?.removeEventListener('click', this.stop)
      reset?.removeEventListener('click', this.reset)
      window.removeEventListener('message', this.onMessage)
      this.destroyFrame()
    })
    for (const cell of this.payload.cells) {
      const button = document.querySelector<HTMLButtonElement>(
        `[data-notebook-run-cell="${CSS.escape(cell.id)}"]`,
      )
      if (!button) continue
      const handler = () => {
        this.runCell(cell)
      }
      button.addEventListener('click', handler)
      window.addCleanup(() => button.removeEventListener('click', handler))
    }
  }

  private runAll = async () => {
    if (this.running) return
    this.stopped = false
    for (const cell of this.payload.cells) {
      if (this.stopped) break
      await this.runCell(cell)
    }
  }

  private runCell = async (cell: RuntimeCell) => {
    if (this.running) return
    const source = this.sourceForCell(cell)
    const executionCount = this.nextExecutionCount(cell.id)
    this.running = true
    this.setStatus(`running ${cell.id}`)
    this.setExecutionLabel(cell.id, '*')
    this.setRunningControls(true)
    this.clearOutput(cell.id)
    this.hideSavedOutput(cell.id)
    const unsupported = unsupportedNotebookRuntimeReason(source)
    if (unsupported) {
      this.renderOutput(cell.id, {
        type: 'error',
        ename: 'UnsupportedRuntimeFeature',
        evalue: unsupported,
        traceback: unsupported,
      })
      this.running = false
      this.setExecutionLabel(cell.id, executionCount)
      this.setRunningControls(false)
      this.setStatus('idle')
      return
    }
    try {
      await this.ensureFrame()
      await this.postRun(cell, source)
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      this.renderOutput(cell.id, {
        type: 'error',
        ename: 'RuntimeError',
        evalue: text,
        traceback: text,
      })
    } finally {
      this.running = false
      this.setExecutionLabel(cell.id, executionCount)
      this.setRunningControls(false)
      if (!this.stopped) this.setStatus('idle')
    }
  }

  private stop = () => {
    this.stopped = true
    this.destroyFrame()
    this.setRunningControls(false)
    this.setStatus('stopped')
    for (const waiter of this.waiters.values()) {
      waiter.reject(new Error('runtime stopped'))
    }
    this.waiters.clear()
  }

  private reset = () => {
    this.stopped = false
    this.destroyFrame()
    for (const waiter of this.waiters.values()) {
      waiter.reject(new Error('runtime reset'))
    }
    this.waiters.clear()
    this.running = false
    this.payload.cells.forEach(cell => {
      this.clearOutput(cell.id)
      this.restoreSavedOutput(cell.id)
      this.setExecutionLabel(cell.id, cell.executionIndex)
    })
    this.executionCounter = Math.max(0, ...this.payload.cells.map(cell => cell.executionIndex ?? 0))
    this.setRunningControls(false)
    this.setStatus('idle')
  }

  private decorateCells() {
    for (const cell of this.payload.cells) {
      this.setExecutionLabel(cell.id, cell.executionIndex)
      const controls = document.querySelector<HTMLElement>(
        `[data-notebook-cell="${CSS.escape(cell.id)}"]`,
      )
      if (!controls || controls.closest('[data-notebook-cell-frame]')) continue
      const frame = document.createElement('div')
      frame.className = 'notebook-code-cell'
      frame.dataset.notebookCellFrame = cell.id
      controls.before(frame)
      frame.append(controls)
      let sibling = frame.nextElementSibling
      while (sibling instanceof HTMLElement) {
        if (sibling.matches('[data-notebook-cell]')) break
        if (
          sibling.matches(
            'figure[data-rehype-pretty-code-figure], .notebook-output, [data-notebook-output]',
          )
        ) {
          const next = sibling.nextElementSibling
          frame.append(sibling)
          sibling = next
          continue
        }
        break
      }
      this.decorateSourceEditor(cell, frame, controls)
    }
  }

  private decorateSourceEditor(cell: RuntimeCell, frame: HTMLElement, controls: HTMLElement) {
    if (this.sourceControls.has(cell.id)) return
    const figure =
      frame.querySelector<HTMLElement>('figure[data-rehype-pretty-code-figure]') ?? undefined
    const actions = document.createElement('div')
    actions.className = 'notebook-cell-actions'
    actions.dataset.notebookCellActions = cell.id

    const runButton = controls.querySelector<HTMLButtonElement>(
      `[data-notebook-run-cell="${CSS.escape(cell.id)}"]`,
    )
    if (runButton) {
      setNotebookIconButton(runButton, 'run', `Run ${cell.id}`)
      actions.append(runButton)
    }

    const editorHost = document.createElement('div')
    editorHost.className = 'notebook-source-editor'
    editorHost.dataset.notebookSourceEditor = cell.id
    editorHost.hidden = true

    const editor = new MarkdownEditor({
      parent: editorHost,
      initialContent: cell.source,
      mode: 'code',
      language: cell.language,
      lineWrapping: false,
      onChange: () => this.syncSourceControls(cell),
      onSubmit: () => this.runCell(cell),
      onCancel: () => this.revertEditorSource(cell),
    })

    const editButton = document.createElement('button')
    editButton.type = 'button'
    editButton.dataset.notebookEditCell = cell.id
    setNotebookIconButton(editButton, 'edit', `Edit ${cell.id}`)
    const editSource = () => this.showSourceEditor(cell, true)
    editButton.addEventListener('click', editSource)

    const saveButton = document.createElement('button')
    saveButton.type = 'button'
    saveButton.dataset.notebookSaveCell = cell.id
    setNotebookIconButton(saveButton, 'save', `Save ${cell.id} locally`)
    saveButton.hidden = true
    const saveSource = () => this.saveEditorSource(cell)
    saveButton.addEventListener('click', saveSource)

    const revertButton = document.createElement('button')
    revertButton.type = 'button'
    revertButton.dataset.notebookRevertCell = cell.id
    setNotebookIconButton(revertButton, 'revert', `Revert ${cell.id} local edit`)
    revertButton.hidden = true
    const revertSource = () => this.revertEditorSource(cell)
    revertButton.addEventListener('click', revertSource)

    const status = document.createElement('span')
    status.className = 'notebook-local-source-status'
    status.dataset.notebookLocalSourceStatus = cell.id
    status.hidden = true

    actions.append(editButton, saveButton, revertButton, status)
    frame.append(actions)
    if (figure) {
      figure.before(editorHost)
    } else {
      frame.append(editorHost)
    }
    this.sourceControls.set(cell.id, {
      frame,
      editor,
      editorHost,
      figure,
      status,
      editButton,
      saveButton,
      revertButton,
    })

    const stored = this.readStoredSource(cell)
    if (stored !== undefined && stored !== cell.source) {
      editor.setValue(stored)
      this.showSourceEditor(cell, true)
    } else if (stored !== undefined) {
      this.clearStoredSource(cell)
    }
    this.syncSourceControls(cell)

    window.addCleanup(() => {
      editButton.removeEventListener('click', editSource)
      saveButton.removeEventListener('click', saveSource)
      revertButton.removeEventListener('click', revertSource)
      editor.destroy()
    })
  }

  private sourceForCell(cell: RuntimeCell): string {
    const controls = this.sourceControls.get(cell.id)
    if (controls && !controls.editorHost.hidden) return controls.editor.getValue()
    return this.readStoredSource(cell) ?? cell.source
  }

  private sourceStorageKey(cell: RuntimeCell): string {
    return notebookRuntimeLocalSourceKey(this.payload.sourcePath, cell.id)
  }

  private readStoredSource(cell: RuntimeCell): string | undefined {
    try {
      const value = window.localStorage.getItem(this.sourceStorageKey(cell))
      return value === null ? undefined : value
    } catch {
      return undefined
    }
  }

  private writeStoredSource(cell: RuntimeCell, source: string): boolean {
    try {
      window.localStorage.setItem(this.sourceStorageKey(cell), source)
      return true
    } catch {
      return false
    }
  }

  private clearStoredSource(cell: RuntimeCell) {
    try {
      window.localStorage.removeItem(this.sourceStorageKey(cell))
    } catch {}
  }

  private showSourceEditor(cell: RuntimeCell, visible: boolean) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    if (visible) controls.editor.setValue(this.sourceForCell(cell))
    controls.editorHost.hidden = !visible
    if (controls.figure) controls.figure.hidden = visible
    controls.frame.toggleAttribute('data-notebook-editing', visible)
    this.syncSourceControls(cell)
    if (visible) controls.editor.focus()
  }

  private saveEditorSource(cell: RuntimeCell) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    const source = controls.editor.getValue()
    if (source === cell.source) {
      this.clearStoredSource(cell)
      this.showSourceEditor(cell, false)
      this.setStatus('idle')
      return
    }
    if (!this.writeStoredSource(cell, source)) {
      this.setStatus('local save failed')
      return
    }
    this.showSourceEditor(cell, true)
    this.setStatus('saved local edit')
  }

  private revertEditorSource(cell: RuntimeCell) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    controls.editor.setValue(cell.source)
    this.clearStoredSource(cell)
    this.showSourceEditor(cell, false)
    this.setStatus('idle')
  }

  private syncSourceControls(cell: RuntimeCell) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    const stored = this.readStoredSource(cell)
    const editing = !controls.editorHost.hidden
    const edited = editing && controls.editor.getValue() !== (stored ?? cell.source)
    const hasStoredSource = stored !== undefined
    controls.status.hidden = !hasStoredSource && !edited
    controls.status.dataset.notebookSourceState = edited ? 'edited' : 'local'
    controls.status.title = edited ? 'edited local source' : 'saved local source'
    controls.status.setAttribute(
      'aria-label',
      edited ? 'edited local source' : 'saved local source',
    )
    controls.editButton.hidden = editing
    controls.saveButton.hidden = !editing
    controls.revertButton.hidden = !(editing || hasStoredSource)
    controls.frame.toggleAttribute('data-notebook-local-source', hasStoredSource)
  }

  private nextExecutionCount(cellId: string): number {
    this.executionCounter += 1
    const controls = document.querySelector<HTMLElement>(
      `[data-notebook-cell="${CSS.escape(cellId)}"]`,
    )
    if (controls) controls.dataset.notebookExecutionCount = String(this.executionCounter)
    return this.executionCounter
  }

  private setExecutionLabel(cellId: string, count: number | '*' | null) {
    const label = document.querySelector<HTMLElement>(
      `[data-notebook-execution-label="${CSS.escape(cellId)}"]`,
    )
    if (!label) return
    label.textContent = count === null ? 'In [ ]:' : `In [${count}]:`
  }

  private ensureFrame(): Promise<void> {
    if (this.ready) return this.ready
    this.ready = new Promise(resolve => {
      this.readyResolve = resolve
    })
    const iframe = document.createElement('iframe')
    iframe.className = 'notebook-runtime-frame'
    iframe.sandbox.add('allow-scripts', 'allow-modals')
    iframe.addEventListener('load', () => {
      iframe.contentWindow?.postMessage(
        { source: 'quartz-notebook-runtime', type: 'init', runtimeId: this.payload.id },
        '*',
      )
    })
    iframe.srcdoc = frameSource()
    this.root.appendChild(iframe)
    this.iframe = iframe
    return this.ready
  }

  private async postRun(cell: RuntimeCell, source: string): Promise<void> {
    const modules = await this.notebookModulesFor(source)
    return new Promise((resolve, reject) => {
      this.waiters.set(cell.id, { resolve, reject })
      this.iframe?.contentWindow?.postMessage(
        {
          source: 'quartz-notebook-runtime',
          type: 'run',
          runtimeId: this.payload.id,
          cellId: cell.id,
          code: source,
          pyodideIndexUrl: this.payload.pyodideIndexUrl,
          modules,
        },
        '*',
      )
    })
  }

  private async notebookModulesFor(source: string): Promise<RuntimeModule[]> {
    const seen = new Set<string>()
    const modules = new Map<string, RuntimeModule>()
    await this.collectNotebookModules(source, seen, modules)
    return [...modules.values()]
  }

  private async collectNotebookModules(
    source: string,
    seen: Set<string>,
    modules: Map<string, RuntimeModule>,
  ): Promise<void> {
    for (const name of notebookRuntimeImportCandidates(source)) {
      if (seen.has(name)) continue
      seen.add(name)
      const module = await this.fetchNotebookModule(name)
      if (!module) continue
      modules.set(name, module)
      await this.collectNotebookModules(module.source, seen, modules)
    }
  }

  private async fetchNotebookModule(name: string): Promise<RuntimeModule | null> {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) return null
    const cached = this.moduleCache.get(name)
    if (cached !== undefined) return cached
    const pending = this.moduleFetches.get(name)
    if (pending) return await pending
    const request = this.loadNotebookModule(name)
    this.moduleFetches.set(name, request)
    try {
      const module = await request
      this.moduleCache.set(name, module)
      return module
    } finally {
      this.moduleFetches.delete(name)
    }
  }

  private async loadNotebookModule(name: string): Promise<RuntimeModule | null> {
    const url = new URL(`${name}.ipynb`, window.location.href)
    const baseDir = new URL('.', window.location.href)
    if (url.origin !== window.location.origin || !url.pathname.startsWith(baseDir.pathname)) {
      throw new Error(`blocked non-sibling notebook import: ${name}`)
    }
    const relative = url.pathname.slice(baseDir.pathname.length)
    if (relative.includes('/')) {
      throw new Error(`blocked nested notebook import: ${name}`)
    }
    const response = await fetch(url)
    if (response.status === 404) return null
    if (!response.ok) {
      throw new Error(`failed to fetch notebook import ${name}.ipynb: ${response.status}`)
    }
    const source = notebookRuntimeModuleSource(await response.text(), `${name}.ipynb`)
    return { name, sourcePath: url.pathname, source }
  }

  private onMessage = (event: MessageEvent<unknown>) => {
    if (this.iframe?.contentWindow && event.source !== this.iframe.contentWindow) return
    const message = readFrameMessage(event.data)
    if (!message || message.runtimeId !== this.payload.id) return
    if (message.type === 'ready') {
      this.readyResolve?.()
      this.readyResolve = undefined
    } else if (message.type === 'output') {
      const output: NotebookRuntimeOutput =
        message.output.type === 'html'
          ? { type: 'html', html: sanitizeHtml(message.output.html) }
          : message.output
      this.renderOutput(message.cellId, output)
    } else if (message.type === 'done') {
      const waiter = this.waiters.get(message.cellId)
      if (!waiter) return
      this.waiters.delete(message.cellId)
      waiter.resolve()
    } else if (message.type === 'asset') {
      this.fetchAsset(message)
    }
  }

  private async fetchAsset(message: Extract<FrameMessage, { type: 'asset' }>) {
    const result = await this.resolveAsset(message)
    const target = this.iframe?.contentWindow
    if (!target) return
    const transfer = result.bytes ? [result.bytes] : []
    target.postMessage(
      { source: 'quartz-notebook-runtime', type: 'asset-result', ...result },
      '*',
      transfer,
    )
  }

  private async resolveAsset(
    message: Extract<FrameMessage, { type: 'asset' }>,
  ): Promise<AssetResult> {
    try {
      const base = new URL(window.location.href)
      const baseDir = new URL('.', base)
      const url = new URL(message.url, base)
      const relative = url.pathname.slice(baseDir.pathname.length)
      if (
        url.origin !== window.location.origin ||
        !url.pathname.startsWith(baseDir.pathname) ||
        relative.includes('/')
      ) {
        throw new Error(`blocked non-sibling notebook asset: ${message.url}`)
      }
      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`missing notebook asset: ${relative || message.url}`)
      }
      return {
        runtimeId: message.runtimeId,
        assetId: message.assetId,
        ok: true,
        status: response.status,
        statusText: response.statusText,
        contentType: response.headers.get('content-type') ?? 'application/octet-stream',
        bytes: await response.arrayBuffer(),
      }
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      return {
        runtimeId: message.runtimeId,
        assetId: message.assetId,
        ok: false,
        status: 404,
        statusText: 'Not Found',
        contentType: 'text/plain',
        error: text,
      }
    }
  }

  private renderOutput(cellId: string, output: NotebookRuntimeOutput) {
    const target = document.querySelector<HTMLElement>(
      `[data-notebook-output="${CSS.escape(cellId)}"]`,
    )
    if (!target) return
    target.hidden = false
    target.insertAdjacentHTML('beforeend', renderNotebookRuntimeOutput(output))
  }

  private clearOutput(cellId: string) {
    const target = document.querySelector<HTMLElement>(
      `[data-notebook-output="${CSS.escape(cellId)}"]`,
    )
    if (!target) return
    target.replaceChildren()
    target.hidden = true
  }

  private hideSavedOutput(cellId: string) {
    const existing = this.savedOutputs.get(cellId)
    if (existing) {
      existing.forEach(element => {
        element.hidden = true
      })
      return
    }
    const target = document.querySelector<HTMLElement>(
      `[data-notebook-output="${CSS.escape(cellId)}"]`,
    )
    if (!target) return
    const saved: HTMLElement[] = []
    let sibling = target.previousElementSibling
    while (sibling instanceof HTMLElement) {
      if (sibling.matches('figure, [data-notebook-cell]')) break
      if (sibling.classList.contains('notebook-output')) {
        saved.unshift(sibling)
      }
      sibling = sibling.previousElementSibling
    }
    if (saved.length === 0) return
    this.savedOutputs.set(cellId, saved)
    saved.forEach(element => {
      element.hidden = true
    })
  }

  private restoreSavedOutput(cellId: string) {
    const saved = this.savedOutputs.get(cellId)
    if (!saved) return
    saved.forEach(element => {
      element.hidden = false
    })
    this.savedOutputs.delete(cellId)
  }

  private setStatus(value: string) {
    const status = this.root.querySelector<HTMLElement>('[data-notebook-status]')
    if (status) status.textContent = value
  }

  private setRunningControls(running: boolean) {
    this.root
      .querySelector<HTMLButtonElement>('[data-notebook-stop]')
      ?.toggleAttribute('disabled', !running)
    this.root
      .querySelector<HTMLButtonElement>('[data-notebook-run-all]')
      ?.toggleAttribute('disabled', running)
    document
      .querySelectorAll<HTMLButtonElement>('[data-notebook-run-cell]')
      .forEach(button => button.toggleAttribute('disabled', running))
  }

  private destroyFrame() {
    this.iframe?.remove()
    this.iframe = undefined
    this.ready = undefined
    this.readyResolve = undefined
  }
}

function mountNotebookRuntime() {
  const root = document.querySelector<HTMLElement>('[data-notebook-runtime]')
  const data = document.querySelector<HTMLScriptElement>('script[data-notebook-runtime-data]')
  if (!root || !data?.textContent || root.dataset.runtimeMounted === 'true') return
  const parsed = parseRuntimeJson(data.textContent)
  if (parsed === undefined) return
  const payload = readRuntimePayload(parsed)
  if (!payload) return
  root.dataset.runtimeMounted = 'true'
  new NotebookRuntime(root, payload).mount()
}

function scheduleNotebookRuntimeMount() {
  const mount = () => {
    if (typeof window.addCleanup !== 'function') {
      window.setTimeout(mount, 0)
      return
    }
    mountNotebookRuntime()
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount, { once: true })
  } else {
    mount()
  }
}

if (!Reflect.get(window, '__quartzNotebookRuntimeRegistered')) {
  Reflect.set(window, '__quartzNotebookRuntimeRegistered', true)
  document.addEventListener('nav', scheduleNotebookRuntimeMount)
  scheduleNotebookRuntimeMount()
}

const notebookRuntimeInlineModule = ''
export default notebookRuntimeInlineModule
