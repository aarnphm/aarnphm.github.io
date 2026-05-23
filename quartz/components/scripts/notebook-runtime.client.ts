import type { NotebookRuntimeOutput } from '../../util/notebook-runtime'
import type { NotebookCodeEditor } from './notebook-code-editor'
import {
  notebookRuntimeImportCandidates,
  notebookRuntimeLocalSourceKey,
  notebookRuntimeModuleSource,
  renderNotebookRuntimeOutput,
  unsupportedNotebookRuntimeReason,
} from '../../util/notebook-runtime'
import notebookDisplayFrameSource from './notebook-runtime.frame.html'

type RuntimeCell = { id: string; source: string; language: string; executionIndex: number | null }

type RuntimeModule = { name: string; sourcePath: string; source: string }

type RuntimeErrorOutput = Extract<NotebookRuntimeOutput, { type: 'error' }>

type RuntimeDebugOutput = NonNullable<RuntimeErrorOutput['debug']>

type SourceControls = {
  frame: HTMLElement
  editor: NotebookCodeEditor | undefined
  editorHost: HTMLElement
  figure: HTMLElement | undefined
  status: HTMLElement
  editButton: HTMLButtonElement
  saveButton: HTMLButtonElement
  revertButton: HTMLButtonElement
  source: string
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
  | {
      type: 'file-result'
      runtimeId: string
      requestId: string
      ok: boolean
      status: number
      statusText: string
      contentType: string
      bytes?: ArrayBuffer
      error?: string
    }
  | { type: 'display-javascript'; runtimeId: string; cellId: string; code: string }
  | { type: 'status'; runtimeId: string; text: string }

type CellWaiter = { resolve: () => void; reject: (error: Error) => void }

type RuntimeFileWaiter = {
  resolve: (message: Extract<FrameMessage, { type: 'file-result' }>) => void
}

type NotebookIcon = 'run' | 'edit' | 'save' | 'revert'

const notebookRuntimeVimModeKey = 'quartz:notebook-vim-mode'

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

function readNumber(record: Record<string, unknown>, key: string): number | undefined {
  const value = record[key]
  return typeof value === 'number' ? value : undefined
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
      const debug = readRuntimeDebugOutput(value.debug)
      return debug ? { type, ename, evalue, traceback, debug } : { type, ename, evalue, traceback }
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

function readRuntimeDebugOutput(value: unknown): RuntimeDebugOutput | undefined {
  if (!isRecord(value)) return undefined
  const phase = readString(value, 'phase')
  if (!phase) return undefined
  const debug: RuntimeDebugOutput = { phase }
  const cellId = readString(value, 'cellId')
  if (cellId !== undefined) debug.cellId = cellId
  const errorName = readString(value, 'errorName')
  if (errorName !== undefined) debug.errorName = errorName
  const errorMessage = readString(value, 'errorMessage')
  if (errorMessage !== undefined) debug.errorMessage = errorMessage
  const stack = readString(value, 'stack')
  if (stack !== undefined) debug.stack = stack
  return debug
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
  if (type === 'display-javascript') {
    const cellId = readString(value, 'cellId')
    const code = readString(value, 'code')
    if (cellId && code !== undefined) return { type, runtimeId, cellId, code }
  }
  if (type === 'status') {
    const text = readString(value, 'text')
    if (text !== undefined) return { type, runtimeId, text }
  }
  if (type === 'asset') {
    const cellId = readString(value, 'cellId')
    const assetId = readString(value, 'assetId')
    const url = readString(value, 'url')
    if (cellId && assetId && url) return { type, runtimeId, cellId, assetId, url }
  }
  if (type === 'file-result') {
    const requestId = readString(value, 'requestId')
    const status = readNumber(value, 'status')
    const statusText = readString(value, 'statusText')
    const contentType = readString(value, 'contentType')
    if (!requestId || status === undefined || !statusText || !contentType) return undefined
    const message: Extract<FrameMessage, { type: 'file-result' }> = {
      type,
      runtimeId,
      requestId,
      ok: value.ok === true,
      status,
      statusText,
      contentType,
    }
    if (value.bytes instanceof ArrayBuffer) message.bytes = value.bytes
    const error = readString(value, 'error')
    if (error !== undefined) message.error = error
    return message
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

function replaceRenderedSource(figure: HTMLElement, source: string) {
  const pre = figure.querySelector('pre')
  if (!pre) {
    figure.textContent = source
    return
  }
  const code = pre.querySelector('code')
  if (!code) {
    pre.textContent = source
    return
  }

  code.replaceChildren()
  const lines = source.split(/\r?\n/)
  lines.forEach((line, index) => {
    const span = document.createElement('span')
    span.dataset.line = ''
    span.textContent = line
    code.append(span)
    if (index < lines.length - 1) code.append(document.createTextNode('\n'))
  })
}

class NotebookRuntime {
  private payload: RuntimePayload
  private root: HTMLElement
  private worker: Worker | undefined
  private displayFrame: HTMLIFrameElement | undefined
  private displayReady: Promise<HTMLIFrameElement> | undefined
  private ready: Promise<void> | undefined
  private readyResolve: (() => void) | undefined
  private waiters = new Map<string, CellWaiter>()
  private runtimeFileWaiters = new Map<string, RuntimeFileWaiter>()
  private moduleCache = new Map<string, RuntimeModule | null>()
  private moduleFetches = new Map<string, Promise<RuntimeModule | null>>()
  private savedOutputs = new Map<string, HTMLElement[]>()
  private sourceControls = new Map<string, SourceControls>()
  private executionCounter: number
  private runtimeFileSequence = 0
  private running = false
  private stopped = false
  private debug = false
  private vimMode: boolean

  constructor(root: HTMLElement, payload: RuntimePayload) {
    this.root = root
    this.payload = payload
    this.executionCounter = Math.max(0, ...payload.cells.map(cell => cell.executionIndex ?? 0))
    this.vimMode = this.readStoredVimMode()
  }

  mount() {
    this.ensureToolbar()
    this.decorateCells()
    const runAll = this.root.querySelector<HTMLButtonElement>('[data-notebook-run-all]')
    const stop = this.root.querySelector<HTMLButtonElement>('[data-notebook-stop]')
    const reset = this.root.querySelector<HTMLButtonElement>('[data-notebook-reset]')
    const debug = this.root.querySelector<HTMLButtonElement>('[data-notebook-debug]')
    const vim = this.root.querySelector<HTMLButtonElement>('[data-notebook-vim-mode]')
    runAll?.addEventListener('click', this.runAll)
    stop?.addEventListener('click', this.stop)
    reset?.addEventListener('click', this.reset)
    debug?.addEventListener('click', this.toggleDebug)
    vim?.addEventListener('click', this.toggleVimMode)
    window.addEventListener('message', this.onDisplayMessage)
    this.addCleanup(() => {
      runAll?.removeEventListener('click', this.runAll)
      stop?.removeEventListener('click', this.stop)
      reset?.removeEventListener('click', this.reset)
      debug?.removeEventListener('click', this.toggleDebug)
      vim?.removeEventListener('click', this.toggleVimMode)
      window.removeEventListener('message', this.onDisplayMessage)
      this.destroyRuntime()
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
      this.addCleanup(() => button.removeEventListener('click', handler))
    }
    this.syncToolbarToggles()
  }

  private ensureToolbar() {
    const toolbar =
      this.root.querySelector<HTMLElement>('[data-notebook-runtime-toolbar]') ??
      document.createElement('div')
    toolbar.className = 'notebook-runtime-toolbar'
    toolbar.dataset.notebookRuntimeToolbar = ''
    toolbar.setAttribute('role', 'toolbar')
    toolbar.setAttribute('aria-label', 'Notebook runtime')

    let status = toolbar.querySelector<HTMLElement>('[data-notebook-status]')
    if (!status) {
      status = document.createElement('span')
    }
    status.className = 'notebook-runtime-status'
    status.dataset.notebookStatus = ''
    status.setAttribute('aria-live', 'polite')
    if (!status.textContent) status.textContent = 'idle'

    const ensureButton = (
      selector: string,
      setup: (button: HTMLButtonElement) => void,
    ): HTMLButtonElement => {
      const existing = toolbar.querySelector<HTMLButtonElement>(selector)
      const button = existing ?? document.createElement('button')
      setup(button)
      if (!existing) toolbar.insertBefore(button, status.isConnected ? status : null)
      return button
    }

    ensureButton('[data-notebook-run-all]', button => {
      button.type = 'button'
      button.dataset.notebookRunAll = ''
      button.textContent = 'Run all'
    })
    ensureButton('[data-notebook-stop]', button => {
      button.type = 'button'
      button.dataset.notebookStop = ''
      button.disabled = true
      button.textContent = 'Stop'
    })
    ensureButton('[data-notebook-reset]', button => {
      button.type = 'button'
      button.dataset.notebookReset = ''
      button.textContent = 'Reset runtime'
    })
    ensureButton('[data-notebook-debug]', button => {
      button.type = 'button'
      button.dataset.notebookDebug = ''
      button.setAttribute('aria-pressed', 'false')
      button.textContent = 'Debug'
    })
    ensureButton('[data-notebook-vim-mode]', button => {
      button.type = 'button'
      button.dataset.notebookVimMode = ''
      button.setAttribute('aria-pressed', 'false')
      button.textContent = 'Vim'
    })

    if (!status.isConnected) toolbar.append(status)
    if (!toolbar.isConnected) this.root.append(toolbar)
  }

  private readStoredVimMode(): boolean {
    try {
      return window.localStorage.getItem(notebookRuntimeVimModeKey) === 'true'
    } catch {
      return false
    }
  }

  private writeStoredVimMode(enabled: boolean) {
    try {
      window.localStorage.setItem(notebookRuntimeVimModeKey, enabled ? 'true' : 'false')
    } catch {}
  }

  private syncToolbarToggles() {
    const debug = this.root.querySelector<HTMLButtonElement>('[data-notebook-debug]')
    debug?.setAttribute('aria-pressed', String(this.debug))
    const vim = this.root.querySelector<HTMLButtonElement>('[data-notebook-vim-mode]')
    vim?.setAttribute('aria-pressed', String(this.vimMode))
  }

  private toggleDebug = () => {
    this.debug = !this.debug
    this.syncToolbarToggles()
  }

  private toggleVimMode = () => {
    this.vimMode = !this.vimMode
    this.writeStoredVimMode(this.vimMode)
    this.syncToolbarToggles()
    for (const controls of this.sourceControls.values()) {
      void controls.editor?.setVimMode(this.vimMode).catch(error => {
        this.setStatus(error instanceof Error ? error.message : 'failed to toggle vim mode')
      })
    }
  }

  private addCleanup(callback: () => void) {
    const runtimeWindow = window as Window & { addCleanup?: (callback: () => void) => void }
    if (typeof runtimeWindow.addCleanup === 'function') {
      runtimeWindow.addCleanup(callback)
    } else {
      window.addEventListener('beforeunload', callback, { once: true })
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
    this.hideSavedOutput(cell.id)
    try {
      await this.ensureWorker()
      await this.postRun(cell, source)
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      const output: RuntimeErrorOutput = {
        type: 'error',
        ename: 'RuntimeError',
        evalue: text,
        traceback: text,
      }
      if (this.debug) output.debug = this.debugOutput('client-runtime', cell.id, error)
      this.renderOutput(cell.id, output)
    } finally {
      this.running = false
      this.setExecutionLabel(cell.id, executionCount)
      this.setRunningControls(false)
      if (!this.stopped) this.setStatus('idle')
    }
  }

  private stop = () => {
    this.stopped = true
    this.destroyRuntime()
    this.setRunningControls(false)
    this.setStatus('stopped')
    for (const waiter of this.waiters.values()) {
      waiter.reject(new Error('runtime stopped'))
    }
    this.waiters.clear()
  }

  private reset = () => {
    this.stopped = false
    this.destroyRuntime()
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
      const controls = document.querySelector<HTMLElement>(
        `[data-notebook-cell="${CSS.escape(cell.id)}"]`,
      )
      if (!controls) continue
      this.ensureCellControls(cell, controls)
      this.setExecutionLabel(cell.id, cell.executionIndex)
      const existingFrame = controls.closest<HTMLElement>('[data-notebook-cell-frame]')
      if (existingFrame) {
        this.decorateSourceEditor(cell, existingFrame)
        continue
      }
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
      this.decorateSourceEditor(cell, frame)
    }
  }

  private ensureCellControls(cell: RuntimeCell, controls: HTMLElement) {
    if (
      !controls.querySelector<HTMLElement>(
        `[data-notebook-execution-label="${CSS.escape(cell.id)}"]`,
      )
    ) {
      const label = document.createElement('span')
      label.className = 'notebook-execution-prompt'
      label.dataset.notebookExecutionLabel = cell.id
      label.setAttribute('aria-live', 'polite')
      controls.append(label)
    }
  }

  private decorateSourceEditor(cell: RuntimeCell, frame: HTMLElement) {
    if (this.sourceControls.has(cell.id)) return
    const figure =
      frame.querySelector<HTMLElement>('figure[data-rehype-pretty-code-figure]') ?? undefined
    const actions =
      frame.querySelector<HTMLElement>(`[data-notebook-cell-actions="${CSS.escape(cell.id)}"]`) ??
      document.createElement('div')
    actions.className = 'notebook-cell-actions'
    actions.dataset.notebookCellActions = cell.id

    const runButton =
      frame.querySelector<HTMLButtonElement>(`[data-notebook-run-cell="${CSS.escape(cell.id)}"]`) ??
      document.createElement('button')
    runButton.type = 'button'
    runButton.dataset.notebookRunCell = cell.id
    setNotebookIconButton(runButton, 'run', `Run ${cell.id}`)

    const editorHost =
      frame.querySelector<HTMLElement>(`[data-notebook-source-editor="${CSS.escape(cell.id)}"]`) ??
      document.createElement('div')
    editorHost.className = 'notebook-source-editor'
    editorHost.dataset.notebookSourceEditor = cell.id
    editorHost.hidden = true

    const editButton =
      frame.querySelector<HTMLButtonElement>(
        `[data-notebook-edit-cell="${CSS.escape(cell.id)}"]`,
      ) ?? document.createElement('button')
    editButton.type = 'button'
    editButton.dataset.notebookEditCell = cell.id
    setNotebookIconButton(editButton, 'edit', `Edit ${cell.id}`)
    const editSource = () => {
      void this.showSourceEditor(cell, true)
    }
    editButton.addEventListener('click', editSource)

    const saveButton =
      frame.querySelector<HTMLButtonElement>(
        `[data-notebook-save-cell="${CSS.escape(cell.id)}"]`,
      ) ?? document.createElement('button')
    saveButton.type = 'button'
    saveButton.dataset.notebookSaveCell = cell.id
    setNotebookIconButton(saveButton, 'save', `Save ${cell.id} locally`)
    saveButton.hidden = true
    const saveSource = () => this.saveEditorSource(cell)
    saveButton.addEventListener('click', saveSource)

    const revertButton =
      frame.querySelector<HTMLButtonElement>(
        `[data-notebook-revert-cell="${CSS.escape(cell.id)}"]`,
      ) ?? document.createElement('button')
    revertButton.type = 'button'
    revertButton.dataset.notebookRevertCell = cell.id
    setNotebookIconButton(revertButton, 'revert', `Revert ${cell.id} local edit`)
    revertButton.hidden = true
    const revertSource = () => this.revertEditorSource(cell)
    revertButton.addEventListener('click', revertSource)

    const status =
      frame.querySelector<HTMLElement>(
        `[data-notebook-local-source-status="${CSS.escape(cell.id)}"]`,
      ) ?? document.createElement('span')
    status.className = 'notebook-local-source-status'
    status.dataset.notebookLocalSourceStatus = cell.id
    status.hidden = true

    actions.replaceChildren(runButton, editButton, saveButton, revertButton, status)
    if (!actions.isConnected) frame.append(actions)
    if (figure) {
      if (!editorHost.isConnected) figure.before(editorHost)
    } else {
      if (!editorHost.isConnected) frame.append(editorHost)
    }
    const stored = this.readStoredSource(cell)
    this.sourceControls.set(cell.id, {
      frame,
      editor: undefined,
      editorHost,
      figure,
      status,
      editButton,
      saveButton,
      revertButton,
      source: stored ?? cell.source,
    })

    if (stored !== undefined && stored === cell.source) {
      this.clearStoredSource(cell)
    } else if (stored !== undefined && figure) {
      replaceRenderedSource(figure, stored)
    }
    this.syncSourceControls(cell)

    this.addCleanup(() => {
      editButton.removeEventListener('click', editSource)
      saveButton.removeEventListener('click', saveSource)
      revertButton.removeEventListener('click', revertSource)
      this.sourceControls.get(cell.id)?.editor?.destroy()
    })
  }

  private sourceForCell(cell: RuntimeCell): string {
    const controls = this.sourceControls.get(cell.id)
    if (controls && !controls.editorHost.hidden && controls.editor)
      return controls.editor.getValue()
    if (controls) return controls.source
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

  private async ensureSourceEditor(cell: RuntimeCell, controls: SourceControls) {
    if (controls.editor) return controls.editor
    const { createNotebookCodeEditor } = await import('./notebook-code-editor')
    controls.editor = await createNotebookCodeEditor({
      parent: controls.editorHost,
      initialContent: controls.source,
      language: cell.language,
      vimMode: this.vimMode,
      onChange: source => {
        controls.source = source
        this.syncSourceControls(cell)
      },
      onSubmit: () => this.runCell(cell),
      onSave: () => this.saveEditorSource(cell),
      onCancel: () => {
        void this.revertEditorSource(cell)
      },
    })
    return controls.editor
  }

  private async showSourceEditor(cell: RuntimeCell, visible: boolean) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    if (visible) {
      const editor = await this.ensureSourceEditor(cell, controls)
      editor.setValue(this.sourceForCell(cell))
      controls.source = editor.getValue()
    }
    controls.editorHost.hidden = !visible
    if (controls.figure) controls.figure.hidden = visible
    controls.frame.toggleAttribute('data-notebook-editing', visible)
    this.syncSourceControls(cell)
    if (visible) controls.editor?.focus()
  }

  private saveEditorSource(cell: RuntimeCell) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    const source = this.sourceForCell(cell)
    if (source === cell.source) {
      controls.source = cell.source
      this.clearStoredSource(cell)
      if (controls.figure) replaceRenderedSource(controls.figure, cell.source)
      void this.showSourceEditor(cell, false)
      this.setStatus('idle')
      return
    }
    if (!this.writeStoredSource(cell, source)) {
      this.setStatus('local save failed')
      return
    }
    controls.source = source
    if (controls.figure) replaceRenderedSource(controls.figure, source)
    void this.showSourceEditor(cell, false)
    this.setStatus('saved local edit')
  }

  private revertEditorSource(cell: RuntimeCell) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    controls.source = cell.source
    controls.editor?.setValue(cell.source)
    this.clearStoredSource(cell)
    if (controls.figure) replaceRenderedSource(controls.figure, cell.source)
    void this.showSourceEditor(cell, false)
    this.setStatus('idle')
  }

  private syncSourceControls(cell: RuntimeCell) {
    const controls = this.sourceControls.get(cell.id)
    if (!controls) return
    const stored = this.readStoredSource(cell)
    const editing = !controls.editorHost.hidden
    const edited = editing && this.sourceForCell(cell) !== (stored ?? cell.source)
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

  private ensureWorker(): Promise<void> {
    if (this.ready) return this.ready
    this.ready = new Promise(resolve => {
      this.readyResolve = resolve
    })
    const worker = new Worker(new URL('notebook-runtime.worker.js', import.meta.url), {
      type: 'module',
    })
    worker.addEventListener('message', this.onWorkerMessage)
    worker.addEventListener('error', this.onWorkerError)
    worker.postMessage({
      source: 'quartz-notebook-runtime',
      type: 'init',
      runtimeId: this.payload.id,
    })
    this.worker = worker
    return this.ready
  }

  private async postRun(cell: RuntimeCell, source: string): Promise<void> {
    const modules = await this.notebookModulesFor(source)
    return new Promise((resolve, reject) => {
      this.waiters.set(cell.id, { resolve, reject })
      this.worker?.postMessage({
        source: 'quartz-notebook-runtime',
        type: 'run',
        runtimeId: this.payload.id,
        cellId: cell.id,
        code: source,
        pyodideIndexUrl: this.payload.pyodideIndexUrl,
        debug: this.debug,
        modules,
      })
    })
  }

  private debugOutput(phase: string, cellId: string, error: unknown): RuntimeDebugOutput {
    const debug: RuntimeDebugOutput = {
      phase,
      cellId,
      errorMessage: error instanceof Error ? error.message : String(error),
    }
    if (error instanceof Error) {
      debug.errorName = error.name
      if (error.stack !== undefined) debug.stack = error.stack
    }
    return debug
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

  private onWorkerError = (event: ErrorEvent) => {
    this.readyResolve?.()
    this.readyResolve = undefined
    for (const waiter of this.waiters.values()) {
      waiter.reject(event.error instanceof Error ? event.error : new Error(event.message))
    }
    this.waiters.clear()
  }

  private onWorkerMessage = (event: MessageEvent<unknown>) => {
    const message = readFrameMessage(event.data)
    if (!message || message.runtimeId !== this.payload.id) return
    this.handleRuntimeMessage(message, this.worker)
  }

  private onDisplayMessage = (event: MessageEvent<unknown>) => {
    if (this.displayFrame?.contentWindow && event.source !== this.displayFrame.contentWindow) {
      return
    }
    const message = readFrameMessage(event.data)
    if (!message || message.runtimeId !== this.payload.id) return
    this.handleRuntimeMessage(message, this.displayFrame?.contentWindow ?? undefined)
  }

  private handleRuntimeMessage(message: FrameMessage, target: Worker | Window | undefined) {
    if (message.type === 'ready') {
      this.readyResolve?.()
      this.readyResolve = undefined
    } else if (message.type === 'output') {
      const output: NotebookRuntimeOutput =
        message.output.type === 'html'
          ? { type: 'html', html: sanitizeHtml(message.output.html) }
          : message.output
      if (output.type === 'error' && output.ename === 'UnsupportedRuntimeFeature') {
        this.restoreSavedOutput(message.cellId)
      }
      this.renderOutput(message.cellId, output)
    } else if (message.type === 'done') {
      const waiter = this.waiters.get(message.cellId)
      if (!waiter) return
      this.waiters.delete(message.cellId)
      waiter.resolve()
    } else if (message.type === 'asset') {
      void this.fetchAsset(message, target)
    } else if (message.type === 'file-result') {
      if (!(target instanceof Worker)) return
      const waiter = this.runtimeFileWaiters.get(message.requestId)
      if (!waiter) return
      this.runtimeFileWaiters.delete(message.requestId)
      waiter.resolve(message)
    } else if (message.type === 'display-javascript') {
      this.runDisplayJavascript(message.cellId, message.code)
    } else if (message.type === 'status') {
      this.setStatus(message.text)
    }
  }

  private ensureDisplayFrame(): Promise<HTMLIFrameElement> {
    if (this.displayFrame) return Promise.resolve(this.displayFrame)
    if (this.displayReady) return this.displayReady
    const iframe = document.createElement('iframe')
    iframe.className = 'notebook-runtime-frame'
    iframe.sandbox.add('allow-scripts', 'allow-modals')
    this.displayReady = new Promise(resolve => {
      iframe.addEventListener('load', () => {
        iframe.contentWindow?.postMessage(
          { source: 'quartz-notebook-runtime', type: 'init', runtimeId: this.payload.id },
          '*',
        )
        resolve(iframe)
      })
    })
    iframe.srcdoc = notebookDisplayFrameSource
    this.root.appendChild(iframe)
    this.displayFrame = iframe
    return this.displayReady
  }

  private runDisplayJavascript(cellId: string, code: string) {
    void this.ensureDisplayFrame().then(iframe => {
      iframe.contentWindow?.postMessage(
        {
          source: 'quartz-notebook-runtime',
          type: 'run-display',
          runtimeId: this.payload.id,
          cellId,
          code,
          debug: this.debug,
        },
        '*',
      )
    })
  }

  private async fetchAsset(
    message: Extract<FrameMessage, { type: 'asset' }>,
    target: Worker | Window | undefined,
  ) {
    const result = await this.resolveAsset(message)
    if (!target) return
    const transfer = result.bytes ? [result.bytes] : []
    const payload = { source: 'quartz-notebook-runtime', type: 'asset-result', ...result }
    if (target instanceof Worker) {
      target.postMessage(payload, transfer)
    } else {
      target.postMessage(payload, '*', transfer)
    }
  }

  private async resolveAsset(
    message: Extract<FrameMessage, { type: 'asset' }>,
  ): Promise<AssetResult> {
    const fallback = (error: unknown): AssetResult => ({
      runtimeId: message.runtimeId,
      assetId: message.assetId,
      ok: false,
      status: 404,
      statusText: 'Not Found',
      contentType: 'text/plain',
      error: error instanceof Error ? error.message : String(error),
    })

    let relative = ''
    let assetUrl = ''
    try {
      const base = new URL(window.location.href)
      const baseDir = new URL('.', base)
      const url = new URL(message.url, base)
      assetUrl = url.href
      relative = url.pathname.slice(baseDir.pathname.length)
      if (
        url.origin !== window.location.origin ||
        !url.pathname.startsWith(baseDir.pathname) ||
        relative.includes('/')
      ) {
        throw new Error(`blocked non-sibling notebook asset: ${message.url}`)
      }
    } catch (error) {
      return fallback(error)
    }

    try {
      const response = await fetch(assetUrl)
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
      return (
        (await this.resolveRuntimeFileAsset(message, relative || message.url)) ?? fallback(error)
      )
    }
  }

  private resolveRuntimeFileAsset(
    message: Extract<FrameMessage, { type: 'asset' }>,
    path: string,
  ): Promise<AssetResult | undefined> {
    if (!this.worker) return Promise.resolve(undefined)
    const requestId = `runtime-file-${++this.runtimeFileSequence}`
    return new Promise(resolve => {
      const timeout = window.setTimeout(() => {
        this.runtimeFileWaiters.delete(requestId)
        resolve(undefined)
      }, 10_000)
      this.runtimeFileWaiters.set(requestId, {
        resolve: result => {
          window.clearTimeout(timeout)
          resolve({
            runtimeId: message.runtimeId,
            assetId: message.assetId,
            ok: result.ok,
            status: result.status,
            statusText: result.statusText,
            contentType: result.contentType,
            bytes: result.bytes,
            error: result.error,
          })
        },
      })
      this.worker?.postMessage({
        source: 'quartz-notebook-runtime',
        type: 'file',
        runtimeId: this.payload.id,
        requestId,
        path,
      })
    })
  }

  private renderOutput(cellId: string, output: NotebookRuntimeOutput) {
    const target = document.querySelector<HTMLElement>(
      `[data-notebook-output="${CSS.escape(cellId)}"]`,
    )
    if (!target) return
    target.hidden = false
    target.insertAdjacentHTML(
      'beforeend',
      renderNotebookRuntimeOutput(output, { debug: this.debug }),
    )
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

  private destroyRuntime() {
    this.worker?.removeEventListener('message', this.onWorkerMessage)
    this.worker?.removeEventListener('error', this.onWorkerError)
    this.worker?.terminate()
    this.worker = undefined
    this.runtimeFileWaiters.clear()
    this.displayFrame?.remove()
    this.displayFrame = undefined
    this.displayReady = undefined
    this.ready = undefined
    this.readyResolve = undefined
  }
}

export function mountNotebookRuntime(root: HTMLElement, text: string) {
  if (root.dataset.runtimeMounted === 'true') return
  const parsed = parseRuntimeJson(text)
  if (parsed === undefined) return
  const payload = readRuntimePayload(parsed)
  if (!payload) return
  root.dataset.runtimeMounted = 'true'
  new NotebookRuntime(root, payload).mount()
}
