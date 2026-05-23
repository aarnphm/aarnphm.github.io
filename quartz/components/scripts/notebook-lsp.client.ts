import type { Completion, CompletionContext, CompletionResult } from '@codemirror/autocomplete'
import type { Extension, Text } from '@codemirror/state'
import type { DecorationSet, EditorView, Tooltip, ViewUpdate } from '@codemirror/view'
import type { Disposable } from 'vscode-jsonrpc'
import type {
  CompletionItem,
  CompletionList,
  CompletionParams,
  DefinitionParams,
  Diagnostic,
  Hover,
  HoverParams,
  Location,
  LocationLink,
  Position,
  ProtocolConnection,
  PublishDiagnosticsParams,
  ServerCapabilities,
  SignatureHelp,
  SignatureHelpParams,
  TextDocumentContentChangeEvent,
} from 'vscode-languageserver-protocol'
import {
  notebookDocumentPath,
  notebookDocumentUri,
  notebookWorkspaceRootUri,
} from '../../util/notebook-lsp-uri'

type RuntimeCell = { id: string; source: string; language: string; executionIndex: number | null }

type RuntimePayload = {
  id: string
  sourcePath: string
  language: string
  pyodideIndexUrl: string
  cells: RuntimeCell[]
}

type NotebookLspBootMessage = {
  type: 'browser/boot'
  mode: 'foreground' | 'background'
  initialData?: unknown
  port?: MessagePort
}

type NotebookLspNewWorkerMessage = {
  type: 'browser/newWorker'
  initialData: unknown
  port: MessagePort
}

type NotebookLspConfig = {
  workerUrl: string
  rootUri: string
  locale?: string
  initializationOptions: { files: Record<string, string>; diagnosticMode: 'openFilesOnly' }
  onDiagnostics?: (params: PublishDiagnosticsParams) => void
  onLogMessage?: (message: string) => void
}

type NotebookCodeEditorLspConfig = { runtimeId: string; cellId: string; language: string }

type NotebookLspBridge = {
  createEditorExtension(config: NotebookCodeEditorLspConfig): Promise<Extension[]>
}

type NotebookCellSpan = { cellId: string; startLine: number; endLine: number }

type NotebookMappedDiagnostic = {
  message: string
  severity: 'error' | 'warning' | 'info' | 'hint'
  fromLine: number
  fromCharacter: number
  toLine: number
  toCharacter: number
}

type NotebookDiagnosticListener = (diagnostics: readonly NotebookMappedDiagnostic[]) => void

let workerCounter = 0

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function readString(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key]
  return typeof value === 'string' ? value : undefined
}

function readRuntimeCell(value: unknown): RuntimeCell | undefined {
  if (!isRecord(value)) return undefined
  const id = readString(value, 'id')
  const source = readString(value, 'source')
  const language = readString(value, 'language')
  const executionIndex = value.executionIndex
  if (!id || source === undefined || !language) return undefined
  if (typeof executionIndex !== 'number' && executionIndex !== null) return undefined
  return { id, source, language, executionIndex }
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
  const cells: RuntimeCell[] = []
  for (const cellValue of value.cells) {
    const cell = readRuntimeCell(cellValue)
    if (!cell) return undefined
    cells.push(cell)
  }
  return { id, sourcePath, language, pyodideIndexUrl, cells }
}

function parseRuntimeJson(text: string): unknown | undefined {
  try {
    return JSON.parse(text)
  } catch {
    return undefined
  }
}

function lineCount(source: string): number {
  return source.split(/\r\n?|\n/).length
}

function diagnosticSeverity(
  severity: Diagnostic['severity'],
): NotebookMappedDiagnostic['severity'] {
  if (severity === 1) return 'error'
  if (severity === 2) return 'warning'
  if (severity === 3) return 'info'
  return 'hint'
}

function cellSpanForLine(
  spans: readonly NotebookCellSpan[],
  line: number,
): NotebookCellSpan | undefined {
  return spans.find(span => line >= span.startLine && line <= span.endLine)
}

function plainInsertText(item: CompletionItem): string {
  const text = item.insertText ?? item.label
  if (item.insertTextFormat !== 2) return text
  return text
    .replace(/\$\{\d+:([^}]*)\}/g, '$1')
    .replace(/\$\d+/g, '')
    .replace(/\${\d+\|([^}]*)\|}/g, '$1')
}

function completionInfo(value: unknown): string | undefined {
  if (typeof value === 'string') return value
  if (isRecord(value)) {
    const markdown = readString(value, 'value')
    if (markdown) return markdown
  }
}

function completionType(kind: CompletionItem['kind']): Completion['type'] | undefined {
  if (kind === 2 || kind === 3) return 'function'
  if (kind === 6 || kind === 10) return 'variable'
  if (kind === 7) return 'class'
  if (kind === 9) return 'namespace'
  if (kind === 14 || kind === 15) return 'keyword'
}

function completionFromLsp(item: CompletionItem): Completion {
  const completion: Completion = { label: item.label, apply: plainInsertText(item) }
  if (item.detail) completion.detail = item.detail
  const info = completionInfo(item.documentation)
  if (info) completion.info = info
  const type = completionType(item.kind)
  if (type) completion.type = type
  return completion
}

function hoverText(value: Hover | null): string | undefined {
  if (!value) return undefined
  const contents = value.contents
  if (typeof contents === 'string') return contents
  if (Array.isArray(contents)) {
    return contents
      .map(item => {
        if (typeof item === 'string') return item
        if (isRecord(item)) return readString(item, 'value') ?? ''
        return ''
      })
      .filter(Boolean)
      .join('\n\n')
  }
  if (isRecord(contents)) return readString(contents, 'value')
}

function positionAt(doc: Text, offset: number): Position {
  const line = doc.lineAt(offset)
  return { line: line.number - 1, character: offset - line.from }
}

function offsetAt(doc: Text, line: number, character: number): number {
  const safeLine = Math.min(Math.max(line, 0), Math.max(doc.lines - 1, 0))
  const lineInfo = doc.line(safeLine + 1)
  return Math.min(lineInfo.to, lineInfo.from + Math.max(character, 0))
}

class NotebookLspClient {
  private connection: ProtocolConnection | undefined
  private workers: Worker[] = []
  private versions = new Map<string, number>()
  private initializePromise: Promise<void> | undefined
  private disposables: Disposable[] = []
  capabilities: ServerCapabilities | undefined

  constructor(private config: NotebookLspConfig) {}

  async start(): Promise<void> {
    if (this.initializePromise) return this.initializePromise
    this.initializePromise = this.boot()
    return this.initializePromise
  }

  private async boot(): Promise<void> {
    const [jsonrpc, protocol] = await Promise.all([
      import('vscode-jsonrpc/browser'),
      import('vscode-languageserver-protocol/browser'),
    ])
    const { BrowserMessageReader, BrowserMessageWriter } = jsonrpc
    const {
      createProtocolConnection,
      InitializeRequest,
      InitializedNotification,
      LogMessageNotification,
      PublishDiagnosticsNotification,
      RegistrationRequest,
      DiagnosticTag,
    } = protocol

    const suffix = ++workerCounter
    const foreground = new Worker(this.config.workerUrl, { name: `Pyright-foreground-${suffix}` })
    foreground.postMessage({
      type: 'browser/boot',
      mode: 'foreground',
    } satisfies NotebookLspBootMessage)
    this.workers.push(foreground)

    let backgroundCount = 0
    foreground.addEventListener('message', event => {
      const data = event.data as NotebookLspNewWorkerMessage | undefined
      if (data?.type !== 'browser/newWorker') return
      const background = new Worker(this.config.workerUrl, {
        name: `Pyright-background-${suffix}-${++backgroundCount}`,
      })
      this.workers.push(background)
      background.postMessage(
        {
          type: 'browser/boot',
          mode: 'background',
          initialData: data.initialData,
          port: data.port,
        } satisfies NotebookLspBootMessage,
        [data.port],
      )
    })

    const connection = createProtocolConnection(
      new BrowserMessageReader(foreground),
      new BrowserMessageWriter(foreground),
    )
    connection.onDispose(() => {
      for (const worker of this.workers) worker.terminate()
      this.workers = []
    })

    this.disposables.push(
      connection.onNotification(LogMessageNotification.type, params => {
        this.config.onLogMessage?.(params.message)
      }),
      connection.onNotification(PublishDiagnosticsNotification.type, params => {
        this.config.onDiagnostics?.(params)
      }),
      connection.onRequest(RegistrationRequest.type, () => undefined),
    )

    this.connection = connection
    connection.listen()

    const result = (await connection.sendRequest(InitializeRequest.type, {
      processId: null,
      locale: this.config.locale ?? 'en',
      rootUri: this.config.rootUri,
      workspaceFolders: [{ name: 'notebook', uri: this.config.rootUri }],
      initializationOptions: this.config.initializationOptions,
      capabilities: {
        textDocument: {
          synchronization: {
            dynamicRegistration: false,
            willSave: false,
            didSave: false,
            willSaveWaitUntil: false,
          },
          completion: {
            dynamicRegistration: false,
            contextSupport: true,
            completionItem: {
              snippetSupport: false,
              commitCharactersSupport: true,
              documentationFormat: ['markdown', 'plaintext'],
              deprecatedSupport: true,
              preselectSupport: false,
              insertReplaceSupport: false,
            },
          },
          hover: { dynamicRegistration: false, contentFormat: ['markdown', 'plaintext'] },
          signatureHelp: {
            dynamicRegistration: false,
            signatureInformation: {
              documentationFormat: ['markdown', 'plaintext'],
              activeParameterSupport: true,
              parameterInformation: { labelOffsetSupport: true },
            },
          },
          definition: { dynamicRegistration: false, linkSupport: true },
          publishDiagnostics: {
            relatedInformation: true,
            tagSupport: { valueSet: [DiagnosticTag.Unnecessary, DiagnosticTag.Deprecated] },
            versionSupport: false,
          },
        },
        workspace: {
          workspaceFolders: true,
          configuration: false,
          didChangeConfiguration: { dynamicRegistration: false },
        },
      },
    })) as { capabilities: ServerCapabilities }
    this.capabilities = result.capabilities
    connection.sendNotification(InitializedNotification.type, {})
  }

  stop(): void {
    for (const disposable of this.disposables) disposable.dispose()
    this.disposables = []
    this.connection?.dispose()
    this.connection = undefined
    this.initializePromise = undefined
    this.capabilities = undefined
    this.versions.clear()
  }

  async didOpen(uri: string, source: string, languageId = 'python'): Promise<void> {
    if (!this.connection) return
    const { DidOpenTextDocumentNotification } =
      await import('vscode-languageserver-protocol/browser')
    this.connection.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: { uri, languageId, version: this.nextVersion(uri), text: source },
    })
  }

  async didChange(uri: string, source: string): Promise<void> {
    if (!this.connection) return
    const { DidChangeTextDocumentNotification } =
      await import('vscode-languageserver-protocol/browser')
    const contentChanges: TextDocumentContentChangeEvent[] = [{ text: source }]
    this.connection.sendNotification(DidChangeTextDocumentNotification.type, {
      textDocument: { uri, version: this.nextVersion(uri) },
      contentChanges,
    })
  }

  async didClose(uri: string): Promise<void> {
    if (!this.connection) return
    const { DidCloseTextDocumentNotification } =
      await import('vscode-languageserver-protocol/browser')
    this.connection.sendNotification(DidCloseTextDocumentNotification.type, {
      textDocument: { uri },
    })
    this.versions.delete(uri)
  }

  async completion(params: CompletionParams): Promise<CompletionList> {
    if (!this.connection) return { items: [], isIncomplete: true }
    const { CompletionRequest } = await import('vscode-languageserver-protocol/browser')
    const result = (await this.connection.sendRequest(CompletionRequest.type, params)) as
      | CompletionList
      | CompletionItem[]
      | null
    if (!result) return { items: [], isIncomplete: true }
    if (Array.isArray(result)) return { items: result, isIncomplete: true }
    return result
  }

  async hover(params: HoverParams): Promise<Hover | null> {
    if (!this.connection) return null
    const { HoverRequest } = await import('vscode-languageserver-protocol/browser')
    return ((await this.connection.sendRequest(HoverRequest.type, params)) as Hover | null) ?? null
  }

  async definition(params: DefinitionParams): Promise<Location[] | LocationLink[] | null> {
    if (!this.connection) return null
    const { DefinitionRequest } = await import('vscode-languageserver-protocol/browser')
    const result = (await this.connection.sendRequest(DefinitionRequest.type, params)) as
      | Location
      | Location[]
      | LocationLink[]
      | null
    if (!result) return null
    if (Array.isArray(result)) return result
    return [result]
  }

  async signatureHelp(params: SignatureHelpParams): Promise<SignatureHelp | null> {
    if (!this.connection) return null
    const { SignatureHelpRequest } = await import('vscode-languageserver-protocol/browser')
    return (
      ((await this.connection.sendRequest(
        SignatureHelpRequest.type,
        params,
      )) as SignatureHelp | null) ?? null
    )
  }

  private nextVersion(uri: string): number {
    const next = (this.versions.get(uri) ?? 0) + 1
    this.versions.set(uri, next)
    return next
  }
}

class NotebookLspSession {
  private client: NotebookLspClient | undefined
  private startPromise: Promise<void> | undefined
  private sources = new Map<string, string>()
  private spans: NotebookCellSpan[] = []
  private documentSource = ''
  private syncTimer: number | undefined
  private diagnosticsByCell = new Map<string, NotebookMappedDiagnostic[]>()
  private diagnosticListeners = new Map<string, Set<NotebookDiagnosticListener>>()

  constructor(private payload: RuntimePayload) {
    this.replacePayload(payload)
  }

  replacePayload(payload: RuntimePayload) {
    this.payload = payload
    const liveCells = new Set<string>()
    for (const cell of payload.cells) {
      liveCells.add(cell.id)
      if (!this.sources.has(cell.id)) this.sources.set(cell.id, cell.source)
    }
    for (const cellId of this.sources.keys()) {
      if (!liveCells.has(cellId)) this.sources.delete(cellId)
    }
    this.rebuildDocument()
  }

  async createEditorExtension(config: NotebookCodeEditorLspConfig): Promise<Extension[]> {
    const [
      { autocompletion },
      { StateEffect, StateField },
      { Decoration, EditorView, ViewPlugin, hoverTooltip },
    ] = await Promise.all([
      import('@codemirror/autocomplete'),
      import('@codemirror/state'),
      import('@codemirror/view'),
    ])

    const diagnosticsForEditor = () => this.diagnosticsForCell(config.cellId)
    const subscribeDiagnostics = (listener: NotebookDiagnosticListener) =>
      this.subscribeDiagnostics(config.cellId, listener)
    const updateEditorCell = (source: string) => this.updateCell(config.cellId, source)
    const setDiagnostics = StateEffect.define<readonly NotebookMappedDiagnostic[]>()
    const diagnosticDecorations = (
      doc: Text,
      diagnostics: readonly NotebookMappedDiagnostic[],
    ): DecorationSet => {
      const ranges = diagnostics.map(diagnostic => {
        const from = offsetAt(doc, diagnostic.fromLine, diagnostic.fromCharacter)
        const to = Math.max(from + 1, offsetAt(doc, diagnostic.toLine, diagnostic.toCharacter))
        return Decoration.mark({
          class: `cm-notebook-lsp-diagnostic cm-notebook-lsp-diagnostic-${diagnostic.severity}`,
        }).range(from, to)
      })
      return Decoration.set(ranges, true)
    }
    const diagnosticField = StateField.define<DecorationSet>({
      create() {
        return Decoration.none
      },
      update(value, transaction) {
        let next = value.map(transaction.changes)
        for (const effect of transaction.effects) {
          if (effect.is(setDiagnostics)) {
            next = diagnosticDecorations(transaction.state.doc, effect.value)
          }
        }
        return next
      },
      provide: field => EditorView.decorations.from(field),
    })
    const diagnosticPlugin = ViewPlugin.fromClass(
      class {
        private dispose: (() => void) | undefined

        constructor(view: EditorView) {
          this.dispose = subscribeDiagnostics(diagnostics => {
            view.dispatch({ effects: setDiagnostics.of(diagnostics) })
          })
          queueMicrotask(() => {
            view.dispatch({ effects: setDiagnostics.of(diagnosticsForEditor()) })
          })
        }

        update(update: ViewUpdate) {
          if (update.docChanged) {
            updateEditorCell(update.state.doc.toString())
          }
        }

        destroy() {
          this.dispose?.()
        }
      },
    )
    const completionSource = async (
      context: CompletionContext,
    ): Promise<CompletionResult | null> => {
      const token = context.matchBefore(/[A-Za-z_][A-Za-z0-9_]*$/)
      if (!context.explicit && !token) return null
      await this.start().catch(error => {
        console.warn('failed to start notebook lsp', error)
      })
      await this.syncNow()
      const client = this.client
      if (!client) return null
      const result = await client.completion({
        textDocument: { uri: this.documentUri },
        position: this.virtualPosition(config.cellId, context.state.doc, context.pos),
      })
      return {
        from: token?.from ?? context.pos,
        options: result.items.map(completionFromLsp),
        filter: false,
      }
    }
    const lspHover = hoverTooltip(
      async (view: EditorView, pos: number): Promise<Tooltip | null> => {
        await this.start().catch(error => {
          console.warn('failed to start notebook lsp', error)
        })
        await this.syncNow()
        const client = this.client
        if (!client) return null
        const text = hoverText(
          await client.hover({
            textDocument: { uri: this.documentUri },
            position: this.virtualPosition(config.cellId, view.state.doc, pos),
          }),
        )
        if (!text) return null
        const word = view.state.wordAt(pos)
        return {
          pos: word?.from ?? pos,
          end: word?.to ?? pos,
          above: true,
          create() {
            const dom = document.createElement('div')
            dom.className = 'cm-notebook-lsp-tooltip'
            dom.textContent = text
            return { dom }
          },
        }
      },
    )
    const diagnosticTooltip = hoverTooltip((view: EditorView, pos: number): Tooltip | null => {
      const diagnostic = diagnosticsForEditor().find(item => {
        const from = offsetAt(view.state.doc, item.fromLine, item.fromCharacter)
        const to = offsetAt(view.state.doc, item.toLine, item.toCharacter)
        return pos >= from && pos <= Math.max(from + 1, to)
      })
      if (!diagnostic) return null
      return {
        pos,
        above: true,
        create() {
          const dom = document.createElement('div')
          dom.className = `cm-notebook-lsp-tooltip cm-notebook-lsp-tooltip-${diagnostic.severity}`
          dom.textContent = diagnostic.message
          return { dom }
        },
      }
    })
    const lspTheme = EditorView.baseTheme({
      '.cm-notebook-lsp-diagnostic-error': { textDecoration: 'underline wavy #c7372f 0.08em' },
      '.cm-notebook-lsp-diagnostic-warning': { textDecoration: 'underline wavy #b87900 0.08em' },
      '.cm-notebook-lsp-diagnostic-info, .cm-notebook-lsp-diagnostic-hint': {
        textDecoration: 'underline dotted currentColor 0.08em',
      },
      '.cm-notebook-lsp-tooltip': {
        maxWidth: 'min(32rem, 80vw)',
        padding: '0.45rem 0.55rem',
        border: '1px solid var(--lightgray)',
        borderRadius: 'var(--radius-3)',
        background: 'var(--light)',
        color: 'var(--dark)',
        fontFamily: 'var(--codeFont)',
        fontSize: '0.78rem',
        lineHeight: '1.45',
        whiteSpace: 'pre-wrap',
      },
    })

    void this.start().catch(error => {
      console.warn('failed to start notebook lsp', error)
    })
    return [
      autocompletion({ override: [completionSource] }),
      diagnosticField,
      diagnosticPlugin,
      diagnosticTooltip,
      lspHover,
      lspTheme,
    ]
  }

  updateCell(cellId: string, source: string) {
    if (this.sources.get(cellId) === source) return
    this.sources.set(cellId, source)
    this.rebuildDocument()
    this.scheduleSync()
  }

  diagnosticsForCell(cellId: string): readonly NotebookMappedDiagnostic[] {
    return this.diagnosticsByCell.get(cellId) ?? []
  }

  subscribeDiagnostics(cellId: string, listener: NotebookDiagnosticListener): () => void {
    const listeners = this.diagnosticListeners.get(cellId) ?? new Set<NotebookDiagnosticListener>()
    listeners.add(listener)
    this.diagnosticListeners.set(cellId, listeners)
    return () => {
      listeners.delete(listener)
      if (listeners.size === 0) this.diagnosticListeners.delete(cellId)
    }
  }

  private get documentUri(): string {
    return notebookDocumentUri(this.payload.id)
  }

  private async start(): Promise<void> {
    if (this.startPromise) return this.startPromise
    this.startPromise = this.startClient().catch(error => {
      this.client?.stop()
      this.client = undefined
      this.startPromise = undefined
      throw error
    })
    return this.startPromise
  }

  private async startClient(): Promise<void> {
    const client = new NotebookLspClient({
      workerUrl: new URL('pyright.worker.js', import.meta.url).href,
      rootUri: notebookWorkspaceRootUri(this.payload.id),
      initializationOptions: {
        files: { [notebookDocumentPath(this.payload.id)]: this.documentSource },
        diagnosticMode: 'openFilesOnly',
      },
      onDiagnostics: params => this.receiveDiagnostics(params),
      onLogMessage: message => {
        if (message.toLowerCase().includes('error')) console.warn(message)
      },
    })
    this.client = client
    await client.start()
    await client.didOpen(this.documentUri, this.documentSource)
  }

  private scheduleSync() {
    if (!this.client || this.syncTimer !== undefined) return
    this.syncTimer = window.setTimeout(() => {
      this.syncTimer = undefined
      void this.syncNow()
    }, 120)
  }

  private async syncNow(): Promise<void> {
    if (this.syncTimer !== undefined) {
      window.clearTimeout(this.syncTimer)
      this.syncTimer = undefined
    }
    await this.client?.didChange(this.documentUri, this.documentSource)
  }

  private virtualPosition(cellId: string, doc: Text, offset: number): Position {
    const local = positionAt(doc, offset)
    const span = this.spans.find(span => span.cellId === cellId)
    return { line: (span?.startLine ?? 0) + local.line, character: local.character }
  }

  private rebuildDocument() {
    const spans: NotebookCellSpan[] = []
    const chunks: string[] = []
    let line = 0
    for (const cell of this.payload.cells) {
      chunks.push(`# %% ${cell.id}`)
      line += 1
      const source = this.sources.get(cell.id) ?? cell.source
      const count = lineCount(source)
      spans.push({ cellId: cell.id, startLine: line, endLine: line + count - 1 })
      chunks.push(source)
      line += count
      chunks.push('')
      line += 1
    }
    this.spans = spans
    this.documentSource = `${chunks.join('\n')}\n`
  }

  private receiveDiagnostics(params: PublishDiagnosticsParams) {
    if (params.uri !== this.documentUri) return
    const next = new Map<string, NotebookMappedDiagnostic[]>()
    for (const diagnostic of params.diagnostics) {
      const span = cellSpanForLine(this.spans, diagnostic.range.start.line)
      if (!span) continue
      const diagnostics = next.get(span.cellId) ?? []
      diagnostics.push({
        message: diagnostic.message,
        severity: diagnosticSeverity(diagnostic.severity),
        fromLine: Math.max(0, diagnostic.range.start.line - span.startLine),
        fromCharacter: diagnostic.range.start.character,
        toLine: Math.max(0, diagnostic.range.end.line - span.startLine),
        toCharacter: diagnostic.range.end.character,
      })
      next.set(span.cellId, diagnostics)
    }
    this.diagnosticsByCell = next
    for (const cell of this.payload.cells) {
      const diagnostics = this.diagnosticsForCell(cell.id)
      for (const listener of this.diagnosticListeners.get(cell.id) ?? []) {
        listener(diagnostics)
      }
    }
  }
}

const sessions = new Map<string, NotebookLspSession>()

function findRuntimePayload(runtimeId: string): RuntimePayload | undefined {
  const scripts = document.querySelectorAll<HTMLScriptElement>('script[data-notebook-runtime-data]')
  for (const script of scripts) {
    const payload = readRuntimePayload(parseRuntimeJson(script.textContent ?? ''))
    if (payload?.id === runtimeId) return payload
  }
}

function sessionForRuntime(runtimeId: string): NotebookLspSession | undefined {
  let session = sessions.get(runtimeId)
  if (session) return session
  const payload = findRuntimePayload(runtimeId)
  if (!payload) return undefined
  session = new NotebookLspSession(payload)
  sessions.set(payload.id, session)
  return session
}

export function mountNotebookLsp(root: HTMLElement, text: string) {
  const payload = readRuntimePayload(parseRuntimeJson(text))
  if (!payload || !payload.language.toLowerCase().startsWith('python')) return
  const session = sessions.get(payload.id) ?? new NotebookLspSession(payload)
  session.replacePayload(payload)
  sessions.set(payload.id, session)
  root.dataset.notebookLspMounted = 'true'
}

const bridge: NotebookLspBridge = {
  async createEditorExtension(config) {
    if (!config.language.toLowerCase().startsWith('python')) return []
    const session = sessionForRuntime(config.runtimeId)
    return session?.createEditorExtension(config) ?? []
  },
}

Reflect.set(globalThis, '__quartzNotebookLsp', bridge)
