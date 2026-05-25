import type {
  Hover,
  HoverParams,
  MarkedString,
  MarkupContent,
  Position,
} from 'vscode-languageserver-protocol'
import { autocompletion, type CompletionSource } from '@codemirror/autocomplete'
import {
  LSPPlugin,
  Workspace,
  type LSPClient,
  type Transport,
  type WorkspaceFile,
} from '@codemirror/lsp-client'
import { EditorState, Text, type Extension } from '@codemirror/state'
import { hoverTooltip, type EditorView, type Tooltip } from '@codemirror/view'
import DOMPurify from 'dompurify'
import type { LspBridge, LspConfig } from './bridge'
import { escapeHTML } from '../../util/escape'
import {
  arrayValue,
  isJsonObject,
  isJsonValue,
  isRecord,
  objectValue,
  stringValue,
  type JsonObject,
  type JsonValue,
  type UnknownRecord,
} from '../../util/type-guards'
import { notebookRuntimeAssetUrl } from '../notebook/assets'
import { notebookPyrightAssetManifestEntry } from './pyright-assets'

export type NotebookLspCell = {
  id: string
  source: string
  language: string
  executionIndex: number | null
}

export type NotebookLspConfig = {
  enabled: boolean
  runtimeId: string
  sourcePath: string
  cellId: string
  language: string
  cells: () => readonly NotebookLspCell[]
}

type JsonRpcId = string | number | null
type NotebookMessageTarget = { postMessage(message: unknown): void }

const notebookPyrightWorkerManifestName = '../notebook-pyright-worker.json'
const notebookPyrightTypeshedManifestName = '../notebook-pyright-typeshed.json'

const notebookAnalysisSettings: JsonObject = {
  typeCheckingMode: 'basic',
  diagnosticMode: 'openFilesOnly',
  typeshedPaths: ['file:///typeshed'],
  diagnosticSeverityOverrides: { reportMissingImports: 'none', reportMissingModuleSource: 'none' },
}

const notebookClientCapabilities: JsonObject = {
  workspace: { configuration: true, didChangeConfiguration: {}, workspaceFolders: true },
}

const notebookServices = new Map<string, NotebookPythonLspService>()

let notebookWorkerUrl: Promise<string> | undefined
let notebookLspAssetWarmup: Promise<void> | undefined
let notebookLspSequence = 0

function messageId(value: unknown): JsonRpcId | undefined {
  return typeof value === 'string' || typeof value === 'number' || value === null
    ? value
    : undefined
}

function pythonLanguage(language: string) {
  const value = language.trim().toLowerCase()
  return value === 'python' || value === 'py' || value === 'ipython'
}

function notebookRootUri(config: NotebookLspConfig) {
  return `file:///quartz-notebook/${encodeURIComponent(config.runtimeId)}/`
}

function notebookRootPath(config: NotebookLspConfig) {
  return `/quartz-notebook/${encodeURIComponent(config.runtimeId)}`
}

function notebookCellUri(rootUri: string, sourcePath: string, cellId: string) {
  return `${rootUri}${encodeURIComponent(sourcePath)}/${encodeURIComponent(cellId)}.py`
}

function notebookFileUri(config: NotebookLspConfig) {
  return notebookCellUri(notebookRootUri(config), config.sourcePath, config.cellId)
}

function notebookServiceKey(config: NotebookLspConfig) {
  return `${config.runtimeId}\u0000${config.sourcePath}`
}

function notebookText(source: string): Text {
  return Text.of(source.split('\n'))
}

function notebookPositionOffset(doc: Text, position: Position): number {
  const line = doc.line(Math.min(doc.lines, Math.max(1, position.line + 1)))
  return Math.min(line.to, line.from + position.character)
}

function notebookLspRequestTimedOut(error: unknown): boolean {
  return error instanceof Error && error.message === 'Request timed out'
}

function notebookLspRequestCancelled(error: unknown): boolean {
  return isRecord(error) && error.code === -32800
}

function notebookIgnoredLspError(error: unknown): boolean {
  return notebookLspRequestTimedOut(error) || notebookLspRequestCancelled(error)
}

function pyrightWorkerManifestUrl() {
  return notebookRuntimeAssetUrl(
    'pyrightWorkerManifestUrl',
    notebookPyrightWorkerManifestName,
    import.meta.url,
  )
}

function pyrightTypeshedManifestUrl() {
  return notebookRuntimeAssetUrl(
    'pyrightTypeshedManifestUrl',
    notebookPyrightTypeshedManifestName,
    import.meta.url,
  )
}

async function fetchJsonObject(url: string, label: string): Promise<JsonObject> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${label} request failed with ${response.status}`)
  const value: unknown = await response.json()
  if (!isJsonObject(value)) throw new Error(`${label} is not a JSON object`)
  return value
}

async function loadNotebookWorkerUrl(): Promise<string> {
  notebookWorkerUrl ??= (async () => {
    const manifestUrl = pyrightWorkerManifestUrl()
    const manifest = await fetchJsonObject(manifestUrl, 'notebook pyright worker manifest')
    const entry = notebookPyrightAssetManifestEntry(manifest, 'notebook pyright worker')
    return new URL(entry, manifestUrl).href
  })()
  return notebookWorkerUrl
}

export async function warmNotebookLspAssets(): Promise<void> {
  notebookLspAssetWarmup ??= Promise.all([
    import('@codemirror/lsp-client'),
    loadNotebookWorkerUrl(),
  ]).then(() => {})
  return notebookLspAssetWarmup
}

function notebookConfiguration(section: string | undefined): JsonValue {
  if (section === 'basedpyright.analysis' || section === 'python.analysis') {
    return notebookAnalysisSettings
  }
  if (section === 'basedpyright' || section === 'python') {
    return { analysis: notebookAnalysisSettings }
  }
  return {
    basedpyright: { analysis: notebookAnalysisSettings },
    python: { analysis: notebookAnalysisSettings },
  }
}

function configurationResult(params: JsonValue | undefined): JsonValue {
  const items = arrayValue(objectValue(params)?.items)
  if (!items) return []
  return items.map(item => notebookConfiguration(stringValue(objectValue(item)?.section)))
}

function responseMessage(id: JsonRpcId, result: JsonValue): JsonObject {
  return { jsonrpc: '2.0', id, result }
}

function notebookWorkspaceFiles(rootPath: string): JsonObject {
  return { files: { [`${rootPath}/.pyright-root`]: '' } }
}

function handleServerRequest(message: UnknownRecord, target: NotebookMessageTarget) {
  const method = stringValue(message.method)
  const id = messageId(message.id)
  if (method === undefined || id === undefined) return false

  if (method === 'workspace/configuration') {
    target.postMessage(
      responseMessage(
        id,
        configurationResult(isJsonValue(message.params) ? message.params : undefined),
      ),
    )
    return true
  }
  if (
    method === 'client/registerCapability' ||
    method === 'client/unregisterCapability' ||
    method === 'window/workDoneProgress/create'
  ) {
    target.postMessage(responseMessage(id, null))
    return true
  }
  if (method === 'workspace/applyEdit') {
    target.postMessage(responseMessage(id, { applied: false }))
    return true
  }
  return false
}

function initializeMessage(
  message: JsonObject,
  rootUri: string,
  initialFiles: JsonObject,
  typeshedManifestUrl: string,
): JsonObject {
  if (message.method !== 'initialize') return message
  const params = objectValue(message.params) ?? {}
  const capabilities = objectValue(params.capabilities) ?? {}
  const workspace = objectValue(capabilities.workspace) ?? {}
  return {
    ...message,
    params: {
      ...params,
      rootUri,
      workspaceFolders: [{ name: 'notebook', uri: rootUri }],
      initializationOptions: { files: initialFiles, typeshedManifestUrl },
      capabilities: {
        ...capabilities,
        workspace: {
          ...workspace,
          configuration: true,
          didChangeConfiguration: {},
          workspaceFolders: true,
        },
      },
    },
  }
}

function notebookServerCompletion(serverCompletionSource: CompletionSource): Extension {
  const quietServerCompletionSource: CompletionSource = context =>
    Promise.resolve(serverCompletionSource(context)).catch(error => {
      if (notebookIgnoredLspError(error)) return null
      return Promise.reject(error)
    })
  return [
    autocompletion({ defaultKeymap: false }),
    EditorState.languageData.of(() => [{ autocomplete: quietServerCompletionSource }]),
  ]
}

function renderNotebookMarkedString(plugin: LSPPlugin, value: MarkedString): string {
  if (typeof value === 'string') return plugin.docToHTML(value, 'markdown')
  return `<pre><code>${escapeHTML(value.value)}</code></pre>`
}

function renderNotebookHoverContent(
  plugin: LSPPlugin,
  value: string | MarkupContent | MarkedString | MarkedString[],
): string {
  if (Array.isArray(value)) {
    return value.map(item => renderNotebookMarkedString(plugin, item)).join('<br>')
  }
  if (typeof value === 'string' || 'language' in value) {
    return renderNotebookMarkedString(plugin, value)
  }
  return plugin.docToHTML(value)
}

function notebookHoverTooltips(hoverTime: number): Extension {
  return hoverTooltip(
    (view, pos): Promise<Tooltip | null> => {
      const plugin = LSPPlugin.get(view)
      if (!plugin || plugin.client.serverCapabilities?.hoverProvider === false) {
        return Promise.resolve(null)
      }
      plugin.client.sync()
      return plugin.client
        .request<HoverParams, Hover | null>('textDocument/hover', {
          position: plugin.toPosition(pos),
          textDocument: { uri: plugin.uri },
        })
        .then(result => {
          if (!result) return null
          return {
            pos: result.range ? notebookPositionOffset(view.state.doc, result.range.start) : pos,
            end: result.range ? notebookPositionOffset(view.state.doc, result.range.end) : pos,
            create() {
              const element = document.createElement('div')
              element.className = 'cm-lsp-hover-tooltip cm-lsp-documentation'
              element.innerHTML = renderNotebookHoverContent(plugin, result.contents)
              return { dom: element }
            },
            above: true,
          }
        })
        .catch(error => {
          if (notebookIgnoredLspError(error)) return null
          return Promise.reject(error)
        })
    },
    { hideOn: transaction => transaction.docChanged, hoverTime },
  )
}

function parseTransportMessage(message: string) {
  const value: unknown = JSON.parse(message)
  if (!isJsonObject(value)) throw new Error('notebook pyright received a non-object message')
  return value
}

function reportPyrightWorkerError(scope: string, serviceId: number, message: string | undefined) {
  if (message) console.warn(`notebook pyright ${scope} ${serviceId} failed`, message)
}

function handleBackgroundWorker(
  message: UnknownRecord,
  workerUrl: string,
  workers: Set<Worker>,
  serviceId: number,
  nextWorkerName: () => string,
) {
  if (message.type !== 'browser/newWorker') return false
  const port = message.port
  if (typeof MessagePort === 'undefined' || !(port instanceof MessagePort)) return true
  const background = new Worker(workerUrl, { name: nextWorkerName(), type: 'module' })
  workers.add(background)
  background.addEventListener('error', event => {
    reportPyrightWorkerError('background', serviceId, event.message)
  })
  background.postMessage(
    { type: 'browser/boot', mode: 'background', initialData: message.initialData, port },
    [port],
  )
  return true
}

function createPyrightTransport(
  workerUrl: string,
  rootUri: string,
  initialFiles: JsonObject,
  typeshedManifestUrl: string,
  serviceId: number,
): Transport {
  if (typeof Worker === 'undefined') throw new Error('browser workers are unavailable')
  const foreground = new Worker(workerUrl, {
    name: `Pyright-foreground-${serviceId}`,
    type: 'module',
  })
  const workers = new Set<Worker>([foreground])
  const handlers = new Set<(value: string) => void>()
  const channel = new MessageChannel()
  const foregroundPort = channel.port2
  const pendingMessages: JsonObject[] = []
  let backgroundCount = 0
  let foregroundReady = false

  const nextWorkerName = () => {
    backgroundCount += 1
    return `Pyright-background-${serviceId}-${backgroundCount}`
  }

  foreground.addEventListener('message', event => {
    if (!isRecord(event.data)) return
    if (event.data.type === 'browser/ready') {
      foregroundReady = true
      for (const message of pendingMessages.splice(0)) foregroundPort.postMessage(message)
      return
    }
    if (handleBackgroundWorker(event.data, workerUrl, workers, serviceId, nextWorkerName)) return
  })
  foregroundPort.addEventListener('message', event => {
    if (!isRecord(event.data)) return
    if (handleServerRequest(event.data, foregroundPort)) return
    const message = JSON.stringify(event.data)
    for (const handler of handlers) handler(message)
  })
  foreground.addEventListener('error', event => {
    reportPyrightWorkerError('foreground', serviceId, event.message)
  })
  foreground.addEventListener('messageerror', () => {
    console.warn(`notebook pyright foreground ${serviceId} message failed`)
  })
  foregroundPort.start()
  foreground.postMessage({ type: 'browser/boot', mode: 'foreground', port: channel.port1 }, [
    channel.port1,
  ])

  return {
    send(message: string) {
      const payload = initializeMessage(
        parseTransportMessage(message),
        rootUri,
        initialFiles,
        typeshedManifestUrl,
      )
      if (foregroundReady) {
        foregroundPort.postMessage(payload)
      } else {
        pendingMessages.push(payload)
      }
    },
    subscribe(handler: (value: string) => void) {
      handlers.add(handler)
    },
    unsubscribe(handler: (value: string) => void) {
      handlers.delete(handler)
      if (handlers.size > 0) return
      foregroundPort.close()
      for (const worker of workers) worker.terminate()
      workers.clear()
    },
  }
}

class NotebookWorkspaceFile implements WorkspaceFile {
  constructor(
    readonly id: string,
    readonly uri: string,
    public languageId: string,
    public version: number,
    public doc: Text,
    public executionIndex: number | null,
    public view: EditorView | null = null,
  ) {}

  getView() {
    return this.view
  }
}

class NotebookPythonWorkspace extends Workspace {
  files: NotebookWorkspaceFile[] = []

  private readonly filesByUri = new Map<string, NotebookWorkspaceFile>()
  private readonly fileVersions = new Map<string, number>()
  private readonly openedUris = new Set<string>()
  private opening: Promise<void> | undefined

  constructor(
    client: LSPClient,
    private readonly rootUri: string,
    private readonly sourcePath: string,
    private readonly cells: () => readonly NotebookLspCell[],
  ) {
    super(client)
    this.refreshFiles()
  }

  override connected() {
    this.queueOpenFiles()
  }

  override disconnected() {
    this.openedUris.clear()
    this.opening = undefined
  }

  override getFile(uri: string) {
    this.refreshFiles()
    return this.filesByUri.get(uri) ?? null
  }

  override requestFile(uri: string) {
    return Promise.resolve(this.getFile(uri))
  }

  override openFile(uri: string, languageId: string, view: EditorView) {
    this.refreshFiles()
    const file = this.filesByUri.get(uri)
    if (!file) return
    file.languageId = languageId
    file.view = view
    const changed = this.updateFileDoc(file, view.state.doc)
    if (!this.openedUris.has(file.uri)) {
      if (this.client.connected) this.queueOpenFiles()
      return
    }
    if (changed) this.sendTextChange(file)
  }

  override closeFile(uri: string, view: EditorView) {
    const file = this.filesByUri.get(uri)
    if (!file || file.view !== view) return
    const changed = this.updateFileDoc(file, view.state.doc)
    file.view = null
    if (changed) this.sendTextChange(file)
  }

  override syncFiles() {
    const updates = []
    for (const file of this.files) {
      const plugin = file.view ? LSPPlugin.get(file.view) : undefined
      if (!file.view || !plugin || plugin.unsyncedChanges.empty) continue
      updates.push({ file, prevDoc: file.doc, changes: plugin.unsyncedChanges })
      file.doc = file.view.state.doc
      file.version = this.nextFileVersion(file.uri)
      plugin.clear()
    }
    return updates
  }

  private updateFileDoc(file: NotebookWorkspaceFile, nextDoc: Text) {
    if (file.doc.eq(nextDoc)) return false
    file.doc = nextDoc
    file.version = this.nextFileVersion(file.uri)
    return true
  }

  private sendTextChange(file: NotebookWorkspaceFile) {
    if (!this.openedUris.has(file.uri)) return
    this.client.notification('textDocument/didChange', {
      textDocument: { uri: file.uri, version: file.version },
      contentChanges: [{ text: file.doc.toString() }],
    })
  }

  private refreshFiles() {
    const nextFiles: NotebookWorkspaceFile[] = []
    const nextByUri = new Map<string, NotebookWorkspaceFile>()
    for (const cell of this.cells()) {
      const uri = notebookCellUri(this.rootUri, this.sourcePath, cell.id)
      const existing = this.filesByUri.get(uri)
      if (existing) {
        existing.languageId = cell.language
        existing.executionIndex = cell.executionIndex
        if (!existing.view) {
          const changed = this.updateFileDoc(existing, notebookText(cell.source))
          if (changed) this.sendTextChange(existing)
        }
        nextFiles.push(existing)
        nextByUri.set(uri, existing)
        continue
      }
      const file = new NotebookWorkspaceFile(
        cell.id,
        uri,
        cell.language,
        this.nextFileVersion(uri),
        notebookText(cell.source),
        cell.executionIndex,
      )
      nextFiles.push(file)
      nextByUri.set(uri, file)
    }
    const removedUris = new Set(this.filesByUri.keys())
    for (const uri of nextByUri.keys()) removedUris.delete(uri)
    for (const uri of removedUris) {
      if (!this.openedUris.delete(uri)) continue
      this.client.didClose(uri)
    }
    this.files = nextFiles
    this.filesByUri.clear()
    nextByUri.forEach((file, uri) => this.filesByUri.set(uri, file))
  }

  private queueOpenFiles() {
    if (this.opening) return
    this.opening = this.client.initializing
      .then(() => {
        this.opening = undefined
        if (this.client.connected) this.openFiles()
      })
      .catch(error => {
        this.opening = undefined
        if (!notebookIgnoredLspError(error)) console.warn('notebook pyright open failed', error)
      })
  }

  private openFiles() {
    this.refreshFiles()
    for (const file of this.files) {
      if (this.openedUris.has(file.uri)) continue
      this.client.didOpen(file)
      this.openedUris.add(file.uri)
    }
  }

  private nextFileVersion(uri: string) {
    const version = (this.fileVersions.get(uri) ?? -1) + 1
    this.fileVersions.set(uri, version)
    return version
  }
}

class NotebookPythonLspService {
  private client: Promise<LSPClient> | undefined
  private cells: () => readonly NotebookLspCell[]

  constructor(
    private readonly rootUri: string,
    private readonly rootPath: string,
    private readonly sourcePath: string,
    cells: () => readonly NotebookLspCell[],
  ) {
    this.cells = cells
  }

  update(config: NotebookLspConfig) {
    this.cells = config.cells
  }

  async warm(): Promise<void> {
    await this.ensureClient()
  }

  async extensions(config: NotebookLspConfig): Promise<readonly Extension[]> {
    const client = await this.ensureClient()
    return [client.plugin(notebookFileUri(config), 'python')]
  }

  private async ensureClient() {
    this.client ??= this.createClient()
    return this.client
  }

  private async createClient() {
    const [{ LSPClient, serverCompletionSource, serverDiagnostics, signatureHelp }, workerUrl] =
      await Promise.all([import('@codemirror/lsp-client'), loadNotebookWorkerUrl()])
    const workspaceFiles = notebookWorkspaceFiles(this.rootPath)
    const client = new LSPClient({
      rootUri: this.rootUri,
      workspace: client =>
        new NotebookPythonWorkspace(client, this.rootUri, this.sourcePath, () => this.cells()),
      timeout: 20000,
      sanitizeHTML: html => DOMPurify.sanitize(html),
      extensions: [
        { clientCapabilities: notebookClientCapabilities },
        notebookServerCompletion(serverCompletionSource),
        notebookHoverTooltips(300),
        signatureHelp({ keymap: false }),
        serverDiagnostics(),
      ],
    })
    client.connect(
      createPyrightTransport(
        workerUrl,
        this.rootUri,
        workspaceFiles,
        pyrightTypeshedManifestUrl(),
        notebookLspSequence,
      ),
    )
    notebookLspSequence += 1
    return client
  }
}

function notebookService(config: NotebookLspConfig) {
  const key = notebookServiceKey(config)
  let service = notebookServices.get(key)
  if (!service) {
    service = new NotebookPythonLspService(
      notebookRootUri(config),
      notebookRootPath(config),
      config.sourcePath,
      config.cells,
    )
    notebookServices.set(key, service)
  } else {
    service.update(config)
  }
  return service
}

export async function notebookLspExtensions(
  config: NotebookLspConfig,
): Promise<readonly Extension[]> {
  if (!config.enabled || !pythonLanguage(config.language)) return []
  try {
    return await notebookService(config).extensions(config)
  } catch (error) {
    console.warn('notebook pyright is unavailable', error)
    return []
  }
}

export async function warmNotebookLspRuntime(config: NotebookLspConfig): Promise<void> {
  if (!config.enabled || !pythonLanguage(config.language)) return
  try {
    await notebookService(config).warm()
  } catch (error) {
    console.warn('notebook pyright warmup failed', error)
  }
}

export const pyrightLspBridge: LspBridge = {
  async extensions(config: LspConfig): Promise<readonly Extension[]> {
    return notebookLspExtensions({
      enabled: config.enabled,
      runtimeId: config.runtimeId,
      sourcePath: config.sourcePath,
      cellId: String(config.cellId),
      language: config.language,
      cells: config.cells ?? (() => []),
    })
  },
}
