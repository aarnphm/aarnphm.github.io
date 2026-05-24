import type { LSPClient, Transport } from '@codemirror/lsp-client'
import { autocompletion, type CompletionSource } from '@codemirror/autocomplete'
import { EditorState, type Extension } from '@codemirror/state'
import DOMPurify from 'dompurify'
import {
  notebookPyrightAssetManifestChunks,
  notebookPyrightTypeshedFiles,
} from '../../util/notebook-pyright-assets'
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

export type NotebookLspConfig = {
  enabled: boolean
  runtimeId: string
  sourcePath: string
  cellId: string
  language: string
}

type JsonRpcId = string | number | null

const notebookPyrightWorkerManifestName = '../notebook-pyright-worker.json'
const notebookPyrightTypeshedManifestName = '../notebook-pyright-typeshed.json'

const notebookAnalysisSettings: JsonObject = {
  typeCheckingMode: 'basic',
  diagnosticMode: 'openFilesOnly',
  diagnosticSeverityOverrides: { reportMissingImports: 'none', reportMissingModuleSource: 'none' },
}

const notebookClientCapabilities: JsonObject = {
  workspace: { configuration: true, didChangeConfiguration: {}, workspaceFolders: true },
}

const notebookServices = new Map<string, NotebookPythonLspService>()

let notebookTypeshed: Promise<JsonObject> | undefined
let notebookWorkerUrl: Promise<string> | undefined
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

function notebookFileUri(config: NotebookLspConfig) {
  return `${notebookRootUri(config)}${encodeURIComponent(config.sourcePath)}/${encodeURIComponent(
    config.cellId,
  )}.py`
}

function notebookServiceKey(config: NotebookLspConfig) {
  return `${config.runtimeId}\u0000${config.sourcePath}`
}

function pyrightWorkerManifestUrl() {
  return new URL(notebookPyrightWorkerManifestName, import.meta.url).href
}

function pyrightTypeshedManifestUrl() {
  return new URL(notebookPyrightTypeshedManifestName, import.meta.url).href
}

async function fetchJsonObject(url: string, label: string): Promise<JsonObject> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${label} request failed with ${response.status}`)
  const value: unknown = await response.json()
  if (!isJsonObject(value)) throw new Error(`${label} is not a JSON object`)
  return value
}

async function fetchText(url: string, label: string): Promise<string> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${label} request failed with ${response.status}`)
  return response.text()
}

async function loadNotebookWorkerUrl(): Promise<string> {
  notebookWorkerUrl ??= (async () => {
    if (typeof URL.createObjectURL !== 'function') {
      throw new Error('notebook pyright worker blob URLs are unavailable')
    }
    const manifestUrl = pyrightWorkerManifestUrl()
    const manifest = await fetchJsonObject(manifestUrl, 'notebook pyright worker manifest')
    const chunks = await Promise.all(
      notebookPyrightAssetManifestChunks(manifest, 'notebook pyright worker').map(
        async chunkName => {
          const chunkUrl = new URL(chunkName, manifestUrl).href
          return fetchText(chunkUrl, `notebook pyright worker chunk ${chunkName}`)
        },
      ),
    )
    return URL.createObjectURL(new Blob(chunks, { type: 'text/javascript' }))
  })()
  return notebookWorkerUrl
}

async function loadNotebookTypeshedChunks(): Promise<JsonObject> {
  const manifestUrl = pyrightTypeshedManifestUrl()
  const manifest = await fetchJsonObject(manifestUrl, 'notebook pyright typeshed manifest')
  const files: Record<string, string> = {}
  await Promise.all(
    notebookPyrightAssetManifestChunks(manifest, 'notebook pyright typeshed').map(
      async chunkName => {
        const chunkUrl = new URL(chunkName, manifestUrl).href
        const chunk = await fetchJsonObject(
          chunkUrl,
          `notebook pyright typeshed chunk ${chunkName}`,
        )
        Object.assign(files, notebookPyrightTypeshedFiles(chunk))
      },
    ),
  )
  return { files }
}

async function loadNotebookTypeshed() {
  notebookTypeshed ??= loadNotebookTypeshedChunks()
  return notebookTypeshed
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

function notebookWorkspaceFiles(typeshed: JsonObject, rootPath: string): JsonObject {
  return {
    ...typeshed,
    files: { ...objectValue(typeshed.files), [`${rootPath}/.pyright-root`]: '' },
  }
}

function handleServerRequest(message: UnknownRecord, worker: Worker) {
  const method = stringValue(message.method)
  const id = messageId(message.id)
  if (method === undefined || id === undefined) return false

  if (method === 'workspace/configuration') {
    worker.postMessage(
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
    worker.postMessage(responseMessage(id, null))
    return true
  }
  if (method === 'workspace/applyEdit') {
    worker.postMessage(responseMessage(id, { applied: false }))
    return true
  }
  return false
}

function initializeMessage(message: JsonObject, rootUri: string, typeshed: JsonObject): JsonObject {
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
      initializationOptions: { files: typeshed },
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
  return [
    autocompletion({ defaultKeymap: false }),
    EditorState.languageData.of(() => [{ autocomplete: serverCompletionSource }]),
  ]
}

function parseTransportMessage(message: string) {
  const value: unknown = JSON.parse(message)
  if (!isJsonObject(value)) throw new Error('notebook pyright received a non-object message')
  return value
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
  const background = new Worker(workerUrl, { name: nextWorkerName() })
  workers.add(background)
  background.addEventListener('error', event => {
    console.error(`notebook pyright background ${serviceId} failed`, event.message)
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
  typeshed: JsonObject,
  serviceId: number,
): Transport {
  if (typeof Worker === 'undefined') throw new Error('browser workers are unavailable')
  const foreground = new Worker(workerUrl, { name: `Pyright-foreground-${serviceId}` })
  const workers = new Set<Worker>([foreground])
  const handlers = new Set<(value: string) => void>()
  let backgroundCount = 0

  const nextWorkerName = () => {
    backgroundCount += 1
    return `Pyright-background-${serviceId}-${backgroundCount}`
  }

  foreground.addEventListener('message', event => {
    if (!isRecord(event.data)) return
    if (handleBackgroundWorker(event.data, workerUrl, workers, serviceId, nextWorkerName)) return
    if (handleServerRequest(event.data, foreground)) return
    const message = JSON.stringify(event.data)
    for (const handler of handlers) handler(message)
  })
  foreground.addEventListener('error', event => {
    console.error(`notebook pyright foreground ${serviceId} failed`, event.message)
  })
  foreground.postMessage({ type: 'browser/boot', mode: 'foreground' })

  return {
    send(message: string) {
      foreground.postMessage(initializeMessage(parseTransportMessage(message), rootUri, typeshed))
    },
    subscribe(handler: (value: string) => void) {
      handlers.add(handler)
    },
    unsubscribe(handler: (value: string) => void) {
      handlers.delete(handler)
      if (handlers.size > 0) return
      for (const worker of workers) worker.terminate()
      workers.clear()
    },
  }
}

class NotebookPythonLspService {
  private client: Promise<LSPClient> | undefined

  constructor(
    private readonly rootUri: string,
    private readonly rootPath: string,
  ) {}

  async extensions(config: NotebookLspConfig): Promise<readonly Extension[]> {
    const client = await this.ensureClient()
    return [client.plugin(notebookFileUri(config), 'python')]
  }

  private async ensureClient() {
    this.client ??= this.createClient()
    return this.client
  }

  private async createClient() {
    const [
      { LSPClient, hoverTooltips, serverCompletionSource, serverDiagnostics, signatureHelp },
      typeshed,
      workerUrl,
    ] = await Promise.all([
      import('@codemirror/lsp-client'),
      loadNotebookTypeshed(),
      loadNotebookWorkerUrl(),
    ])
    const workspaceFiles = notebookWorkspaceFiles(typeshed, this.rootPath)
    const client = new LSPClient({
      rootUri: this.rootUri,
      timeout: 8000,
      sanitizeHTML: html => DOMPurify.sanitize(html),
      extensions: [
        { clientCapabilities: notebookClientCapabilities },
        notebookServerCompletion(serverCompletionSource),
        hoverTooltips({ hoverTime: 300 }),
        signatureHelp({ keymap: false }),
        serverDiagnostics(),
      ],
    })
    client.connect(
      createPyrightTransport(workerUrl, this.rootUri, workspaceFiles, notebookLspSequence),
    )
    notebookLspSequence += 1
    return client
  }
}

function notebookService(config: NotebookLspConfig) {
  const key = notebookServiceKey(config)
  let service = notebookServices.get(key)
  if (!service) {
    service = new NotebookPythonLspService(notebookRootUri(config), notebookRootPath(config))
    notebookServices.set(key, service)
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
