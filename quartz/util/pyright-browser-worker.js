import { InvalidatedReason } from 'pyright-internal/analyzer/backgroundAnalysisProgram'
import { ImportResolver } from 'pyright-internal/analyzer/importResolver'
import {
  BackgroundAnalysisBase,
  BackgroundAnalysisRunnerBase,
} from 'pyright-internal/backgroundAnalysisBase'
import { DefaultCancellationProvider } from 'pyright-internal/common/cancellationUtils'
import { nullFileWatcherHandler } from 'pyright-internal/common/fileWatcher'
import { NoAccessHost } from 'pyright-internal/common/host'
import { normalizeSlashes } from 'pyright-internal/common/pathUtils'
import { ServiceKeys } from 'pyright-internal/common/serviceKeys'
import { Uri } from 'pyright-internal/common/uri/uri'
import { getRootUri } from 'pyright-internal/common/uri/uriUtils'
import {
  createWorker,
  initializeWorkersHost,
  parentPort,
  shallowReplace,
} from 'pyright-internal/common/workersHost'
import { RealLanguageServer } from 'pyright-internal/realLanguageServer'
import { TestFileSystem } from 'pyright-internal/tests/harness/vfs/filesystem'
import {
  BrowserMessageReader,
  BrowserMessageWriter,
  createConnection,
} from 'vscode-languageserver/browser'

const workerScope = self

function isRecord(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringValue(value) {
  return typeof value === 'string' ? value : undefined
}

function stringArrayValue(value) {
  if (!Array.isArray(value)) return undefined
  return value.every(item => typeof item === 'string') ? value : undefined
}

function initialFilesFromOptions(value) {
  if (!isRecord(value)) return undefined
  return isRecord(value.files) ? value.files : value
}

function typeshedManifestUrlFromOptions(value) {
  return isRecord(value) ? stringValue(value.typeshedManifestUrl) : undefined
}

function typeshedFilesFromChunk(value, label) {
  const files = isRecord(value) && isRecord(value.files) ? value.files : undefined
  if (!files) throw new Error(`${label} has invalid files`)
  for (const [path, content] of Object.entries(files)) {
    if (typeof content !== 'string') throw new Error(`${label} contains invalid file ${path}`)
  }
  return files
}

async function fetchJsonObject(url, label) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${label} request failed with ${response.status}`)
  const value = await response.json()
  if (!isRecord(value)) throw new Error(`${label} is not a JSON object`)
  return value
}

async function loadTypeshedFiles(manifestUrl) {
  const manifest = await fetchJsonObject(manifestUrl, 'notebook pyright typeshed manifest')
  const chunks = stringArrayValue(manifest.chunks)
  if (!chunks) throw new Error('notebook pyright typeshed manifest has invalid chunks')
  const files = {}
  await Promise.all(
    chunks.map(async chunkName => {
      const chunkUrl = new URL(chunkName, manifestUrl).href
      const chunk = await fetchJsonObject(chunkUrl, `notebook pyright typeshed chunk ${chunkName}`)
      Object.assign(files, typeshedFilesFromChunk(chunk, chunkName))
    }),
  )
  return files
}

function createNotebookFileSystem() {
  return new TestFileSystem(false, { cwd: normalizeSlashes('/') })
}

let notebookTypeshedFiles

class NotebookPyrightBrowserServer extends RealLanguageServer {
  #initialFiles

  constructor(connection) {
    const fileSystem = createNotebookFileSystem()
    super(
      connection,
      0,
      fileSystem,
      new DefaultCancellationProvider(),
      fileSystem,
      nullFileWatcherHandler,
    )
  }

  createBackgroundAnalysis(_programRoot, workspaceRoot) {
    const analysis = new NotebookPyrightBackgroundAnalysis(workspaceRoot, this.serviceProvider)
    if (this.#initialFiles) analysis.initializeFileSystem(this.#initialFiles)
    return analysis
  }

  async getSettings(workspace) {
    const settings = await super.getSettings(workspace)
    settings.typeshedPath = Uri.parse('file:///typeshed', this.serverOptions.serviceProvider)
    settings.extraPaths = [
      ...(Array.isArray(settings.extraPaths) ? settings.extraPaths : []),
      Uri.parse('file:///typeshed/stdlib', this.serverOptions.serviceProvider),
    ]
    return settings
  }

  setupConnection(supportedCommands, supportedCodeActions) {
    super.setupConnection(supportedCommands, supportedCodeActions)
    this.connection.onNotification('pyright/createFile', params => {
      const filePath = Uri.parse(params.uri, this.serverOptions.serviceProvider).getPath()
      this.serverOptions.serviceProvider.fs().apply({ [filePath]: '' })
      for (const workspace of this.workspaceFactory.items()) {
        const backgroundAnalysis = workspace.service.backgroundAnalysisProgram.backgroundAnalysis
        backgroundAnalysis?.createFile(params)
        workspace.service.invalidateAndForceReanalysis(InvalidatedReason.Nunya)
      }
    })
    this.connection.onNotification('pyright/deleteFile', params => {
      const fileUri = Uri.parse(params.uri, this.serverOptions.serviceProvider)
      this.serverOptions.serviceProvider.fs().unlinkSync(fileUri)
      for (const workspace of this.workspaceFactory.items()) {
        const backgroundAnalysis = workspace.service.backgroundAnalysisProgram.backgroundAnalysis
        backgroundAnalysis?.deleteFile(params)
        workspace.service.invalidateAndForceReanalysis(InvalidatedReason.Nunya)
      }
    })
  }

  async initialize(params, supportedCommands, supportedCodeActions) {
    const options = params.initializationOptions
    const initialFiles = initialFilesFromOptions(options?.files) ?? {}
    const typeshedManifestUrl = typeshedManifestUrlFromOptions(options)
    const typeshedFiles = typeshedManifestUrl
      ? (notebookTypeshedFiles ??= loadTypeshedFiles(typeshedManifestUrl))
      : undefined
    const files = { ...(typeshedFiles ? await typeshedFiles : {}), ...initialFiles }
    this.#initialFiles = files
    this.serverOptions.serviceProvider.fs().apply(files)
    return super.initialize(params, supportedCommands, supportedCodeActions)
  }

  createHost() {
    return new NoAccessHost()
  }
}

class NotebookPyrightBackgroundAnalysis extends BackgroundAnalysisBase {
  static workerIndex = 0

  constructor(workspaceRoot, serviceProvider) {
    super(serviceProvider.console())
    NotebookPyrightBackgroundAnalysis.workerIndex += 1
    const index = NotebookPyrightBackgroundAnalysis.workerIndex
    const initialData = {
      rootUri: getRootUri(serviceProvider)?.toString() ?? '',
      tempFileName: serviceProvider.get(ServiceKeys.tempFile).tmpdir().getFilePath(),
      serviceId: index.toString(),
      cancellationFolderName: undefined,
      runner: undefined,
      workerIndex: index,
      workspaceRootUri: workspaceRoot.toString(),
    }
    this.setup(createWorker(initialData))
  }
}

class NotebookPyrightBackgroundAnalysisRunner extends BackgroundAnalysisRunnerBase {
  constructor(initialData, serviceProvider) {
    super(parentPort(), initialData, serviceProvider)
  }

  createRealFileSystem() {
    return createNotebookFileSystem()
  }

  createRealTempFile() {
    return this.createRealFileSystem()
  }

  createHost() {
    return new NoAccessHost()
  }

  createImportResolver(serviceProvider, options, host) {
    return new ImportResolver(serviceProvider, options, host)
  }
}

class NotebookPyrightWorkersHost {
  #parentPort

  constructor(parentPort) {
    this.#parentPort = parentPort
  }

  threadId() {
    return workerScope.name
  }

  parentPort() {
    return this.#parentPort ? new NotebookPyrightMessagePort(this.#parentPort) : null
  }

  createWorker(initialData) {
    const channel = new MessageChannel()
    workerScope.postMessage({ type: 'browser/newWorker', initialData, port: channel.port1 }, [
      channel.port1,
    ])
    channel.port1.start()
    channel.port2.start()
    return new NotebookPyrightMessagePort(channel.port2)
  }

  createMessageChannel() {
    const channel = new MessageChannel()
    return {
      port1: new NotebookPyrightMessagePort(channel.port1),
      port2: new NotebookPyrightMessagePort(channel.port2),
    }
  }
}

class NotebookPyrightMessagePort {
  #delegate

  constructor(delegate) {
    this.#delegate = delegate
  }

  unwrap() {
    return this.#delegate
  }

  postMessage(value, transferList) {
    if (transferList) {
      this.#delegate.postMessage(unwrapForSend(value), unwrapForSend(transferList))
      return
    }
    this.#delegate.postMessage(value)
  }

  on(type, listener) {
    if (type !== 'message') return
    this.#delegate.addEventListener('message', event => listener(wrapOnReceive(event.data)))
  }

  start() {
    this.#delegate.start()
  }

  close() {
    this.#delegate.close()
  }

  terminate() {
    this.#delegate.close()
    return Promise.resolve(0)
  }
}

function unwrapForSend(value) {
  return shallowReplace(value, item =>
    item instanceof NotebookPyrightMessagePort ? item.unwrap() : item,
  )
}

function wrapOnReceive(value) {
  return shallowReplace(value, item =>
    item instanceof MessagePort ? new NotebookPyrightMessagePort(item) : item,
  )
}

function bootForeground(port) {
  if (!(port instanceof MessagePort))
    throw new Error('notebook pyright foreground boot has invalid port')
  initializeWorkersHost(new NotebookPyrightWorkersHost())
  port.start()
  workerScope.notebookPyrightApp = new NotebookPyrightBrowserServer(
    createConnection(new BrowserMessageReader(port), new BrowserMessageWriter(port)),
  )
  workerScope.postMessage({ type: 'browser/ready' })
}

function bootBackground(initialData, port) {
  if (!initialData) throw new Error('notebook pyright background boot is missing initialData')
  if (!(port instanceof MessagePort))
    throw new Error('notebook pyright background boot has invalid port')
  initializeWorkersHost(new NotebookPyrightWorkersHost(port))
  workerScope.notebookPyrightApp = new NotebookPyrightBackgroundAnalysisRunner(initialData)
  workerScope.notebookPyrightApp.start()
}

workerScope.addEventListener('message', event => {
  if (!isRecord(event.data) || event.data.type !== 'browser/boot') return
  try {
    if (event.data.mode === 'foreground') {
      bootForeground(event.data.port)
      return
    }
    if (event.data.mode === 'background') {
      bootBackground(event.data.initialData, event.data.port)
      return
    }
    throw new Error(`notebook pyright worker received invalid boot mode ${event.data.mode}`)
  } catch (error) {
    workerScope.close()
    throw error
  }
})
