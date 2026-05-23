import {
  BrowserMessageReader,
  BrowserMessageWriter,
  createMessageConnection,
  type MessageConnection,
} from 'vscode-jsonrpc/lib/browser/main.js'
import {
  InitializeRequest,
  InitializedNotification,
  DidOpenTextDocumentNotification,
  DidChangeTextDocumentNotification,
  DidCloseTextDocumentNotification,
  CompletionRequest,
  HoverRequest,
  DefinitionRequest,
  SignatureHelpRequest,
  PublishDiagnosticsNotification,
  LogMessageNotification,
  RegistrationRequest,
  DiagnosticTag,
  type InitializeParams,
  type ServerCapabilities,
  type CompletionParams,
  type CompletionList,
  type CompletionItem,
  type HoverParams,
  type Hover,
  type DefinitionParams,
  type Location,
  type LocationLink,
  type SignatureHelpParams,
  type SignatureHelp,
  type TextDocumentContentChangeEvent,
  type PublishDiagnosticsParams,
} from 'vscode-languageserver-protocol/lib/common/api.js'

export type NotebookLspBootMessage = {
  type: 'browser/boot'
  mode: 'foreground' | 'background'
  initialData?: unknown
  port?: MessagePort
}

export type NotebookLspNewWorkerMessage = {
  type: 'browser/newWorker'
  initialData: unknown
  port: MessagePort
}

export type NotebookLspConfig = {
  workerUrl: string
  rootUri: string
  locale?: string
  initializationOptions?: unknown
  onDiagnostics?: (params: PublishDiagnosticsParams) => void
  onLogMessage?: (message: string) => void
}

let workerCounter = 0

export class NotebookLspClient {
  private connection: MessageConnection | undefined
  private workers: Worker[] = []
  private versions = new Map<string, number>()
  private initializePromise: Promise<void> | undefined
  capabilities: ServerCapabilities | undefined

  constructor(private config: NotebookLspConfig) {}

  async start(): Promise<void> {
    if (this.initializePromise) return this.initializePromise
    this.initializePromise = this.boot()
    return this.initializePromise
  }

  private async boot(): Promise<void> {
    const suffix = ++workerCounter
    const foreground = new Worker(this.config.workerUrl, {
      name: `Pyright-foreground-${suffix}`,
    })
    foreground.postMessage({ type: 'browser/boot', mode: 'foreground' } satisfies NotebookLspBootMessage)
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

    const connection = createMessageConnection(
      new BrowserMessageReader(foreground),
      new BrowserMessageWriter(foreground),
    )
    connection.onDispose(() => {
      for (const worker of this.workers) worker.terminate()
      this.workers = []
    })

    connection.onNotification(LogMessageNotification.type, params => {
      this.config.onLogMessage?.(params.message)
    })
    connection.onNotification(PublishDiagnosticsNotification.type, params => {
      this.config.onDiagnostics?.(params)
    })
    connection.onRequest(RegistrationRequest.type, () => {
      // basedpyright sends registrations even though we don't claim the capability; acknowledge with empty.
    })

    this.connection = connection
    connection.listen()

    const initializeParams: InitializeParams = {
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
          hover: {
            dynamicRegistration: false,
            contentFormat: ['markdown', 'plaintext'],
          },
          signatureHelp: {
            dynamicRegistration: false,
            signatureInformation: {
              documentationFormat: ['markdown', 'plaintext'],
              activeParameterSupport: true,
              parameterInformation: { labelOffsetSupport: true },
            },
          },
          definition: {
            dynamicRegistration: false,
            linkSupport: true,
          },
          publishDiagnostics: {
            relatedInformation: true,
            tagSupport: {
              valueSet: [DiagnosticTag.Unnecessary, DiagnosticTag.Deprecated],
            },
            versionSupport: false,
          },
        },
        workspace: {
          workspaceFolders: true,
          configuration: true,
          didChangeConfiguration: { dynamicRegistration: false },
        },
      },
    }
    const result = await connection.sendRequest(InitializeRequest.type, initializeParams)
    this.capabilities = result.capabilities
    connection.sendNotification(InitializedNotification.type, {})
  }

  stop(): void {
    this.connection?.dispose()
    this.connection = undefined
    this.initializePromise = undefined
    this.capabilities = undefined
    this.versions.clear()
  }

  didOpen(uri: string, source: string, languageId = 'python'): void {
    if (!this.connection) return
    this.connection.sendNotification(DidOpenTextDocumentNotification.type, {
      textDocument: {
        uri,
        languageId,
        version: this.nextVersion(uri),
        text: source,
      },
    })
  }

  didChange(uri: string, source: string): void {
    if (!this.connection) return
    const contentChanges: TextDocumentContentChangeEvent[] = [{ text: source }]
    this.connection.sendNotification(DidChangeTextDocumentNotification.type, {
      textDocument: { uri, version: this.nextVersion(uri) },
      contentChanges,
    })
  }

  didClose(uri: string): void {
    if (!this.connection) return
    this.connection.sendNotification(DidCloseTextDocumentNotification.type, {
      textDocument: { uri },
    })
    this.versions.delete(uri)
  }

  async completion(params: CompletionParams): Promise<CompletionList> {
    if (!this.connection) return { items: [], isIncomplete: true }
    const result = await this.connection.sendRequest(CompletionRequest.type, params)
    if (!result) return { items: [], isIncomplete: true }
    if (Array.isArray(result)) {
      return { items: result as CompletionItem[], isIncomplete: true }
    }
    return result
  }

  async hover(params: HoverParams): Promise<Hover | null> {
    if (!this.connection) return null
    return (await this.connection.sendRequest(HoverRequest.type, params)) ?? null
  }

  async definition(params: DefinitionParams): Promise<Location[] | LocationLink[] | null> {
    if (!this.connection) return null
    const result = await this.connection.sendRequest(DefinitionRequest.type, params)
    if (!result) return null
    return Array.isArray(result) ? result : [result]
  }

  async signatureHelp(params: SignatureHelpParams): Promise<SignatureHelp | null> {
    if (!this.connection) return null
    return (await this.connection.sendRequest(SignatureHelpRequest.type, params)) ?? null
  }

  private nextVersion(uri: string): number {
    const next = (this.versions.get(uri) ?? 0) + 1
    this.versions.set(uri, next)
    return next
  }
}

export function notebookCellUri(runtimeId: string, cellIndex: number): string {
  return `file:///notebook/${runtimeId}/cell-${cellIndex}.py`
}
