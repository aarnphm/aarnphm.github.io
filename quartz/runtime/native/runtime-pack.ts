import type { CellId } from '../../util/notebook/types'
import type { KernelFactoryOptions } from '../notebook/backend'
import type {
  Kernel,
  KernelExecuteOptions,
  KernelInitOptions,
  RuntimeAssetRequest,
  RuntimeAssetResult,
  RuntimeDownload,
  RuntimeEvent,
  RuntimeFileResult,
} from '../notebook/kernel'
import type { NotebookRuntimeOutput } from '../notebook/types'
import { isRecord, readNumber, readString } from '../../util/type-guards'
import { AsyncEventQueue } from '../notebook/async-event-queue'

export type NativeRuntimeLanguage = 'go' | 'haskell' | 'mojo' | 'ocaml' | 'rust'

export type NativeRuntimePackEntry = { readonly worker: string; readonly assets: readonly string[] }

export type NativeRuntimePackManifest = {
  readonly version: 1
  readonly runtimes: Partial<Record<NativeRuntimeLanguage, NativeRuntimePackEntry>>
}

type ReadyMessage = { type: 'ready'; runtimeId: string }
type OutputMessage = {
  type: 'output'
  runtimeId: string
  cellId: CellId
  output: NotebookRuntimeOutput
}
type DoneMessage = { type: 'done'; runtimeId: string; cellId: CellId; failed: boolean }
type AssetMessage = {
  type: 'asset'
  runtimeId: string
  cellId: CellId
  assetId: string
  url: string
}
type DownloadMessage = RuntimeDownload & { type: 'download' }
type FileResultMessage = RuntimeFileResult & {
  type: 'file-result'
  runtimeId: string
  requestId: string
}
type StatusMessage = { type: 'status'; runtimeId: string; text: string }

type NativeRuntimePackWorkerMessage =
  | ReadyMessage
  | OutputMessage
  | DoneMessage
  | AssetMessage
  | DownloadMessage
  | FileResultMessage
  | StatusMessage

type ActiveExecution = { readonly cellId: CellId; readonly queue: AsyncEventQueue<RuntimeEvent> }
type RuntimeDebugOutput = NonNullable<Extract<NotebookRuntimeOutput, { type: 'error' }>['debug']>

const source = 'quartz-notebook-runtime'
const nativeRuntimeLanguages: readonly NativeRuntimeLanguage[] = [
  'go',
  'haskell',
  'mojo',
  'ocaml',
  'rust',
]

export const emptyNativeRuntimePackManifest: NativeRuntimePackManifest = {
  version: 1,
  runtimes: {},
}

function isNativeRuntimeLanguage(value: string): value is NativeRuntimeLanguage {
  switch (value) {
    case 'go':
    case 'haskell':
    case 'mojo':
    case 'ocaml':
    case 'rust':
      return true
    default:
      return false
  }
}

function isCellId(value: unknown): value is CellId {
  return typeof value === 'string' && value.length > 0
}

function readStringArray(value: unknown): readonly string[] | undefined {
  if (!Array.isArray(value)) return undefined
  return value.every(item => typeof item === 'string') ? value : undefined
}

function readRuntimeDebugOutput(value: unknown): RuntimeDebugOutput | undefined {
  if (!isRecord(value)) return undefined
  const phase = readString(value, 'phase')
  if (!phase) return undefined
  return {
    phase,
    cellId: readString(value, 'cellId'),
    errorName: readString(value, 'errorName'),
    errorMessage: readString(value, 'errorMessage'),
    stack: readString(value, 'stack'),
  }
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
  if (type === 'json') {
    const text = readString(value, 'text')
    if (text !== undefined) return { type, text }
  }
  if (type === 'html') {
    const html = readString(value, 'html')
    if (html !== undefined) return { type, html }
  }
  if (type === 'success') return { type }
}

function readPackEntry(value: unknown): NativeRuntimePackEntry | undefined {
  if (!isRecord(value)) return undefined
  const worker = readString(value, 'worker')
  if (!worker) return undefined
  return { worker, assets: readStringArray(value.assets) ?? [] }
}

export function readNativeRuntimePackManifest(
  value: unknown,
): NativeRuntimePackManifest | undefined {
  if (!isRecord(value) || value.version !== 1 || !isRecord(value.runtimes)) return undefined
  const runtimes: Partial<Record<NativeRuntimeLanguage, NativeRuntimePackEntry>> = {}
  for (const language of nativeRuntimeLanguages) {
    const entry = readPackEntry(value.runtimes[language])
    if (entry) runtimes[language] = entry
  }
  return { version: 1, runtimes }
}

function readWorkerMessage(value: unknown): NativeRuntimePackWorkerMessage | undefined {
  if (!isRecord(value) || value.source !== source) return undefined
  const type = readString(value, 'type')
  const runtimeId = readString(value, 'runtimeId')
  if (!type || !runtimeId) return undefined
  if (type === 'ready') return { type, runtimeId }
  if (type === 'done') {
    const cellId = value.cellId
    if (isCellId(cellId)) return { type, runtimeId, cellId, failed: value.failed === true }
  }
  if (type === 'output') {
    const cellId = value.cellId
    const output = readRuntimeOutput(value.output)
    if (isCellId(cellId) && output) return { type, runtimeId, cellId, output }
  }
  if (type === 'asset') {
    const cellId = value.cellId
    const assetId = readString(value, 'assetId')
    const url = readString(value, 'url')
    if (isCellId(cellId) && assetId && url) return { type, runtimeId, cellId, assetId, url }
  }
  if (type === 'download') {
    const cellId = value.cellId
    const filename = readString(value, 'filename')
    const contentType = readString(value, 'contentType')
    if (isCellId(cellId) && filename && contentType && value.bytes instanceof ArrayBuffer) {
      return { type, runtimeId, cellId, filename, contentType, bytes: value.bytes }
    }
  }
  if (type === 'file-result') {
    const requestId = readString(value, 'requestId')
    const status = readNumber(value, 'status')
    const statusText = readString(value, 'statusText')
    const contentType = readString(value, 'contentType')
    if (!requestId || status === undefined || !statusText || !contentType) return undefined
    return {
      type,
      runtimeId,
      requestId,
      ok: value.ok === true,
      status,
      statusText,
      contentType,
      ...(value.bytes instanceof ArrayBuffer ? { bytes: value.bytes } : {}),
      ...(readString(value, 'error') !== undefined ? { error: readString(value, 'error') } : {}),
    }
  }
  if (type === 'status') {
    const text = readString(value, 'text')
    if (text !== undefined) return { type, runtimeId, text }
  }
}

function transferableBytes(bytes: ArrayBuffer | undefined): Transferable[] {
  return bytes ? [bytes] : []
}

function resolveManifestAsset(manifestUrl: string, asset: string): string {
  return new URL(asset, manifestUrl).href
}

async function runtimePackManifest(manifestUrl: string): Promise<NativeRuntimePackManifest> {
  const response = await fetch(manifestUrl)
  if (!response.ok) {
    throw new Error(
      `native runtime pack manifest request failed with ${response.status}: ${response.statusText}`,
    )
  }
  const manifest = readNativeRuntimePackManifest(await response.json())
  if (!manifest) throw new Error('native runtime pack manifest is invalid')
  return manifest
}

export class NativeRuntimePackKernel implements Kernel {
  readonly language: NativeRuntimeLanguage
  private readonly runtimeId: string
  private readonly manifestUrl: string
  private readonly resolveAsset?: KernelFactoryOptions['resolveAsset']
  private readonly download?: KernelFactoryOptions['download']
  private readonly status?: KernelFactoryOptions['status']
  private worker: Worker | undefined
  private ready: Promise<void> | undefined
  private readyResolve: (() => void) | undefined
  private readyReject: ((error: unknown) => void) | undefined
  private active: ActiveExecution | undefined
  private fileSequence = 0
  private fileWaiters = new Map<string, (result: RuntimeFileResult | undefined) => void>()

  constructor(language: NativeRuntimeLanguage, opts: KernelFactoryOptions) {
    this.language = language
    this.runtimeId = opts.runtimeId
    if (!opts.workerUrl) throw new Error(`${language} runtime pack manifest URL is not configured`)
    this.manifestUrl = String(opts.workerUrl)
    this.resolveAsset = opts.resolveAsset
    this.download = opts.download
    this.status = opts.status
  }

  async init(_opts: KernelInitOptions): Promise<void> {
    if (this.ready) return this.ready
    this.ready = this.start()
    return this.ready
  }

  async *execute(
    cellId: CellId,
    sourceText: string,
    opts: KernelExecuteOptions = {},
  ): AsyncIterable<RuntimeEvent> {
    if (this.active) throw new Error(`runtime is already executing ${this.active.cellId}`)
    await this.init({ signal: new AbortController().signal })
    const queue = new AsyncEventQueue<RuntimeEvent>()
    this.active = { cellId, queue }
    queue.push({ type: 'started', cellId })
    this.worker?.postMessage({
      source,
      type: 'run',
      runtimeId: this.runtimeId,
      cellId,
      code: sourceText,
      debug: opts.debug === true,
      modules: opts.modules ?? [],
    })
    try {
      for await (const event of queue) yield event
    } finally {
      if (this.active?.queue === queue) this.active = undefined
    }
  }

  interrupt(): void {
    this.active?.queue.push({ type: 'interrupted', cellId: this.active.cellId })
    this.terminate(new Error('runtime stopped'))
  }

  async reset(): Promise<void> {
    this.terminate(new Error('runtime reset'))
  }

  async dispose(): Promise<void> {
    this.terminate(new Error('runtime disposed'))
  }

  private async start(): Promise<void> {
    const manifest = await runtimePackManifest(this.manifestUrl)
    const entry = manifest.runtimes[this.language]
    if (!entry) {
      throw new Error(
        `${this.language} notebook cells need a self-hosted WebAssembly runtime pack. ${this.manifestUrl} does not list ${this.language}.`,
      )
    }
    if (typeof Worker === 'undefined') throw new Error('browser workers are unavailable')
    await new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve
      this.readyReject = reject
      const worker = new Worker(resolveManifestAsset(this.manifestUrl, entry.worker), {
        type: 'module',
      })
      worker.addEventListener('message', this.onWorkerMessage)
      worker.addEventListener('error', this.onWorkerError)
      this.worker = worker
      worker.postMessage({
        source,
        type: 'init',
        runtimeId: this.runtimeId,
        language: this.language,
        manifestUrl: this.manifestUrl,
        assets: entry.assets.map(asset => resolveManifestAsset(this.manifestUrl, asset)),
      })
    })
  }

  private runtimeFile(path: string): Promise<RuntimeFileResult | undefined> {
    if (!this.worker) return Promise.resolve(undefined)
    const requestId = `runtime-file-${++this.fileSequence}`
    return new Promise(resolve => {
      const timeout = globalThis.setTimeout(() => {
        this.fileWaiters.delete(requestId)
        resolve(undefined)
      }, 10_000)
      this.fileWaiters.set(requestId, result => {
        globalThis.clearTimeout(timeout)
        resolve(result)
      })
      this.worker?.postMessage({ source, type: 'file', runtimeId: this.runtimeId, requestId, path })
    })
  }

  private onWorkerMessage = (event: MessageEvent<unknown>) => {
    const message = readWorkerMessage(event.data)
    if (!message || message.runtimeId !== this.runtimeId) return
    void this.handleWorkerMessage(message)
  }

  private onWorkerError = (event: ErrorEvent) => {
    const error = event.error instanceof Error ? event.error : new Error(event.message)
    this.readyReject?.(error)
    this.active?.queue.fail(error)
  }

  private async handleWorkerMessage(message: NativeRuntimePackWorkerMessage): Promise<void> {
    if (message.type === 'ready') {
      this.readyResolve?.()
      this.readyResolve = undefined
      this.readyReject = undefined
      return
    }
    if (message.type === 'status') {
      this.status?.(message.text)
      this.active?.queue.push({ type: 'status', text: message.text })
      return
    }
    if (message.type === 'file-result') {
      const waiter = this.fileWaiters.get(message.requestId)
      if (!waiter) return
      this.fileWaiters.delete(message.requestId)
      waiter(message)
      return
    }
    if (message.type === 'asset') {
      await this.handleAssetRequest(message)
      return
    }
    if (message.type === 'download') {
      this.download?.(message)
      this.active?.queue.push({ type: 'download', download: message })
      return
    }
    const active = this.active
    if (!active || active.cellId !== message.cellId) return
    if (message.type === 'output') {
      active.queue.push({ type: 'output', cellId: message.cellId, output: message.output })
      return
    }
    if (message.type === 'done') {
      active.queue.push({
        type: 'done',
        cellId: message.cellId,
        executionCount: null,
        failed: message.failed,
      })
      active.queue.close()
      this.active = undefined
    }
  }

  private async handleAssetRequest(message: AssetMessage): Promise<void> {
    const request: RuntimeAssetRequest = {
      runtimeId: message.runtimeId,
      cellId: message.cellId,
      assetId: message.assetId,
      url: message.url,
    }
    if (!this.resolveAsset) {
      this.active?.queue.push({ type: 'asset', ...request })
      return
    }
    const result = await this.resolveAsset(request, path => this.runtimeFile(path))
    this.postAssetResult(result)
  }

  private postAssetResult(result: RuntimeAssetResult): void {
    this.worker?.postMessage(
      { source, type: 'asset-result', ...result },
      transferableBytes(result.bytes),
    )
  }

  private terminate(error: Error): void {
    this.worker?.removeEventListener('message', this.onWorkerMessage)
    this.worker?.removeEventListener('error', this.onWorkerError)
    this.worker?.terminate()
    this.worker = undefined
    this.ready = undefined
    this.readyResolve = undefined
    this.readyReject = undefined
    this.active?.queue.fail(error)
    this.active = undefined
    for (const waiter of this.fileWaiters.values()) waiter(undefined)
    this.fileWaiters.clear()
  }
}

export function nativeRuntimeCanExecute(): { readonly ok: true } {
  return { ok: true }
}

export function nativeRuntimePackKernelFactory(
  language: string,
  opts: KernelFactoryOptions,
): Kernel {
  if (!isNativeRuntimeLanguage(language)) {
    throw new Error(`${language} is not a native runtime pack language`)
  }
  return new NativeRuntimePackKernel(language, opts)
}
