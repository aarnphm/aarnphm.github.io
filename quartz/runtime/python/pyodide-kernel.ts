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

type WorkerMessage =
  | ReadyMessage
  | OutputMessage
  | DoneMessage
  | AssetMessage
  | DownloadMessage
  | FileResultMessage
  | StatusMessage

type ActiveExecution = { readonly cellId: CellId; readonly queue: AsyncEventQueue<RuntimeEvent> }

function readRuntimeDebugOutput(
  value: unknown,
): NotebookRuntimeOutput extends infer O
  ? O extends { type: 'error'; debug?: infer Debug }
    ? Debug
    : never
  : never {
  if (!isRecord(value)) return undefined as never
  const phase = readString(value, 'phase')
  if (!phase) return undefined as never
  return {
    phase,
    cellId: readString(value, 'cellId'),
    errorName: readString(value, 'errorName'),
    errorMessage: readString(value, 'errorMessage'),
    stack: readString(value, 'stack'),
  } as never
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

function readWorkerMessage(value: unknown): WorkerMessage | undefined {
  if (!isRecord(value) || value.source !== 'quartz-notebook-runtime') return undefined
  const type = readString(value, 'type')
  const runtimeId = readString(value, 'runtimeId')
  if (!type || !runtimeId) return undefined
  if (type === 'ready') return { type, runtimeId }
  if (type === 'done') {
    const cellId = readString(value, 'cellId') as CellId | undefined
    if (cellId) return { type, runtimeId, cellId, failed: value.failed === true }
  }
  if (type === 'output') {
    const cellId = readString(value, 'cellId') as CellId | undefined
    const output = readRuntimeOutput(value.output)
    if (cellId && output) return { type, runtimeId, cellId, output }
  }
  if (type === 'asset') {
    const cellId = readString(value, 'cellId') as CellId | undefined
    const assetId = readString(value, 'assetId')
    const url = readString(value, 'url')
    if (cellId && assetId && url) return { type, runtimeId, cellId, assetId, url }
  }
  if (type === 'download') {
    const cellId = readString(value, 'cellId') as CellId | undefined
    const filename = readString(value, 'filename')
    const contentType = readString(value, 'contentType')
    if (cellId && filename && contentType && value.bytes instanceof ArrayBuffer) {
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

export class PyodideKernel implements Kernel {
  readonly language = 'python'
  private readonly runtimeId: string
  private readonly indexUrl?: string
  private readonly workerUrl: string | URL
  private readonly resolveAsset?: KernelFactoryOptions['resolveAsset']
  private readonly download?: KernelFactoryOptions['download']
  private readonly status?: KernelFactoryOptions['status']
  private worker: Worker | undefined
  private ready: Promise<void> | undefined
  private readyResolve: (() => void) | undefined
  private readyReject: ((error: unknown) => void) | undefined
  private active: ActiveExecution | undefined
  private interruptBuffer: SharedArrayBuffer | undefined
  private interruptView: Int32Array | undefined
  private fileSequence = 0
  private fileWaiters = new Map<string, (result: RuntimeFileResult | undefined) => void>()

  constructor(opts: KernelFactoryOptions) {
    this.runtimeId = opts.runtimeId
    this.indexUrl = opts.indexUrl
    if (!opts.workerUrl) throw new Error('python notebook runtime worker URL is not configured')
    this.workerUrl = opts.workerUrl
    this.resolveAsset = opts.resolveAsset
    this.download = opts.download
    this.status = opts.status
  }

  async init(opts: KernelInitOptions): Promise<void> {
    if (this.ready) return this.ready
    if (typeof Worker === 'undefined') throw new Error('browser workers are unavailable')
    this.ready = new Promise((resolve, reject) => {
      this.readyResolve = resolve
      this.readyReject = reject
    })
    const worker = new Worker(this.workerUrl, { type: 'module' })
    worker.addEventListener('message', this.onWorkerMessage)
    worker.addEventListener('error', this.onWorkerError)
    this.worker = worker
    worker.postMessage({
      source: 'quartz-notebook-runtime',
      type: 'init',
      runtimeId: this.runtimeId,
      indexUrl: opts.indexUrl ?? this.indexUrl,
      interruptBuffer: this.ensureInterruptBuffer(),
    })
    return this.ready
  }

  async *execute(
    cellId: CellId,
    source: string,
    opts: KernelExecuteOptions = {},
  ): AsyncIterable<RuntimeEvent> {
    if (this.active) throw new Error(`runtime is already executing ${this.active.cellId}`)
    await this.init({ signal: new AbortController().signal, indexUrl: this.indexUrl })
    const queue = new AsyncEventQueue<RuntimeEvent>()
    this.active = { cellId, queue }
    this.clearInterrupt()
    queue.push({ type: 'started', cellId })
    this.worker?.postMessage({
      source: 'quartz-notebook-runtime',
      type: 'run',
      runtimeId: this.runtimeId,
      cellId,
      code: source,
      indexUrl: this.indexUrl,
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
    if (this.interruptView) {
      if (typeof Atomics !== 'undefined' && typeof Atomics.store === 'function') {
        Atomics.store(this.interruptView, 0, 2)
      } else {
        this.interruptView[0] = 2
      }
      return
    }
    this.active?.queue.push({ type: 'interrupted', cellId: this.active.cellId })
    this.terminate(new Error('runtime stopped'))
  }

  async reset(): Promise<void> {
    this.terminate(new Error('runtime reset'))
  }

  async dispose(): Promise<void> {
    this.terminate(new Error('runtime disposed'))
  }

  async runtimeFile(path: string): Promise<RuntimeFileResult | undefined> {
    if (!this.worker) return undefined
    const requestId = `runtime-file-${++this.fileSequence}`
    return await new Promise(resolve => {
      const timeout = globalThis.setTimeout(() => {
        this.fileWaiters.delete(requestId)
        resolve(undefined)
      }, 10_000)
      this.fileWaiters.set(requestId, result => {
        globalThis.clearTimeout(timeout)
        resolve(result)
      })
      this.worker?.postMessage({
        source: 'quartz-notebook-runtime',
        type: 'file',
        runtimeId: this.runtimeId,
        requestId,
        path,
      })
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

  private async handleWorkerMessage(message: WorkerMessage): Promise<void> {
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
      { source: 'quartz-notebook-runtime', type: 'asset-result', ...result },
      transferableBytes(result.bytes),
    )
  }

  private ensureInterruptBuffer(): SharedArrayBuffer | undefined {
    if (this.interruptBuffer) return this.interruptBuffer
    const isolated = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true
    if (!isolated || typeof SharedArrayBuffer === 'undefined') return undefined
    try {
      this.interruptBuffer = new SharedArrayBuffer(4)
      this.interruptView = new Int32Array(this.interruptBuffer)
      return this.interruptBuffer
    } catch {
      return undefined
    }
  }

  private clearInterrupt(): void {
    if (!this.interruptView) return
    if (typeof Atomics !== 'undefined' && typeof Atomics.store === 'function') {
      Atomics.store(this.interruptView, 0, 0)
    } else {
      this.interruptView[0] = 0
    }
  }

  private terminate(error: Error): void {
    this.worker?.removeEventListener('message', this.onWorkerMessage)
    this.worker?.removeEventListener('error', this.onWorkerError)
    this.worker?.terminate()
    this.worker = undefined
    this.ready = undefined
    this.readyResolve = undefined
    this.readyReject = undefined
    this.interruptBuffer = undefined
    this.interruptView = undefined
    this.active?.queue.fail(error)
    this.active = undefined
    for (const waiter of this.fileWaiters.values()) waiter(undefined)
    this.fileWaiters.clear()
  }
}

export function createPyodideKernel(opts: KernelFactoryOptions): Kernel {
  return new PyodideKernel(opts)
}
