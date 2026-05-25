import type { CellId } from '../../util/notebook/types'
import type { KernelFactoryOptions } from '../notebook/backend'
import type {
  Kernel,
  KernelExecuteOptions,
  KernelInitOptions,
  RuntimeAssetRequest,
  RuntimeAssetResult,
  RuntimeEvent,
  RuntimeFileResult,
} from '../notebook/kernel'
import type { NotebookRuntimeOutput } from '../notebook/types'
import { isRecord, readString } from '../../util/type-guards'
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
type StatusMessage = { type: 'status'; runtimeId: string; text: string }

type BrowserJavaScriptWorkerMessage =
  | ReadyMessage
  | OutputMessage
  | DoneMessage
  | AssetMessage
  | StatusMessage

type ActiveExecution = { readonly cellId: CellId; readonly queue: AsyncEventQueue<RuntimeEvent> }
type RuntimeDebugOutput = NonNullable<Extract<NotebookRuntimeOutput, { type: 'error' }>['debug']>

function isCellId(value: unknown): value is CellId {
  return typeof value === 'string' && value.length > 0
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
    if ((name === 'stdout' || name === 'stderr') && text !== undefined) return { type, name, text }
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

function readWorkerMessage(value: unknown): BrowserJavaScriptWorkerMessage | undefined {
  if (!isRecord(value) || value.source !== 'quartz-notebook-runtime') return undefined
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
  if (type === 'status') {
    const text = readString(value, 'text')
    if (text !== undefined) return { type, runtimeId, text }
  }
}

function readAssetResult(value: RuntimeAssetResult): RuntimeFileResult {
  return {
    ok: value.ok,
    status: value.status,
    statusText: value.statusText,
    contentType: value.contentType,
    bytes: value.bytes,
    error: value.error,
  }
}

function transferableBytes(bytes: ArrayBuffer | undefined): Transferable[] {
  return bytes ? [bytes] : []
}

export class BrowserJavaScriptKernel implements Kernel {
  readonly language = 'javascript'
  private readonly runtimeId: string
  private readonly workerUrl: string | URL
  private readonly resolveAsset?: KernelFactoryOptions['resolveAsset']
  private readonly status?: KernelFactoryOptions['status']
  private worker: Worker | undefined
  private ready: Promise<void> | undefined
  private readyResolve: (() => void) | undefined
  private readyReject: ((error: unknown) => void) | undefined
  private active: ActiveExecution | undefined

  constructor(opts: KernelFactoryOptions) {
    this.runtimeId = opts.runtimeId
    if (!opts.workerUrl) throw new Error('javascript notebook runtime worker URL is not configured')
    this.workerUrl = opts.workerUrl
    this.resolveAsset = opts.resolveAsset
    this.status = opts.status
  }

  async init(_opts: KernelInitOptions): Promise<void> {
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
    })
    return this.ready
  }

  async *execute(
    cellId: CellId,
    source: string,
    _opts: KernelExecuteOptions = {},
  ): AsyncIterable<RuntimeEvent> {
    if (this.active) throw new Error(`runtime is already executing ${this.active.cellId}`)
    await this.init({ signal: new AbortController().signal })
    const queue = new AsyncEventQueue<RuntimeEvent>()
    this.active = { cellId, queue }
    queue.push({ type: 'started', cellId })
    this.worker?.postMessage({
      source: 'quartz-notebook-runtime',
      type: 'run',
      runtimeId: this.runtimeId,
      cellId,
      code: source,
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

  private async handleWorkerMessage(message: BrowserJavaScriptWorkerMessage): Promise<void> {
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
    if (message.type === 'asset') {
      await this.handleAssetRequest(message)
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
    const result = await this.resolveAsset(request, async () => undefined)
    this.postAssetResult(result)
  }

  private postAssetResult(result: RuntimeAssetResult): void {
    const file = readAssetResult(result)
    this.worker?.postMessage(
      {
        source: 'quartz-notebook-runtime',
        type: 'asset-result',
        runtimeId: result.runtimeId,
        assetId: result.assetId,
        ...file,
      },
      transferableBytes(file.bytes),
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
  }
}

export function createBrowserJavaScriptKernel(opts: KernelFactoryOptions): Kernel {
  return new BrowserJavaScriptKernel(opts)
}
