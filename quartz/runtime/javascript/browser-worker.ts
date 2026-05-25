import { isRecord, readNumber, readString } from '../../util/type-guards'
import { stripJavaScriptCellMagics } from './can-execute'

type PendingAsset = {
  readonly resolve: (response: Response) => void
  readonly reject: (error: Error) => void
}

type RuntimeElement = {
  append: (...values: readonly unknown[]) => void
  text: (...values: readonly unknown[]) => void
  html: (value: unknown) => void
  empty: () => void
}

const source = 'quartz-notebook-runtime'
const nativeFetch = globalThis.fetch.bind(globalThis)
const nativeConsole = globalThis.console
const pendingAssets = new Map<string, PendingAsset>()
let runtimeId = ''
let currentCellId = ''
let assetSequence = 0

function post(message: Record<string, unknown>, transfer?: Transferable[]): void {
  globalThis.postMessage({ source, runtimeId, ...message }, { transfer: transfer ?? [] })
}

function textOf(value: unknown): string {
  if (value === undefined) return 'undefined'
  if (value === null) return 'null'
  if (typeof value === 'string') return value
  if (typeof value === 'function') return `[Function ${value.name || 'anonymous'}]`
  try {
    return String(value)
  } catch {
    return Object.prototype.toString.call(value)
  }
}

function jsonTextOf(value: unknown): string | undefined {
  if (!isRecord(value) && !Array.isArray(value)) return undefined
  const seen = new WeakSet<object>()
  try {
    return JSON.stringify(
      value,
      (_key, item: unknown) => {
        if (typeof item !== 'object' || item === null) return item
        if (seen.has(item)) return '[Circular]'
        seen.add(item)
        return item
      },
      2,
    )
  } catch {
    return undefined
  }
}

function outputText(values: readonly unknown[], separator: string): string {
  return values.map(textOf).join(separator)
}

function emitOutputForCell(cellId: string, output: Record<string, unknown>): void {
  if (!cellId) return
  post({ type: 'output', cellId, output })
}

function emitOutput(output: Record<string, unknown>): void {
  emitOutputForCell(currentCellId, output)
}

function emitStream(name: 'stdout' | 'stderr', text: string): void {
  if (text.length === 0) return
  emitOutput({ type: 'stream', name, text })
}

function debugOutput(phase: string, cellId: string, error: unknown): Record<string, unknown> {
  return {
    phase,
    cellId,
    errorName: error instanceof Error ? error.name : 'Error',
    errorMessage: error instanceof Error ? error.message : textOf(error),
    stack: error instanceof Error ? error.stack : undefined,
  }
}

function emitError(error: unknown, phase = 'javascript'): void {
  const text = error instanceof Error ? error.message : textOf(error)
  emitOutput({
    type: 'error',
    ename: error instanceof Error ? error.name : 'Error',
    evalue: text,
    traceback: error instanceof Error && error.stack ? error.stack : text,
    debug: debugOutput(phase, currentCellId, error),
  })
}

function emitResult(result: unknown): void {
  if (result === undefined) return
  const json = jsonTextOf(result)
  if (json !== undefined) {
    emitOutput({ type: 'json', text: json })
    return
  }
  emitOutput({ type: 'text', text: textOf(result) })
}

const element: RuntimeElement = {
  append: (...values) => emitStream('stdout', outputText(values, '')),
  text: (...values) => emitOutput({ type: 'text', text: outputText(values, '') }),
  html: value => emitOutput({ type: 'html', html: textOf(value) }),
  empty: () => {},
}

function display(value: unknown): void {
  emitResult(value)
}

function requestInputUrl(input: RequestInfo | URL): string | undefined {
  if (typeof input === 'string') return input
  if (input instanceof URL) return input.href
  if (typeof Request !== 'undefined' && input instanceof Request) return input.url
}

function cellFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  if (init !== undefined) {
    return Promise.reject(
      new Error('fetch options are unavailable in the JavaScript notebook runtime'),
    )
  }
  const url = requestInputUrl(input)
  if (url === undefined) return Promise.reject(new Error('unsupported fetch input'))
  if (!currentCellId) return nativeFetch(input)
  const assetId = 'asset-' + ++assetSequence
  post({ type: 'asset', cellId: currentCellId, assetId, url })
  return new Promise<Response>((resolve, reject) => {
    pendingAssets.set(assetId, { resolve, reject })
  })
}

function installRuntimeGlobals(): void {
  globalThis.console = {
    ...nativeConsole,
    log: (...values: readonly unknown[]) => emitStream('stdout', `${outputText(values, ' ')}\n`),
    info: (...values: readonly unknown[]) => emitStream('stdout', `${outputText(values, ' ')}\n`),
    warn: (...values: readonly unknown[]) => emitStream('stderr', `${outputText(values, ' ')}\n`),
    error: (...values: readonly unknown[]) => emitStream('stderr', `${outputText(values, ' ')}\n`),
  }
  globalThis.fetch = cellFetch
  Object.defineProperty(globalThis, 'element', { value: element, configurable: true })
  Object.defineProperty(globalThis, 'display', { value: display, configurable: true })
}

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return isRecord(value) && typeof value.then === 'function'
}

async function evaluateJavaScript(sourceText: string): Promise<unknown> {
  const evaluate: (code: string) => unknown = globalThis.eval
  const result = evaluate(stripJavaScriptCellMagics(sourceText).source)
  return isPromiseLike(result) ? await result : result
}

async function runCell(message: Record<string, unknown>): Promise<void> {
  const cellId = readString(message, 'cellId')
  const code = readString(message, 'code')
  if (!cellId || code === undefined) return
  currentCellId = cellId
  let failed = false
  try {
    emitResult(await evaluateJavaScript(code))
  } catch (error) {
    failed = true
    emitError(error)
  } finally {
    post({ type: 'done', cellId, failed })
    currentCellId = ''
  }
}

function handleAssetResult(message: Record<string, unknown>): void {
  const assetId = readString(message, 'assetId')
  if (!assetId) return
  const pending = pendingAssets.get(assetId)
  if (!pending) return
  pendingAssets.delete(assetId)
  if (message.ok !== true) {
    pending.reject(new Error(readString(message, 'error') ?? 'failed to fetch notebook asset'))
    return
  }
  const status = readNumber(message, 'status') ?? 200
  const statusText = readString(message, 'statusText') ?? 'OK'
  const contentType = readString(message, 'contentType') ?? 'application/octet-stream'
  const body = message.bytes instanceof ArrayBuffer ? message.bytes : undefined
  pending.resolve(
    new Response(body, { status, statusText, headers: { 'content-type': contentType } }),
  )
}

installRuntimeGlobals()

globalThis.addEventListener('message', event => {
  const origin = typeof event.origin === 'string' ? event.origin : ''
  if (origin && origin !== globalThis.location.origin) return
  const message = event.data
  if (!isRecord(message) || message.source !== source) return
  const type = readString(message, 'type')
  if (type === 'init') {
    const nextRuntimeId = readString(message, 'runtimeId')
    if (!nextRuntimeId) return
    runtimeId = nextRuntimeId
    post({ type: 'ready' })
  } else if (type === 'run' && message.runtimeId === runtimeId) {
    void runCell(message)
  } else if (type === 'asset-result' && message.runtimeId === runtimeId) {
    handleAssetResult(message)
  }
})
