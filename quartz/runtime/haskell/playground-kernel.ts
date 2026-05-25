import type { CellId } from '../../util/notebook/types'
import type {
  Kernel,
  KernelExecuteOptions,
  KernelInitOptions,
  RuntimeEvent,
} from '../notebook/kernel'

export type HaskellPlaygroundExecuteResult = {
  readonly ec: number
  readonly ghcout: string
  readonly sout: string
  readonly serr: string
  readonly err?: string
  readonly timesecs?: number
}

type FetchHaskellPlayground = typeof globalThis.fetch

const haskellPlaygroundUpstreamUrl = 'https://play.haskell.org/submit'
const haskellPlaygroundDefaultVersion = '9.14.1'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function readString(value: Record<string, unknown>, key: string): string | undefined {
  const item = value[key]
  return typeof item === 'string' ? item : undefined
}

function readNumber(value: Record<string, unknown>, key: string): number | undefined {
  const item = value[key]
  return typeof item === 'number' && Number.isFinite(item) ? item : undefined
}

export function readHaskellPlaygroundExecuteResult(
  value: unknown,
): HaskellPlaygroundExecuteResult | undefined {
  if (!isRecord(value)) return undefined
  const ec = readNumber(value, 'ec')
  const ghcout = readString(value, 'ghcout')
  const sout = readString(value, 'sout')
  const serr = readString(value, 'serr')
  if (ec === undefined || ghcout === undefined || sout === undefined || serr === undefined) {
    return undefined
  }
  const err = readString(value, 'err')
  const timesecs = readNumber(value, 'timesecs')
  return {
    ec,
    ghcout,
    sout,
    serr,
    ...(err !== undefined ? { err } : {}),
    ...(timesecs !== undefined ? { timesecs } : {}),
  }
}

function defaultHaskellPlaygroundUrl(): string {
  const location = globalThis.location
  return location
    ? new URL('/api/haskell-playground', location.href).href
    : haskellPlaygroundUpstreamUrl
}

export function haskellPlaygroundFailureText(result: HaskellPlaygroundExecuteResult): string {
  const parts = [
    result.err ? `playground error: ${result.err}` : '',
    result.ghcout,
    result.serr,
    result.sout,
    result.ec === 0 ? '' : `Command exited with code ${result.ec}.`,
  ].filter(part => part.length > 0)
  return parts.length > 0 ? parts.join('\n').trimEnd() : 'haskell playground execution failed'
}

export async function executeHaskellPlayground(
  source: string,
  fetchHaskellPlayground: FetchHaskellPlayground = globalThis.fetch.bind(globalThis),
  signal?: AbortSignal,
  url = defaultHaskellPlaygroundUrl(),
): Promise<HaskellPlaygroundExecuteResult> {
  const response = await fetchHaskellPlayground(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      code: source,
      version: haskellPlaygroundDefaultVersion,
      opt: 'O0',
      output: 'run',
    }),
    signal,
  })
  const text = await response.text()
  if (!response.ok) {
    throw new Error(`haskell playground request failed with ${response.status}: ${text}`)
  }
  let value: unknown
  try {
    value = JSON.parse(text)
  } catch {
    throw new Error('haskell playground returned invalid JSON')
  }
  const result = readHaskellPlaygroundExecuteResult(value)
  if (!result) throw new Error('haskell playground returned an invalid response')
  return result
}

export class HaskellPlaygroundKernel implements Kernel {
  readonly language = 'haskell'
  private readonly fetchHaskellPlayground: FetchHaskellPlayground
  private active: AbortController | undefined

  constructor(fetchHaskellPlayground: FetchHaskellPlayground = globalThis.fetch.bind(globalThis)) {
    this.fetchHaskellPlayground = fetchHaskellPlayground
  }

  async init(_opts: KernelInitOptions): Promise<void> {}

  async *execute(
    cellId: CellId,
    source: string,
    _opts: KernelExecuteOptions = {},
  ): AsyncIterable<RuntimeEvent> {
    if (this.active) throw new Error(`runtime is already executing ${cellId}`)
    const abort = new AbortController()
    this.active = abort
    yield { type: 'started', cellId }
    yield { type: 'status', text: 'running haskell playground' }
    let failed = false
    try {
      const result = await executeHaskellPlayground(
        source,
        this.fetchHaskellPlayground,
        abort.signal,
      )
      failed = result.ec !== 0 || result.err !== undefined
      if (failed) {
        const message = haskellPlaygroundFailureText(result)
        yield {
          type: 'output',
          cellId,
          output: {
            type: 'error',
            ename: 'HaskellPlaygroundError',
            evalue: result.err ?? `Command exited with code ${result.ec}.`,
            traceback: message,
          },
        }
      } else {
        if (result.ghcout.length > 0) {
          yield { type: 'stream', cellId, name: 'stderr', text: result.ghcout }
        }
        if (result.serr.length > 0) {
          yield { type: 'stream', cellId, name: 'stderr', text: result.serr }
        }
        if (result.sout.length > 0) {
          yield { type: 'stream', cellId, name: 'stdout', text: result.sout }
        }
      }
    } catch (error) {
      failed = true
      const message = error instanceof Error ? error.message : String(error)
      yield {
        type: 'output',
        cellId,
        output: {
          type: 'error',
          ename: 'HaskellPlaygroundError',
          evalue: message,
          traceback: message,
        },
      }
    } finally {
      if (this.active === abort) this.active = undefined
    }
    yield { type: 'done', cellId, executionCount: null, failed }
  }

  interrupt(): void {
    this.active?.abort()
    this.active = undefined
  }

  async reset(): Promise<void> {
    this.interrupt()
  }

  async dispose(): Promise<void> {
    this.interrupt()
  }
}

export function createHaskellPlaygroundKernel(): Kernel {
  return new HaskellPlaygroundKernel()
}
