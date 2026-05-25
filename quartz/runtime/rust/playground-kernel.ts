import type { CellId } from '../../util/notebook/types'
import type {
  Kernel,
  KernelExecuteOptions,
  KernelInitOptions,
  RuntimeEvent,
} from '../notebook/kernel'

export type RustPlaygroundExecuteResult = {
  readonly success: boolean
  readonly exitDetail: string
  readonly stdout: string
  readonly stderr: string
}

type FetchRustPlayground = typeof globalThis.fetch

const rustPlaygroundExecuteUrl = 'https://play.rust-lang.org/execute'
const rustPlaygroundCargoNoise = [
  /^\s*Compiling playground v[^\n]*$/,
  /^\s*Finished `[^`]+` profile \[[^\]]+\] target\(s\) in [^\n]+$/,
  /^\s*Running `target\/[^`]+`$/,
]

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function readString(value: Record<string, unknown>, key: string): string | undefined {
  const item = value[key]
  return typeof item === 'string' ? item : undefined
}

export function readRustPlaygroundExecuteResult(
  value: unknown,
): RustPlaygroundExecuteResult | undefined {
  if (!isRecord(value) || typeof value.success !== 'boolean') return undefined
  const exitDetail = readString(value, 'exitDetail')
  const stdout = readString(value, 'stdout')
  const stderr = readString(value, 'stderr')
  if (exitDetail === undefined || stdout === undefined || stderr === undefined) return undefined
  return { success: value.success, exitDetail, stdout, stderr }
}

export function rustPlaygroundVisibleStderr(result: RustPlaygroundExecuteResult): string {
  if (!result.success) return result.stderr
  return result.stderr
    .split(/\r?\n/)
    .filter(
      line => line.length > 0 && rustPlaygroundCargoNoise.every(pattern => !pattern.test(line)),
    )
    .join('\n')
}

export async function executeRustPlayground(
  source: string,
  fetchRustPlayground: FetchRustPlayground = globalThis.fetch.bind(globalThis),
  signal?: AbortSignal,
): Promise<RustPlaygroundExecuteResult> {
  const response = await fetchRustPlayground(rustPlaygroundExecuteUrl, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      channel: 'stable',
      mode: 'debug',
      edition: '2024',
      crateType: 'bin',
      tests: false,
      backtrace: false,
      code: source,
    }),
    signal,
  })
  const text = await response.text()
  if (!response.ok) {
    throw new Error(`rust playground request failed with ${response.status}: ${text}`)
  }
  let value: unknown
  try {
    value = JSON.parse(text)
  } catch {
    throw new Error('rust playground returned invalid JSON')
  }
  const result = readRustPlaygroundExecuteResult(value)
  if (!result) throw new Error('rust playground returned an invalid response')
  return result
}

export class RustPlaygroundKernel implements Kernel {
  readonly language = 'rust'
  private readonly fetchRustPlayground: FetchRustPlayground
  private active: AbortController | undefined

  constructor(fetchRustPlayground: FetchRustPlayground = globalThis.fetch.bind(globalThis)) {
    this.fetchRustPlayground = fetchRustPlayground
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
    yield { type: 'status', text: 'running rust playground' }
    let failed = false
    try {
      const result = await executeRustPlayground(source, this.fetchRustPlayground, abort.signal)
      failed = !result.success
      if (result.stdout.length > 0) {
        yield { type: 'stream', cellId, name: 'stdout', text: result.stdout }
      }
      const stderr = rustPlaygroundVisibleStderr(result)
      if (result.success) {
        if (stderr.length > 0) yield { type: 'stream', cellId, name: 'stderr', text: stderr }
      } else {
        yield {
          type: 'output',
          cellId,
          output: {
            type: 'error',
            ename: 'RustPlaygroundError',
            evalue: result.exitDetail,
            traceback: stderr || result.exitDetail,
          },
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
          ename: 'RustPlaygroundError',
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

export function createRustPlaygroundKernel(): Kernel {
  return new RustPlaygroundKernel()
}
