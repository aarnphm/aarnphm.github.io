import type { CellId, ErrorOutput } from '../../util/notebook/types'
import type { KernelFactoryOptions } from '../notebook/backend'
import type {
  Kernel,
  KernelExecuteOptions,
  KernelInitOptions,
  RuntimeEvent,
} from '../notebook/kernel'
import { isRecord, readNumber, readString } from '../../util/type-guards'

const PLAYGROUND_URL = 'https://live.lean-lang.org/'
const DEFAULT_IMPORTS = 'mathlib'

type LeanSeverity = 'error' | 'warning' | 'info' | 'unknown'

type LeanMessage = {
  readonly severity: LeanSeverity
  readonly line?: number
  readonly column?: number
  readonly text: string
}

function escapeHtmlAttr(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}

function playgroundUrl(source: string, imports: string): string {
  const code = encodeURIComponent(source)
  const project = encodeURIComponent(imports)
  return `${PLAYGROUND_URL}#code=${code}&project=${project}`
}

function readSeverity(value: unknown): LeanSeverity {
  if (value === 'error' || value === 'warning' || value === 'info') return value
  if (value === 1) return 'error'
  if (value === 2) return 'warning'
  if (value === 3 || value === 4) return 'info'
  return 'unknown'
}

function readPosition(value: unknown): { line?: number; column?: number } {
  if (!isRecord(value)) return {}
  return { line: readNumber(value, 'line'), column: readNumber(value, 'column') }
}

function readLeanMessage(value: unknown): LeanMessage | undefined {
  if (!isRecord(value)) return undefined
  const text =
    readString(value, 'data') ?? readString(value, 'message') ?? readString(value, 'text')
  if (text === undefined) return undefined
  const pos = readPosition(value.pos ?? value.position)
  return { severity: readSeverity(value.severity), line: pos.line, column: pos.column, text }
}

function formatMessage(message: LeanMessage): string {
  const at = message.line !== undefined ? `${message.line}:${message.column ?? 0}: ` : ''
  return `${at}${message.text}`.replace(/\s+$/, '')
}

const SORRY_PATTERN = /\b(sorry|admit)\b/i

function isSorryMessage(message: LeanMessage): boolean {
  return SORRY_PATTERN.test(message.text)
}

function readSorryGoal(value: unknown): string {
  if (!isRecord(value)) return 'unsolved goal'
  const goal = readString(value, 'goal') ?? readString(value, 'data')
  const pos = readPosition(value.pos ?? value.position)
  const at = pos.line !== undefined ? `${pos.line}:${pos.column ?? 0}: ` : ''
  return `${at}unsolved goal (sorry)${goal ? `\n${goal}` : ''}`
}

export class LeanKernel implements Kernel {
  readonly language = 'lean'
  private indexUrl: string | undefined
  private controller: AbortController | undefined

  constructor(opts: KernelFactoryOptions) {
    this.indexUrl = opts.indexUrl?.trim() || undefined
  }

  async init(opts: KernelInitOptions): Promise<void> {
    const configured = opts.indexUrl?.trim()
    if (configured) this.indexUrl = configured
  }

  private endpoint(): string | undefined {
    if (!this.indexUrl) return undefined
    try {
      return new URL(this.indexUrl, window.location.href).href
    } catch {
      return undefined
    }
  }

  async *execute(
    cellId: CellId,
    source: string,
    _opts: KernelExecuteOptions = {},
  ): AsyncIterable<RuntimeEvent> {
    yield { type: 'started', cellId }
    this.controller = new AbortController()
    const endpoint = this.endpoint()
    if (endpoint) {
      const handled = yield* this.verifyRemote(cellId, source, endpoint)
      if (handled) return
    }
    yield* this.handoff(cellId, source)
  }

  private async *verifyRemote(
    cellId: CellId,
    source: string,
    endpoint: string,
  ): AsyncGenerator<RuntimeEvent, boolean> {
    yield { type: 'status', text: 'checking proof against mathlib…' }
    let response: Response
    try {
      response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ code: source, imports: DEFAULT_IMPORTS }),
        signal: this.controller?.signal,
      })
    } catch (error) {
      if (this.controller?.signal.aborted) {
        yield { type: 'interrupted', cellId }
        yield { type: 'done', cellId, executionCount: null, failed: false }
        return true
      }
      console.warn('lean verify request failed, falling back to playground', error)
      return false
    }
    if (!response.ok) return false
    let payload: unknown
    try {
      payload = await response.json()
    } catch {
      yield* this.fail(cellId, ['lean verifier returned a non-JSON response'])
      return true
    }
    if (!isRecord(payload)) {
      yield* this.fail(cellId, ['lean verifier returned an unexpected response'])
      return true
    }

    const fatal = readString(payload, 'error')
    if (fatal !== undefined) {
      yield* this.fail(cellId, [fatal])
      return true
    }

    const messagesField = payload.messages
    const sorriesField = payload.sorries
    const messagesMalformed = messagesField !== undefined && !Array.isArray(messagesField)
    const sorriesMalformed = sorriesField !== undefined && !Array.isArray(sorriesField)
    const recognized =
      typeof payload.env === 'number' || Array.isArray(messagesField) || Array.isArray(sorriesField)
    if (messagesMalformed || sorriesMalformed || !recognized) {
      console.warn('lean verifier returned an unrecognized payload', payload)
      yield* this.fail(cellId, ['lean verifier returned an unrecognized response'])
      return true
    }

    const rawMessages = Array.isArray(messagesField) ? messagesField : []
    const parsed = rawMessages.map(readLeanMessage)
    const unparsed = parsed.filter(message => message === undefined).length
    const messages = parsed.filter((message): message is LeanMessage => message !== undefined)
    const sorryGoals = (Array.isArray(sorriesField) ? sorriesField : []).map(readSorryGoal)

    const blocking = messages.filter(
      message =>
        message.severity === 'error' || message.severity === 'unknown' || isSorryMessage(message),
    )
    const notes = messages.filter(
      message =>
        (message.severity === 'warning' || message.severity === 'info') && !isSorryMessage(message),
    )

    if (notes.length > 0) {
      yield {
        type: 'output',
        cellId,
        output: { type: 'stream', name: 'stdout', text: notes.map(formatMessage).join('\n') },
      }
    }

    const failures = [...blocking.map(formatMessage), ...sorryGoals]
    if (failures.length > 0 || unparsed > 0) {
      const lines =
        failures.length > 0 ? failures : ['lean reported diagnostics that could not be parsed']
      yield* this.fail(cellId, lines)
      return true
    }
    yield { type: 'output', cellId, output: { type: 'success' } }
    yield { type: 'done', cellId, executionCount: null, failed: false }
    return true
  }

  private async *fail(cellId: CellId, lines: string[]): AsyncGenerator<RuntimeEvent> {
    yield { type: 'error', cellId, output: this.errorOutput(lines) }
    yield { type: 'done', cellId, executionCount: null, failed: true }
  }

  private async *handoff(cellId: CellId, source: string): AsyncGenerator<RuntimeEvent> {
    const url = playgroundUrl(source, DEFAULT_IMPORTS)
    const html = `<p class="lean-handoff">In-browser checking isn't wired up for this page. <a href="${escapeHtmlAttr(
      url,
    )}" target="_blank" rel="noopener noreferrer">Open in the Lean 4 playground ↗</a> to verify against mathlib.</p>`
    yield { type: 'output', cellId, output: { type: 'html', html } }
    yield { type: 'done', cellId, executionCount: null, failed: false }
  }

  private errorOutput(lines: string[]): ErrorOutput {
    const traceback = lines.length > 0 ? lines : ['verification failed']
    return { kind: 'error', ename: 'LeanError', evalue: traceback[0], traceback }
  }

  interrupt(): void {
    this.controller?.abort()
  }

  async reset(): Promise<void> {
    this.controller?.abort()
    this.controller = undefined
  }

  async dispose(): Promise<void> {
    this.controller?.abort()
    this.controller = undefined
  }
}

export function createLeanKernel(opts: KernelFactoryOptions): Kernel {
  return new LeanKernel(opts)
}
