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

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}

const escapeHtmlAttr = escapeHtml

function playgroundUrl(source: string, imports: string): string {
  const code = encodeURIComponent(source)
  const project = encodeURIComponent(imports)
  return `${PLAYGROUND_URL}#code=${code}&project=${project}`
}

function readSeverity(value: unknown): LeanSeverity {
  if (value === 'error' || value === 'warning' || value === 'info') return value
  if (value === 'information') return 'info'
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
const GOAL_STATE_PATTERN = /(?:^|\n)\s*unsolved goals\b|⊢/i

function isSorryMessage(message: LeanMessage): boolean {
  return SORRY_PATTERN.test(message.text)
}

function isGoalMessage(message: LeanMessage): boolean {
  return GOAL_STATE_PATTERN.test(message.text)
}

type LeanSorryItem = {
  readonly line?: number
  readonly column?: number
  readonly goals: readonly string[]
  readonly message?: string
}

function readSorryGoalsList(value: Record<string, unknown>): string[] {
  const single = readString(value, 'goal') ?? readString(value, 'data')
  if (single !== undefined && single.trim().length > 0) return [single.trim()]
  if (Array.isArray(value.goals)) {
    return value.goals
      .filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
      .map(item => item.trim())
  }
  return []
}

function readSorryItem(value: unknown): LeanSorryItem {
  if (!isRecord(value)) return { goals: [] }
  const pos = readPosition(value.pos ?? value.position)
  const goals = readSorryGoalsList(value)
  const message = readString(value, 'message')
  return { line: pos.line, column: pos.column, goals, message }
}

function formatSorryItem(item: LeanSorryItem, raw: unknown): string {
  if (!isRecord(raw)) return 'unsolved goal'
  const goalText = item.goals.join('\n\n')
  const at = item.line !== undefined ? `${item.line}:${item.column ?? 0}: ` : ''
  return `${at}unsolved goal (sorry)${goalText ? `\n${goalText}` : ''}`
}

type ParsedGoalBlock = {
  readonly caseName?: string
  readonly hypotheses: ReadonlyArray<{ readonly name: string; readonly type: string }>
  readonly target: string
}

function parseGoalBlocks(rawText: string): ParsedGoalBlock[] {
  const cleaned = rawText.replace(/^\s*unsolved goals\s*\r?\n?/i, '').trim()
  if (!cleaned) return []

  const lines = cleaned.split(/\r?\n/)
  const blocks: ParsedGoalBlock[] = []
  let caseName: string | undefined
  let hypotheses: Array<{ name: string; type: string }> = []
  let targetLines: string[] | null = null

  const flushTarget = () => {
    if (targetLines !== null) {
      blocks.push({ caseName, hypotheses, target: targetLines.join('\n').trim() })
      caseName = undefined
      hypotheses = []
      targetLines = null
    }
  }

  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) {
      flushTarget()
      continue
    }
    const turnstileIndex = line.indexOf('⊢')
    if (turnstileIndex !== -1) {
      flushTarget()
      targetLines = [line.slice(turnstileIndex + 1).trim()]
      continue
    }
    if (targetLines !== null) {
      if (/^case\s+\S+/.test(trimmed) || /^[^:⊢]+?\s*:\s*.+$/.test(trimmed)) {
        flushTarget()
      } else {
        targetLines.push(trimmed)
        continue
      }
    }
    const caseMatch = trimmed.match(/^case\s+(.+)$/)
    if (caseMatch) {
      caseName = caseMatch[1].trim()
      continue
    }
    const hypMatch = trimmed.match(/^([^:⊢]+?)\s*:\s*(.+)$/)
    if (hypMatch) {
      hypotheses.push({ name: hypMatch[1].trim(), type: hypMatch[2].trim() })
    }
  }
  flushTarget()

  if (blocks.length === 0 && cleaned.length > 0) {
    blocks.push({ hypotheses: [], target: cleaned })
  }
  return blocks
}

function renderGoalSectionHtml(
  badge: string,
  pos: { line?: number; column?: number },
  goal: ParsedGoalBlock,
): string {
  const posHtml =
    pos.line !== undefined
      ? `<span class="lean-infoview-pos">${pos.line}:${pos.column ?? 0}</span>`
      : ''
  const caseHtml = goal.caseName
    ? `<span class="lean-infoview-case">case ${escapeHtml(goal.caseName)}</span>`
    : ''
  const hypsHtml =
    goal.hypotheses.length > 0
      ? `<div class="lean-infoview-hypotheses">${goal.hypotheses
          .map(
            hyp =>
              `<div class="lean-infoview-hypothesis"><span class="lean-infoview-hyp-name">${escapeHtml(
                hyp.name,
              )}</span><span class="lean-infoview-hyp-sep"> : </span><code class="lean-infoview-hyp-type">${escapeHtml(
                hyp.type,
              )}</code></div>`,
          )
          .join('')}</div>`
      : ''
  return `<section class="lean-infoview-goal" data-lean-infoview-kind="goal"><header class="lean-infoview-header"><span class="lean-infoview-badge">${escapeHtml(
    badge,
  )}</span>${caseHtml}${posHtml}</header>${hypsHtml}<div class="lean-infoview-target"><span class="lean-infoview-turnstile" aria-label="turnstile">⊢</span><code class="lean-infoview-target-type">${escapeHtml(
    goal.target,
  )}</code></div></section>`
}

function renderInfoViewHtml(
  messages: readonly LeanMessage[],
  sorryItems: readonly LeanSorryItem[],
): string | undefined {
  const goalSections: string[] = []

  for (const message of messages) {
    if (!isGoalMessage(message)) continue
    const badge = message.severity === 'error' ? 'unsolved goal' : 'tactic state'
    for (const block of parseGoalBlocks(message.text)) {
      goalSections.push(
        renderGoalSectionHtml(badge, { line: message.line, column: message.column }, block),
      )
    }
  }

  for (const item of sorryItems) {
    for (const rawGoal of item.goals) {
      for (const block of parseGoalBlocks(rawGoal)) {
        goalSections.push(
          renderGoalSectionHtml('sorry goal', { line: item.line, column: item.column }, block),
        )
      }
    }
  }

  if (goalSections.length === 0) return undefined

  const noteSections = messages
    .filter(
      message =>
        (message.severity === 'info' || message.severity === 'warning') &&
        !isSorryMessage(message) &&
        !isGoalMessage(message),
    )
    .map(note => {
      const posHtml =
        note.line !== undefined
          ? `<span class="lean-infoview-pos">${note.line}:${note.column ?? 0}</span>`
          : ''
      return `<section class="lean-infoview-message" data-lean-infoview-kind="${escapeHtml(
        note.severity,
      )}"><header class="lean-infoview-header"><span class="lean-infoview-badge">${escapeHtml(
        note.severity,
      )}</span>${posHtml}</header><pre class="lean-infoview-code"><code>${escapeHtml(
        note.text,
      )}</code></pre></section>`
    })

  return `<div class="lean-infoview" data-lean-infoview>${[...goalSections, ...noteSections].join(
    '',
  )}</div>`
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
      const base =
        typeof window !== 'undefined' && window.location?.href
          ? window.location.href
          : 'http://localhost'
      return new URL(this.indexUrl, base).href
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
    const rawSorries = Array.isArray(sorriesField) ? sorriesField : []
    const sorryItems = rawSorries.map(readSorryItem)
    const sorryGoals = rawSorries.map((raw, index) => formatSorryItem(sorryItems[index], raw))

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

    const infoViewHtml = renderInfoViewHtml(messages, sorryItems)
    if (infoViewHtml) {
      yield { type: 'output', cellId, output: { type: 'html', html: infoViewHtml } }
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
