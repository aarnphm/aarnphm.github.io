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

const LEAN4WEB_WS_URL = 'wss://live.lean-lang.org/websocket/mathlib-stable'
const LEAN4WEB_PROJECT = 'mathlib-stable'
const LEAN4WEB_DOC_URI = `file:///${LEAN4WEB_PROJECT}/${LEAN4WEB_PROJECT}.lean`

type LspPos = { line: number; character: number }
type LspDiag = {
  range?: { start?: LspPos; end?: LspPos }
  severity?: number
  message?: string
  isSilent?: boolean
}

function findSorryPositionsInSource(source: string): LspPos[] {
  const lines = source.split('\n')
  const out: LspPos[] = []
  for (let line = 0; line < lines.length; line++) {
    const clean = lines[line].replace(/--.*$/, '')
    for (const match of clean.matchAll(/\b(?:sorry|admit)\b/g)) {
      if (match.index !== undefined) {
        out.push({ line, character: match.index })
        if (out.length >= 8) return out
      }
    }
  }
  return out
}

function flattenLeanTaggedText(node: unknown): string {
  if (!isRecord(node)) return ''
  if (typeof node.text === 'string') return node.text
  if (Array.isArray(node.append)) return node.append.map(flattenLeanTaggedText).join('')
  if (Array.isArray(node.tag) && node.tag.length >= 2) return flattenLeanTaggedText(node.tag[1])
  return ''
}

function formatLeanInteractiveGoal(goal: unknown): string {
  if (!isRecord(goal)) return ''
  const lines: string[] = []
  const userName = readString(goal, 'userName')?.trim()
  if (userName) lines.push(`case ${userName}`)
  if (Array.isArray(goal.hyps)) {
    for (const hyp of goal.hyps) {
      if (!isRecord(hyp)) continue
      const names = Array.isArray(hyp.names)
        ? hyp.names.filter((n): n is string => typeof n === 'string').join(' ')
        : ''
      const typeStr = flattenLeanTaggedText(hyp.type)
      if (names && typeStr) lines.push(`${names} : ${typeStr}`)
    }
  }
  const prefix = readString(goal, 'goalPrefix') ?? '⊢ '
  const target = flattenLeanTaggedText(goal.type)
  if (target) lines.push(`${prefix}${target}`)
  return lines.join('\n')
}

class LeanWebSocketSession {
  private ws: WebSocket | undefined
  private readyPromise: Promise<boolean> | undefined
  private isReadyState = false
  private version = 0
  private nextRpcId = 1000
  private rpcSessionId: string | undefined
  private latestDiagnostics: LspDiag[] = []
  private pendingReplies = new Map<number, (result: unknown) => void>()
  private queue: Promise<unknown> = Promise.resolve()

  isWarm(): boolean {
    return this.isReadyState && this.ws?.readyState === WebSocket.OPEN
  }

  warm(): Promise<boolean> {
    if (this.isWarm()) return Promise.resolve(true)
    if (this.readyPromise) return this.readyPromise
    if (typeof WebSocket === 'undefined') return Promise.resolve(false)

    this.readyPromise = new Promise<boolean>(resolve => {
      let ws: WebSocket
      try {
        ws = new WebSocket(LEAN4WEB_WS_URL)
      } catch {
        this.readyPromise = undefined
        resolve(false)
        return
      }
      this.ws = ws
      let settled = false
      const finish = (ok: boolean) => {
        if (settled) return
        settled = true
        if (!ok) {
          this.isReadyState = false
          this.readyPromise = undefined
        }
        resolve(ok)
      }

      ws.addEventListener('open', () => {
        const initId = this.allocId()
        this.pendingReplies.set(initId, () => {
          ws.send(JSON.stringify({ jsonrpc: '2.0', method: 'initialized', params: {} }))
          this.version = 1
          ws.send(
            JSON.stringify({
              jsonrpc: '2.0',
              method: 'textDocument/didOpen',
              params: {
                textDocument: {
                  uri: LEAN4WEB_DOC_URI,
                  languageId: 'lean4',
                  version: this.version,
                  text: 'import Mathlib\n',
                },
              },
            }),
          )
          const waitId = this.allocId()
          this.pendingReplies.set(waitId, () => {
            this.isReadyState = true
            finish(true)
          })
          ws.send(
            JSON.stringify({
              jsonrpc: '2.0',
              id: waitId,
              method: 'textDocument/waitForDiagnostics',
              params: { uri: LEAN4WEB_DOC_URI, version: this.version },
            }),
          )
        })
        ws.send(
          JSON.stringify({
            jsonrpc: '2.0',
            id: initId,
            method: 'initialize',
            params: {
              processId: null,
              clientInfo: { name: 'quartz-lean-kernel', version: '1.0.0' },
              rootUri: `file:///${LEAN4WEB_PROJECT}`,
              capabilities: { lean: { silentDiagnosticSupport: true } },
              initializationOptions: { editDelay: 50, hasWidgets: true },
              workspaceFolders: [{ uri: `file:///${LEAN4WEB_PROJECT}`, name: LEAN4WEB_PROJECT }],
            },
          }),
        )
      })

      ws.addEventListener('message', event => {
        const raw = typeof event.data === 'string' ? event.data : ''
        if (!raw) return
        let msg: Record<string, unknown>
        try {
          msg = JSON.parse(raw) as Record<string, unknown>
        } catch {
          return
        }
        if (typeof msg.method === 'string' && msg.id !== undefined) {
          ws.send(JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: null }))
          return
        }
        if (msg.method === 'textDocument/publishDiagnostics') {
          const params = msg.params as { diagnostics?: LspDiag[] } | undefined
          if (Array.isArray(params?.diagnostics)) {
            this.latestDiagnostics = params.diagnostics
          }
          return
        }
        if (!msg.method && typeof msg.id === 'number') {
          const cb = this.pendingReplies.get(msg.id)
          if (cb) {
            this.pendingReplies.delete(msg.id)
            cb(msg.result)
          }
        }
      })

      const reset = () => {
        this.isReadyState = false
        this.readyPromise = undefined
        this.rpcSessionId = undefined
        for (const [, cb] of this.pendingReplies) cb(undefined)
        this.pendingReplies.clear()
        finish(false)
      }
      ws.addEventListener('error', reset)
      ws.addEventListener('close', reset)
    })

    return this.readyPromise
  }

  verify(source: string, signal?: AbortSignal): Promise<Record<string, unknown> | undefined> {
    const run = async (): Promise<Record<string, unknown> | undefined> => {
      const ok = await this.warm()
      if (!ok || !this.ws || this.ws.readyState !== WebSocket.OPEN || signal?.aborted) {
        return undefined
      }
      const ws = this.ws
      this.version += 1
      const version = this.version
      this.latestDiagnostics = []

      ws.send(
        JSON.stringify({
          jsonrpc: '2.0',
          method: 'textDocument/didChange',
          params: {
            textDocument: { uri: LEAN4WEB_DOC_URI, version },
            contentChanges: [{ text: source }],
          },
        }),
      )

      await this.request(
        ws,
        'textDocument/waitForDiagnostics',
        { uri: LEAN4WEB_DOC_URI, version },
        signal,
      )
      if (signal?.aborted) return undefined

      const diagnostics = [...this.latestDiagnostics]
      const sorryPositions = findSorryPositionsInSource(source)
      const sorryGoalsByIndex = new Map<number, string[]>()

      if (sorryPositions.length > 0) {
        if (!this.rpcSessionId) {
          const connected = await this.request(
            ws,
            '$/lean/rpc/connect',
            { uri: LEAN4WEB_DOC_URI },
            signal,
          )
          if (isRecord(connected) && typeof connected.sessionId === 'string') {
            this.rpcSessionId = connected.sessionId
          }
        }
        if (this.rpcSessionId) {
          await Promise.all(
            sorryPositions.map(async (pos, idx) => {
              const res = await this.request(
                ws,
                '$/lean/rpc/call',
                {
                  sessionId: this.rpcSessionId,
                  method: 'Lean.Widget.getInteractiveGoals',
                  textDocument: { uri: LEAN4WEB_DOC_URI },
                  position: pos,
                  params: { textDocument: { uri: LEAN4WEB_DOC_URI }, position: pos },
                },
                signal,
              )
              if (isRecord(res) && Array.isArray(res.goals)) {
                sorryGoalsByIndex.set(idx, res.goals.map(formatLeanInteractiveGoal).filter(Boolean))
              }
            }),
          )
        }
      }

      const messages: Array<Record<string, unknown>> = []
      const sorries: Array<Record<string, unknown>> = []
      for (const diag of diagnostics) {
        if (diag.isSilent) continue
        const text = typeof diag.message === 'string' ? diag.message : ''
        if (!text) continue
        const startLine = (diag.range?.start?.line ?? 0) + 1
        const startCol = diag.range?.start?.character ?? 0
        const endLine = (diag.range?.end?.line ?? startLine - 1) + 1
        const endCol = diag.range?.end?.character ?? startCol
        const severity = diag.severity === 1 ? 'error' : diag.severity === 2 ? 'warning' : 'info'
        const pos = { line: startLine, column: startCol }
        const endPos = { line: endLine, column: endCol }
        messages.push({ severity, pos, endPos, data: text })
        if (GOAL_STATE_PATTERN.test(text)) {
          sorries.push({ pos, endPos, goal: text, goals: [text], message: text })
        }
      }

      sorryPositions.forEach((pos, idx) => {
        const goals = sorryGoalsByIndex.get(idx) ?? []
        const line1 = pos.line + 1
        const startPos = { line: line1, column: pos.character }
        const endPos = { line: line1, column: pos.character + 5 }
        const goalText = goals.join('\n\n')
        sorries.push({
          pos: startPos,
          endPos,
          ...(goalText ? { goal: goalText, goals } : { goals: [] }),
        })
      })

      return { messages, sorries }
    }

    const next = this.queue.then(run, run)
    this.queue = next.catch(() => undefined)
    return next
  }

  dispose(): void {
    try {
      this.ws?.close()
    } catch {
      // Ignore close errors
    }
    this.ws = undefined
    this.isReadyState = false
    this.readyPromise = undefined
    this.rpcSessionId = undefined
    for (const [, cb] of this.pendingReplies) cb(undefined)
    this.pendingReplies.clear()
  }

  private allocId(): number {
    this.nextRpcId += 1
    return this.nextRpcId
  }

  private request(
    ws: WebSocket,
    method: string,
    params: unknown,
    signal?: AbortSignal,
  ): Promise<unknown> {
    return new Promise<unknown>(resolve => {
      if (signal?.aborted || ws.readyState !== WebSocket.OPEN) {
        resolve(undefined)
        return
      }
      const id = this.allocId()
      const onAbort = () => {
        this.pendingReplies.delete(id)
        resolve(undefined)
      }
      signal?.addEventListener('abort', onAbort, { once: true })
      this.pendingReplies.set(id, result => {
        signal?.removeEventListener('abort', onAbort)
        resolve(result)
      })
      ws.send(JSON.stringify({ jsonrpc: '2.0', id, method, params }))
    })
  }
}

export class LeanKernel implements Kernel {
  readonly language = 'lean'
  private indexUrl: string | undefined
  private controller: AbortController | undefined
  private wsSession: LeanWebSocketSession | undefined

  constructor(opts: KernelFactoryOptions) {
    this.indexUrl = opts.indexUrl?.trim() || undefined
  }

  async init(opts: KernelInitOptions): Promise<void> {
    const configured = opts.indexUrl?.trim()
    if (configured) this.indexUrl = configured
  }

  private isRemoteBrowser(): boolean {
    if (typeof window === 'undefined' || typeof WebSocket === 'undefined') return false
    const host = window.location?.hostname
    return Boolean(host && host !== 'localhost' && host !== '127.0.0.1')
  }

  private ensureWsSession(): LeanWebSocketSession {
    if (!this.wsSession) {
      this.wsSession = new LeanWebSocketSession()
    }
    return this.wsSession
  }

  private endpoints(): string[] {
    if (!this.indexUrl) return []
    try {
      const hasWindow = typeof window !== 'undefined' && Boolean(window.location?.href)
      const base = hasWindow ? window.location.href : 'http://localhost'
      const primary = new URL(this.indexUrl, base).href
      const list = [primary]
      if (
        hasWindow &&
        (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
      ) {
        const fallback = 'http://127.0.0.1:8082/api/lean/verify'
        if (!list.includes(fallback)) list.push(fallback)
      }
      return list
    } catch {
      return []
    }
  }

  async *execute(
    cellId: CellId,
    source: string,
    _opts: KernelExecuteOptions = {},
  ): AsyncIterable<RuntimeEvent> {
    yield { type: 'started', cellId }
    this.controller = new AbortController()
    const endpoints = this.endpoints()
    const canUseBrowserWs = this.isRemoteBrowser()

    if (canUseBrowserWs && this.wsSession?.isWarm()) {
      yield { type: 'status', text: 'checking proof against mathlib…' }
      const payload = await this.wsSession.verify(source, this.controller.signal)
      if (this.controller.signal.aborted) {
        yield { type: 'interrupted', cellId }
        yield { type: 'done', cellId, executionCount: null, failed: false }
        return
      }
      if (payload) {
        yield* this.emitVerifiedPayload(cellId, payload)
        return
      }
    }

    if (canUseBrowserWs) {
      void this.ensureWsSession().warm()
    }

    if (endpoints.length > 0) {
      yield { type: 'status', text: 'checking proof against mathlib…' }
      for (const endpoint of endpoints) {
        const handled = yield* this.verifyRemote(cellId, source, endpoint)
        if (handled) return
      }
    }

    if (canUseBrowserWs) {
      if (endpoints.length === 0) {
        yield { type: 'status', text: 'checking proof against mathlib…' }
      }
      const payload = await this.ensureWsSession().verify(source, this.controller.signal)
      if (this.controller.signal.aborted) {
        yield { type: 'interrupted', cellId }
        yield { type: 'done', cellId, executionCount: null, failed: false }
        return
      }
      if (payload) {
        yield* this.emitVerifiedPayload(cellId, payload)
        return
      }
    }

    yield* this.handoff(cellId, source)
  }

  private async *verifyRemote(
    cellId: CellId,
    source: string,
    endpoint: string,
  ): AsyncGenerator<RuntimeEvent, boolean> {
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
    yield* this.emitVerifiedPayload(cellId, payload)
    return true
  }

  private async *emitVerifiedPayload(
    cellId: CellId,
    payload: Record<string, unknown>,
  ): AsyncGenerator<RuntimeEvent> {
    const fatal = readString(payload, 'error')
    if (fatal !== undefined) {
      yield* this.fail(cellId, [fatal])
      return
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
      return
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
      return
    }
    yield { type: 'output', cellId, output: { type: 'success' } }
    yield { type: 'done', cellId, executionCount: null, failed: false }
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
    this.wsSession?.dispose()
    this.wsSession = undefined
  }
}

export function createLeanKernel(opts: KernelFactoryOptions): Kernel {
  return new LeanKernel(opts)
}
