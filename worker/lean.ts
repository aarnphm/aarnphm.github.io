type LeanEnv = { LEAN_VERIFY_ORIGIN?: string; LEAN_VERIFY_TOKEN?: string }
type CfCacheStorage = CacheStorage & { readonly default?: Cache }

const MAX_CODE_LENGTH = 100_000
const MAX_IMPORTS_LENGTH = 256
const UPSTREAM_TIMEOUT_MS = 45_000
const CACHE_CONTROL_VALUE = 'public, max-age=86400, s-maxage=604800'

const JSON_HEADERS = { 'Content-Type': 'application/json' }

function jsonError(message: string, status: number, extra?: Record<string, string>): Response {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { ...JSON_HEADERS, ...extra },
  })
}

function defaultCache(): Cache | undefined {
  if (typeof caches === 'undefined' || caches === null) return undefined
  const storage = caches as CfCacheStorage
  return storage.default && typeof storage.default.match === 'function'
    ? storage.default
    : undefined
}

async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value))
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')
}

function isValidJson(text: string): boolean {
  if (!text.trim()) return false
  try {
    JSON.parse(text)
    return true
  } catch {
    return false
  }
}

const DEFAULT_LEAN4WEB_WS_URL = 'https://live.lean-lang.org/websocket/mathlib-stable'
const DEFAULT_LEAN4WEB_PROJECT = 'mathlib-stable'

type CfWebSocket = WebSocket & { accept?: () => void }
type CfUpgradeResponse = Response & { readonly webSocket?: CfWebSocket }

interface LspPosition {
  line: number
  character: number
}

interface LspDiagnostic {
  range?: { start?: LspPosition; end?: LspPosition }
  severity?: number
  message?: string
  isSilent?: boolean
}

function localDevFallbackOrigin(request: Request): string | undefined {
  try {
    const url = new URL(request.url)
    if (url.hostname !== 'localhost' && url.hostname !== '127.0.0.1') return undefined
    const port = Number(url.port)
    if (!Number.isInteger(port) || port <= 0 || port > 65533) return undefined
    return `http://127.0.0.1:${port + 2}/api/lean/verify`
  } catch {
    return undefined
  }
}

function findSorryPositions(source: string): LspPosition[] {
  const lines = source.split('\n')
  const positions: LspPosition[] = []
  for (let line = 0; line < lines.length; line++) {
    const clean = lines[line].replace(/--.*$/, '')
    for (const match of clean.matchAll(/\b(?:sorry|admit)\b/g)) {
      if (match.index !== undefined) {
        positions.push({ line, character: match.index })
        if (positions.length >= 8) return positions
      }
    }
  }
  return positions
}

function flattenTaggedText(node: unknown): string {
  if (!node || typeof node !== 'object') return ''
  const rec = node as Record<string, unknown>
  if (typeof rec.text === 'string') return rec.text
  if (Array.isArray(rec.append)) return rec.append.map(flattenTaggedText).join('')
  if (Array.isArray(rec.tag) && rec.tag.length >= 2) return flattenTaggedText(rec.tag[1])
  return ''
}

function formatInteractiveGoal(goal: unknown): string {
  if (!goal || typeof goal !== 'object') return ''
  const rec = goal as Record<string, unknown>
  const lines: string[] = []
  if (typeof rec.userName === 'string' && rec.userName.trim()) {
    lines.push(`case ${rec.userName.trim()}`)
  }
  if (Array.isArray(rec.hyps)) {
    for (const hyp of rec.hyps) {
      if (!hyp || typeof hyp !== 'object') continue
      const hypRec = hyp as Record<string, unknown>
      const names = Array.isArray(hypRec.names)
        ? hypRec.names.filter((n): n is string => typeof n === 'string').join(' ')
        : ''
      const typeStr = flattenTaggedText(hypRec.type)
      if (names && typeStr) lines.push(`${names} : ${typeStr}`)
    }
  }
  const prefix = typeof rec.goalPrefix === 'string' ? rec.goalPrefix : '⊢ '
  const target = flattenTaggedText(rec.type)
  if (target) lines.push(`${prefix}${target}`)
  return lines.join('\n')
}

function mapLspSeverity(severity: number | undefined): 'error' | 'warning' | 'info' {
  if (severity === 1) return 'error'
  if (severity === 2) return 'warning'
  return 'info'
}

export async function verifyLeanViaWebSocket(
  code: string,
  wsEndpoint: string = DEFAULT_LEAN4WEB_WS_URL,
  signal?: AbortSignal,
): Promise<string | null> {
  const httpUrl = wsEndpoint.replace(/^wss:\/\//i, 'https://').replace(/^ws:\/\//i, 'http://')
  const upgradeResp = (await fetch(httpUrl, {
    headers: { Upgrade: 'websocket', Origin: 'https://live.lean-lang.org' },
    signal,
  })) as CfUpgradeResponse

  const ws = upgradeResp.webSocket
  if (!ws) return null
  if (typeof ws.accept === 'function') {
    ws.accept()
  }

  const docUri = `file:///${DEFAULT_LEAN4WEB_PROJECT}/${DEFAULT_LEAN4WEB_PROJECT}.lean`
  const sorryPositions = findSorryPositions(code)

  return new Promise<string | null>((resolve, reject) => {
    let settled = false
    let diagnostics: LspDiagnostic[] = []
    const sorryGoalsByIndex = new Map<number, string[]>()

    const finish = (payload: string | null, err?: unknown) => {
      if (settled) return
      settled = true
      try {
        ws.close()
      } catch {
        // Ignore close errors
      }
      if (err) reject(err)
      else resolve(payload)
    }

    if (signal) {
      if (signal.aborted) {
        finish(null, new DOMException('Aborted', 'AbortError'))
        return
      }
      signal.addEventListener(
        'abort',
        () => finish(null, new DOMException('Aborted', 'AbortError')),
        { once: true },
      )
    }

    const emitResult = () => {
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
        const severity = mapLspSeverity(diag.severity)
        const pos = { line: startLine, column: startCol }
        const endPos = { line: endLine, column: endCol }
        messages.push({ severity, pos, endPos, data: text })

        if (/(?:^|\n)\s*unsolved goals\b|⊢/i.test(text)) {
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

      finish(JSON.stringify({ messages, sorries }))
    }

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
        const params = msg.params as { diagnostics?: LspDiagnostic[] } | undefined
        if (Array.isArray(params?.diagnostics)) {
          diagnostics = params.diagnostics
        }
        return
      }

      if (!msg.method && msg.id === 100) {
        ws.send(JSON.stringify({ jsonrpc: '2.0', method: 'initialized', params: {} }))
        ws.send(
          JSON.stringify({
            jsonrpc: '2.0',
            method: 'textDocument/didOpen',
            params: { textDocument: { uri: docUri, languageId: 'lean4', version: 1, text: code } },
          }),
        )
        ws.send(
          JSON.stringify({
            jsonrpc: '2.0',
            id: 200,
            method: 'textDocument/waitForDiagnostics',
            params: { uri: docUri, version: 1 },
          }),
        )
        return
      }

      if (!msg.method && msg.id === 200) {
        if (sorryPositions.length === 0) {
          emitResult()
          return
        }
        ws.send(
          JSON.stringify({
            jsonrpc: '2.0',
            id: 300,
            method: '$/lean/rpc/connect',
            params: { uri: docUri },
          }),
        )
        return
      }

      if (!msg.method && msg.id === 300) {
        const result = msg.result as { sessionId?: string } | undefined
        const sessionId = result?.sessionId
        if (!sessionId) {
          emitResult()
          return
        }
        sorryPositions.forEach((pos, idx) => {
          ws.send(
            JSON.stringify({
              jsonrpc: '2.0',
              id: 400 + idx,
              method: '$/lean/rpc/call',
              params: {
                sessionId,
                method: 'Lean.Widget.getInteractiveGoals',
                textDocument: { uri: docUri },
                position: pos,
                params: { textDocument: { uri: docUri }, position: pos },
              },
            }),
          )
        })
        return
      }

      if (
        !msg.method &&
        typeof msg.id === 'number' &&
        msg.id >= 400 &&
        msg.id < 400 + sorryPositions.length
      ) {
        const idx = msg.id - 400
        const result = msg.result as { goals?: unknown[] } | undefined
        const goals = Array.isArray(result?.goals)
          ? result.goals.map(formatInteractiveGoal).filter(Boolean)
          : []
        sorryGoalsByIndex.set(idx, goals)
        if (sorryGoalsByIndex.size >= sorryPositions.length) {
          emitResult()
        }
      }
    })

    ws.addEventListener('error', () => finish(null))
    ws.addEventListener('close', () => finish(null))

    ws.send(
      JSON.stringify({
        jsonrpc: '2.0',
        id: 100,
        method: 'initialize',
        params: {
          processId: null,
          clientInfo: { name: 'quartz-lean-verifier', version: '1.0.0' },
          rootUri: `file:///${DEFAULT_LEAN4WEB_PROJECT}`,
          capabilities: { lean: { silentDiagnosticSupport: true } },
          initializationOptions: { editDelay: 50, hasWidgets: true },
          workspaceFolders: [
            { uri: `file:///${DEFAULT_LEAN4WEB_PROJECT}`, name: DEFAULT_LEAN4WEB_PROJECT },
          ],
        },
      }),
    )
  })
}

async function cacheAndReturnJson(
  text: string,
  cache: Cache | undefined,
  cacheKey: Request,
): Promise<Response> {
  const cacheable = isValidJson(text)
  const responseHeaders: Record<string, string> = {
    ...JSON_HEADERS,
    ...(cacheable ? { 'Cache-Control': CACHE_CONTROL_VALUE, 'X-Lean-Verify-Cache': 'MISS' } : {}),
  }
  const response = new Response(text, { status: 200, headers: responseHeaders })
  if (cache && cacheable) {
    try {
      await cache.put(cacheKey, response.clone())
    } catch {
      // Ignore cache write errors
    }
  }
  return response
}

export async function handleLeanVerify(request: Request, env: LeanEnv): Promise<Response> {
  if (request.method !== 'POST') {
    return jsonError('method not allowed', 405, { Allow: 'POST, OPTIONS' })
  }

  let body: { code?: unknown; imports?: unknown }
  try {
    body = (await request.json()) as typeof body
  } catch {
    return jsonError('invalid json', 400)
  }

  const code = typeof body.code === 'string' ? body.code : null
  if (!code) return jsonError('code required', 400)
  if (code.length > MAX_CODE_LENGTH) return jsonError('code too large', 413)
  const imports =
    typeof body.imports === 'string' && body.imports.length <= MAX_IMPORTS_LENGTH
      ? body.imports
      : 'mathlib'

  const hash = await sha256Hex(`${imports}\0${code}`)
  const cacheKey = new Request(`https://lean-verify.internal/${hash}`, { method: 'GET' })
  const cache = defaultCache()
  if (cache) {
    try {
      const cached = await cache.match(cacheKey)
      if (cached) {
        const headers = new Headers(cached.headers)
        headers.set('Content-Type', 'application/json')
        headers.set('Cache-Control', cached.headers.get('Cache-Control') ?? CACHE_CONTROL_VALUE)
        headers.set('X-Lean-Verify-Cache', 'HIT')
        return new Response(cached.body, { status: 200, headers })
      }
    } catch {
      // Proceed to upstream if cache read fails
    }
  }

  const configuredOrigin = env.LEAN_VERIFY_ORIGIN?.trim()
  const fallbackOrigin = configuredOrigin ? undefined : localDevFallbackOrigin(request)
  const httpOrigin =
    configuredOrigin && !/^wss?:\/\//i.test(configuredOrigin) ? configuredOrigin : fallbackOrigin
  const wsOrigin =
    configuredOrigin && /^wss?:\/\//i.test(configuredOrigin)
      ? configuredOrigin
      : DEFAULT_LEAN4WEB_WS_URL

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS)
  try {
    if (httpOrigin) {
      const headers: Record<string, string> = { ...JSON_HEADERS, Accept: 'application/json' }
      if (env.LEAN_VERIFY_TOKEN) headers.Authorization = `Bearer ${env.LEAN_VERIFY_TOKEN}`
      try {
        const upstream = await fetch(httpOrigin, {
          method: 'POST',
          headers,
          body: JSON.stringify({ code, imports }),
          signal: controller.signal,
        })
        const text = await upstream.text()
        if (upstream.ok) {
          return await cacheAndReturnJson(text, cache, cacheKey)
        }
        if (configuredOrigin) {
          return new Response(text, { status: 502, headers: JSON_HEADERS })
        }
      } catch (err) {
        if (configuredOrigin) throw err
      }
    }

    const wsPayload = await verifyLeanViaWebSocket(code, wsOrigin, controller.signal)
    if (wsPayload) {
      return await cacheAndReturnJson(wsPayload, cache, cacheKey)
    }
    return jsonError('lean verification is not configured', 503)
  } catch (error) {
    const aborted = error instanceof Error && error.name === 'AbortError'
    return jsonError(aborted ? 'verification timed out' : 'verification upstream error', 504)
  } finally {
    clearTimeout(timeout)
  }
}
