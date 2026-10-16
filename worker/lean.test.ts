import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import test, { describe } from 'node:test'
import { handleLeanVerify } from './lean'

function createMemoryCache() {
  const store = new Map<string, Response>()
  const cache: Cache = {
    async match(request: RequestInfo | URL): Promise<Response | undefined> {
      const key =
        typeof request === 'string' ? request : request instanceof URL ? request.href : request.url
      const hit = store.get(key)
      return hit ? hit.clone() : undefined
    },
    async put(request: RequestInfo | URL, response: Response): Promise<void> {
      const key =
        typeof request === 'string' ? request : request instanceof URL ? request.href : request.url
      store.set(key, response.clone())
    },
    async delete(): Promise<boolean> {
      return false
    },
    async add(): Promise<void> {},
    async addAll(): Promise<void> {},
    async keys(): Promise<readonly Request[]> {
      return []
    },
    async matchAll(): Promise<readonly Response[]> {
      return []
    },
  }
  return { cache, store }
}

describe('worker/lean handleLeanVerify', () => {
  test('caches valid 200 JSON responses in caches.default by sha256(imports + NUL + code) and serves HIT on repeat', async () => {
    const { cache, store } = createMemoryCache()
    const previousCaches = globalThis.caches
    const previousFetch = globalThis.fetch
    Object.defineProperty(globalThis, 'caches', {
      value: { default: cache },
      configurable: true,
      writable: true,
    })

    let upstreamCalls = 0
    globalThis.fetch = async (input, init) => {
      upstreamCalls += 1
      assert.equal(String(input), 'https://verify.example/verify')
      assert.equal(init?.method, 'POST')
      return new Response(JSON.stringify({ messages: [], sorries: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    try {
      const code = '#eval 1 + 1'
      const imports = 'mathlib'
      const expectedHash = createHash('sha256').update(`${imports}\0${code}`).digest('hex')
      const expectedCacheUrl = `https://lean-verify.internal/${expectedHash}`

      const req1 = new Request('https://aarnphm.xyz/api/lean/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, imports }),
      })
      const res1 = await handleLeanVerify(req1, {
        LEAN_VERIFY_ORIGIN: 'https://verify.example/verify',
      })
      assert.equal(res1.status, 200)
      assert.equal(res1.headers.get('X-Lean-Verify-Cache'), 'MISS')
      assert.equal(res1.headers.get('Cache-Control'), 'public, max-age=86400, s-maxage=604800')
      assert.deepEqual(await res1.json(), { messages: [], sorries: [] })
      assert.equal(upstreamCalls, 1)
      assert.equal(store.has(expectedCacheUrl), true)

      const req2 = new Request('https://aarnphm.xyz/api/lean/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, imports }),
      })
      const res2 = await handleLeanVerify(req2, {
        LEAN_VERIFY_ORIGIN: 'https://verify.example/verify',
      })
      assert.equal(res2.status, 200)
      assert.equal(res2.headers.get('X-Lean-Verify-Cache'), 'HIT')
      assert.deepEqual(await res2.json(), { messages: [], sorries: [] })
      assert.equal(upstreamCalls, 1)
    } finally {
      globalThis.fetch = previousFetch
      Object.defineProperty(globalThis, 'caches', {
        value: previousCaches,
        configurable: true,
        writable: true,
      })
    }
  })

  test('does not cache upstream 502 or invalid JSON responses', async () => {
    const { cache, store } = createMemoryCache()
    const previousCaches = globalThis.caches
    const previousFetch = globalThis.fetch
    Object.defineProperty(globalThis, 'caches', {
      value: { default: cache },
      configurable: true,
      writable: true,
    })

    let mode: 'fail' | 'non-json' = 'fail'
    globalThis.fetch = async () => {
      if (mode === 'fail') {
        return new Response('bad gateway', { status: 500 })
      }
      return new Response('not valid json', { status: 200 })
    }

    try {
      const makeReq = () =>
        new Request('https://aarnphm.xyz/api/lean/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: '#check Nat' }),
        })

      const resFail = await handleLeanVerify(makeReq(), {
        LEAN_VERIFY_ORIGIN: 'https://verify.example/verify',
      })
      assert.equal(resFail.status, 502)
      assert.equal(store.size, 0)

      mode = 'non-json'
      const resNonJson = await handleLeanVerify(makeReq(), {
        LEAN_VERIFY_ORIGIN: 'https://verify.example/verify',
      })
      assert.equal(resNonJson.status, 200)
      assert.equal(store.size, 0)
    } finally {
      globalThis.fetch = previousFetch
      Object.defineProperty(globalThis, 'caches', {
        value: previousCaches,
        configurable: true,
        writable: true,
      })
    }
  })

  test('verifies via live.lean-lang.org WebSocket LSP when LEAN_VERIFY_ORIGIN is unset and caches result', async () => {
    const { cache, store } = createMemoryCache()
    const previousCaches = globalThis.caches
    const previousFetch = globalThis.fetch
    Object.defineProperty(globalThis, 'caches', {
      value: { default: cache },
      configurable: true,
      writable: true,
    })

    let wsUpgrades = 0
    globalThis.fetch = async (input, init) => {
      wsUpgrades += 1
      assert.equal(String(input), 'https://live.lean-lang.org/websocket/mathlib-stable')
      const headers = new Headers(init?.headers)
      assert.equal(headers.get('Upgrade'), 'websocket')

      const listeners = new Map<string, Array<(ev: { data?: string }) => void>>()
      const emit = (type: string, data?: string) => {
        for (const fn of listeners.get(type) ?? []) {
          fn({ data })
        }
      }

      const fakeWs = {
        accept() {},
        close() {},
        addEventListener(type: string, cb: (ev: { data?: string }) => void) {
          const list = listeners.get(type) ?? []
          list.push(cb)
          listeners.set(type, list)
        },
        send(raw: string) {
          const msg = JSON.parse(raw) as { id?: number; method?: string }
          if (msg.id === 100 && msg.method === 'initialize') {
            queueMicrotask(() => {
              emit(
                'message',
                JSON.stringify({ jsonrpc: '2.0', id: 100, result: { capabilities: {} } }),
              )
            })
          } else if (msg.id === 200 && msg.method === 'textDocument/waitForDiagnostics') {
            queueMicrotask(() => {
              emit(
                'message',
                JSON.stringify({
                  jsonrpc: '2.0',
                  method: 'textDocument/publishDiagnostics',
                  params: {
                    diagnostics: [
                      {
                        range: { start: { line: 1, character: 0 }, end: { line: 1, character: 5 } },
                        severity: 3,
                        message: '42',
                      },
                      {
                        range: { start: { line: 2, character: 0 }, end: { line: 3, character: 7 } },
                        severity: 3,
                        isSilent: true,
                        message: 'Goals accomplished!',
                      },
                    ],
                  },
                }),
              )
              emit('message', JSON.stringify({ jsonrpc: '2.0', id: 200, result: null }))
            })
          } else if (msg.id === 300 && msg.method === '$/lean/rpc/connect') {
            queueMicrotask(() => {
              emit(
                'message',
                JSON.stringify({ jsonrpc: '2.0', id: 300, result: { sessionId: 's1' } }),
              )
            })
          } else if (msg.id === 400 && msg.method === '$/lean/rpc/call') {
            queueMicrotask(() => {
              emit(
                'message',
                JSON.stringify({
                  jsonrpc: '2.0',
                  id: 400,
                  result: {
                    goals: [
                      {
                        goalPrefix: '⊢ ',
                        hyps: [{ names: ['P'], type: { text: 'Prop' } }],
                        type: { text: 'P → P' },
                      },
                    ],
                  },
                }),
              )
            })
          }
        },
      }

      const resp = new Response(null, { status: 200 })
      Object.defineProperty(resp, 'webSocket', { value: fakeWs })
      return resp
    }

    try {
      const code = 'import Mathlib\n#eval 21 + 21\ntheorem t (P : Prop) : P → P := by\n  sorry'
      const req1 = new Request('https://aarnphm.xyz/api/lean/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, imports: 'mathlib' }),
      })
      const res1 = await handleLeanVerify(req1, {})
      assert.equal(res1.status, 200)
      assert.equal(res1.headers.get('X-Lean-Verify-Cache'), 'MISS')
      const body1 = (await res1.json()) as {
        messages: Array<{ severity: string; data: string }>
        sorries: Array<{ goal?: string }>
      }
      assert.equal(body1.messages.length, 1)
      assert.equal(body1.messages[0].data, '42')
      assert.equal(body1.sorries.length, 1)
      assert.match(body1.sorries[0].goal ?? '', /P : Prop\n⊢ P → P/)
      assert.equal(wsUpgrades, 1)
      assert.equal(store.size, 1)

      const req2 = new Request('https://aarnphm.xyz/api/lean/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, imports: 'mathlib' }),
      })
      const res2 = await handleLeanVerify(req2, {})
      assert.equal(res2.status, 200)
      assert.equal(res2.headers.get('X-Lean-Verify-Cache'), 'HIT')
      assert.equal(wsUpgrades, 1)
    } finally {
      globalThis.fetch = previousFetch
      Object.defineProperty(globalThis, 'caches', {
        value: previousCaches,
        configurable: true,
        writable: true,
      })
    }
  })
})
