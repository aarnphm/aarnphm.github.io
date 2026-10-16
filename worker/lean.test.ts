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
})
