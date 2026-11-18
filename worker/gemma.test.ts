import assert from 'node:assert/strict'
import test from 'node:test'
import { boundReadBody, handleGemmaRead, type GemmaEnv } from './gemma'

const passages = [{ row: 3837, slug: 'thoughts/Nietzsche', text: 'beyond good and evil' }]
const env: GemmaEnv = {
  GEMMA_MODAL_URL: 'https://example--garden-gemma-gemma-web.modal.run/',
  GEMMA_MODAL_KEY: 'wk-test',
  GEMMA_MODAL_SECRET: 'ws-test',
}

function post(body: unknown, origin = 'https://aarnphm.xyz') {
  return new Request('https://aarnphm.xyz/api/gemma/read', {
    method: 'POST',
    headers: {
      Origin: origin,
      'Content-Type': 'application/json',
      'CF-Connecting-IP': '203.0.113.9',
    },
    body: JSON.stringify(body),
  })
}

test('bounds the body: known fields only, clipped text, default token budget', () => {
  const out = boundReadBody({
    query: '  what   is  evil ',
    passages: [{ row: 1, slug: 's', text: 'x'.repeat(10_000), extra: 1 }],
    max_tokens: 5000,
    lora_scale: 3,
  })
  assert.ok(typeof out !== 'string')
  assert.equal(out.query, 'what is evil')
  assert.equal(out.passages[0].text.length, 2400)
  assert.equal(out.max_tokens, 120)
  assert.equal('lora_scale' in out, false)
})

test('refuses steering outside the transcoders', () => {
  const out = boundReadBody({ query: 'q', passages, steer: [{ layer: 42, id: 1, s: 1 }] })
  assert.equal(typeof out, 'string')
})

test('refuses other origins before spending anything', async () => {
  let called = false
  const res = await handleGemmaRead(post({ query: 'q', passages }, 'https://evil.example'), {
    ...env,
    GEMMA_RATE_LIMITER: {
      limit: async () => ((called = true), { success: true }),
    } as unknown as RateLimit,
  })
  assert.equal(res.status, 403)
  assert.equal(called, false)
})

test('says the model is not deployed when the secrets are absent', async () => {
  const res = await handleGemmaRead(post({ query: 'q', passages }), {})
  assert.equal(res.status, 503)
})

test('429 when the visitor is over the limit', async () => {
  const res = await handleGemmaRead(post({ query: 'q', passages }), {
    ...env,
    GEMMA_RATE_LIMITER: { limit: async () => ({ success: false }) } as unknown as RateLimit,
  })
  assert.equal(res.status, 429)
  assert.equal(res.headers.get('Retry-After'), '60')
})

test('passes the Modal token and streams the body through', async () => {
  const real = globalThis.fetch
  let seen: Request | undefined
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    seen = new Request(input, init)
    return new Response('{"t":" a","i":0}\n{"done":true}\n', { status: 200 })
  }) as typeof fetch
  try {
    const res = await handleGemmaRead(post({ query: 'q', passages }), env)
    assert.equal(res.status, 200)
    assert.equal(res.headers.get('Content-Type'), 'application/x-ndjson; charset=utf-8')
    assert.equal(await res.text(), '{"t":" a","i":0}\n{"done":true}\n')
    assert.equal(seen?.url, 'https://example--garden-gemma-gemma-web.modal.run/read')
    assert.equal(seen?.headers.get('Modal-Key'), 'wk-test')
    assert.equal(seen?.headers.get('Modal-Secret'), 'ws-test')
  } finally {
    globalThis.fetch = real
  }
})

test('GET probes whether the model is deployed', async () => {
  const probe = new Request('https://aarnphm.xyz/api/gemma/read')
  assert.equal((await handleGemmaRead(probe, {})).status, 503)
  assert.equal((await handleGemmaRead(probe, env)).status, 204)
})
