type LeanEnv = { LEAN_VERIFY_ORIGIN?: string; LEAN_VERIFY_TOKEN?: string }

const MAX_CODE_LENGTH = 100_000
const MAX_IMPORTS_LENGTH = 256
const UPSTREAM_TIMEOUT_MS = 45_000

const JSON_HEADERS = { 'Content-Type': 'application/json' }

function jsonError(message: string, status: number, extra?: Record<string, string>): Response {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { ...JSON_HEADERS, ...extra },
  })
}

export async function handleLeanVerify(request: Request, env: LeanEnv): Promise<Response> {
  if (request.method !== 'POST') {
    return jsonError('method not allowed', 405, { Allow: 'POST, OPTIONS' })
  }

  const origin = env.LEAN_VERIFY_ORIGIN?.trim()
  if (!origin) return jsonError('lean verification is not configured', 503)

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

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS)
  try {
    const headers: Record<string, string> = { ...JSON_HEADERS, Accept: 'application/json' }
    if (env.LEAN_VERIFY_TOKEN) headers.Authorization = `Bearer ${env.LEAN_VERIFY_TOKEN}`
    const upstream = await fetch(origin, {
      method: 'POST',
      headers,
      body: JSON.stringify({ code, imports }),
      signal: controller.signal,
    })
    const text = await upstream.text()
    return new Response(text, { status: upstream.ok ? 200 : 502, headers: JSON_HEADERS })
  } catch (error) {
    const aborted = error instanceof Error && error.name === 'AbortError'
    return jsonError(aborted ? 'verification timed out' : 'verification upstream error', 504)
  } finally {
    clearTimeout(timeout)
  }
}
