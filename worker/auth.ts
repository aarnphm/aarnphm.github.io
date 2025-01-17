const te = new TextEncoder()

function b64urlEncode(data: Uint8Array): string {
  let bin = ""
  for (let i = 0; i < data.length; i++) bin += String.fromCharCode(data[i])
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "")
}

function b64urlEncodeString(s: string): string {
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "")
}

function b64urlDecodeString(s: string): string {
  s = s.replace(/-/g, "+").replace(/_/g, "/")
  const pad = s.length % 4
  if (pad) s += "=".repeat(4 - pad)
  return atob(s)
}

async function importHmacKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    te.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"],
  ) as Promise<CryptoKey>
}

async function hmac(secret: string, data: string): Promise<string> {
  const key = await importHmacKey(secret)
  const sig = await crypto.subtle.sign("HMAC", key, te.encode(data))
  return b64urlEncode(new Uint8Array(sig))
}

export type SessionPayload = {
  iss: string
  sub: string | number
  login: string
  name?: string | null
  avatarUrl?: string | null
  iat: number
  exp: number
}

export type TokenPayload = SessionPayload & { aud: string }

export function getCookie(req: Request, name: string): string | null {
  const header = req.headers.get("Cookie")
  if (!header) return null
  const parts = header.split(/;\s*/)
  for (const p of parts) {
    const idx = p.indexOf("=")
    if (idx === -1) continue
    const k = decodeURIComponent(p.slice(0, idx))
    if (k === name) return decodeURIComponent(p.slice(idx + 1))
  }
  return null
}

export function makeCookie(
  name: string,
  value: string,
  opts: {
    httpOnly?: boolean
    secure?: boolean
    sameSite?: "Lax" | "Strict" | "None"
    path?: string
    domain?: string
    maxAge?: number
  } = {},
): string {
  const pairs = [`${encodeURIComponent(name)}=${encodeURIComponent(value)}`]
  if (opts.path) pairs.push(`Path=${opts.path}`)
  if (opts.domain) pairs.push(`Domain=${opts.domain}`)
  if (opts.maxAge !== undefined) pairs.push(`Max-Age=${opts.maxAge}`)
  if (opts.httpOnly !== false) pairs.push("HttpOnly")
  if (opts.secure !== false) pairs.push("Secure")
  pairs.push(`SameSite=${opts.sameSite ?? "Lax"}`)
  return pairs.join("; ")
}

function nowSeconds(): number {
  return Math.floor(Date.now() / 1000)
}

export async function signStructured(
  prefix: string,
  payload: Record<string, unknown>,
  secret: string,
): Promise<string> {
  const body = b64urlEncodeString(JSON.stringify(payload))
  const ver = "v1"
  const toSign = `${ver}.${prefix}.${body}`
  const sig = await hmac(secret, toSign)
  return `${ver}.${body}.${sig}`
}

export async function verifyStructured(
  prefix: string,
  token: string,
  secret: string,
): Promise<Record<string, unknown> | null> {
  const parts = token.split(".")
  if (parts.length !== 3) return null
  const [ver, body, sig] = parts
  if (ver !== "v1") return null
  const expected = await hmac(secret, `${ver}.${prefix}.${body}`)
  if (sig !== expected) return null
  try {
    const json = b64urlDecodeString(body)
    return JSON.parse(json) as Record<string, unknown>
  } catch {
    return null
  }
}

export async function createState(secret: string, next: string): Promise<string> {
  const payload = {
    next,
    n: crypto.getRandomValues(new Uint32Array(1))[0].toString(16),
    iat: nowSeconds(),
  }
  return signStructured("state", payload, secret)
}

export async function verifyState(secret: string, state: string): Promise<{ next: string } | null> {
  const data = await verifyStructured("state", state, secret)
  if (!data) return null
  const next = (data["next"] as string) || "/"
  return { next }
}

export async function createSession(secret: string, payload: SessionPayload): Promise<string> {
  return signStructured("session", payload, secret)
}

export async function verifySession(secret: string, token: string): Promise<SessionPayload | null> {
  const data = await verifyStructured("session", token, secret)
  if (!data) return null
  const exp = data["exp"] as number
  if (typeof exp !== "number" || nowSeconds() >= exp) return null
  return data as SessionPayload
}

export async function createAccessToken(
  secret: string,
  session: SessionPayload,
  aud: string,
  ttlSeconds: number,
): Promise<string> {
  const payload: TokenPayload = {
    ...session,
    aud,
    iat: nowSeconds(),
    exp: nowSeconds() + ttlSeconds,
  }
  return signStructured("token", payload, secret)
}

export async function verifyAccessToken(
  secret: string,
  token: string,
  aud: string,
): Promise<TokenPayload | null> {
  const data = await verifyStructured("token", token, secret)
  if (!data) return null
  if (data["aud"] !== aud) return null
  const exp = data["exp"] as number
  if (typeof exp !== "number" || nowSeconds() >= exp) return null
  return data as TokenPayload
}
