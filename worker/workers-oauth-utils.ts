import type { AuthRequest, ClientInfo } from "@cloudflare/workers-oauth-provider"

const COOKIE_NAME = "mcp-approved-clients"
const ONE_YEAR_IN_SECONDS = 31536000

function encodeState(data: unknown): string {
  return btoa(JSON.stringify(data))
}

function decodeState<T = unknown>(encoded: string): T {
  return JSON.parse(atob(encoded)) as T
}

async function importKey(secret: string): Promise<CryptoKey> {
  const enc = new TextEncoder()
  return crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"],
  ) as Promise<CryptoKey>
}

async function signData(key: CryptoKey, data: string): Promise<string> {
  const enc = new TextEncoder()
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(data))
  return Array.from(new Uint8Array(sig))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
}

async function verifySignature(
  key: CryptoKey,
  signatureHex: string,
  data: string,
): Promise<boolean> {
  const enc = new TextEncoder()
  const bytes = new Uint8Array(signatureHex.match(/.{1,2}/g)!.map((x) => Number.parseInt(x, 16)))
  return crypto.subtle.verify("HMAC", key, bytes.buffer, enc.encode(data))
}

async function getApprovedClientsFromCookie(
  cookieHeader: string | null,
  secret: string,
): Promise<string[] | null> {
  if (!cookieHeader) return null
  const cookies = cookieHeader.split(";").map((c) => c.trim())
  const target = cookies.find((c) => c.startsWith(`${COOKIE_NAME}=`))
  if (!target) return null
  const cookieValue = target.substring(COOKIE_NAME.length + 1)
  const parts = cookieValue.split(".")
  if (parts.length !== 2) return null
  const [signatureHex, base64Payload] = parts
  const payload = atob(base64Payload)
  const key = await importKey(secret)
  const ok = await verifySignature(key, signatureHex, payload)
  if (!ok) return null
  try {
    const list = JSON.parse(payload)
    if (!Array.isArray(list)) return null
    if (!list.every((x) => typeof x === "string")) return null
    return list
  } catch {
    return null
  }
}

export async function clientIdAlreadyApproved(
  request: Request,
  clientId: string,
  cookieSecret: string,
): Promise<boolean> {
  if (!clientId) return false
  const cookieHeader = request.headers.get("Cookie")
  const approved = await getApprovedClientsFromCookie(cookieHeader, cookieSecret)
  return approved?.includes(clientId) ?? false
}

export interface ApprovalDialogOptions {
  client: ClientInfo | null
  server: { name: string; logo?: string; description?: string }
  state: Record<string, unknown>
}

function escapeHtml(unsafe: string): string {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;")
}

export function renderApprovalDialog(request: Request, options: ApprovalDialogOptions): Response {
  const { client, server, state } = options
  const encodedState = encodeState(state)
  const serverName = escapeHtml(server.name)
  const clientName = client?.clientName ? escapeHtml(client.clientName) : "Unknown MCP Client"
  const serverDesc = server.description ? escapeHtml(server.description) : ""
  const logoUrl = server.logo ? escapeHtml(server.logo) : ""
  const clientUri = client?.clientUri ? escapeHtml(client.clientUri) : ""
  const policyUri = client?.policyUri ? escapeHtml(client.policyUri) : ""
  const tosUri = client?.tosUri ? escapeHtml(client.tosUri) : ""
  const contacts =
    client?.contacts && client.contacts.length > 0 ? escapeHtml(client.contacts.join(", ")) : ""
  const redirectUris =
    client?.redirectUris && client.redirectUris.length > 0
      ? client.redirectUris.map((u) => escapeHtml(u))
      : []
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>${clientName} | Authorization Request</title>
<style>
body { font-family: var(--bodyFont, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif); line-height: 1.6; margin: 0; background: var(--light, #ffffff); color: var(--dark, #111827); }
a { color: var(--darkgray, #475569); text-decoration: underline; }
a:hover { color: var(--tertiary, #b4637a); }
.container { max-width: 640px; margin: 2rem auto; padding: 0 1rem; }
.card { background: var(--light, #ffffff); border: 1px solid var(--lightgray, #e5e7eb); border-radius: 0; box-shadow: none; padding: 1.5rem; }
.header { display: flex; align-items: center; justify-content: center; gap: .75rem; margin-bottom: .5rem; }
.logo { width: 48px; height: 48px; object-fit: contain; }
.title { margin: 0; font-size: 1.3rem; font-weight: 700; font-family: var(--headerFont, inherit); color: var(--dark, #111827); }
.desc { color: var(--darkgray, #475569); text-align: center; margin: .25rem 0 1rem; }
.section { margin: 1rem 0; }
.info { border: 1px solid var(--lightgray, #e5e7eb); border-radius: 0; padding: 1rem; }
.row { display: flex; gap: .5rem; margin: .35rem 0; align-items: baseline; }
.label { min-width: 120px; font-weight: 600; color: var(--darkgray, #475569); }
.actions { display: flex; justify-content: flex-end; gap: .75rem; margin-top: 1.25rem; }
.btn { padding: .6rem 1rem; border-radius: 0; border: 1px solid var(--lightgray, #e5e7eb); background: transparent; color: var(--dark, #111827); font-weight: 500; }
.btn.primary { background: var(--secondary, #4385be); border-color: var(--secondary, #4385be); color: var(--light, #ffffff); }
.btn.primary:hover { filter: brightness(0.95); }
.client-name { font-weight: 600; }
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <div class="header">
      ${logoUrl ? `<img src="${logoUrl}" alt="Logo" class="logo"/>` : ""}
      <h1 class="title">${serverName}</h1>
    </div>
    ${serverDesc ? `<p class="desc">${serverDesc}</p>` : ""}
    <div class="section info">
      <div class="row"><div class="label">Name</div><div class="client-name">${clientName}</div></div>
      ${clientUri ? `<div class="row"><div class="label">Website</div><div><a href="${clientUri}" target="_blank" rel="noopener noreferrer">${clientUri}</a></div></div>` : ""}
      ${policyUri ? `<div class="row"><div class="label">Privacy</div><div><a href="${policyUri}" target="_blank" rel="noopener noreferrer">${policyUri}</a></div></div>` : ""}
      ${tosUri ? `<div class="row"><div class="label">Terms</div><div><a href="${tosUri}" target="_blank" rel="noopener noreferrer">${tosUri}</a></div></div>` : ""}
      ${redirectUris.length > 0 ? `<div class="row"><div class="label">Redirect URIs</div><div>${redirectUris.map((u) => `<div>${u}</div>`).join("")}</div></div>` : ""}
      ${contacts ? `<div class="row"><div class="label">Contact</div><div>${contacts}</div></div>` : ""}
    </div>
    <form method="post" action="${new URL(request.url).pathname}">
      <input type="hidden" name="state" value="${encodedState}" />
      <div class="actions">
        <button type="button" class="btn" onclick="window.history.back()">Cancel</button>
        <button type="submit" class="btn primary">Approve</button>
      </div>
    </form>
  </div>
</div>
</body>
</html>`
  return new Response(html, { headers: { "Content-Type": "text/html; charset=utf-8" } })
}

type ApprovalState = { oauthReqInfo?: AuthRequest }

export interface ParsedApprovalResult {
  state: ApprovalState
  headers: Record<string, string>
}

export async function parseRedirectApproval(
  request: Request,
  cookieSecret: string,
): Promise<ParsedApprovalResult> {
  if (request.method !== "POST") throw new Error("Invalid method")
  const form = await request.formData()
  const encodedState = form.get("state")
  if (typeof encodedState !== "string" || !encodedState) throw new Error("Missing state")
  const state = decodeState<ApprovalState>(encodedState)
  const clientId = state?.oauthReqInfo?.clientId
  if (!clientId) throw new Error("Missing clientId in state")
  const cookieHeader = request.headers.get("Cookie")
  const prior = (await getApprovedClientsFromCookie(cookieHeader, cookieSecret)) || []
  const updated = Array.from(new Set([...prior, clientId]))
  const payload = JSON.stringify(updated)
  const key = await importKey(cookieSecret)
  const signature = await signData(key, payload)
  const cookieVal = `${signature}.${btoa(payload)}`
  const headers: Record<string, string> = {
    "Set-Cookie": `${COOKIE_NAME}=${cookieVal}; HttpOnly; Secure; Path=/; SameSite=Lax; Max-Age=${ONE_YEAR_IN_SECONDS}`,
  }
  return { state, headers }
}
