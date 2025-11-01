import type { AuthRequest, ClientInfo } from "@cloudflare/workers-oauth-provider"

export class OAuthError extends Error {
  constructor(
    public code: string,
    public description: string,
    public statusCode = 400,
  ) {
    super(description)
    this.name = "OAuthError"
  }

  toResponse(): Response {
    return new Response(
      JSON.stringify({
        error: this.code,
        error_description: this.description,
      }),
      {
        status: this.statusCode,
        headers: { "Content-Type": "application/json" },
      },
    )
  }
}

export interface OAuthStateResult {
  stateToken: string
}

export interface ValidateStateResult {
  oauthReqInfo: AuthRequest
  clearCookie: string
}

export interface BindStateResult {
  setCookie: string
}

export interface CSRFProtectionResult {
  token: string
  setCookie: string
}

export interface ValidateCSRFResult {
  clearCookie: string
}

export function sanitizeText(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;")
}

export function sanitizeUrl(url: string): string {
  const normalized = url.trim()

  if (normalized.length === 0) {
    return ""
  }

  for (let i = 0; i < normalized.length; i++) {
    const code = normalized.charCodeAt(i)
    if ((code >= 0x00 && code <= 0x1f) || (code >= 0x7f && code <= 0x9f)) {
      return ""
    }
  }

  let parsedUrl: URL
  try {
    parsedUrl = new URL(normalized)
  } catch {
    return ""
  }

  const allowedSchemes = ["https", "http"]
  const scheme = parsedUrl.protocol.slice(0, -1).toLowerCase()
  if (!allowedSchemes.includes(scheme)) {
    return ""
  }

  return normalized
}

export function generateCSRFProtection(): CSRFProtectionResult {
  const csrfCookieName = "__Host-CSRF_TOKEN"
  const token = crypto.randomUUID()
  const setCookie = `${csrfCookieName}=${token}; HttpOnly; Secure; Path=/; SameSite=Lax; Max-Age=600`
  return { token, setCookie }
}

export function validateCSRFToken(formData: FormData, request: Request): ValidateCSRFResult {
  const csrfCookieName = "__Host-CSRF_TOKEN"

  const tokenFromForm = formData.get("csrf_token")

  if (!tokenFromForm || typeof tokenFromForm !== "string") {
    throw new OAuthError("invalid_request", "Missing CSRF token in form data", 400)
  }

  const cookieHeader = request.headers.get("Cookie") || ""
  const cookies = cookieHeader.split(";").map((c) => c.trim())
  const csrfCookie = cookies.find((c) => c.startsWith(`${csrfCookieName}=`))
  const tokenFromCookie = csrfCookie ? csrfCookie.substring(csrfCookieName.length + 1) : null

  if (!tokenFromCookie) {
    throw new OAuthError("invalid_request", "Missing CSRF token cookie", 400)
  }

  if (tokenFromForm !== tokenFromCookie) {
    throw new OAuthError("invalid_request", "CSRF token mismatch", 400)
  }

  const clearCookie = `${csrfCookieName}=; HttpOnly; Secure; Path=/; SameSite=Lax; Max-Age=0`

  return { clearCookie }
}

export async function createOAuthState(
  oauthReqInfo: AuthRequest,
  kv: KVNamespace,
  stateTTL = 600,
): Promise<OAuthStateResult> {
  const stateToken = crypto.randomUUID()

  await kv.put(`oauth:state:${stateToken}`, JSON.stringify(oauthReqInfo), {
    expirationTtl: stateTTL,
  })

  return { stateToken }
}

export async function bindStateToSession(stateToken: string): Promise<BindStateResult> {
  const consentedStateCookieName = "__Host-CONSENTED_STATE"

  const encoder = new TextEncoder()
  const data = encoder.encode(stateToken)
  const hashBuffer = await crypto.subtle.digest("SHA-256", data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("")

  const setCookie = `${consentedStateCookieName}=${hashHex}; HttpOnly; Secure; Path=/; SameSite=Lax; Max-Age=600`

  return { setCookie }
}

export async function validateOAuthState(
  request: Request,
  kv: KVNamespace,
): Promise<ValidateStateResult> {
  const consentedStateCookieName = "__Host-CONSENTED_STATE"
  const url = new URL(request.url)
  const stateFromQuery = url.searchParams.get("state")

  if (!stateFromQuery) {
    throw new OAuthError("invalid_request", "Missing state parameter", 400)
  }

  const storedDataJson = await kv.get(`oauth:state:${stateFromQuery}`)
  if (!storedDataJson) {
    throw new OAuthError("invalid_request", "Invalid or expired state", 400)
  }

  const cookieHeader = request.headers.get("Cookie") || ""
  const cookies = cookieHeader.split(";").map((c) => c.trim())
  const consentedStateCookie = cookies.find((c) => c.startsWith(`${consentedStateCookieName}=`))
  const consentedStateHash = consentedStateCookie
    ? consentedStateCookie.substring(consentedStateCookieName.length + 1)
    : null

  if (!consentedStateHash) {
    throw new OAuthError(
      "invalid_request",
      "Missing session binding cookie - authorization flow must be restarted",
      400,
    )
  }

  const encoder = new TextEncoder()
  const data = encoder.encode(stateFromQuery)
  const hashBuffer = await crypto.subtle.digest("SHA-256", data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  const stateHash = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("")

  if (stateHash !== consentedStateHash) {
    throw new OAuthError(
      "invalid_request",
      "State token does not match session - possible CSRF attack detected",
      400,
    )
  }

  let oauthReqInfo: AuthRequest
  try {
    oauthReqInfo = JSON.parse(storedDataJson) as AuthRequest
  } catch (_e) {
    throw new OAuthError("server_error", "Invalid state data", 500)
  }

  await kv.delete(`oauth:state:${stateFromQuery}`)

  const clearCookie = `${consentedStateCookieName}=; HttpOnly; Secure; Path=/; SameSite=Lax; Max-Age=0`

  return { oauthReqInfo, clearCookie }
}

export async function isClientApproved(
  request: Request,
  clientId: string,
  cookieSecret: string,
): Promise<boolean> {
  const approvedClients = await getApprovedClientsFromCookie(request, cookieSecret)
  return approvedClients?.includes(clientId) ?? false
}

export async function addApprovedClient(
  request: Request,
  clientId: string,
  cookieSecret: string,
): Promise<string> {
  const approvedClientsCookieName = "__Host-APPROVED_CLIENTS"
  const THIRTY_DAYS_IN_SECONDS = 2592000

  const existingApprovedClients = (await getApprovedClientsFromCookie(request, cookieSecret)) || []
  const updatedApprovedClients = Array.from(new Set([...existingApprovedClients, clientId]))

  const payload = JSON.stringify(updatedApprovedClients)
  const signature = await signData(payload, cookieSecret)
  const cookieValue = `${signature}.${btoa(payload)}`

  return `${approvedClientsCookieName}=${cookieValue}; HttpOnly; Secure; Path=/; SameSite=Lax; Max-Age=${THIRTY_DAYS_IN_SECONDS}`
}

export interface ApprovalDialogOptions {
  client: ClientInfo | null
  server: {
    name: string
    logo?: string
    description?: string
  }
  state: Record<string, any>
  csrfToken: string
  setCookie: string
}

export function renderApprovalDialog(request: Request, options: ApprovalDialogOptions): Response {
  const { client, server, state, csrfToken, setCookie } = options

  const encodedState = btoa(JSON.stringify(state))

  const serverName = sanitizeText(server.name)
  const clientName = client?.clientName ? sanitizeText(client.clientName) : "Unknown MCP Client"
  const serverDescription = server.description ? sanitizeText(server.description) : ""

  const logoUrl = server.logo ? sanitizeText(sanitizeUrl(server.logo)) : ""
  const clientUri = client?.clientUri ? sanitizeText(sanitizeUrl(client.clientUri)) : ""
  const policyUri = client?.policyUri ? sanitizeText(sanitizeUrl(client.policyUri)) : ""
  const tosUri = client?.tosUri ? sanitizeText(sanitizeUrl(client.tosUri)) : ""

  const contacts =
    client?.contacts && client.contacts.length > 0 ? sanitizeText(client.contacts.join(", ")) : ""

  const redirectUris =
    client?.redirectUris && client.redirectUris.length > 0
      ? client.redirectUris
          .map((uri) => {
            const validated = sanitizeUrl(uri)
            return validated ? sanitizeText(validated) : ""
          })
          .filter((uri) => uri !== "")
      : []

  const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${clientName} | Authorization Request</title>
<style>
:root {
  --light: rgb(255, 252, 240);
  --lightgray: rgb(230, 228, 217);
  --gray: rgb(183, 181, 172);
  --darkgray: rgb(111, 110, 105);
  --dark: rgb(16, 15, 15);
  --iris: rgb(139, 126, 200);
  --foam: rgb(67, 133, 190);
  --highlight: rgb(218, 216, 206);
  --card-shadow: 0 14px 50px rgba(27, 33, 48, 0.12), 0 10px 30px rgba(27, 33, 48, 0.16);
}
body {
  font-family: "PP Neue Montreal", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.6;
  color: var(--dark);
  background-color: var(--light);
  margin: 0;
  padding: 0;
  text-transform: lowercase;
}
.container {
  max-width: 600px;
  margin: 2rem auto;
  padding: 1rem;
}
.precard {
  padding: 2rem;
}
.card {
  background-color: var(--light);
  border-radius: 0px;
  border: 1px solid var(--lightgray);
  box-shadow: var(--card-shadow);
  padding: 2rem;
}
.header {
  display: flex;
  align-items: center;
  margin-bottom: 1.5rem;
}
.logo {
  width: 48px;
  height: 48px;
  margin-right: 1rem;
  border-radius: 0px;
  object-fit: contain;
}
.title {
  margin: 0;
  font-size: 1.3rem;
  font-weight: 400;
  font-family: "Parclo Serif", serif;
}
.alert {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 400;
  margin: 1rem 0;
  font-family: "Parclo Serif", serif;
}
.description {
  color: var(--darkgray);
}
.client-info {
  border: 1px solid var(--lightgray);
  border-radius: 0px;
  padding: 1rem 1rem 0.5rem;
  margin-bottom: 1.5rem;
  background-color: rgba(255, 255, 255, 0.3);
}
.client-name {
  font-weight: 600;
  font-size: 1.2rem;
  margin: 0 0 0.5rem 0;
  font-family: "Parclo Serif", serif;
}
.client-detail {
  display: flex;
  margin-bottom: 0.5rem;
  align-items: baseline;
}
.detail-label {
  font-weight: 500;
  min-width: 120px;
  color: var(--darkgray);
}
.detail-value {
  font-family: "Berkeley Mono", ui-monospace, SFMono-Regular, SF Mono, Menlo, monospace;
  word-break: break-all;
  color: var(--dark);
}
.detail-value a {
  color: var(--dark);
  text-decoration: underline;
}
.detail-value a:hover {
  background: var(--dark);
  color: var(--light);
}
.detail-value.small {
  font-size: 0.8em;
}
.actions {
  display: flex;
  justify-content: flex-end;
  gap: 1rem;
  margin-top: 2rem;
}
.button {
  padding: 0.75rem 1.5rem;
  border-radius: 0px;
  font-weight: 500;
  cursor: pointer;
  border: none;
  font-size: 1rem;
  transition: all 0.2s ease;
}
.button-primary {
  background-color: var(--dark);
  color: var(--light);
  border: 1px solid var(--dark);
}
.button-primary:hover {
  background-color: var(--iris);
  border-color: var(--iris);
}
.button-secondary {
  background-color: transparent;
  border: 1px solid var(--lightgray);
  color: var(--dark);
}
.button-secondary:hover {
  background-color: var(--highlight);
  border-color: var(--gray);
}
@media (max-width: 640px) {
  .container {
    margin: 1rem auto;
    padding: 0.5rem;
  }
  .card {
    padding: 1.5rem;
  }
  .client-detail {
    flex-direction: column;
  }
  .detail-label {
    min-width: unset;
    margin-bottom: 0.25rem;
  }
  .actions {
    flex-direction: column;
  }
  .button {
    width: 100%;
  }
}
</style>
</head>
<body>
<div class="container">
  <div class="precard">
    <div class="header">
      ${logoUrl ? `<img src="${logoUrl}" alt="${serverName} Logo" class="logo">` : ""}
      <h1 class="title"><strong>${serverName}</strong></h1>
    </div>
    ${serverDescription ? `<p class="description">${serverDescription}</p>` : ""}
  </div>
  <div class="card">
    <h2 class="alert"><strong>${clientName || "A new MCP Client"}</strong> is requesting access</h2>
    <div class="client-info">
      <div class="client-detail">
        <div class="detail-label">Name:</div>
        <div class="detail-value">${clientName}</div>
      </div>
      ${
        clientUri
          ? `<div class="client-detail">
        <div class="detail-label">Website:</div>
        <div class="detail-value small">
          <a href="${clientUri}" target="_blank" rel="noopener noreferrer">${clientUri}</a>
        </div>
      </div>`
          : ""
      }
      ${
        policyUri
          ? `<div class="client-detail">
        <div class="detail-label">Privacy Policy:</div>
        <div class="detail-value">
          <a href="${policyUri}" target="_blank" rel="noopener noreferrer">${policyUri}</a>
        </div>
      </div>`
          : ""
      }
      ${
        tosUri
          ? `<div class="client-detail">
        <div class="detail-label">Terms of Service:</div>
        <div class="detail-value">
          <a href="${tosUri}" target="_blank" rel="noopener noreferrer">${tosUri}</a>
        </div>
      </div>`
          : ""
      }
      ${
        redirectUris.length > 0
          ? `<div class="client-detail">
        <div class="detail-label">Redirect URIs:</div>
        <div class="detail-value small">
          ${redirectUris.map((uri) => `<div>${uri}</div>`).join("")}
        </div>
      </div>`
          : ""
      }
      ${
        contacts
          ? `<div class="client-detail">
        <div class="detail-label">Contact:</div>
        <div class="detail-value">${contacts}</div>
      </div>`
          : ""
      }
    </div>
    <p><strong>${clientName || "this MCP Client"}</strong> is requesting to be authorized on <em>${serverName}</em>. If you approve, you will be redirected to complete authentication.</p>
    <form method="post" action="${new URL(request.url).pathname}">
      <input type="hidden" name="state" value="${encodedState}">
      <input type="hidden" name="csrf_token" value="${csrfToken}">
      <div class="actions">
        <button type="button" class="button button-secondary" onclick="window.history.back()">Cancel</button>
        <button type="submit" class="button button-primary">Approve</button>
      </div>
    </form>
  </div>
</div>
</body>
</html>`

  return new Response(htmlContent, {
    headers: {
      "Content-Security-Policy": "frame-ancestors 'none'",
      "Content-Type": "text/html; charset=utf-8",
      "Set-Cookie": setCookie,
      "X-Frame-Options": "DENY",
    },
  })
}

async function getApprovedClientsFromCookie(
  request: Request,
  cookieSecret: string,
): Promise<string[] | null> {
  const approvedClientsCookieName = "__Host-APPROVED_CLIENTS"

  const cookieHeader = request.headers.get("Cookie")
  if (!cookieHeader) return null

  const cookies = cookieHeader.split(";").map((c) => c.trim())
  const targetCookie = cookies.find((c) => c.startsWith(`${approvedClientsCookieName}=`))

  if (!targetCookie) return null

  const cookieValue = targetCookie.substring(approvedClientsCookieName.length + 1)
  const parts = cookieValue.split(".")

  if (parts.length !== 2) return null

  const [signatureHex, base64Payload] = parts
  const payload = atob(base64Payload)

  const isValid = await verifySignature(signatureHex, payload, cookieSecret)

  if (!isValid) return null

  try {
    const approvedClients = JSON.parse(payload)
    if (
      !Array.isArray(approvedClients) ||
      !approvedClients.every((item) => typeof item === "string")
    ) {
      return null
    }
    return approvedClients as string[]
  } catch (_e) {
    return null
  }
}

async function signData(data: string, secret: string): Promise<string> {
  const key = await importKey(secret)
  const enc = new TextEncoder()
  const signatureBuffer = await crypto.subtle.sign("HMAC", key, enc.encode(data))
  return Array.from(new Uint8Array(signatureBuffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
}

async function verifySignature(
  signatureHex: string,
  data: string,
  secret: string,
): Promise<boolean> {
  const key = await importKey(secret)
  const enc = new TextEncoder()
  try {
    const signatureBytes = new Uint8Array(
      signatureHex.match(/.{1,2}/g)!.map((byte) => Number.parseInt(byte, 16)),
    )
    return await crypto.subtle.verify("HMAC", key, signatureBytes.buffer, enc.encode(data))
  } catch (_e) {
    return false
  }
}

async function importKey(secret: string): Promise<CryptoKey> {
  if (!secret) {
    throw new Error("cookieSecret is required for signing cookies")
  }
  const enc = new TextEncoder()
  return crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { hash: "SHA-256", name: "HMAC" },
    false,
    ["sign", "verify"],
  )
}
