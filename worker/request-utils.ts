type BaseUrlEnv = {
  PUBLIC_BASE_URL?: string
}

export function resolveBaseUrl(env: BaseUrlEnv, request: Request): string {
  if (env.PUBLIC_BASE_URL) return env.PUBLIC_BASE_URL.replace(/\/$/, "")
  const u = new URL(request.url)
  u.pathname = ""
  u.search = ""
  u.hash = ""
  return u.toString().replace(/\/$/, "")
}

function isLocalHostname(hostname: string): boolean {
  const normalized = hostname.toLowerCase()
  return (
    normalized === "localhost" ||
    normalized === "127.0.0.1" ||
    normalized === "::1" ||
    normalized.endsWith(".localhost")
  )
}

function isLoopbackIp(ip: string): boolean {
  const normalized = ip.trim().toLowerCase()
  return normalized === "127.0.0.1" || normalized === "::1" || normalized === "0:0:0:0:0:0:0:1"
}

function getRequestHostname(request: Request, fallbackUrl: URL): string {
  const hostHeader = request.headers.get("X-Forwarded-Host") ?? request.headers.get("Host")
  const hostValue = hostHeader?.split(",")[0]?.trim() ?? ""
  if (hostValue.length > 0) return new URL(`http://${hostValue}`).hostname
  return fallbackUrl.hostname
}

export function isLocalRequest(request: Request): boolean {
  const url = new URL(request.url)
  const requestHostname = getRequestHostname(request, url)
  const connectingIp = request.headers.get("CF-Connecting-IP") ?? ""
  return (
    isLocalHostname(requestHostname) ||
    isLocalHostname(url.hostname) ||
    isLoopbackIp(connectingIp)
  )
}
