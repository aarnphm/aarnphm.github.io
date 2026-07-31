export function resolveBaseUrl(env: Env, request: Request): string {
  if (env.PUBLIC_BASE_URL) return env.PUBLIC_BASE_URL.replace(/\/$/, '')
  const u = new URL(request.url)
  u.pathname = ''
  u.search = ''
  u.hash = ''
  return u.toString().replace(/\/$/, '')
}

const AGENT_USER_AGENT_MARKERS = [
  'ai2bot',
  'anthropic',
  'bytespider',
  'chatgpt',
  'claude',
  'cohere-ai',
  'codex',
  'diffbot',
  'gemini',
  'gptbot',
  'meta-externalagent',
  'oai-searchbot',
  'openai',
  'perplexity',
]

function getExtension(pathname: string): string | null {
  const last = pathname.split('/').pop() ?? ''
  const index = last.lastIndexOf('.')
  return index === -1 ? null : last.slice(index + 1).toLowerCase()
}

export function shouldTreatAsDocument(pathname: string): boolean {
  const extension = getExtension(pathname)
  if (!extension) return true
  return extension === 'html' || extension === 'htm'
}

export function isAgentUserAgent(request: Request): boolean {
  const userAgent = request.headers.get('User-Agent')?.toLowerCase() ?? ''
  return AGENT_USER_AGENT_MARKERS.some(marker => userAgent.includes(marker))
}

export function wantsMarkdown(request: Request): boolean {
  const accept = request.headers.get('Accept')?.toLowerCase() ?? ''
  if (accept.includes('text/markdown')) return true
  return isAgentUserAgent(request)
}

export function markdownPathname(pathname: string): string {
  if (pathname === '/') return '/index.md'
  if (pathname.endsWith('/')) return `${pathname.slice(0, -1)}.md`
  return `${pathname}.md`
}

export function shouldRewriteMarkdown(request: Request, url: URL): boolean {
  if (request.method !== 'GET' && request.method !== 'HEAD') return false
  if (!wantsMarkdown(request)) return false
  if (url.pathname.endsWith('.md')) return false
  if (getExtension(url.pathname)) return false
  if (url.pathname.startsWith('/api/')) return false
  if (url.pathname === '/triathlon/data') return false
  if (url.pathname.startsWith('/comments/')) return false
  if (url.pathname.startsWith('/mcp')) return false
  if (url.pathname.startsWith('/sse')) return false
  if (url.pathname.startsWith('/authorize')) return false
  if (url.pathname.startsWith('/register')) return false
  if (url.pathname.startsWith('/token')) return false
  if (url.pathname.startsWith('/.well-known/')) return false
  if (url.pathname.startsWith('/_plausible/')) return false
  if (url.pathname.startsWith('/fonts/')) return false
  return true
}

function isLocalHostname(hostname: string): boolean {
  const normalized = hostname.toLowerCase()
  return (
    normalized === 'localhost' ||
    normalized === '127.0.0.1' ||
    normalized === '::1' ||
    normalized.startsWith('appl-mbp16') ||
    normalized.endsWith('.localhost')
  )
}

function isLoopbackIp(ip: string): boolean {
  const normalized = ip.trim().toLowerCase()
  return normalized === '127.0.0.1' || normalized === '::1' || normalized === '0:0:0:0:0:0:0:1'
}

function getRequestHostname(request: Request, fallbackUrl: URL): string {
  const hostHeader = request.headers.get('X-Forwarded-Host') ?? request.headers.get('Host')
  const hostValue = hostHeader?.split(',')[0]?.trim() ?? ''
  if (hostValue.length > 0) return new URL(`http://${hostValue}`).hostname
  return fallbackUrl.hostname
}

export function isLocalRequest(request: Request): boolean {
  const url = new URL(request.url)
  const requestHostname = getRequestHostname(request, url)
  const connectingIp = request.headers.get('CF-Connecting-IP') ?? ''
  return (
    isLocalHostname(requestHostname) || isLocalHostname(url.hostname) || isLoopbackIp(connectingIp)
  )
}
