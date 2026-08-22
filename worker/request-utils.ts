export function resolveBaseUrl(env: { PUBLIC_BASE_URL?: string }, request: Request): string {
  if (env.PUBLIC_BASE_URL) return env.PUBLIC_BASE_URL.replace(/\/$/, '')
  const u = new URL(request.url)
  u.pathname = ''
  u.search = ''
  u.hash = ''
  return u.toString().replace(/\/$/, '')
}

const AGENT_USER_AGENT_MARKERS = [
  'ai2bot',
  'applebot-extended',
  'anthropic',
  'bytespider',
  'chatgpt',
  'claude',
  'cohere-ai',
  'codex',
  'deepseekbot',
  'diffbot',
  'gemini',
  'google-extended',
  'gptbot',
  'meta-externalagent',
  'oai-searchbot',
  'openai',
  'ora-agent',
]

type MediaRange = { mediaType: string; quality: number }

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

function parseAccept(request: Request): MediaRange[] {
  const accept = request.headers.get('Accept')?.toLowerCase().trim()
  if (!accept) return [{ mediaType: '*/*', quality: 1 }]
  return accept.split(',').map(value => {
    const [mediaType, ...parameters] = value.split(';').map(part => part.trim())
    const qualityParameter = parameters.find(parameter => parameter.startsWith('q='))
    const parsedQuality = qualityParameter ? Number(qualityParameter.slice(2)) : 1
    const quality = Number.isFinite(parsedQuality) ? Math.min(1, Math.max(0, parsedQuality)) : 0
    return { mediaType, quality }
  })
}

function mediaQuality(ranges: MediaRange[], mediaType: string): number {
  const [type] = mediaType.split('/')
  const exact = ranges.find(range => range.mediaType === mediaType)
  if (exact) return exact.quality
  const typeWildcard = ranges.find(range => range.mediaType === `${type}/*`)
  if (typeWildcard) return typeWildcard.quality
  return ranges.find(range => range.mediaType === '*/*')?.quality ?? 0
}

export function acceptsMarkdown(request: Request): boolean {
  return mediaQuality(parseAccept(request), 'text/markdown') > 0
}

export function wantsMarkdown(request: Request): boolean {
  const ranges = parseAccept(request)
  const markdownQuality = mediaQuality(ranges, 'text/markdown')
  const htmlQuality = mediaQuality(ranges, 'text/html')
  if (markdownQuality === 0) return false
  if (markdownQuality > htmlQuality) return true
  return markdownQuality === htmlQuality && isAgentUserAgent(request)
}

export function markdownPathname(pathname: string): string {
  if (pathname === '/') return '/llms.txt'
  if (pathname.endsWith('/')) return `${pathname.slice(0, -1)}.md`
  return `${pathname}.md`
}

function isNegotiableDocumentPath(pathname: string): boolean {
  if (pathname.endsWith('.md')) return false
  if (getExtension(pathname)) return false
  if (pathname.startsWith('/api/')) return false
  if (pathname === '/triathlon/data') return false
  if (pathname.startsWith('/comments/')) return false
  if (pathname.startsWith('/mcp')) return false
  if (pathname.startsWith('/sse')) return false
  if (pathname.startsWith('/authorize')) return false
  if (pathname.startsWith('/register')) return false
  if (pathname.startsWith('/token')) return false
  if (pathname.startsWith('/.well-known/')) return false
  if (pathname.startsWith('/_plausible/')) return false
  if (pathname.startsWith('/fonts/')) return false
  return true
}

export function shouldRewriteMarkdown(request: Request, url: URL): boolean {
  if (request.method !== 'GET' && request.method !== 'HEAD') return false
  if (!wantsMarkdown(request)) return false
  return isNegotiableDocumentPath(url.pathname)
}

export function shouldRejectDocumentResponse(request: Request, url: URL): boolean {
  if (request.method !== 'GET' && request.method !== 'HEAD') return false
  if (!isNegotiableDocumentPath(url.pathname)) return false
  const ranges = parseAccept(request)
  return mediaQuality(ranges, 'text/markdown') === 0 && mediaQuality(ranges, 'text/html') === 0
}

export function documentDiscoveryLink(pathname: string): string {
  const documentPathname = pathname.replace(/\.html?$/, '')
  const markdownPath =
    pathname === '/' || pathname === '/index.html'
      ? '/llms.txt'
      : markdownPathname(documentPathname)
  const describedBy = '</llms.txt>; rel="describedby"; type="text/markdown"'
  if (markdownPath === '/llms.txt') {
    return '</llms.txt>; rel="alternate describedby"; type="text/markdown"'
  }
  return `<${markdownPath}>; rel="alternate"; type="text/markdown", ${describedBy}`
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
