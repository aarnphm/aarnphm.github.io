const lineRangeRegex = /^L(\d+)(?:C\d+)?(?:-L?(\d+)(?:C\d+)?)?$/i

export interface GithubBlobRef {
  owner: string
  repo: string
  ref: string
  filePath: string
  rawUrl: string
}

export interface LineRange {
  start: number
  end: number
}

export function parseGithubBlobUrl(input: string | undefined): GithubBlobRef | null {
  if (!input) return null

  let url: URL
  try {
    url = new URL(input.trim())
  } catch {
    return null
  }

  if (url.protocol !== 'https:' && url.protocol !== 'http:') return null
  if (url.hostname.toLowerCase() !== 'github.com') return null

  const parts = url.pathname.split('/').filter(part => part.length > 0)
  if (parts.length < 5 || parts[2] !== 'blob') return null

  const [owner, repo, _blob, ref, ...fileParts] = parts
  const filePath = fileParts.join('/')
  if (!owner || !repo || !ref || !filePath) return null

  const rawUrl = `https://raw.githubusercontent.com/${owner}/${repo}/${ref}/${filePath}`
  return { owner, repo, ref, filePath, rawUrl }
}

export function parseLineRange(anchorText: string | undefined): LineRange | null {
  if (!anchorText) return null
  const match = lineRangeRegex.exec(anchorText.trim())
  if (!match) return null
  const start = Number(match[1])
  const end = match[2] ? Number(match[2]) : start
  if (!Number.isFinite(start) || start < 1) return null
  return { start: Math.min(start, end), end: Math.max(start, end) }
}

export function buildBlobUrl(ref: GithubBlobRef, anchorText: string | undefined): string {
  const base = `https://github.com/${ref.owner}/${ref.repo}/blob/${ref.ref}/${ref.filePath}`
  return anchorText ? `${base}#${anchorText}` : base
}

export function lineRangeLabel(range: LineRange | null): string {
  if (!range) return ''
  if (range.start === range.end) return String(range.start)
  return `${range.start}-${range.end}`
}

export function lineRangeMeta(range: LineRange | null): string {
  const label = lineRangeLabel(range)
  return label ? `showLineNumbers {${label}}` : ''
}

const cache = new Map<string, Promise<string[]>>()

export function fetchGithubFileLines(rawUrl: string): Promise<string[]> {
  const cached = cache.get(rawUrl)
  if (cached) return cached

  const promise = (async () => {
    const res = await fetch(rawUrl, { headers: { 'user-agent': 'quartz-github-embed' } })
    if (!res.ok) {
      throw new Error(`failed to fetch GitHub blob ${rawUrl}: ${res.status} ${res.statusText}`)
    }
    return (await res.text()).split(/\r?\n/)
  })()

  cache.set(rawUrl, promise)
  promise.catch(() => cache.delete(rawUrl))
  return promise
}

export function joinFileLines(lines: string[]): string {
  const trimmed = lines.length > 0 && lines[lines.length - 1] === '' ? lines.slice(0, -1) : lines
  return trimmed.join('\n')
}
