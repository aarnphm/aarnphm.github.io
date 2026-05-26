import { isRecord, readNumber, readString } from './type-guards'

export interface WikipediaTarget {
  lang: string
  title: string
}

export interface WikipediaThumbnail {
  source: string
  width?: number
  height?: number
}

export interface WikipediaPreview {
  title: string
  extract: string
  pageUrl: string
  description?: string
  thumbnail?: WikipediaThumbnail
}

const wikipediaLanguage = /^[a-z][a-z0-9-]{0,15}$/
const ignoredLanguageLabels = new Set(['m', 'www'])

function wikipediaHostLanguage(hostname: string): string | undefined {
  const labels = hostname.toLowerCase().split('.')
  const wikipediaIndex = labels.lastIndexOf('wikipedia')
  if (wikipediaIndex < 1 || labels[wikipediaIndex + 1] !== 'org') return undefined

  const lang = labels[0]
  if (ignoredLanguageLabels.has(lang) || !wikipediaLanguage.test(lang)) return undefined
  return lang
}

function wikipediaPathTitle(pathname: string): string | undefined {
  if (!pathname.startsWith('/wiki/')) return undefined

  const rawTitle = pathname.slice('/wiki/'.length)
  if (rawTitle.length === 0) return undefined

  try {
    return decodeURIComponent(rawTitle).replace(/ /g, '_')
  } catch {
    return rawTitle.replace(/ /g, '_')
  }
}

export function parseWikipediaTarget(href: string): WikipediaTarget | undefined {
  let url: URL
  try {
    url = new URL(href)
  } catch {
    return undefined
  }

  const lang = wikipediaHostLanguage(url.hostname)
  const title = wikipediaPathTitle(url.pathname)
  if (!lang || !title) return undefined

  return { lang, title }
}

export function wikipediaArticleUrl({ lang, title }: WikipediaTarget): string {
  const normalizedTitle = title.replace(/ /g, '_')
  const pathTitle = normalizedTitle.split('/').map(encodeURIComponent).join('/')
  return `https://${lang}.wikipedia.org/wiki/${pathTitle}`
}

export function wikipediaActionApiUrl(target: WikipediaTarget): URL {
  const url = new URL(`https://${target.lang}.wikipedia.org/w/api.php`)
  url.search = new URLSearchParams({
    action: 'query',
    exintro: '1',
    explaintext: '1',
    exsentences: '3',
    format: 'json',
    formatversion: '2',
    origin: '*',
    pithumbsize: '320',
    prop: 'extracts|pageimages|description',
    redirects: '1',
    titles: target.title,
  }).toString()
  return url
}

export function readWikipediaPreviewResponse(
  value: unknown,
  target: WikipediaTarget,
): WikipediaPreview | undefined {
  if (!isRecord(value) || !isRecord(value.query) || !Array.isArray(value.query.pages)) {
    return undefined
  }

  for (const item of value.query.pages) {
    if (!isRecord(item) || item.missing === true) continue

    const extract = readString(item, 'extract')?.trim()
    if (!extract) continue

    const title = readString(item, 'title') ?? target.title
    const description = readString(item, 'description')
    const thumbnail = readWikipediaThumbnail(item.thumbnail)

    return {
      title,
      extract,
      pageUrl: wikipediaArticleUrl({ ...target, title }),
      ...(description ? { description } : {}),
      ...(thumbnail ? { thumbnail } : {}),
    }
  }

  return undefined
}

function readWikipediaThumbnail(value: unknown): WikipediaThumbnail | undefined {
  if (!isRecord(value)) return undefined

  const source = readString(value, 'source')
  if (!source) return undefined

  const width = readNumber(value, 'width')
  const height = readNumber(value, 'height')

  return {
    source,
    ...(width !== undefined ? { width } : {}),
    ...(height !== undefined ? { height } : {}),
  }
}
