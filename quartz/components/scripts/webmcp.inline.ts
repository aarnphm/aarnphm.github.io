import { type FullSlug, getFullSlug, isFullSlug, resolveRelative } from '../../util/path'
import { isRecord } from '../../util/type-guards'
import { populateSearchIndex, querySiteSearchIndex } from './search-index'

const defaultSearchLimit = 5
const defaultReadCharacters = 12_000

async function loadSearchData(): Promise<ContentIndex> {
  if (typeof fetchSearchData === 'undefined') return fetchData

  try {
    return await fetchSearchData
  } catch {
    return fetchData
  }
}

function requireInput(input: unknown): Record<string, unknown> {
  if (!isRecord(input)) throw new TypeError('Tool input must be an object')
  return input
}

function requireString(input: Record<string, unknown>, key: string, maximumLength: number): string {
  const value = input[key]
  if (typeof value !== 'string' || value.trim() === '') {
    throw new TypeError(`${key} must be a non-empty string`)
  }
  const normalized = value.trim()
  if (normalized.length > maximumLength) {
    throw new TypeError(`${key} must contain at most ${maximumLength} characters`)
  }
  return normalized
}

function optionalInteger(
  input: Record<string, unknown>,
  key: string,
  fallback: number,
  minimum: number,
  maximum: number,
): number {
  const value = input[key]
  if (value === undefined) return fallback
  if (typeof value !== 'number' || !Number.isInteger(value) || value < minimum || value > maximum) {
    throw new TypeError(`${key} must be an integer from ${minimum} to ${maximum}`)
  }
  return value
}

function pageUrl(slug: FullSlug): URL {
  return new URL(resolveRelative(getFullSlug(window), slug), window.location.toString())
}

function excerpt(content: string, query: string): string {
  const normalized = content.replace(/\s+/g, ' ').trim()
  const match = normalized.toLocaleLowerCase().indexOf(query.toLocaleLowerCase())
  const start = Math.max(0, match < 0 ? 0 : match - 120)
  const prefix = start > 0 ? '…' : ''
  const end = Math.min(normalized.length, start + 360)
  const suffix = end < normalized.length ? '…' : ''
  return `${prefix}${normalized.slice(start, end)}${suffix}`
}

const searchSiteTool: WebMcpTool = {
  name: 'search_site',
  description:
    "Search aarnphm's public notes and files by title, path, alias, or content. Returns stable page slugs and URLs for read_page or open_page.",
  inputSchema: {
    type: 'object',
    properties: {
      query: {
        type: 'string',
        minLength: 1,
        maxLength: 200,
        description: 'Words or a phrase to find in the garden.',
      },
      limit: {
        type: 'integer',
        minimum: 1,
        maximum: 10,
        default: defaultSearchLimit,
        description: 'Maximum number of results to return.',
      },
    },
    required: ['query'],
    additionalProperties: false,
  },
  annotations: { readOnlyHint: true, untrustedContentHint: true },
  async execute(rawInput) {
    const input = requireInput(rawInput)
    const query = requireString(input, 'query', 200)
    const limit = optionalInteger(input, 'limit', defaultSearchLimit, 1, 10)
    const data = await loadSearchData()
    await populateSearchIndex(data)
    const results = await querySiteSearchIndex(query, limit)

    return {
      query,
      results: results.map(result => ({
        slug: result.slug,
        title: result.title || result.name,
        url: pageUrl(result.slug).toString(),
        excerpt: excerpt(result.content, query),
      })),
    }
  },
}

const readPageTool: WebMcpTool = {
  name: 'read_page',
  description:
    "Read a public page from aarnphm's garden by the slug returned from search_site. Omit slug to read the page currently open in the browser.",
  inputSchema: {
    type: 'object',
    properties: {
      slug: {
        type: 'string',
        minLength: 1,
        maxLength: 512,
        description: 'A page slug returned by search_site.',
      },
      maxCharacters: {
        type: 'integer',
        minimum: 500,
        maximum: 50_000,
        default: defaultReadCharacters,
        description: 'Maximum page-content characters to return.',
      },
    },
    additionalProperties: false,
  },
  annotations: { readOnlyHint: true, untrustedContentHint: true },
  async execute(rawInput) {
    const input = requireInput(rawInput)
    const requestedSlug = input.slug
    if (requestedSlug !== undefined && typeof requestedSlug !== 'string') {
      throw new TypeError('slug must be a string')
    }
    const slug = requestedSlug?.trim() || getFullSlug(window)
    if (slug.length > 512) throw new TypeError('slug must contain at most 512 characters')
    if (!isFullSlug(slug)) throw new TypeError('slug is not a valid garden page slug')

    const maxCharacters = optionalInteger(
      input,
      'maxCharacters',
      defaultReadCharacters,
      500,
      50_000,
    )
    const data = await loadSearchData()
    const page = data[slug]
    if (!page) throw new Error(`No public garden page exists at slug: ${slug}`)

    const content = page.content ?? ''
    return {
      slug,
      title: page.title,
      url: pageUrl(slug).toString(),
      description: page.description ?? '',
      tags: page.tags,
      content: content.slice(0, maxCharacters),
      totalCharacters: content.length,
      truncated: content.length > maxCharacters,
    }
  },
}

const openPageTool: WebMcpTool = {
  name: 'open_page',
  description:
    'Open a same-origin garden URL in the visible browser tab through the site router. Use a URL returned by search_site or read_page.',
  inputSchema: {
    type: 'object',
    properties: {
      url: {
        type: 'string',
        minLength: 1,
        maxLength: 2048,
        description: 'An absolute or relative same-origin garden URL.',
      },
    },
    required: ['url'],
    additionalProperties: false,
  },
  annotations: { readOnlyHint: false, untrustedContentHint: false },
  async execute(rawInput) {
    const input = requireInput(rawInput)
    const requestedUrl = requireString(input, 'url', 2048)
    const url = new URL(requestedUrl, window.location.toString())
    if (url.origin !== window.location.origin || url.username !== '' || url.password !== '') {
      throw new TypeError('url must use the current garden origin')
    }

    await window.spaNavigate(url)
    return { url: url.toString() }
  },
}

const tools = [searchSiteTool, readPageTool, openPageTool]
navigator.modelContext?.provideContext({ tools })
