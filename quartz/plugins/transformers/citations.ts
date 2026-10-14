import type { Element, ElementContent, Root as HastRoot, Text as HastText } from 'hast'
import type { Link, Parent, Root, Text } from 'mdast'
import type { VFile } from 'vfile'
import { h } from 'hastscript'
import { unified } from 'unified'
import { visit } from 'unist-util-visit'
import type { QuartzTransformerPlugin } from '../../types/plugin'
import type { FrontmatterLink } from './frontmatter'
import { logBuildSpan, PerfTimer } from '../../util/perf'
import { isRecord } from '../../util/type-guards'
import { hostnameMatches, parseExternalUrl } from '../../util/url'
import {
  cachedRehypeCitation,
  citationBibliography,
  hydrateCitationDataCache,
  persistCitationDataCache,
  readCitationTreeCache,
  resolveCitationLocale,
  setCitationTreeCache,
  stringFingerprint,
} from '../stores/citation-cache'
import {
  cacheState,
  ensurePendingPaper,
  extractArxivId,
  normalizeArxivId,
} from '../stores/citations'
export { resolveCitationLocale } from '../stores/citation-cache'

const URL_PATTERN = /https?:\/\/[^\s<>)"]+/g
interface LinkType {
  type: string
  pattern: (url: URL, rawUrl: string) => boolean | string | null
  label: string
}

const LINK_TYPES: LinkType[] = [
  { type: 'arxiv', pattern: (_url, rawUrl) => extractArxivId(rawUrl), label: '[arXiv]' },
  {
    type: 'lesswrong',
    pattern: (url: URL) => hostnameMatches(url, 'lesswrong.com'),
    label: '[lesswrong]',
  },
  { type: 'github', pattern: (url: URL) => hostnameMatches(url, 'github.com'), label: '[GitHub]' },
  {
    type: 'transformer',
    pattern: (url: URL) => hostnameMatches(url, 'transformer-circuits.pub'),
    label: '[transformer circuit]',
  },
  {
    type: 'alignment',
    pattern: (url: URL) => hostnameMatches(url, 'alignmentforum.org'),
    label: '[alignment forum]',
  },
]

function createTextNode(value: string): HastText {
  return { type: 'text', value }
}

function isElement(node: unknown): node is Element {
  return isRecord(node) && node.type === 'element' && typeof node.tagName === 'string'
}

function isMdastParent(node: unknown): node is Parent {
  return isRecord(node) && Array.isArray(node.children)
}

type HastParent = { children: Array<Element | ElementContent> }

function isHastParent(node: unknown): node is HastParent {
  return isRecord(node) && Array.isArray(node.children)
}

function hasClass(node: unknown, className: string): node is Element {
  return (
    isElement(node) &&
    Array.isArray(node.properties.className) &&
    node.properties.className.includes(className)
  )
}

function readFrontmatterBoolean(frontmatter: unknown, key: string): boolean | undefined {
  if (!isRecord(frontmatter)) return undefined
  const value = frontmatter[key]
  return typeof value === 'boolean' ? value : undefined
}

function hasFrontmatterValue(frontmatter: unknown, key: string): boolean {
  return isRecord(frontmatter) && frontmatter[key] !== undefined
}

function citationKeyFromSlug(slug: string): string | undefined {
  if (!slug.startsWith('@')) return undefined
  const key = slug.slice(1).trim()
  return key.length > 0 ? key : undefined
}

export function collectSeeAlsoCitationKeys(links: FrontmatterLink[] | undefined): string[] {
  if (!links) return []

  const keys = new Set<string>()
  for (const link of links) {
    const key = citationKeyFromSlug(link.slug)
    if (key) keys.add(key)
  }

  return Array.from(keys)
}

function normalizedNoCite(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === 'string')
  }
  return typeof value === 'string' ? [value] : []
}

export function mergeCitationKeysIntoNoCite(frontmatter: unknown, keys: Iterable<string>): boolean {
  if (!isRecord(frontmatter)) return false

  const originalNoCite = frontmatter.noCite
  const noCite = normalizedNoCite(frontmatter.noCite)
  if (noCite.includes('@*')) {
    frontmatter.noCite = ['@*']
    return true
  }

  const merged = new Set(noCite)
  for (const key of keys) {
    const normalized = key.trim()
    if (normalized.length > 0)
      merged.add(normalized.startsWith('@') ? normalized : `@${normalized}`)
  }

  if (merged.size === noCite.length && Array.isArray(originalNoCite)) return noCite.length > 0
  frontmatter.noCite = Array.from(merged)
  return true
}

function frontmatterCacheValue(frontmatter: unknown, key: string): string {
  if (!isRecord(frontmatter)) return ''
  const value = frontmatter[key]
  return value === undefined ? '' : JSON.stringify(value)
}

function markdownBody(source: string): string {
  if (!source.startsWith('---')) return source
  const marker = source.indexOf('\n---', 3)
  if (marker === -1) return source
  const afterMarker = source.indexOf('\n', marker + 4)
  return afterMarker === -1 ? '' : source.slice(afterMarker + 1)
}

export function citationTreeCacheKey(
  file: VFile,
  bibliographyKey: string,
  locale: string,
  tree: HastRoot,
): string | undefined {
  if (file.data.citationsDisabled || !file.data.hasCitationSyntax) return undefined
  const source = typeof file.value === 'string' ? file.value : file.value?.toString()
  if (typeof source !== 'string') return undefined
  const frontmatter = file.data.frontmatter
  return [
    stringFingerprint(markdownBody(source)),
    // Cache hits replace the whole tree, including output from upstream transformers.
    stringFingerprint(JSON.stringify(tree)),
    bibliographyKey,
    locale,
    frontmatterCacheValue(frontmatter, 'noCite'),
  ].join(':')
}

function getLinkType(url: string): LinkType | undefined {
  const parsed = parseExternalUrl(url)
  if (!parsed) return undefined
  return LINK_TYPES.find(type => type.pattern(parsed, url))
}

function createLinkElement(href: string): Element {
  const linkType = getLinkType(href)
  const displayText = linkType ? linkType.label : href
  const arxivId = extractArxivId(href)
  const properties: Record<string, string> = { href, target: '_blank', rel: 'noopener noreferrer' }
  if (arxivId) properties.dataArxivId = arxivId

  return h('a.csl-external-link', properties, createTextNode(displayText))
}

function processTextNode(node: HastText): ElementContent[] {
  const text = node.value
  const matches = Array.from(text.matchAll(URL_PATTERN))

  if (matches.length === 0) {
    return [node]
  }

  const result: ElementContent[] = []
  let lastIndex = 0

  matches.forEach(match => {
    const href = match[0]
    const startIndex = match.index!

    if (startIndex > lastIndex) {
      result.push(createTextNode(text.slice(lastIndex, startIndex)))
    }

    const arxivId = extractArxivId(href)
    if (arxivId) {
      result.push(createTextNode(`arXiv preprint arXiv:${arxivId} `))
    }

    result.push(createLinkElement(href))
    lastIndex = startIndex + href.length
  })

  if (lastIndex < text.length) {
    result.push(createTextNode(text.slice(lastIndex)))
  }

  return result
}

function processNodes(nodes: ElementContent[]): ElementContent[] {
  return nodes.flatMap(node => {
    if (node.type === 'text') {
      return processTextNode(node)
    }
    if (node.type === 'element') {
      return { ...node, children: processNodes(node.children) }
    }
    return [node]
  })
}

export function normalizeCitationBibliography(tree: HastRoot): void {
  visit(tree, (node, index, parent) => {
    if (!hasClass(node, 'references') || typeof index !== 'number' || !isHastParent(parent)) return

    const entries: Element[] = []
    visit(node, entry => {
      if (!hasClass(entry, 'csl-entry')) return
      entries.push(h('li', entry.properties, processNodes(entry.children)))
    })

    parent.children.splice(
      index,
      1,
      h(
        'section.bibliography',
        { dataReferences: '' },
        h('h2#reference-label', [{ type: 'text', value: 'bibliographie' }]),
        h('ul', ...entries),
      ),
    )
  })
}

export const checkBib = ({ tagName, properties }: Element) =>
  tagName === 'a' && typeof properties?.href === 'string' && properties.href.startsWith('#bib')

export const checkBibSection = ({ type, tagName, properties }: Element) =>
  type === 'element' && tagName === 'section' && properties.dataReferences == ''

interface Options {
  bibliography: string
}

declare module 'vfile' {
  interface DataMap {
    citations?: { arxivIds: string[] }
    citationsDisabled?: boolean
    hasCitationSyntax?: boolean
  }
}

export const Citations: QuartzTransformerPlugin<Options> = (opts?: Options) => {
  const bibliography = opts?.bibliography ?? 'content/References.bib'
  return {
    name: 'Citations',
    fileDependencies: (_ctx, file) =>
      !file || (file.data.hasCitationSyntax && !file.data.citationsDisabled) ? [bibliography] : [],
    markdownPlugins: () => [
      () => (tree: Root, file: VFile) => {
        const frontmatter = file.data.frontmatter
        const disableCitations =
          readFrontmatterBoolean(frontmatter, 'citations') === false ||
          readFrontmatterBoolean(frontmatter, 'noCitations') === true
        if (disableCitations) {
          file.data.citationsDisabled = true
          delete file.data.citations
          delete file.data.hasCitationSyntax
          return
        }
        file.data.citationsDisabled = false
        const seeAlsoCitationKeys = collectSeeAlsoCitationKeys(file.data.frontmatterLinks?.seealso)
        if (seeAlsoCitationKeys.length > 0) {
          mergeCitationKeysIntoNoCite(frontmatter, seeAlsoCitationKeys)
        }
        let hasCitationSyntax = hasFrontmatterValue(frontmatter, 'noCite')
        const arxivNodes: { node: Link; index: number; parent: Parent; id: string }[] = []

        visit(tree, (node, index, parent) => {
          if (node.type === 'text') {
            if (!hasCitationSyntax && node.value.includes('@')) {
              hasCitationSyntax = true
            }
            return
          }
          if (node.type === 'link') {
            if (typeof index !== 'number' || !isMdastParent(parent)) return
            const arxivId = extractArxivId(node.url)
            if (!arxivId) return
            arxivNodes.push({ node, index, parent, id: normalizeArxivId(arxivId) })
          }
        })

        const docIds = Array.from(new Set(arxivNodes.map(entry => entry.id))).sort()
        if (docIds.length > 0) {
          file.data.citations = { arxivIds: docIds }
          hasCitationSyntax = true
        } else {
          delete file.data.citations
        }
        file.data.hasCitationSyntax = hasCitationSyntax

        if (arxivNodes.length === 0) return

        for (const id of docIds) {
          ensurePendingPaper(id)
        }

        for (const { node, index, parent, id } of arxivNodes) {
          const entry = cacheState.papers.get(id)
          if (!entry) continue

          const titleText: Text = { type: 'text', value: entry.title }
          const citationText: Text = { type: 'text', value: ` [@${entry.bibkey}] ` }
          node.children = [titleText]
          parent.children.splice(index, 1, node, citationText)
        }
      },
    ],
    htmlPlugins: ctx => [
      () => {
        const locale = resolveCitationLocale(ctx.cfg.configuration.locale)
        const renderCitations = unified().use(cachedRehypeCitation, {
          bibliography,
          suppressBibliography: false,
          linkCitations: true,
          csl: 'apa',
          lang: locale,
        })
        return async (tree: HastRoot, file: VFile) => {
          if (file.data.citationsDisabled || !file.data.hasCitationSyntax) return
          const keys = await citationBibliography(ctx, bibliography)
          file.data.citationCache = keys
          await hydrateCitationDataCache(keys.data)
          const cacheKey = citationTreeCacheKey(file, keys.bibliography, locale, tree)
          const cached = cacheKey ? await readCitationTreeCache(cacheKey) : undefined
          if (cached) {
            tree.children = cached.children
            return
          }
          const perf = new PerfTimer()
          await renderCitations.run(tree, file)
          await persistCitationDataCache(keys.data)
          if (cacheKey) await setCitationTreeCache(cacheKey, tree)
          const slug = typeof file.data.slug === 'string' ? file.data.slug : file.path
          logBuildSpan(ctx.argv, 'html:Citations', slug, perf.elapsedMs())
        }
      },
      () => (tree: HastRoot, file: VFile) => {
        if (file.data.citationsDisabled || !file.data.hasCitationSyntax) return
        visit(tree, (node, _index, parent) => {
          if (!isElement(node) || !checkBib(node) || !isElement(parent)) return
          node.properties['data-bib'] = true
          parent.tagName = 'cite'
        })
      },
      () => (tree: HastRoot, file: VFile) => {
        if (file.data.citationsDisabled || !file.data.hasCitationSyntax) return
        normalizeCitationBibliography(tree)
      },
    ],
  }
}
