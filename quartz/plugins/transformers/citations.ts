import type { Element, ElementContent, Root as HastRoot, Text as HastText } from 'hast'
import type { Link, Parent, Root, Text } from 'mdast'
import type { VFile } from 'vfile'
import { h } from 'hastscript'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import { createRequire } from 'node:module'
import path from 'node:path'
import { Cite, rehypeCitationGenerator } from 'rehype-citation'
import { unified } from 'unified'
import { visit } from 'unist-util-visit'
import type { QuartzTransformerPlugin } from '../../types/plugin'
import { logBuildSpan, PerfTimer } from '../../util/perf'
import { isRecord } from '../../util/type-guards'
import { hostnameMatches, parseExternalUrl } from '../../util/url'
import {
  cacheState,
  ensurePendingPaper,
  extractArxivId,
  normalizeArxivId,
} from '../stores/citations'
import '@citation-js/plugin-bibtex'
import '@citation-js/plugin-doi'

const URL_PATTERN = /https?:\/\/[^\s<>)"]+/g
const MAX_CITATION_DATA_CACHE_ENTRIES = 8
const MAX_CITATION_TREE_CACHE_ENTRIES = 256
const CITATION_DATA_CACHE_DIR = path.join('.quartz-cache', 'citation-data')
const CITATION_TREE_CACHE_DIR = path.join('.quartz-cache', 'citation-tree')
const citationDataCache = new Map<string, unknown[]>()
const citationTreeCache = new Map<string, HastRoot>()
const citationInputPartKeys = new Map<string, string>()
const require = createRequire(import.meta.url)

interface CachedCiteInstance {
  _options?: unknown
  log?: unknown
  data: unknown[]
}

interface CitationLocaleRegister {
  has(locale: string): boolean
  add(locale: string, localeXml: string): unknown
}

interface CitationPluginConfig {
  get(name: string): unknown
}

function isCitationLocaleRegister(value: unknown): value is CitationLocaleRegister {
  return isRecord(value) && typeof value.has === 'function' && typeof value.add === 'function'
}

function citationPluginConfig(): CitationPluginConfig | undefined {
  const plugins: unknown = Reflect.get(Cite, 'plugins')
  if (!isRecord(plugins) || !isRecord(plugins.config)) return undefined

  const config = plugins.config
  const get = config.get
  if (typeof get !== 'function') return undefined

  return {
    get(name) {
      return get.call(config, name)
    },
  }
}

function citationLocaleRegister(): CitationLocaleRegister | undefined {
  const config = citationPluginConfig()?.get('@csl')
  if (!isRecord(config) || !isCitationLocaleRegister(config.locales)) return undefined
  return config.locales
}

function bundledLocale(locale: string): string | undefined {
  try {
    const rehypeCitationPath = require.resolve('rehype-citation')
    const localesPath = require.resolve('@citation-js/plugin-csl/lib/locales.json', {
      paths: [path.dirname(rehypeCitationPath)],
    })
    const parsed: unknown = JSON.parse(fs.readFileSync(localesPath, 'utf8'))
    if (!isRecord(parsed)) return undefined
    const value = parsed[locale]
    return typeof value === 'string' ? value : undefined
  } catch {
    return undefined
  }
}

export function resolveCitationLocale(locale: string): string {
  const register = citationLocaleRegister()
  if (!register || register.has(locale)) return locale

  const localeXml = bundledLocale(locale)
  if (localeXml) {
    register.add(locale, localeXml)
    return locale
  }

  return `https://raw.githubusercontent.com/citation-style-language/locales/master/locales-${locale}.xml`
}

function citationInputPartKey(part: string): string {
  const cached = citationInputPartKeys.get(part)
  if (cached) return cached
  try {
    const stat = fs.statSync(part)
    if (stat.isFile()) {
      const key = `${part}:${stat.size}:${Math.trunc(stat.mtimeMs)}`
      citationInputPartKeys.set(part, key)
      return key
    }
  } catch {
    return part
  }
  return part
}

async function citationInputPartKeyAsync(part: string): Promise<string> {
  const cached = citationInputPartKeys.get(part)
  if (cached) return cached
  try {
    const stat = await fsp.stat(part)
    if (stat.isFile()) {
      const key = `${part}:${stat.size}:${Math.trunc(stat.mtimeMs)}`
      citationInputPartKeys.set(part, key)
      return key
    }
  } catch {
    return part
  }
  return part
}

function citationInputParts(data: unknown): string[] | undefined {
  if (!Array.isArray(data)) return undefined
  const parts: string[] = []
  for (const item of data) {
    if (typeof item !== 'string') return undefined
    parts.push(citationInputPartKey(item))
  }
  return parts
}

async function citationInputPartsAsync(data: unknown): Promise<string[] | undefined> {
  if (!Array.isArray(data)) return undefined
  const parts: string[] = []
  for (const item of data) {
    if (typeof item !== 'string') return undefined
    parts.push(await citationInputPartKeyAsync(item))
  }
  return parts
}

function citationDataCacheKey(data: unknown): string | undefined {
  const parts = citationInputParts(data)
  if (!parts) return undefined
  return citationDataCacheKeyFromParts(parts)
}

async function citationDataCacheKeyAsync(data: unknown): Promise<string | undefined> {
  const parts = await citationInputPartsAsync(data)
  if (!parts) return undefined
  return citationDataCacheKeyFromParts(parts)
}

function citationDataCacheKeyFromParts(parts: string[]): string {
  let hash = 2166136261
  let length = 0
  for (const part of parts) {
    length += part.length
    for (let index = 0; index < part.length; index += 1) {
      hash ^= part.charCodeAt(index)
      hash = Math.imul(hash, 16777619)
    }
  }
  return `${parts.length}:${length}:${hash >>> 0}`
}

function stringFingerprint(value: string): string {
  let hash = 2166136261
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return `${value.length}:${hash >>> 0}`
}

function cloneCitationData(data: unknown[]): unknown[] | undefined {
  try {
    return structuredClone(data)
  } catch {
    return undefined
  }
}

function cloneCitationTree(tree: HastRoot): HastRoot | undefined {
  try {
    return structuredClone(tree)
  } catch {
    return undefined
  }
}

function isHastRoot(node: unknown): node is HastRoot {
  return isRecord(node) && node.type === 'root' && Array.isArray(node.children)
}

function persistentCitationDataCachePath(key: string): string {
  return path.join(CITATION_DATA_CACHE_DIR, `${key.replace(/[^A-Za-z0-9_.-]/g, '_')}.json`)
}

function persistentCitationTreeCachePath(key: string): string {
  return path.join(CITATION_TREE_CACHE_DIR, `${key.replace(/[^A-Za-z0-9_.-]/g, '_')}.json`)
}

async function readPersistentCitationDataCache(key: string): Promise<unknown[] | undefined> {
  try {
    const parsed = JSON.parse(await fsp.readFile(persistentCitationDataCachePath(key), 'utf8'))
    return Array.isArray(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}

async function readPersistentCitationTreeCache(key: string): Promise<HastRoot | undefined> {
  try {
    const parsed = JSON.parse(await fsp.readFile(persistentCitationTreeCachePath(key), 'utf8'))
    return isHastRoot(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}

async function writePersistentCitationDataCache(key: string, data: unknown[]): Promise<void> {
  try {
    await fsp.mkdir(CITATION_DATA_CACHE_DIR, { recursive: true })
    await fsp.writeFile(persistentCitationDataCachePath(key), JSON.stringify(data))
  } catch {
    return
  }
}

async function writePersistentCitationTreeCache(key: string, tree: HastRoot): Promise<void> {
  try {
    await fsp.mkdir(CITATION_TREE_CACHE_DIR, { recursive: true })
    await fsp.writeFile(persistentCitationTreeCachePath(key), JSON.stringify(tree))
  } catch {
    return
  }
}

function setCitationDataCache(key: string, data: unknown[]): void {
  citationDataCache.set(key, data)
  void writePersistentCitationDataCache(key, data)
  if (citationDataCache.size <= MAX_CITATION_DATA_CACHE_ENTRIES) return
  const first = citationDataCache.keys().next().value
  if (first) citationDataCache.delete(first)
}

const hydratedCitationDataCacheKeys = new Set<string>()

async function hydrateCitationDataCache(key: string): Promise<void> {
  if (citationDataCache.has(key) || hydratedCitationDataCacheKeys.has(key)) return
  hydratedCitationDataCacheKeys.add(key)
  const persisted = await readPersistentCitationDataCache(key)
  if (persisted) {
    citationDataCache.set(key, persisted)
  }
}

async function setCitationTreeCache(key: string, tree: HastRoot): Promise<void> {
  const cloned = cloneCitationTree(tree)
  if (!cloned) return
  citationTreeCache.set(key, cloned)
  await writePersistentCitationTreeCache(key, cloned)
  if (citationTreeCache.size <= MAX_CITATION_TREE_CACHE_ENTRIES) return
  const first = citationTreeCache.keys().next().value
  if (first) citationTreeCache.delete(first)
}

function CachedCite(this: CachedCiteInstance, data: unknown, opts: unknown): void {
  const key = citationDataCacheKey(data)
  if (key) {
    const cached = citationDataCache.get(key)
    if (cached) {
      this._options = {}
      this.log = []
      this.data = cloneCitationData(cached) ?? cached
      return
    }
  }

  const citation = new Cite(data, opts)
  this._options = citation._options
  this.log = citation.log
  this.data = citation.data

  const cloned = cloneCitationData(citation.data)
  if (key && cloned) {
    setCitationDataCache(key, cloned)
  }
}

const cachedCite = Object.assign(CachedCite, { plugins: Cite.plugins })
const cachedRehypeCitation = rehypeCitationGenerator(cachedCite)

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

function citationTreeCacheKey(
  file: VFile,
  bibliographyKey: string,
  locale: string,
): string | undefined {
  if (file.data.citationsDisabled || !file.data.hasCitationSyntax) return undefined
  const source = typeof file.value === 'string' ? file.value : file.value?.toString()
  if (typeof source !== 'string') return undefined
  const frontmatter = file.data.frontmatter
  return [
    stringFingerprint(markdownBody(source)),
    bibliographyKey,
    locale,
    frontmatterCacheValue(frontmatter, 'noCite'),
  ].join(':')
}

async function readCitationTreeCache(key: string): Promise<HastRoot | undefined> {
  const cached = citationTreeCache.get(key)
  if (cached) return cloneCitationTree(cached)

  const persisted = await readPersistentCitationTreeCache(key)
  if (!persisted) return undefined
  citationTreeCache.set(key, persisted)
  return cloneCitationTree(persisted)
}

function getLinkType(url: string): LinkType | undefined {
  const parsed = parseExternalUrl(url)
  if (!parsed) return undefined
  return LINK_TYPES.find(type => type.pattern(parsed, url))
}

function createLinkElement(href: string): Element {
  const linkType = getLinkType(href)
  const displayText = linkType ? linkType.label : href

  return h(
    'a.csl-external-link',
    { href, target: '_blank', rel: 'noopener noreferrer' },
    createTextNode(displayText),
  )
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
        let hasCitationSyntax = hasFrontmatterValue(frontmatter, 'noCite')
        const arxivNodes: { node: Link; index: number; parent: Parent; id: string }[] = []

        visit(
          tree,
          (node): node is Text => node.type === 'text',
          node => {
            if (hasCitationSyntax) return
            hasCitationSyntax = node.value.includes('@')
          },
        )

        visit(
          tree,
          (node): node is Link => node.type === 'link',
          (node, index, parent) => {
            if (typeof index !== 'number' || !isMdastParent(parent)) return
            const arxivId = extractArxivId(node.url)
            if (!arxivId) return
            arxivNodes.push({ node, index, parent, id: normalizeArxivId(arxivId) })
          },
        )

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
    htmlPlugins: ({ argv, cfg }) => [
      () => {
        const locale = resolveCitationLocale(cfg.configuration.locale)
        const bibliographyKey = citationInputPartKeyAsync(bibliography)
        const citationDataKey = citationDataCacheKeyAsync([bibliography])
        const renderCitations = unified().use(cachedRehypeCitation, {
          bibliography,
          suppressBibliography: false,
          linkCitations: true,
          csl: 'apa',
          lang: locale,
        })
        return async (tree: HastRoot, file: VFile) => {
          if (file.data.citationsDisabled || !file.data.hasCitationSyntax) return
          const [resolvedBibliographyKey, resolvedCitationDataKey] = await Promise.all([
            bibliographyKey,
            citationDataKey,
          ])
          if (resolvedCitationDataKey) {
            await hydrateCitationDataCache(resolvedCitationDataKey)
          }
          const cacheKey = citationTreeCacheKey(file, resolvedBibliographyKey, locale)
          const cached = cacheKey ? await readCitationTreeCache(cacheKey) : undefined
          if (cached) {
            tree.children = cached.children
            return
          }
          const perf = new PerfTimer()
          await renderCitations.run(tree, file)
          if (cacheKey) await setCitationTreeCache(cacheKey, tree)
          const slug = typeof file.data.slug === 'string' ? file.data.slug : file.path
          logBuildSpan(argv, 'html:Citations', slug, perf.elapsedMs())
        }
      },
      () => (tree: HastRoot, file: VFile) => {
        if (file.data.citationsDisabled) return
        visit(tree, (node, _index, parent) => {
          if (!isElement(node) || !checkBib(node) || !isElement(parent)) return
          node.properties['data-bib'] = true
          parent.tagName = 'cite'
        })
      },
      () => (tree: HastRoot, file: VFile) => {
        if (file.data.citationsDisabled) return
        normalizeCitationBibliography(tree)
      },
    ],
  }
}
