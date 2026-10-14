import type { Root as HastRoot } from 'hast'
import { randomUUID } from 'node:crypto'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import { createRequire } from 'node:module'
import path from 'node:path'
import { Cite, rehypeCitationGenerator } from 'rehype-citation'
import type { BuildCtx } from '../../util/ctx'
import type { ProcessedContent } from '../vfile'
import { releasedOnReload } from '../../util/module-caches'
import { isRecord } from '../../util/type-guards'
import '@citation-js/plugin-bibtex'
import '@citation-js/plugin-doi'

declare module 'vfile' {
  interface DataMap {
    citationCache?: { bibliography: string; data: string }
  }
}

const MAX_CITATION_DATA_CACHE_ENTRIES = 8
const MAX_CITATION_TREE_CACHE_ENTRIES = 256
const CITATION_DATA_CACHE_DIR = path.join('.quartz-cache', 'citation-data')
const CITATION_TREE_CACHE_DIR = path.join('.quartz-cache', 'citation-tree')
const citationDataCache = releasedOnReload(new Map<string, unknown[]>())
const dirtyCitationDataKeys = releasedOnReload(new Set<string>())
const citationTreeCache = releasedOnReload(new Map<string, HastRoot>())
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

function citationDataCacheKey(data: unknown): string | undefined {
  if (!Array.isArray(data) || !data.every((part): part is string => typeof part === 'string'))
    return
  return citationDataCacheKeyFromParts(data)
}

const bibliographySnapshots = new WeakMap<
  BuildCtx,
  { buildId: string; keys: Map<string, Promise<{ bibliography: string; data: string }>> }
>()

export function citationBibliography(ctx: BuildCtx, filename: string) {
  let snapshot = bibliographySnapshots.get(ctx)
  if (!snapshot || snapshot.buildId !== ctx.buildId) {
    snapshot = { buildId: ctx.buildId, keys: new Map() }
    bibliographySnapshots.set(ctx, snapshot)
  }
  let keys = snapshot.keys.get(filename)
  if (!keys) {
    keys = fsp
      .readFile(filename, 'utf8')
      .then(contents => ({
        bibliography: `${filename}:${stringFingerprint(contents)}`,
        data: citationDataCacheKeyFromParts([contents]),
      }))
    snapshot.keys.set(filename, keys)
  }
  return keys
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

export function stringFingerprint(value: string): string {
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

async function writeCache(filename: string, value: unknown): Promise<void> {
  const temporary = `${filename}.${randomUUID()}.tmp`
  try {
    await fsp.mkdir(path.dirname(filename), { recursive: true })
    await fsp.writeFile(temporary, JSON.stringify(value))
    await fsp.rename(temporary, filename)
  } catch {
    return
  } finally {
    await fsp.rm(temporary, { force: true }).catch(() => undefined)
  }
}

export async function persistCitationDataCache(key: string): Promise<void> {
  if (!dirtyCitationDataKeys.delete(key)) return
  const data = citationDataCache.get(key)
  if (data) await writeCache(persistentCitationDataCachePath(key), data)
}

function setCitationDataCache(key: string, data: unknown[]): void {
  citationDataCache.set(key, data)
  dirtyCitationDataKeys.add(key)
  if (citationDataCache.size <= MAX_CITATION_DATA_CACHE_ENTRIES) return
  const first = citationDataCache.keys().next().value
  if (first) citationDataCache.delete(first)
}

const hydratedCitationDataCacheKeys = releasedOnReload(new Set<string>())

export async function hydrateCitationDataCache(key: string): Promise<void> {
  if (citationDataCache.has(key) || hydratedCitationDataCacheKeys.has(key)) return
  hydratedCitationDataCacheKeys.add(key)
  const persisted = await readPersistentCitationDataCache(key)
  if (persisted) {
    citationDataCache.set(key, persisted)
  }
}

export async function pruneCitationCaches(content: ProcessedContent[]): Promise<void> {
  const active = content.flatMap(([, file]) =>
    file.data.citationCache ? [file.data.citationCache] : [],
  )
  if (active.length === 0) return
  const treeTokens = [
    ...new Set(active.map(key => `_${key.bibliography.replace(/[^A-Za-z0-9_.-]/g, '_')}_`)),
  ]
  const dataFiles = new Set(
    active.map(key => path.basename(persistentCitationDataCachePath(key.data))),
  )
  for (const [directory, keep] of [
    [CITATION_TREE_CACHE_DIR, (entry: string) => treeTokens.some(token => entry.includes(token))],
    [CITATION_DATA_CACHE_DIR, (entry: string) => dataFiles.has(entry)],
  ] satisfies [string, (entry: string) => boolean][]) {
    const entries = await fsp.readdir(directory).catch(() => [])
    await Promise.all(
      entries
        .filter(entry => entry.endsWith('.json') && !keep(entry))
        .map(entry => fsp.unlink(path.join(directory, entry)).catch(() => undefined)),
    )
  }
}

export async function setCitationTreeCache(key: string, tree: HastRoot): Promise<void> {
  const cloned = cloneCitationTree(tree)
  if (!cloned) return
  citationTreeCache.set(key, cloned)
  await writeCache(persistentCitationTreeCachePath(key), cloned)
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
export const cachedRehypeCitation = rehypeCitationGenerator(cachedCite)

export async function readCitationTreeCache(key: string): Promise<HastRoot | undefined> {
  const cached = citationTreeCache.get(key)
  if (cached) return cloneCitationTree(cached)

  const persisted = await readPersistentCitationTreeCache(key)
  if (!persisted) return undefined
  citationTreeCache.set(key, persisted)
  return cloneCitationTree(persisted)
}
