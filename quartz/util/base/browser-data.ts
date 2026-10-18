import yaml from 'js-yaml'
import type { QuartzPluginData } from '../../plugins/vfile'
import type { BaseFile } from './types'
import { isFilePath, isFullSlug, isSimpleSlug, simplifySlug } from '../path'
import { isJsonObject, isRecord, type JsonObject, type JsonValue } from '../type-guards'

export const BASE_QUERY_DATA_SLUG = 'static/base-query'
export const BASE_QUERY_DATA_PATH = `/${BASE_QUERY_DATA_SLUG}.json`

export interface BrowserBaseRecord {
  slug: string
  filePath: string
  frontmatter: JsonObject
  datePaths: string[][]
  dates?: { created: string; modified: string; published: string }
  links: string[]
  embeds: string[]
  size?: number
}

export interface BaseQueryData {
  version: 1
  files: BrowserBaseRecord[]
}

export interface BaseQueryDocument {
  source: string
  baseFile: BrowserBaseRecord
  thisFile?: BrowserBaseRecord
}

const sensitiveKeys = new Set([
  'password',
  'protectedpassword',
  'secret',
  'token',
  'authtoken',
  'accesstoken',
  'refreshtoken',
  'apikey',
  'authorization',
  'cookie',
])

import { ReloadableWeakMap, releasedOnReload } from '../module-caches'

const sensitiveKeyCache = releasedOnReload(new Map<string, boolean>())

const isSensitiveKey = (key: string): boolean => {
  let cached = sensitiveKeyCache.get(key)
  if (cached === undefined) {
    cached = sensitiveKeys.has(key.replace(/[-_]/g, '').toLowerCase())
    sensitiveKeyCache.set(key, cached)
  }
  return cached
}

export const isPublicBaseFile = (file: QuartzPluginData): boolean =>
  !file.frontmatter?.draft &&
  !file.frontmatter?.private &&
  !file.frontmatter?.protected &&
  !file.frontmatter?.password &&
  !file.protectedPassword

function projectValue(
  value: unknown,
  path: string[],
  datePaths: string[][],
  ancestors: Set<object>,
): JsonValue | undefined {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value
  if (typeof value === 'number') return Number.isFinite(value) ? value : undefined
  if (value instanceof Date) {
    if (!Number.isFinite(value.getTime())) return undefined
    datePaths.push(path)
    return value.toISOString()
  }
  if (typeof value !== 'object' || ancestors.has(value)) return undefined
  ancestors.add(value)
  let result: JsonValue | undefined
  if (Array.isArray(value)) {
    result = value.map(
      (entry, index) => projectValue(entry, [...path, String(index)], datePaths, ancestors) ?? null,
    )
  } else if (isRecord(value)) {
    result = Object.fromEntries(
      Object.entries(value).flatMap(([key, entry]) => {
        if (isSensitiveKey(key)) return []
        const projected = projectValue(entry, [...path, key], datePaths, ancestors)
        return projected === undefined ? [] : [[key, projected]]
      }),
    )
  }
  ancestors.delete(value)
  return result
}

function sourcePath(file: QuartzPluginData): string {
  for (const path of [file.relativePath, file.filePath]) {
    if (typeof path !== 'string') continue
    const relative = path.replace(/^content\//, '')
    if (isSourcePath(relative)) return relative
  }
  return `${file.slug ?? 'untitled'}.${file.bases ? 'base' : 'md'}`
}

function isSourcePath(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    isFilePath(value) &&
    !value.startsWith('/') &&
    !value.includes('\\') &&
    !value.includes(':') &&
    !value.split('/').includes('..')
  )
}

function serializeRecord(file: QuartzPluginData): BrowserBaseRecord | undefined {
  if (!file.slug || !isFullSlug(file.slug) || !isPublicBaseFile(file)) return undefined
  const datePaths: string[][] = []
  const projected = projectValue(file.frontmatter ?? {}, [], datePaths, new Set())
  const frontmatter = isJsonObject(projected) ? projected : {}
  const dates = file.dates
  const validDates =
    dates &&
    [dates.created, dates.modified, dates.published].every(
      date => date instanceof Date && Number.isFinite(date.getTime()),
    )
  return {
    slug: file.slug,
    filePath: sourcePath(file),
    frontmatter,
    datePaths,
    dates: validDates
      ? {
          created: dates.created.toISOString(),
          modified: dates.modified.toISOString(),
          published: dates.published.toISOString(),
        }
      : undefined,
    links: Array.isArray(file.links) ? file.links.filter(link => typeof link === 'string') : [],
    embeds: Array.isArray(file.embeds)
      ? file.embeds.filter((entry): entry is string => typeof entry === 'string')
      : [],
    size: typeof file.size === 'number' && Number.isFinite(file.size) ? file.size : undefined,
  }
}

/** The emitter passes its already-published inventory; this projection also excludes locked notes. */
export function createBaseQueryData(files: QuartzPluginData[]): BaseQueryData {
  const records = files.flatMap(file => {
    const record = serializeRecord(file)
    return record ? [record] : []
  })
  const visible = new Set(
    records.flatMap(record => (isFullSlug(record.slug) ? [simplifySlug(record.slug)] : [])),
  )
  for (const record of records)
    record.links = record.links.filter(link => isSimpleSlug(link) && visible.has(link))
  return { version: 1, files: records }
}

export function baseQuerySource(config: BaseFile): string {
  return yaml.dump(config, { noRefs: true, lineWidth: -1, sortKeys: false })
}

export function createBaseQueryDocument(
  baseFileData: QuartzPluginData,
  thisFile?: QuartzPluginData,
): BaseQueryDocument | undefined {
  const config = baseFileData.basesConfig
  const baseFile = serializeRecord(baseFileData)
  if (!config || !baseFile) return undefined
  const context = thisFile ? serializeRecord(thisFile) : undefined
  if (thisFile && !context) return undefined
  return {
    source: baseFileData.basesSource ?? baseQuerySource(config),
    baseFile,
    thisFile: context,
  }
}

const strings = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every(entry => typeof entry === 'string')

const validDate = (value: unknown): value is string =>
  typeof value === 'string' && Number.isFinite(Date.parse(value))

const parsedDates = new ReloadableWeakMap<
  NonNullable<BrowserBaseRecord['dates']>,
  NonNullable<QuartzPluginData['dates']>
>()

function readRecord(value: unknown): BrowserBaseRecord | undefined {
  if (
    !isRecord(value) ||
    typeof value.slug !== 'string' ||
    value.slug.length === 0 ||
    !isFullSlug(value.slug) ||
    !isSourcePath(value.filePath) ||
    !isJsonObject(value.frontmatter) ||
    !Array.isArray(value.datePaths) ||
    !value.datePaths.every(strings) ||
    !strings(value.links) ||
    !strings(value.embeds) ||
    (value.size !== undefined && (typeof value.size !== 'number' || !Number.isFinite(value.size)))
  )
    return undefined
  const rawDates = value.dates
  let dates: BrowserBaseRecord['dates']
  if (rawDates !== undefined) {
    if (
      !isRecord(rawDates) ||
      typeof rawDates.created !== 'string' ||
      typeof rawDates.modified !== 'string' ||
      typeof rawDates.published !== 'string'
    )
      return undefined
    const created = new Date(rawDates.created)
    const modified = new Date(rawDates.modified)
    const published = new Date(rawDates.published)
    if (
      !Number.isFinite(created.getTime()) ||
      !Number.isFinite(modified.getTime()) ||
      !Number.isFinite(published.getTime())
    )
      return undefined
    dates = {
      created: rawDates.created,
      modified: rawDates.modified,
      published: rawDates.published,
    }
    parsedDates.set(dates, { created, modified, published })
  }
  if (value.frontmatter.draft || value.frontmatter.private || value.frontmatter.protected)
    return undefined
  for (const key in value.frontmatter) {
    if (isSensitiveKey(key)) return undefined
  }
  return {
    slug: value.slug,
    filePath: value.filePath,
    frontmatter: value.frontmatter,
    datePaths: value.datePaths,
    dates,
    links: value.links,
    embeds: value.embeds,
    size: value.size,
  }
}

export function parseBaseQueryDocument(value: unknown): BaseQueryDocument | undefined {
  if (!isRecord(value) || typeof value.source !== 'string') return undefined
  const baseFile = readRecord(value.baseFile)
  const thisFile = value.thisFile === undefined ? undefined : readRecord(value.thisFile)
  if (!baseFile || (value.thisFile !== undefined && !thisFile)) return undefined
  return { source: value.source, baseFile, thisFile }
}

function reviveValue(value: JsonValue, path: string[], datePaths: Set<string>): unknown {
  if (typeof value === 'string' && datePaths.has(JSON.stringify(path)) && validDate(value))
    return new Date(value)
  if (Array.isArray(value))
    return value.map((entry, index) => reviveValue(entry, [...path, String(index)], datePaths))
  if (isJsonObject(value))
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [
        key,
        reviveValue(entry, [...path, key], datePaths),
      ]),
    )
  return value
}

export function hydrateBaseQueryRecord(record: BrowserBaseRecord): QuartzPluginData {
  const slug = record.slug
  const filePath = record.filePath
  if (!isFullSlug(slug) || !isFilePath(filePath)) throw new Error('Invalid base record path')
  let frontmatter: QuartzPluginData['frontmatter']
  if (record.datePaths.length === 0) {
    frontmatter = {
      title: slug,
      pageLayout: 'default',
      ...record.frontmatter,
    } as QuartzPluginData['frontmatter']
  } else {
    frontmatter = { title: slug, pageLayout: 'default' }
    const revived = reviveValue(
      record.frontmatter,
      [],
      new Set(record.datePaths.map(path => JSON.stringify(path))),
    )
    if (isRecord(revived)) Object.assign(frontmatter, revived)
  }
  let dates: QuartzPluginData['dates']
  if (record.dates) {
    dates = parsedDates.get(record.dates)
    if (!dates) {
      dates = {
        created: new Date(record.dates.created),
        modified: new Date(record.dates.modified),
        published: new Date(record.dates.published),
      }
      parsedDates.set(record.dates, dates)
    }
  }
  return {
    slug,
    filePath,
    relativePath: filePath,
    frontmatter,
    dates,
    links: record.links.filter(isSimpleSlug),
    embeds: record.embeds,
    size: record.size,
  }
}

export function readBaseQueryData(value: unknown): QuartzPluginData[] {
  if (!isRecord(value) || value.version !== 1 || !Array.isArray(value.files))
    throw new Error('Invalid base query data')
  return value.files.map(entry => {
    const record = readRecord(entry)
    if (!record) throw new Error('Invalid base query record')
    return hydrateBaseQueryRecord(record)
  })
}
