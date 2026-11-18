import matter from 'gray-matter'
import yaml from 'js-yaml'
import katex from 'katex'
import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import { deserialize, serialize } from 'node:v8'
import path from 'path'
import { VFile } from 'vfile'
import type { MarkdownContent, ProcessedContent, QuartzPluginData } from '../plugins/vfile'
import type { BuildCtx } from '../util/ctx'
import type { QuartzHtmlProcessor, QuartzMdProcessor } from './markdown-processors'
import { isParseSourceInput } from '../plugins/emitters/component-resources/change-classifier'
import { ReloadableWeakMap } from '../util/module-caches'
import { type FilePath, slugifyFilePath } from '../util/path'
import { isRecord } from '../util/type-guards'
import { createHtmlProcessor, createMdProcessor } from './markdown-processors'

type ProcessorCache = {
  cfg: BuildCtx['cfg']
  profile: string
  md: QuartzMdProcessor
  html: QuartzHtmlProcessor
}
type ProcessedContentCacheEntry = {
  signature: string
  dependencySignature?: string
  fileDependencySignature?: string
  content: ProcessedContent
}
type ProcessedContentCacheState = {
  sourceSalt: string | undefined
  content: Map<FilePath, ProcessedContentCacheEntry>
}

const processorCache = new ReloadableWeakMap<BuildCtx, ProcessorCache>()

declare global {
  var __quartzProcessedContentCache: ProcessedContentCacheState | undefined
  var __quartzParseSourceInputs: string[] | undefined
  var __quartzParseSourceSalt: string | undefined
}

const processedContentCache = (globalThis.__quartzProcessedContentCache ??= {
  sourceSalt: undefined,
  content: new Map<FilePath, ProcessedContentCacheEntry>(),
})

type CachedError = {
  value: Error
  kind: 'native' | 'vfile-message' | 'katex-parse-error'
  properties: { key: string; value: unknown; enumerable: boolean }[]
}
const vfileMessagePrototype: object | null = Object.getPrototypeOf(new VFile().message(''))
const errorPrototypes = new Set<unknown>([
  Error.prototype,
  EvalError.prototype,
  RangeError.prototype,
  ReferenceError.prototype,
  SyntaxError.prototype,
  TypeError.prototype,
  URIError.prototype,
])

export const PROCESSED_CONTENT_CACHE_VERSION = 4
export type ProcessedContentSnapshot =
  | { bytes: Buffer }
  | { reason: 'unsupported-metadata' | 'serialization-failed'; detail: string }

class UnsupportedCacheMetadata extends Error {
  constructor(
    readonly location: string,
    readonly prototypeName: string,
  ) {
    super(
      `Plugin metadata contains a prototype the parse cache cannot restore at ${location}: ${prototypeName}`,
    )
  }
}

export function resetProcessedContentCache(): void {
  processedContentCache.sourceSalt = undefined
  processedContentCache.content.clear()
}

// V8 preserves Dates, Maps, Sets and shared references between OFM blocks and their HAST.
// Restore the VFile prototype after deserialization, before any filter or emitter sees it.
export function restoreProcessedContentCache(bytes: Uint8Array): boolean {
  const state: unknown = deserialize(bytes)
  if (
    !isRecord(state) ||
    state.version !== PROCESSED_CONTENT_CACHE_VERSION ||
    typeof state.sourceSalt !== 'string' ||
    !(state.content instanceof Map) ||
    !Array.isArray(state.errors)
  )
    return false
  for (const error of state.errors) {
    if (
      !isRecord(error) ||
      !(error.value instanceof Error) ||
      (error.kind !== 'native' &&
        error.kind !== 'vfile-message' &&
        error.kind !== 'katex-parse-error') ||
      !Array.isArray(error.properties)
    )
      return false
    for (const property of error.properties) {
      if (
        !isRecord(property) ||
        typeof property.key !== 'string' ||
        typeof property.enumerable !== 'boolean'
      )
        return false
      Object.defineProperty(error.value, property.key, {
        value: property.value,
        enumerable: property.enumerable,
        configurable: true,
        writable: true,
      })
    }
    if (error.kind === 'vfile-message') Object.setPrototypeOf(error.value, vfileMessagePrototype)
    if (error.kind === 'katex-parse-error')
      Object.setPrototypeOf(error.value, katex.ParseError.prototype)
  }
  for (const [fp, entry] of state.content) {
    if (
      typeof fp !== 'string' ||
      !isRecord(entry) ||
      typeof entry.signature !== 'string' ||
      !Array.isArray(entry.content) ||
      entry.content.length !== 2 ||
      !isRecord(entry.content[0]) ||
      entry.content[0].type !== 'root' ||
      !Array.isArray(entry.content[0].children) ||
      !isRecord(entry.content[1]) ||
      !isRecord(entry.content[1].data) ||
      !Array.isArray(entry.content[1].history) ||
      typeof entry.content[1].cwd !== 'string'
    )
      return false
    Object.setPrototypeOf(entry.content[1], VFile.prototype)
  }
  processedContentCache.sourceSalt = state.sourceSalt
  processedContentCache.content = state.content
  return true
}

export function snapshotProcessedContentCache(fps: FilePath[]): ProcessedContentSnapshot {
  try {
    const errors: CachedError[] = []
    const snapshot = {
      version: PROCESSED_CONTENT_CACHE_VERSION,
      sourceSalt: processedContentCache.sourceSalt,
      content: new Map(
        fps.flatMap(fp => {
          const entry = processedContentCache.content.get(fp)
          return entry ? [[fp, entry] as const] : []
        }),
      ),
    }
    assertPersistable(snapshot, new WeakSet(), errors)
    return { bytes: serialize({ ...snapshot, errors }) }
  } catch (error) {
    console.warn('Unable to serialize persistent parse cache', error)
    return error instanceof UnsupportedCacheMetadata
      ? { reason: 'unsupported-metadata', detail: `${error.location}: ${error.prototypeName}` }
      : { reason: 'serialization-failed', detail: 'Unable to encode the processed content graph' }
  }
}

function assertPersistable(
  value: unknown,
  seen: WeakSet<object>,
  errors: CachedError[],
  location = 'snapshot',
): void {
  if (value === null || typeof value !== 'object' || seen.has(value)) return
  seen.add(value)
  const prototype: unknown = Object.getPrototypeOf(value)
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer || value instanceof Date) return
  const kind =
    prototype === vfileMessagePrototype
      ? 'vfile-message'
      : prototype === katex.ParseError.prototype
        ? 'katex-parse-error'
        : errorPrototypes.has(prototype)
          ? 'native'
          : undefined
  if (value instanceof Error && kind) {
    // V8 retains Error identity and causes, but drops custom diagnostic fields and subclasses.
    const properties: CachedError['properties'] = Object.getOwnPropertyNames(value).map(key => ({
      key,
      value: Reflect.get(value, key),
      enumerable: Object.prototype.propertyIsEnumerable.call(value, key),
    }))
    errors.push({ value, properties, kind })
    for (const property of properties)
      assertPersistable(property.value, seen, errors, `${location}.${property.key}`)
    return
  }
  if (
    prototype !== null &&
    prototype !== Object.prototype &&
    prototype !== Array.prototype &&
    prototype !== Map.prototype &&
    prototype !== Set.prototype &&
    prototype !== VFile.prototype
  )
    throw new UnsupportedCacheMetadata(
      location,
      isRecord(prototype) && typeof prototype.constructor === 'function'
        ? prototype.constructor.name
        : 'unknown',
    )
  if (value instanceof Map) {
    for (const [key, item] of value) {
      assertPersistable(key, seen, errors, `${location}.key`)
      assertPersistable(
        item,
        seen,
        errors,
        `${location}[${typeof key === 'string' ? key : 'entry'}]`,
      )
    }
  } else if (value instanceof Set) {
    for (const item of value) assertPersistable(item, seen, errors, `${location}.item`)
  } else {
    for (const [key, item] of Object.entries(value))
      assertPersistable(item, seen, errors, `${location}.${key}`)
  }
}

export function restoredProcessedContent(fps: FilePath[]): ProcessedContent[] {
  return fps.flatMap(fp => {
    const entry = processedContentCache.content.get(fp)
    return entry ? [entry.content] : []
  })
}

const markdownFrontmatterPattern = /^---\r?\n[\s\S]*?\r?\n---\r?\n?/
const htmlReuseIgnoredFrontmatterKeys = new Set([
  'title',
  'tags',
  'date',
  'created',
  'modified',
  'published',
])

function markdownBody(source: unknown): string {
  const value = typeof source === 'string' ? source : source?.toString()
  if (!value) return ''
  const frontmatter = markdownFrontmatterPattern.exec(value)
  return frontmatter ? value.slice(frontmatter[0].length) : value
}

function stableSerialize(value: unknown): string {
  if (value instanceof Date) return JSON.stringify(value.toISOString())
  if (value === null || typeof value !== 'object') return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map(stableSerialize).join(',')}]`
  if (!isRecord(value)) return JSON.stringify(value)

  return `{${Object.keys(value)
    .sort()
    .map(key => `${JSON.stringify(key)}:${stableSerialize(value[key])}`)
    .join(',')}}`
}

function htmlReuseFrontmatterSignature(frontmatter: QuartzPluginData['frontmatter']): string {
  if (!isRecord(frontmatter)) return stableSerialize(frontmatter)
  const retained: Record<string, unknown> = {}
  for (const key of Object.keys(frontmatter).sort()) {
    if (!htmlReuseIgnoredFrontmatterKeys.has(key)) {
      retained[key] = frontmatter[key]
    }
  }
  return stableSerialize(retained)
}

function frontmatterTitleOnlySignature(frontmatter: Record<string, unknown>): string {
  const retained: Record<string, unknown> = {}
  for (const key of Object.keys(frontmatter).sort()) {
    if (key !== 'title') {
      retained[key] = frontmatter[key]
    }
  }
  return stableSerialize(retained)
}

function parseFrontmatterSource(
  source: string,
): { content: string; data: Record<string, unknown> } | undefined {
  const parsed = matter(Buffer.from(source), {
    delimiters: '---',
    language: 'yaml',
    engines: { yaml: s => yaml.load(s, { schema: yaml.JSON_SCHEMA }) as object },
  })
  if (!parsed.matter) return undefined
  return { content: parsed.content, data: parsed.data }
}

export function titleOnlyFrontmatterChange(
  currentSource: string,
  previousSource: string,
): string | undefined {
  const current = parseFrontmatterSource(currentSource)
  const previous = parseFrontmatterSource(previousSource)
  if (!current || !previous) return undefined
  if (current.content !== previous.content) return undefined
  if (
    frontmatterTitleOnlySignature(current.data) !== frontmatterTitleOnlySignature(previous.data)
  ) {
    return undefined
  }
  const currentTitle = current.data.title
  const previousTitle = previous.data.title
  if (currentTitle == null || previousTitle == null) return undefined
  const currentTitleString = currentTitle.toString()
  if (currentTitleString === '' || currentTitleString === previousTitle.toString()) {
    return undefined
  }
  return currentTitleString
}

function htmlReuseBodySignature(source: unknown): string {
  const body = markdownBody(source)
  let hash = 2166136261
  for (let index = 0; index < body.length; index += 1) {
    hash ^= body.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return `${body.length}:${hash >>> 0}`
}

export function recordHtmlReuseSignature(file: VFile): void {
  file.data.htmlReuseBody = htmlReuseBodySignature(file.value)
  file.data.htmlReuseFrontmatter = htmlReuseFrontmatterSignature(file.data.frontmatter)
}

export function canReuseProcessedHtml(file: VFile, previous: ProcessedContent): boolean {
  const previousFile = previous[1]
  const body = file.data.htmlReuseBody ?? htmlReuseBodySignature(file.value)
  const previousBody = previousFile.data.htmlReuseBody ?? htmlReuseBodySignature(previousFile.value)
  const frontmatter =
    file.data.htmlReuseFrontmatter ?? htmlReuseFrontmatterSignature(file.data.frontmatter)
  const previousFrontmatter =
    previousFile.data.htmlReuseFrontmatter ??
    htmlReuseFrontmatterSignature(previousFile.data.frontmatter)

  return body === previousBody && frontmatter === previousFrontmatter
}

function reuseProcessedHtml(file: VFile, previous: ProcessedContent): ProcessedContent {
  file.data = { ...previous[1].data, ...file.data }
  return [previous[0], file]
}

function reuseMarkdownForTitleOnlyChange(
  ctx: BuildCtx,
  file: VFile,
  previous: ProcessedContent,
  rawSource: string,
): MarkdownContent | undefined {
  const previousSource = previous[1].data.rawMarkdownSource
  if (!previousSource) return undefined
  const title = titleOnlyFrontmatterChange(rawSource, previousSource)
  if (!title) return undefined
  const previousFrontmatter = previous[1].data.frontmatter
  if (!previousFrontmatter?.pageLayout) return undefined

  const relativePath = path.posix.relative(ctx.argv.directory, file.path) as FilePath
  const frontmatterBlock = markdownFrontmatterPattern.exec(rawSource)?.[0]
  if (!frontmatterBlock) return undefined
  file.value = frontmatterBlock + markdownBody(previous[1].value)
  const nextFrontmatter: QuartzPluginData['frontmatter'] = {
    ...previousFrontmatter,
    pageLayout: previousFrontmatter.pageLayout,
    title,
  }
  file.data = {
    ...previous[1].data,
    filePath: file.path as FilePath,
    relativePath,
    slug: slugifyFilePath(relativePath),
    frontmatter: nextFrontmatter,
    rawMarkdownSource: rawSource,
  }
  recordHtmlReuseSignature(file)
  return [{ type: 'root', children: [] }, file]
}

export function reuseProcessedTitleOnlyChange(
  ctx: BuildCtx,
  fullPath: FilePath,
  rawSource: string,
  previous: ProcessedContent,
): ProcessedContent | undefined {
  const file = new VFile({ path: fullPath, value: rawSource })
  const markdown = reuseMarkdownForTitleOnlyChange(ctx, file, previous, rawSource)
  if (!markdown) return undefined
  return reuseProcessedHtml(markdown[1], previous)
}

export function cachedProcessors(ctx: BuildCtx): ProcessorCache {
  const cached = processorCache.get(ctx)
  const profile = transformerProfile(ctx)
  if (cached?.cfg === ctx.cfg && cached.profile === profile) {
    return cached
  }

  const next = { cfg: ctx.cfg, profile, md: createMdProcessor(ctx), html: createHtmlProcessor(ctx) }
  processorCache.set(ctx, next)
  return next
}

function transformerProfile(ctx: BuildCtx): string {
  return JSON.stringify(
    ctx.cfg.plugins.transformers.flatMap(plugin =>
      plugin.cacheKey === undefined ? [] : [[plugin.name, plugin.cacheKey]],
    ),
  )
}

async function fileStatSignature(fp: string): Promise<string> {
  try {
    const stat = await fs.stat(fp, { bigint: true })
    return `${stat.mtimeNs}:${stat.ctimeNs}:${stat.size}:${stat.ino}`
  } catch {
    return 'missing'
  }
}

async function sourceFileSignature(fp: string): Promise<string> {
  return `${fp}:${await fileStatSignature(fp)}`
}

async function processedContentCacheSourceSalt(): Promise<string> {
  if (typeof globalThis.__quartzParseSourceSalt === 'string') {
    return globalThis.__quartzParseSourceSalt
  }
  const signatures = await Promise.all(
    (globalThis.__quartzParseSourceInputs ?? [])
      .filter(isParseSourceInput)
      .map(sourceFileSignature),
  )
  return signatures.join('|')
}

export async function syncProcessedContentCacheSource(ctx: BuildCtx): Promise<void> {
  const sourceSalt = ctx.argv.watch
    ? JSON.stringify([
        await processedContentCacheSourceSalt(),
        path.resolve(ctx.argv.directory),
        PROCESSED_CONTENT_CACHE_VERSION,
        transformerProfile(ctx),
        ctx.allFiles.slice().sort(),
      ])
    : undefined
  if (processedContentCache.sourceSalt === sourceSalt) return
  processedContentCache.sourceSalt = sourceSalt
  processedContentCache.content.clear()
}

function hasDirtyCodeDependency(
  ctx: BuildCtx,
  content: ProcessedContent,
  dirtyPaths: ReadonlySet<FilePath>,
): boolean {
  const dependencies = content[1].data.codeDependencies ?? []
  for (const dependency of dependencies) {
    const resolved = path.posix.join(ctx.argv.directory, dependency) as FilePath
    if (dirtyPaths.has(resolved)) return true
  }
  return false
}

export async function cachedProcessedContent(
  ctx: BuildCtx,
  fp: FilePath,
  fileDependencies: Map<string, Promise<string>>,
  dirtyPaths?: ReadonlySet<FilePath>,
): Promise<{ fp: FilePath; signature: string; content?: ProcessedContent }> {
  const cached = processedContentCache.content.get(fp)
  if (
    cached &&
    (cached.fileDependencySignature ?? '') !==
      (await fileDependencySignature(ctx, cached.content, fileDependencies))
  ) {
    return { fp, signature: await fileStatSignature(fp) }
  }
  if (
    cached &&
    dirtyPaths !== undefined &&
    !dirtyPaths.has(fp) &&
    !hasDirtyCodeDependency(ctx, cached.content, dirtyPaths)
  ) {
    return { fp, signature: cached.signature, content: cached.content }
  }
  const signature = await fileStatSignature(fp)
  if (
    cached?.signature === signature &&
    (cached.dependencySignature ?? '') === (await codeDependencySignature(ctx, cached.content))
  ) {
    return { fp, signature, content: cached.content }
  }
  return { fp, signature }
}

export async function fileDependencySignature(
  ctx: BuildCtx,
  content: ProcessedContent,
  fingerprints: Map<string, Promise<string>>,
): Promise<string> {
  const dependencies = ctx.cfg.plugins.transformers.flatMap(
    plugin => plugin.fileDependencies?.(ctx, content[1]) ?? [],
  )
  const signatures = await Promise.all(
    dependencies.map(filename => {
      const resolved = path.resolve(filename)
      let fingerprint = fingerprints.get(resolved)
      if (!fingerprint) {
        fingerprint = fs.readFile(resolved).then(
          bytes => createHash('sha256').update(bytes).digest('hex'),
          () => 'missing',
        )
        fingerprints.set(resolved, fingerprint)
      }
      return fingerprint.then(value => `${resolved}:${value}`)
    }),
  )
  return signatures.join('|')
}

export async function codeDependencySignature(
  ctx: BuildCtx,
  content: ProcessedContent,
): Promise<string> {
  const dependencies = content[1].data.codeDependencies ?? []
  const signatures = await Promise.all(
    dependencies.map(dependency =>
      sourceFileSignature(path.resolve(ctx.argv.directory, dependency)),
    ),
  )
  return signatures.join('|')
}

export function filePathForProcessedContent(content: ProcessedContent): FilePath | undefined {
  const fp = content[1].data.filePath ?? content[1].path
  return typeof fp === 'string' ? (fp as FilePath) : undefined
}

export async function storeProcessedContent(
  ctx: BuildCtx,
  fp: FilePath,
  signature: string,
  content: ProcessedContent,
  fileDependencies: Map<string, Promise<string>>,
): Promise<void> {
  processedContentCache.content.set(fp, {
    signature,
    dependencySignature: await codeDependencySignature(ctx, content),
    fileDependencySignature: await fileDependencySignature(ctx, content, fileDependencies),
    content,
  })
}
