import type { Element, ElementContent, Root as HtmlRoot } from 'hast'
import type { Parents as MdastParents, PhrasingContent, Root as MdRoot } from 'mdast'
import type { Handler, Options as ToHastOptions, State } from 'mdast-util-to-hast'
import esbuild from 'esbuild'
import matter from 'gray-matter'
import yaml from 'js-yaml'
import { toHast } from 'mdast-util-to-hast'
import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import { deserialize, serialize } from 'node:v8'
import path from 'path'
import remarkParse from 'remark-parse'
import { read } from 'to-vfile'
import {
  Processor,
  type Pluggable,
  type PluggableList,
  type Plugin,
  type PluginTuple,
  type Preset,
  type Transformer,
  unified,
} from 'unified'
import { styleText } from 'util'
import { VFile } from 'vfile'
import workerpool, { Promise as WorkerPromise } from 'workerpool'
import type {
  Sidenote,
  SidenoteDefinition,
  SidenoteReference,
} from '../extensions/micromark-extension-ofm-sidenotes'
import { MarkdownContent, ProcessedContent, QuartzPluginData } from '../plugins/vfile'
import { BuildCtx, WorkerSerializableBuildCtx } from '../util/ctx'
import { QuartzLogger } from '../util/log'
import { FilePath, QUARTZ, slugifyFilePath } from '../util/path'
import { logBuildSpan, PerfTimer } from '../util/perf'
import { trace } from '../util/trace'
import { isRecord } from '../util/type-guards'

export type QuartzMdProcessor = Processor<MdRoot, MdRoot, MdRoot>
export type QuartzHtmlProcessor = Processor<HtmlRoot, HtmlRoot, HtmlRoot>

function oneAsElementChildren(
  state: State,
  nodes: PhrasingContent[] | undefined,
  parent: MdastParents,
): ElementContent[] {
  if (!nodes) return []
  const children: ElementContent[] = []
  for (const node of nodes) {
    const result = state.one(node, parent)
    if (Array.isArray(result)) {
      children.push(...result)
    } else if (result) {
      children.push(result)
    }
  }
  return children
}

function isSidenote(node: unknown): node is Sidenote {
  return typeof node === 'object' && node !== null && 'type' in node && node.type === 'sidenote'
}

function isSidenoteReference(node: unknown): node is SidenoteReference {
  return (
    typeof node === 'object' && node !== null && 'type' in node && node.type === 'sidenoteReference'
  )
}

function isSidenoteDefinition(node: unknown): node is SidenoteDefinition {
  return (
    typeof node === 'object' &&
    node !== null &&
    'type' in node &&
    node.type === 'sidenoteDefinition'
  )
}

function sidenoteElement(state: State, node: Sidenote): Element {
  const content = state.all(node)
  const label = oneAsElementChildren(state, node.data?.sidenoteParsed?.labelNodes, node)

  return {
    type: 'element',
    tagName: 'span',
    properties: {
      className: ['sidenote-placeholder'],
      dataType: 'sidenote',
      sidenoteId: node.data?.sidenoteId,
      baseId: node.data?.baseId,
      forceInline: node.data?.forceInline,
      allowLeft: node.data?.allowLeft,
      allowRight: node.data?.allowRight,
      label: node.data?.label,
      internal: node.data?.internal,
    },
    children: [
      {
        type: 'element',
        tagName: 'span',
        properties: { className: ['sidenote-label-hast'] },
        children: label,
      },
      {
        type: 'element',
        tagName: 'span',
        properties: { className: ['sidenote-content-hast'] },
        children: content,
      },
    ],
  }
}

function sidenoteReferenceElement(state: State, node: SidenoteReference): Element {
  const label = oneAsElementChildren(state, node.labelNodes, node)

  return {
    type: 'element',
    tagName: 'span',
    properties: {
      className: ['sidenote-ref-placeholder'],
      dataType: 'sidenote-ref',
      label: node.label,
      sidenoteId: node.data?.sidenoteId,
      baseId: node.data?.baseId,
    },
    children: [
      {
        type: 'element',
        tagName: 'span',
        properties: { className: ['sidenote-label-hast'] },
        children: label,
      },
    ],
  }
}

function sidenoteDefinitionElement(state: State, node: SidenoteDefinition): Element {
  const content = state.all(node)
  const label = oneAsElementChildren(state, node.labelNodes, node)

  return {
    type: 'element',
    tagName: 'div',
    properties: {
      className: ['sidenote-def-placeholder'],
      dataType: 'sidenote-def',
      label: node.label,
    },
    children: [
      {
        type: 'element',
        tagName: 'span',
        properties: { className: ['sidenote-label-hast'] },
        children: label,
      },
      {
        type: 'element',
        tagName: 'div',
        properties: { className: ['sidenote-content-hast'] },
        children: content,
      },
    ],
  }
}

const sidenoteHandler: Handler = (state, node) =>
  isSidenote(node) ? sidenoteElement(state, node) : undefined

const sidenoteReferenceHandler: Handler = (state, node) =>
  isSidenoteReference(node) ? sidenoteReferenceElement(state, node) : undefined

const sidenoteDefinitionHandler: Handler = (state, node) =>
  isSidenoteDefinition(node) ? sidenoteDefinitionElement(state, node) : undefined

const astToHastOptions: ToHastOptions = {
  allowDangerousHtml: true,
  handlers: {
    sidenote: sidenoteHandler,
    sidenoteReference: sidenoteReferenceHandler,
    sidenoteDefinition: sidenoteDefinitionHandler,
  },
}

export function mdastToHastRoot(ast: MdRoot): HtmlRoot {
  const tree = toHast(ast, astToHastOptions)
  if (tree.type !== 'root') {
    throw new Error('Expected Markdown root to convert to HTML root')
  }
  return tree
}

export function createMdProcessor(ctx: BuildCtx): QuartzMdProcessor {
  const transformers = ctx.cfg.plugins.transformers

  return (
    unified()
      // base Markdown -> MD AST
      .use(remarkParse)
      // MD AST -> MD AST transforms
      .use(
        transformers.flatMap(plugin => plugin.markdownPlugins?.(ctx) ?? []),
      ) as unknown as QuartzMdProcessor
    //  ^ sadly the typing of `use` is not smart enough to infer the correct type from our plugin list
  )
}

export function createHtmlProcessor(ctx: BuildCtx): QuartzHtmlProcessor {
  const transformers = ctx.cfg.plugins.transformers
  return unified().use(
    transformers.flatMap(plugin =>
      timedHtmlPlugins(ctx, plugin.name, plugin.htmlPlugins?.(ctx) ?? []),
    ),
  ) as unknown as QuartzHtmlProcessor
}

function timedHtmlPlugins(
  ctx: BuildCtx,
  transformerName: string,
  plugins: PluggableList,
): PluggableList {
  return plugins.map((plugin, index) => timedPluggable(ctx, `${transformerName}:${index}`, plugin))
}

function timedPluggable(ctx: BuildCtx, label: string, plugin: Pluggable): Pluggable {
  if (Array.isArray(plugin)) {
    const [attacher, ...parameters] = plugin
    return [timedPlugin(ctx, label, attacher), ...parameters] as PluginTuple
  }

  if (typeof plugin === 'function') {
    return timedPlugin(ctx, label, plugin)
  }

  if (isPreset(plugin)) {
    return {
      ...plugin,
      plugins: plugin.plugins?.map((item, index) => timedPluggable(ctx, `${label}.${index}`, item)),
    }
  }

  return plugin
}

function isPreset(plugin: Pluggable): plugin is Preset {
  return isRecord(plugin)
}

function timedPlugin(ctx: BuildCtx, label: string, plugin: Plugin<unknown[]>): Plugin<unknown[]> {
  return function wrappedPlugin(this: Processor, ...parameters: unknown[]) {
    const transformer = plugin.apply(this, parameters)
    if (!transformer) return transformer
    return timedTransformer(ctx, label, transformer)
  }
}

function timedTransformer(ctx: BuildCtx, label: string, transformer: Transformer): Transformer {
  return (tree, file) => {
    const perf = new PerfTimer()
    let logged = false
    const log = (targetFile: VFile) => {
      if (logged) return
      logged = true
      const slug = typeof targetFile.data.slug === 'string' ? targetFile.data.slug : targetFile.path
      logBuildSpan(ctx.argv, `hast:${label}`, slug, perf.elapsedMs())
    }

    const result = transformer(tree, file, () => undefined)
    if (result instanceof Promise) {
      return result.then(
        value => {
          log(file)
          return value
        },
        error => {
          log(file)
          throw error
        },
      )
    }

    log(file)
    return result
  }
}

type ProcessorCache = { cfg: BuildCtx['cfg']; md: QuartzMdProcessor; html: QuartzHtmlProcessor }
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

import { ReloadableWeakMap } from '../util/module-caches'

const processorCache = new ReloadableWeakMap<BuildCtx, ProcessorCache>()

declare global {
  var __quartzProcessedContentCache: ProcessedContentCacheState | undefined
  var __quartzParseSourceInputs: string[] | undefined
}

const processedContentCache = (globalThis.__quartzProcessedContentCache ??= {
  sourceSalt: undefined,
  content: new Map<FilePath, ProcessedContentCacheEntry>(),
})

type CachedError = {
  value: Error
  vfileMessage: boolean
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
  vfileMessagePrototype,
])

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
    typeof state.sourceSalt !== 'string' ||
    !(state.content instanceof Map) ||
    !Array.isArray(state.errors)
  )
    return false
  for (const error of state.errors) {
    if (
      !isRecord(error) ||
      !(error.value instanceof Error) ||
      typeof error.vfileMessage !== 'boolean' ||
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
    if (error.vfileMessage) Object.setPrototypeOf(error.value, vfileMessagePrototype)
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

export function snapshotProcessedContentCache(fps: FilePath[]): Buffer | undefined {
  try {
    const errors: CachedError[] = []
    const snapshot = {
      sourceSalt: processedContentCache.sourceSalt,
      content: new Map(
        fps.flatMap(fp => {
          const entry = processedContentCache.content.get(fp)
          return entry ? [[fp, entry] as const] : []
        }),
      ),
    }
    assertPersistable(snapshot, new WeakSet(), errors)
    return serialize({ ...snapshot, errors })
  } catch (error) {
    console.warn('Unable to serialize persistent parse cache', error)
    return undefined
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
  if (value instanceof Error && errorPrototypes.has(prototype)) {
    // V8 retains Error identity and causes, but drops custom diagnostic fields and subclasses.
    const properties: CachedError['properties'] = Object.getOwnPropertyNames(value).map(key => ({
      key,
      value: Reflect.get(value, key),
      enumerable: Object.prototype.propertyIsEnumerable.call(value, key),
    }))
    errors.push({ value, properties, vfileMessage: prototype === vfileMessagePrototype })
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
    throw new Error(
      `Plugin metadata contains a prototype the parse cache cannot restore at ${location}: ${isRecord(prototype) && typeof prototype.constructor === 'function' ? prototype.constructor.name : 'unknown'}`,
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

export function transformMarkdownSource(ctx: BuildCtx, value: string): string {
  let source = value.trim()
  for (const plugin of ctx.cfg.plugins.transformers.filter(plugin => plugin.textTransform)) {
    source = plugin.textTransform!(ctx, source)
  }
  return source
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

function recordHtmlReuseSignature(file: VFile): void {
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
  if (cached?.cfg === ctx.cfg) {
    return cached
  }

  const next = { cfg: ctx.cfg, md: createMdProcessor(ctx), html: createHtmlProcessor(ctx) }
  processorCache.set(ctx, next)
  return next
}

function* chunks<T>(arr: T[], n: number) {
  for (let i = 0; i < arr.length; i += n) {
    yield arr.slice(i, i + n)
  }
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
  const signatures = await Promise.all(
    (globalThis.__quartzParseSourceInputs ?? []).map(sourceFileSignature),
  )
  return signatures.join('|')
}

async function syncProcessedContentCacheSource(ctx: BuildCtx): Promise<void> {
  const sourceSalt = ctx.argv.watch
    ? JSON.stringify([
        await processedContentCacheSourceSalt(),
        path.resolve(ctx.argv.directory),
        ctx.allSlugs.slice().sort(),
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

async function cachedProcessedContent(
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

async function fileDependencySignature(
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

async function codeDependencySignature(ctx: BuildCtx, content: ProcessedContent): Promise<string> {
  const dependencies = content[1].data.codeDependencies ?? []
  const signatures = await Promise.all(
    dependencies.map(dependency =>
      sourceFileSignature(path.resolve(ctx.argv.directory, dependency)),
    ),
  )
  return signatures.join('|')
}

function filePathForProcessedContent(content: ProcessedContent): FilePath | undefined {
  const fp = content[1].data.filePath ?? content[1].path
  return typeof fp === 'string' ? (fp as FilePath) : undefined
}

export const TEXT_PARSE_CHUNK_SIZE = 128
export const HTML_PARSE_CHUNK_SIZE = 16

export function parseWorkerConcurrency(fileCount: number, requestedConcurrency?: number): number {
  if (fileCount <= 1) return 1
  const requested = requestedConcurrency ?? clamp(fileCount / TEXT_PARSE_CHUNK_SIZE, 1, 4)
  const textJobs = Math.ceil(fileCount / TEXT_PARSE_CHUNK_SIZE)
  const htmlJobs = Math.ceil(fileCount / HTML_PARSE_CHUNK_SIZE)
  return clamp(requested, 1, Math.max(textJobs, htmlJobs))
}

async function transpileWorkerScript() {
  // transpile worker script
  const cacheFile = './.quartz-cache/transpiled-worker.mjs'
  const fp = './quartz/worker.ts'
  return esbuild.build({
    entryPoints: [fp],
    outfile: path.join(QUARTZ, cacheFile),
    bundle: true,
    keepNames: true,
    platform: 'node',
    format: 'esm',
    packages: 'external',
    sourcemap: true,
    sourcesContent: false,
    plugins: [
      {
        name: 'css-and-scripts-as-text',
        setup(build) {
          build.onLoad({ filter: /\.scss$/ }, () => ({ contents: '', loader: 'text' }))
          build.onLoad({ filter: /\.inline\.(ts|js)$/ }, () => ({ contents: '', loader: 'text' }))
        },
      },
    ],
  })
}

async function parseNotebookFileToMdast(file: {
  path: string
  value: unknown
}): Promise<MdRoot | undefined> {
  if (!file.path.endsWith('.ipynb')) return undefined
  const { parseNotebookDoc } = await import('../util/notebook/parse')
  const { isNotebookParseError } = await import('../util/notebook/types')
  const { notebookDocToFlatMdast } = await import('../extensions/mdast-util-notebook/flatten')
  const raw = typeof file.value === 'string' ? file.value : (file.value?.toString() ?? '')
  const parsed = parseNotebookDoc(raw, file.path)
  if (isNotebookParseError(parsed)) {
    throw new Error(`Failed to parse notebook ${file.path}: ${parsed.reason}`)
  }
  return notebookDocToFlatMdast(parsed)
}

export function createFileParser(ctx: BuildCtx, fps: FilePath[]) {
  const { argv } = ctx
  return async (processor: QuartzMdProcessor) => {
    const res: MarkdownContent[] = []
    for (const fp of fps) {
      try {
        const perf = new PerfTimer()
        const file = await read(fp)
        const rawSource = file.value.toString()

        // base data properties that plugins may use
        file.data.filePath = file.path as FilePath
        file.data.relativePath = path.posix.relative(argv.directory, file.path) as FilePath
        file.data.slug = slugifyFilePath(file.data.relativePath)
        file.data.rawMarkdownSource = rawSource

        file.value = transformMarkdownSource(ctx, rawSource)

        const notebookAst = await parseNotebookFileToMdast({ path: file.path, value: file.value })
        const ast = notebookAst ?? processor.parse(file)
        const newAst = await processor.run(ast, file)
        recordHtmlReuseSignature(file)
        res.push([newAst, file])

        logBuildSpan(argv, 'markdown', `${fp} -> ${file.data.slug}`, perf.elapsedMs())
      } catch (err) {
        if (ctx.argv.watch) throw new Error(`Failed to process markdown ${fp}`, { cause: err })
        trace(`\nFailed to process markdown \`${fp}\``, err as Error)
      }
    }

    return res
  }
}

declare module 'vfile' {
  interface DataMap {
    htmlReuseBody: string
    htmlReuseFrontmatter: string
    rawMarkdownSource: string
  }
}

export function createMarkdownParser(ctx: BuildCtx, mdContent: MarkdownContent[]) {
  return async (processor: QuartzHtmlProcessor) => {
    const res: ProcessedContent[] = []
    for (const [ast, file] of mdContent) {
      try {
        const perf = new PerfTimer()
        const hastPerf = new PerfTimer()
        const hast = mdastToHastRoot(ast)
        logBuildSpan(ctx.argv, 'ast:hast', `${file.data.slug}`, hastPerf.elapsedMs())

        const pluginPerf = new PerfTimer()
        const newAst = await processor.run(hast, file)
        logBuildSpan(ctx.argv, 'hast:plugins', `${file.data.slug}`, pluginPerf.elapsedMs())
        res.push([newAst, file])

        logBuildSpan(ctx.argv, 'html', `${file.data.slug}`, perf.elapsedMs())
      } catch (err) {
        if (ctx.argv.watch)
          throw new Error(`Failed to process html ${file.data.filePath}`, { cause: err })
        trace(`\nFailed to process html \`${file.data.filePath}\``, err as Error)
      }
    }

    return res
  }
}

const clamp = (num: number, min: number, max: number) =>
  Math.min(Math.max(Math.round(num), min), max)

const HEAVY_PARSE_SUFFIXES = ['/are.na.md', '/stream.md', '/triathlon.md']

function isHeavyParseFile(fp: FilePath): boolean {
  const normalized = fp.replace(/\\/g, '/')
  return HEAVY_PARSE_SUFFIXES.some(
    suffix => normalized.endsWith(suffix) || normalized === suffix.slice(1),
  )
}

export async function parseMarkdown(
  ctx: BuildCtx,
  fps: FilePath[],
  dirtyPaths?: ReadonlySet<FilePath>,
): Promise<ProcessedContent[]> {
  const { argv } = ctx
  if (fps.length === 0) return []

  const perf = new PerfTimer()
  const log = new QuartzLogger(argv.verbose)
  const useCache = argv.watch
  await syncProcessedContentCacheSource(ctx)
  const fileDependencies = new Map<string, Promise<string>>()
  const cached = useCache
    ? await Promise.all(
        fps.map(fp => cachedProcessedContent(ctx, fp, fileDependencies, dirtyPaths)),
      )
    : []
  const cacheHits = cached.filter(entry => entry.content !== undefined).length
  const missSignatures = new Map<FilePath, string>()
  const cachedContent = new Map<FilePath, ProcessedContent>()
  if (useCache) {
    for (const entry of cached) {
      if (entry.content) {
        cachedContent.set(entry.fp, entry.content)
      } else {
        missSignatures.set(entry.fp, entry.signature)
      }
    }
  }

  const fpsToParse = useCache ? fps.filter(fp => !cachedContent.has(fp)) : fps
  const concurrency = parseWorkerConcurrency(fpsToParse.length, ctx.argv.concurrency)

  let res: ProcessedContent[] = []
  if (fpsToParse.length === 0) {
    res = fps.flatMap(fp => {
      const content = cachedContent.get(fp)
      return content ? [content] : []
    })
    log.start(`Parsing input files using cache`)
    log.end(`Parsed ${res.length} Markdown files in ${perf.timeSince()} (${cacheHits} cached)`)
    return res
  }

  log.start(`Parsing input files using ${concurrency} threads`)
  if (concurrency === 1) {
    try {
      const processors = cachedProcessors(ctx)
      const mdRes = await createFileParser(ctx, fpsToParse)(processors.md)
      res = await createMarkdownParser(ctx, mdRes)(processors.html)
    } catch (error) {
      log.end()
      throw error
    }
  } else {
    await transpileWorkerScript()
    const pool = workerpool.pool('./quartz/bootstrap-worker.mjs', {
      minWorkers: 'max',
      maxWorkers: concurrency,
      workerType: 'thread',
    })
    const serializableCtx: WorkerSerializableBuildCtx = {
      buildId: ctx.buildId,
      argv: ctx.argv,
      allSlugs: ctx.allSlugs,
      allFiles: ctx.allFiles,
      incremental: ctx.incremental,
    }

    const heavyFps = fpsToParse.filter(isHeavyParseFile)
    const regularFps = fpsToParse.filter(fp => !isHeavyParseFile(fp))
    const scheduledChunks: FilePath[][] = [
      ...heavyFps.map(fp => [fp]),
      ...chunks(regularFps, HTML_PARSE_CHUNK_SIZE),
    ]

    const parsePromises: WorkerPromise<ProcessedContent[]>[] = []
    let processedFiles = 0
    for (const chunk of scheduledChunks) {
      parsePromises.push(pool.exec('parseFiles', [serializableCtx, chunk]))
    }

    try {
      const results: ProcessedContent[][] = await Promise.all(
        parsePromises.map(async promise => {
          const result = await promise
          processedFiles += result.length
          log.updateText(
            `markdown->html ${styleText('gray', `${processedFiles}/${fpsToParse.length}`)}`,
          )
          return result
        }),
      )

      const parsedByFp = new Map<FilePath, ProcessedContent>()
      for (const content of results.flat()) {
        const fp = filePathForProcessedContent(content)
        if (fp) parsedByFp.set(fp, content)
      }
      res = fpsToParse.flatMap(fp => {
        const content = parsedByFp.get(fp)
        return content ? [content] : []
      })
    } finally {
      await pool.terminate()
    }
  }

  if (useCache) {
    for (const content of res) {
      const fp = filePathForProcessedContent(content)
      if (!fp) continue
      const signature = missSignatures.get(fp)
      if (!signature) continue
      const dependencySignature = await codeDependencySignature(ctx, content)
      processedContentCache.content.set(fp, {
        signature,
        dependencySignature,
        fileDependencySignature: await fileDependencySignature(ctx, content, fileDependencies),
        content,
      })
      cachedContent.set(fp, content)
    }
    res = fps.flatMap(fp => {
      const content = cachedContent.get(fp)
      return content ? [content] : []
    })
  }

  log.end(`Parsed ${res.length} Markdown files in ${perf.timeSince()} (${cacheHits} cached)`)
  return res
}
