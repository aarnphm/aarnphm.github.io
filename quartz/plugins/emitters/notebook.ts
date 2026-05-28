import { Root as MdRoot } from 'mdast'
import fs from 'node:fs/promises'
import { availableParallelism } from 'node:os'
import path from 'node:path'
import { styleText } from 'node:util'
import { VFile } from 'vfile'
import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import * as Component from '../../components'
import HeaderConstructor from '../../components/Header'
import { pageResources, renderPage } from '../../components/renderPage'
import { createHtmlProcessor, createMdProcessor, mdastToHastRoot } from '../../processors/parse'
import '../../runtime/notebook/registry'
import { backendFor } from '../../runtime/notebook/backend'
import { ChangeEvent, QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import { extractInlineNotebookAssets } from '../../util/notebook/extract-assets'
import { notebookToMarkdownChunks } from '../../util/notebook/markdown'
import { parseNotebookDoc } from '../../util/notebook/parse'
import { isNotebookParseError } from '../../util/notebook/types'
import { FilePath, FullSlug, joinSegments, pathToRoot, slugifyFilePath } from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'
import { StaticResources } from '../../util/resources'
import { ProcessedContent, QuartzPluginData } from '../vfile'
import { write } from './helpers'

type NotebookRuntimeOptions = false | { enabled?: boolean; indexUrl?: string }

type Options = { executeTimeoutSeconds?: number; runtime?: NotebookRuntimeOptions }

type ResolvedOptions = {
  executeTimeoutSeconds: number
  runtime: false | { enabled: true; indexUrl?: string }
}

type NotebookDependencies = { imports: Set<FilePath>; assets: Set<FilePath> }
type NotebookProcessors = {
  md: ReturnType<typeof createMdProcessor>
  html: ReturnType<typeof createHtmlProcessor>
}
type NotebookPageCacheEntry = { key: string; page: NotebookPage }
type NotebookPageCacheOptions = { salt: string | undefined; read: boolean; write: boolean }
type NotebookPageResult = { page: NotebookPage; cacheHit: boolean }

declare global {
  var __quartzNotebookPageCache: Map<FilePath, NotebookPageCacheEntry> | undefined
}

const defaultOptions: ResolvedOptions = { executeTimeoutSeconds: 120, runtime: false }

const notebookPageCache = (globalThis.__quartzNotebookPageCache ??= new Map<
  FilePath,
  NotebookPageCacheEntry
>())
const notebookCacheSourcePaths: FilePath[] = [
  'quartz/plugins/emitters/notebook.ts' as FilePath,
  'quartz/processors/parse.ts' as FilePath,
  'quartz/util/notebook/cell-html.ts' as FilePath,
  'quartz/util/notebook/extract-assets.ts' as FilePath,
  'quartz/util/notebook/markdown.ts' as FilePath,
  'quartz/util/notebook/parse.ts' as FilePath,
  'quartz/util/notebook/render/icons.ts' as FilePath,
]

const htmlResourceSrcPattern =
  /<(?:img|video|audio|iframe)\b[^>]*\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/gi
const markdownImageTargetPattern = /!\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)[^)]*\)/g

function resolveOptions(userOpts?: Options): ResolvedOptions {
  const runtime: ResolvedOptions['runtime'] =
    userOpts?.runtime === false ||
    userOpts?.runtime === undefined ||
    userOpts.runtime.enabled === false
      ? false
      : { enabled: true, indexUrl: userOpts.runtime.indexUrl }

  return { ...defaultOptions, ...userOpts, runtime }
}

const notebookPageLayout: FullPageLayout = {
  ...sharedPageComponents,
  beforeBody: [Component.NotebookRuntimeLoader(), Component.ArticleTitle()],
  pageBody: Component.Content(),
  sidebar: defaultContentPageLayout.sidebar,
}

type NotebookPage = { fp: FilePath; slug: FullSlug; content: ProcessedContent }

function isNotebookPath(fp: FilePath): boolean {
  return path.extname(fp) === '.ipynb'
}

function notebookFiles(ctx: BuildCtx): FilePath[] {
  return ctx.allFiles.filter(isNotebookPath).sort()
}

function isMarkdownContentPath(fp: FilePath): boolean {
  const ext = path.extname(fp)
  return ext === '.md' || ext === '.base' || ext === '.canvas'
}

function mayAffectNotebookPages(changeEvents: ChangeEvent[]): boolean {
  return changeEvents.some(
    changeEvent => isNotebookPath(changeEvent.path) || !isMarkdownContentPath(changeEvent.path),
  )
}

function applyTextTransforms(ctx: BuildCtx, markdown: string): string {
  let transformed = markdown
  for (const plugin of ctx.cfg.plugins.transformers) {
    if (plugin.textTransform) {
      transformed = plugin.textTransform(ctx, transformed)
    }
  }
  return transformed
}

function parseNotebookMarkdownChunks(
  mdProcessor: ReturnType<typeof createMdProcessor>,
  chunks: string[],
): MdRoot {
  const children: MdRoot['children'] = []
  for (const chunk of chunks) {
    children.push(...mdProcessor.parse(chunk).children)
  }
  return { type: 'root', children }
}

function localNotebookResourcePath(raw: string, fp: FilePath): FilePath | undefined {
  const trimmed = raw.trim()
  const value = trimmed.startsWith('<') && trimmed.endsWith('>') ? trimmed.slice(1, -1) : trimmed
  if (
    !value ||
    value.startsWith('#') ||
    value.startsWith('/') ||
    /^[A-Za-z][A-Za-z0-9+.-]*:/.test(value)
  ) {
    return undefined
  }

  const match = value.match(/^([^?#]*)([?#].*)?$/)
  let pathname = match?.[1] ?? value
  if (!pathname) return undefined
  try {
    pathname = decodeURI(pathname)
  } catch {}

  const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(fp), pathname))
  if (resolved === '.' || resolved.startsWith('../')) return undefined
  return resolved as FilePath
}

function notebookDependencies(raw: string, fp: FilePath): NotebookDependencies {
  const doc = parseNotebookDoc(raw, fp)
  const imports = new Set<FilePath>()
  const assets = new Set<FilePath>()
  if (isNotebookParseError(doc)) return { imports, assets }
  const dir = path.posix.dirname(fp)
  const backend = backendFor(doc.language)
  const moduleResolver = backend?.moduleResolver

  for (const cell of doc.cells) {
    const source = cell.source
    if (cell.cellType === 'code' && moduleResolver) {
      for (const name of moduleResolver.importNames(source)) {
        imports.add(path.posix.join(dir, `${name}.ipynb`) as FilePath)
      }
    }
    if (cell.cellType !== 'markdown') continue

    for (const match of source.matchAll(htmlResourceSrcPattern)) {
      const candidate = match[1] ?? match[2] ?? match[3]
      if (candidate === undefined) continue
      const dependency = localNotebookResourcePath(candidate, fp)
      if (dependency !== undefined) assets.add(dependency)
    }

    for (const match of source.matchAll(markdownImageTargetPattern)) {
      const dependency = localNotebookResourcePath(match[1], fp)
      if (dependency !== undefined) assets.add(dependency)
    }
  }

  return { imports, assets }
}

async function readNotebookDependencies(
  ctx: BuildCtx,
  fp: FilePath,
): Promise<NotebookDependencies> {
  const src = joinSegments(ctx.argv.directory, fp) as FilePath
  try {
    return notebookDependencies(await fs.readFile(src, 'utf8'), fp)
  } catch {
    return { imports: new Set(), assets: new Set() }
  }
}

async function affectedNotebookFiles(
  ctx: BuildCtx,
  fps: FilePath[],
  changeEvents: ChangeEvent[],
): Promise<FilePath[]> {
  const available = new Set(fps)
  const changedPaths = new Set(changeEvents.map(event => event.path))
  const changedNotebooks = new Set(
    changeEvents.filter(event => isNotebookPath(event.path)).map(event => event.path),
  )
  const affected = new Set<FilePath>()

  for (const event of changeEvents) {
    if (isNotebookPath(event.path) && event.type !== 'delete' && available.has(event.path)) {
      affected.add(event.path)
    }
  }

  const dependencyPairs = await Promise.all(
    fps.map(async fp => [fp, await readNotebookDependencies(ctx, fp)] as const),
  )
  const reverseImports = new Map<FilePath, Set<FilePath>>()
  for (const [fp, deps] of dependencyPairs) {
    for (const target of deps.imports) {
      if (!available.has(target)) continue
      const dependents = reverseImports.get(target) ?? new Set<FilePath>()
      dependents.add(fp)
      reverseImports.set(target, dependents)
    }
    for (const target of deps.assets) {
      if (changedPaths.has(target)) affected.add(fp)
    }
  }

  const queue = [...changedNotebooks]
  const seen = new Set<FilePath>()
  while (queue.length > 0) {
    const current = queue.shift()
    if (current === undefined) break
    if (seen.has(current)) continue
    seen.add(current)
    for (const dependent of reverseImports.get(current) ?? []) {
      if (!affected.has(dependent)) {
        affected.add(dependent)
        queue.push(dependent)
      }
    }
  }

  return fps.filter(fp => affected.has(fp))
}

function notebookContext(ctx: BuildCtx, fps: FilePath[]): BuildCtx {
  const notebookSlugs = fps.map(fp => slugifyFilePath(fp, true))
  return { ...ctx, allSlugs: [...ctx.allSlugs, ...notebookSlugs] }
}

function importableNotebookModules(fp: FilePath, fps: FilePath[]): string[] {
  const dir = path.posix.dirname(fp)
  const modules = new Set<string>()
  for (const candidate of fps) {
    if (candidate === fp || path.posix.dirname(candidate) !== dir) continue
    if (path.extname(candidate) !== '.ipynb') continue
    const name = path.basename(candidate, '.ipynb')
    if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) modules.add(name)
  }
  return [...modules].sort()
}

async function notebookSourceSignature(fp: FilePath): Promise<string> {
  try {
    const stat = await fs.stat(fp)
    return `${fp}:${stat.mtimeMs}:${stat.size}`
  } catch {
    return `${fp}:missing`
  }
}

async function notebookPageCacheSalt(ctx: BuildCtx): Promise<string | undefined> {
  if (!ctx.argv.watch) return undefined
  return (await Promise.all(notebookCacheSourcePaths.map(notebookSourceSignature))).join('|')
}

function notebookRuntimeCacheKey(opts: ResolvedOptions): string {
  if (opts.runtime === false) return 'runtime:false'
  return `runtime:${opts.runtime.indexUrl ?? ''}`
}

function notebookPageCacheKey(
  sourceSignature: string,
  opts: ResolvedOptions,
  sourceSalt: string,
  importableModules: readonly string[],
): string {
  return [
    sourceSalt,
    sourceSignature,
    notebookRuntimeCacheKey(opts),
    importableModules.join('\0'),
  ].join('\u0001')
}

async function notebookProcessedContent(
  ctx: BuildCtx,
  fp: FilePath,
  opts: ResolvedOptions,
  contextFps: FilePath[],
  cacheOptions: NotebookPageCacheOptions,
  processors: () => NotebookProcessors,
): Promise<NotebookPageResult> {
  const src = joinSegments(ctx.argv.directory, fp) as FilePath
  const slug = slugifyFilePath(fp, true)
  const importableModules = importableNotebookModules(fp, contextFps)
  const sourceSignature = await notebookSourceSignature(src)
  const cacheKey = cacheOptions.salt
    ? notebookPageCacheKey(sourceSignature, opts, cacheOptions.salt, importableModules)
    : undefined
  if (cacheKey && cacheOptions.read) {
    const cached = notebookPageCache.get(fp)
    if (cached?.key === cacheKey) return { page: cached.page, cacheHit: true }
  }

  const raw = await fs.readFile(src, 'utf8')
  const notebook = parseNotebookDoc(raw, src)
  if (isNotebookParseError(notebook))
    throw new Error(`${src} is not a valid notebook: ${notebook.reason}`)
  const rawChunks = notebookToMarkdownChunks(notebook, fp, {
    runtime:
      opts.runtime === false ? false : { ...opts.runtime, sourcePath: fp, importableModules },
  }).map(chunk => applyTextTransforms(ctx, chunk))
  const { chunks: markdownChunks } = await extractInlineNotebookAssets(rawChunks, ctx)
  const markdown = markdownChunks.join('\n\n').trim()

  const file = new VFile({ path: src, value: markdown })
  file.data.filePath = src
  file.data.relativePath = fp
  file.data.slug = slug

  const { md: mdProcessor, html: htmlProcessor } = processors()
  const ast = parseNotebookMarkdownChunks(mdProcessor, markdownChunks)
  const mdAst = await mdProcessor.run(ast, file)
  const htmlAst = await htmlProcessor.run(mdastToHastRoot(mdAst), file)
  if (file.data.frontmatter) {
    file.data.frontmatter.description = ''
    file.data.frontmatter.socialDescription = ''
  }
  file.data.description = ''

  const page = { fp, slug, content: [htmlAst, file] } satisfies NotebookPage
  if (cacheKey && cacheOptions.write) {
    notebookPageCache.set(fp, { key: cacheKey, page })
  }
  return { page, cacheHit: false }
}

async function parseNotebookQueue(
  ctx: BuildCtx,
  fps: FilePath[],
  opts: ResolvedOptions,
  contextFps: FilePath[],
  cacheOptions: NotebookPageCacheOptions,
): Promise<Array<NotebookPageResult | { fp: FilePath; error: unknown }>> {
  let processorCache: NotebookProcessors | undefined
  const processors = () => {
    processorCache ??= { md: createMdProcessor(ctx), html: createHtmlProcessor(ctx) }
    return processorCache
  }
  const pages: Array<NotebookPageResult | { fp: FilePath; error: unknown }> = []
  for (const fp of fps) {
    try {
      pages.push(
        await notebookProcessedContent(ctx, fp, opts, contextFps, cacheOptions, processors),
      )
    } catch (error) {
      pages.push({ fp, error })
    }
  }
  return pages
}

async function parseNotebookPages(
  ctx: BuildCtx,
  fps: FilePath[],
  maxWorkers: number,
  opts: ResolvedOptions,
  contextFps: FilePath[],
): Promise<{ pages: NotebookPage[]; cacheHits: number }> {
  const pages: NotebookPage[] = []
  let cacheHits = 0
  const workerCount = Math.max(1, Math.min(maxWorkers, fps.length))
  const queues = Array.from({ length: workerCount }, () => [] as FilePath[])
  for (let i = 0; i < fps.length; i += 1) {
    queues[i % workerCount].push(fps[i])
  }
  const salt = await notebookPageCacheSalt(ctx)
  const cacheOptions = { salt, read: Boolean(salt && ctx.argv.watch), write: Boolean(salt) }
  const results = await Promise.all(
    queues.map(queue => parseNotebookQueue(ctx, queue, opts, contextFps, cacheOptions)),
  )

  for (const result of results.flat()) {
    if ('page' in result) {
      pages.push(result.page)
      if (result.cacheHit) cacheHits += 1
    } else {
      console.error(
        styleText('red', `\n[emit:NotebookViewer] Error processing ${result.fp}:`),
        result.error,
      )
    }
  }

  return { pages, cacheHits }
}

async function emitNotebookPage(
  ctx: BuildCtx,
  page: NotebookPage,
  allFiles: QuartzPluginData[],
  resources: StaticResources,
): Promise<FilePath> {
  const [tree, file] = page.content
  const externalResources = pageResources(pathToRoot(page.slug), resources, ctx)
  const componentData = {
    ctx,
    fileData: file.data,
    externalResources,
    cfg: ctx.cfg.configuration,
    children: [],
    tree,
    allFiles,
  }

  const html = renderPage(
    ctx,
    page.slug,
    componentData,
    notebookPageLayout,
    externalResources,
    false,
  )
  return write({ ctx, content: html, slug: page.slug, ext: '.html' })
}

async function* emitNotebookPages(
  ctx: BuildCtx,
  content: ProcessedContent[],
  resources: StaticResources,
  fps: FilePath[],
  opts: ResolvedOptions,
  contextFps = fps,
): AsyncGenerator<FilePath> {
  if (fps.length === 0) {
    return
  }

  const { argv } = ctx
  const perf = new PerfTimer()
  const resolveWorkerLimit = () => {
    if (argv.concurrency && argv.concurrency > 0) {
      return argv.concurrency
    }

    try {
      return availableParallelism()
    } catch {
      return 1
    }
  }

  const maxWorkers = Math.max(1, Math.min(resolveWorkerLimit(), fps.length))
  const localCtx = notebookContext(ctx, contextFps)
  const { pages, cacheHits } = await parseNotebookPages(localCtx, fps, maxWorkers, opts, contextFps)
  logBuildSpan(
    ctx.argv,
    'notebook:parse',
    `${pages.length} pages, ${cacheHits} cache hits`,
    perf.elapsedMs(),
  )
  const allFiles = [...contentDataFor(content), ...pages.map(page => page.content[1].data)]
  perf.addEvent('render')
  const files = await mapConcurrent(pages, defaultIoConcurrency, async page => {
    try {
      return await emitNotebookPage(localCtx, page, allFiles, resources)
    } catch (error) {
      console.error(styleText('red', `\n[emit:NotebookViewer] Error processing ${page.fp}:`), error)
      return undefined
    }
  })
  logBuildSpan(ctx.argv, 'notebook:render', `${pages.length} pages`, perf.elapsedMs('render'))

  for (const file of files) {
    if (file) yield file
  }
}

async function deleteNotebookPage(ctx: BuildCtx, fp: FilePath): Promise<void> {
  const slug = slugifyFilePath(fp, true)
  const dest = joinSegments(ctx.argv.output, `${slug}.html`) as FilePath

  try {
    await fs.unlink(dest)
  } catch (error) {
    if (!isRecordNotFound(error)) {
      throw error
    }
  }
}

function isRecordNotFound(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

function notebookChangeEvents(changeEvents: ChangeEvent[]): ChangeEvent[] {
  return changeEvents.filter(changeEvent => isNotebookPath(changeEvent.path))
}

export const NotebookViewer: QuartzEmitterPlugin<Options> = userOpts => {
  const opts = resolveOptions(userOpts)
  const {
    head: Head,
    header,
    beforeBody,
    pageBody,
    afterBody,
    sidebar,
    footer: Footer,
  } = notebookPageLayout
  const Header = HeaderConstructor()
  const Headings = Component.HeadingsConstructor()

  return {
    name: 'NotebookViewer',
    getQuartzComponents() {
      return [
        Head,
        Header,
        Headings,
        ...header,
        ...beforeBody,
        pageBody,
        ...afterBody,
        ...sidebar,
        Footer,
      ]
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      if (!mayAffectNotebookPages(changeEvents)) {
        return
      }

      for (const changeEvent of notebookChangeEvents(changeEvents)) {
        if (changeEvent.type === 'delete') {
          await deleteNotebookPage(ctx, changeEvent.path)
        }
      }

      const fps = notebookFiles(ctx)
      const affected = await affectedNotebookFiles(ctx, fps, changeEvents)
      yield* emitNotebookPages(ctx, content, resources, affected, opts, fps)
    },
    async *emit(ctx, content, resources) {
      const fps = notebookFiles(ctx)
      yield* emitNotebookPages(ctx, content, resources, fps, opts)
    },
  }
}
