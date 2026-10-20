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
import { ChangeEvent, QuartzEmitterPlugin } from '../../types/plugin'
import { mapConcurrent } from '../../util/async-pool'
import { BuildCtx, contentDataFor } from '../../util/ctx'
import {
  ensureExtractedNotebookAssets,
  extractInlineNotebookAssets,
} from '../../util/notebook/extract-assets'
import { embedLocalSvgImages, themeNotebookSvgs } from '../../util/notebook/inline-svg'
import { notebookToMarkdownChunks } from '../../util/notebook/markdown'
import { parseNotebookDoc } from '../../util/notebook/parse'
import { isNotebookParseError } from '../../util/notebook/types'
import { FilePath, FullSlug, joinSegments, pathToRoot, slugifyFilePath } from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'
import { StaticResources } from '../../util/resources'
import { ProcessedContent, QuartzPluginData } from '../vfile'
import { write } from './helpers'
import { affectedNotebookFiles, isNotebookPath } from './notebook-dependencies'
import {
  compactNotebookPluginData,
  notebookPageCacheEntries,
  readNotebookPageBlob,
  writeNotebookPageBlob,
  writeNotebookPageCacheIndex,
  type NotebookPage,
  type NotebookPageCacheEntry,
} from './notebook-page-cache'

export {
  notebookPageCacheEntries,
  resetNotebookCaches,
  type NotebookPageCacheEntry,
} from './notebook-page-cache'
export {
  affectedNotebookFiles,
  readNotebookDependencies,
  type NotebookDependencyRecord,
} from './notebook-dependencies'

type NotebookRuntimeOptions = false | { enabled?: boolean; indexUrl?: string }

type Options = { executeTimeoutSeconds?: number; runtime?: NotebookRuntimeOptions }

type ResolvedOptions = {
  executeTimeoutSeconds: number
  runtime: false | { enabled: true; indexUrl?: string }
}

type NotebookProcessors = {
  md: ReturnType<typeof createMdProcessor>
  html: ReturnType<typeof createHtmlProcessor>
}
type NotebookPageCacheOptions = {
  salt: string | undefined
  read: boolean
  write: boolean
  entries: Map<FilePath, NotebookPageCacheEntry>
}
type NotebookPageResult = { page: NotebookPage; cacheHit: boolean }

const defaultOptions: ResolvedOptions = { executeTimeoutSeconds: 120, runtime: false }

const notebookCacheSourcePaths: FilePath[] = [
  'quartz/util/notebook/cell-html.ts' as FilePath,
  'quartz/util/notebook/extract-assets.ts' as FilePath,
  'quartz/util/notebook/inline-svg.ts' as FilePath,
  'quartz/util/notebook/markdown.ts' as FilePath,
  'quartz/util/notebook/parse.ts' as FilePath,
  'quartz/util/notebook/render/icons.ts' as FilePath,
]

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
  beforeBody: [
    Component.NotebookRuntimeLoader(),
    // notebooks inside a university course carry the course trail, like its notes
    Component.UniversityCrumbBar(),
    Component.ArticleTitle(),
  ],
  pageBody: Component.Content(),
  sidebar: defaultContentPageLayout.sidebar,
}

function notebookFiles(ctx: BuildCtx): FilePath[] {
  return ctx.allFiles.filter(isNotebookPath).sort()
}

function isMarkdownContentPath(fp: FilePath): boolean {
  const ext = path.extname(fp)
  return (
    ext === '.md' || ext === '.base' || ext === '.canvas' || ext === '.fc' || ext === '.flashcards'
  )
}

export function mayAffectNotebookPages(changeEvents: ChangeEvent[]): boolean {
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
  if (ctx.argv.force) return undefined
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

function notebookCacheKeyMatches(cachedKey: string | undefined, expectedKey: string): boolean {
  return cachedKey === expectedKey || Boolean(cachedKey?.endsWith(`|${expectedKey}`))
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
    const cached = cacheOptions.entries.get(fp)
    const page = notebookCacheKeyMatches(cached?.key, cacheKey)
      ? await readNotebookPageBlob(ctx, cached!, fp, src, slug)
      : undefined
    if (page) return { page, cacheHit: true }
  }

  const raw = await fs.readFile(src, 'utf8')
  const notebook = parseNotebookDoc(raw, src)
  if (isNotebookParseError(notebook))
    throw new Error(`${src} is not a valid notebook: ${notebook.reason}`)
  const rawChunks = notebookToMarkdownChunks(notebook, fp, {
    runtime:
      opts.runtime === false ? false : { ...opts.runtime, sourcePath: fp, importableModules },
  }).map(chunk => applyTextTransforms(ctx, chunk))
  const svgChunks = await embedLocalSvgImages(rawChunks, ctx.argv.directory)
  const { chunks: markdownChunks, extracted: extractedAssets } = await extractInlineNotebookAssets(
    svgChunks,
    ctx,
  )
  const markdown = markdownChunks.join('\n\n').trim()

  const file = new VFile({ path: src, value: markdown })
  file.data.filePath = src
  file.data.relativePath = fp
  file.data.slug = slug

  const { md: mdProcessor, html: htmlProcessor } = processors()
  const ast = parseNotebookMarkdownChunks(mdProcessor, markdownChunks)
  const mdAst = await mdProcessor.run(ast, file)
  const htmlAst = await htmlProcessor.run(mdastToHastRoot(mdAst), file)
  themeNotebookSvgs(htmlAst)
  if (file.data.frontmatter) {
    file.data.frontmatter.description = ''
    file.data.frontmatter.socialDescription = ''
  }
  file.data.description = ''

  const page = { fp, slug, content: [htmlAst, file] } satisfies NotebookPage
  if (cacheKey && cacheOptions.write) {
    await writeNotebookPageBlob(cacheOptions.entries, cacheKey, page, extractedAssets)
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

export async function parseNotebookPages(
  ctx: BuildCtx,
  fps: FilePath[],
  maxWorkers: number,
  opts: ResolvedOptions = defaultOptions,
  contextFps: FilePath[] = fps,
): Promise<{ pages: NotebookPage[]; cacheHits: number }> {
  const pages: NotebookPage[] = []
  let cacheHits = 0
  let cacheWrites = 0
  const workerCount = Math.max(1, Math.min(maxWorkers, fps.length))
  const queues = Array.from({ length: workerCount }, () => [] as FilePath[])
  for (let i = 0; i < fps.length; i += 1) {
    queues[i % workerCount].push(fps[i])
  }
  const salt = await notebookPageCacheSalt(ctx)
  const cacheOptions = {
    salt,
    read: Boolean(salt && !ctx.argv.force),
    write: Boolean(salt),
    entries: salt ? await notebookPageCacheEntries() : new Map<FilePath, NotebookPageCacheEntry>(),
  }
  const results = await Promise.all(
    queues.map(queue => parseNotebookQueue(ctx, queue, opts, contextFps, cacheOptions)),
  )

  for (const result of results.flat()) {
    if ('page' in result) {
      pages.push(result.page)
      if (result.cacheHit) {
        cacheHits += 1
      } else {
        cacheWrites += 1
      }
    } else {
      console.error(
        styleText('red', `\n[emit:NotebookViewer] Error processing ${result.fp}:`),
        result.error,
      )
    }
  }

  if (cacheOptions.write && cacheWrites > 0) {
    await writeNotebookPageCacheIndex(cacheOptions.entries)
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

type PreparedNotebookEmitItem = {
  fp: FilePath
  src: FilePath
  slug: FullSlug
  data: QuartzPluginData
  cached?: NotebookPageCacheEntry
  page?: NotebookPage
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
  const salt = await notebookPageCacheSalt(localCtx)
  const cacheOptions: NotebookPageCacheOptions = {
    salt,
    read: Boolean(salt && !localCtx.argv.force),
    write: Boolean(salt),
    entries: salt ? await notebookPageCacheEntries() : new Map<FilePath, NotebookPageCacheEntry>(),
  }

  let cacheHits = 0
  let cacheWrites = 0
  let cacheMetadataDirty = false
  let processorCache: NotebookProcessors | undefined
  const processors = () => {
    processorCache ??= { md: createMdProcessor(localCtx), html: createHtmlProcessor(localCtx) }
    return processorCache
  }

  const preparedResults = await mapConcurrent(
    fps,
    maxWorkers,
    async (fp): Promise<PreparedNotebookEmitItem | undefined> => {
      try {
        const src = joinSegments(localCtx.argv.directory, fp) as FilePath
        const slug = slugifyFilePath(fp, true)
        const importableModules = importableNotebookModules(fp, contextFps)
        const sourceSignature = await notebookSourceSignature(src)
        const cacheKey = cacheOptions.salt
          ? notebookPageCacheKey(sourceSignature, opts, cacheOptions.salt, importableModules)
          : undefined

        if (cacheKey && cacheOptions.read) {
          const cached = cacheOptions.entries.get(fp)
          if (cached && notebookCacheKeyMatches(cached.key, cacheKey)) {
            if (
              cached.data &&
              (await ensureExtractedNotebookAssets(localCtx, cached.extractedAssets))
            ) {
              cacheHits += 1
              return { fp, src, slug, data: cached.data, cached }
            }
            const page = await readNotebookPageBlob(localCtx, cached, fp, src, slug)
            if (page && cached.data) {
              cacheHits += 1
              cacheMetadataDirty = true
              ;(page.content as unknown[])[0] = undefined
              return { fp, src, slug, data: cached.data, cached }
            }
          }
        }

        const result = await notebookProcessedContent(
          localCtx,
          fp,
          opts,
          contextFps,
          { ...cacheOptions, read: false },
          processors,
        )
        cacheWrites += 1
        const cached = cacheOptions.entries.get(fp)
        const data = cached?.data ?? compactNotebookPluginData(result.page.content[1].data)
        if (cached) {
          ;(result.page.content as unknown[])[0] = undefined
          return { fp, src, slug, data, cached }
        }
        return { fp, src, slug, data, page: result.page }
      } catch (error) {
        console.error(styleText('red', `\n[emit:NotebookViewer] Error processing ${fp}:`), error)
        return undefined
      }
    },
  )

  if (cacheOptions.write && (cacheWrites > 0 || cacheMetadataDirty)) {
    await writeNotebookPageCacheIndex(cacheOptions.entries)
  }

  const prepared = preparedResults.filter(
    (item): item is PreparedNotebookEmitItem => item !== undefined,
  )
  logBuildSpan(
    ctx.argv,
    'notebook:parse',
    `${prepared.length} pages, ${cacheHits} cache hits`,
    perf.elapsedMs(),
  )
  const allFiles = [...contentDataFor(content), ...prepared.map(item => item.data)]
  perf.addEvent('render')
  const files = await mapConcurrent(prepared, maxWorkers, async item => {
    try {
      let page = item.page
      if (!page && item.cached) {
        page = await readNotebookPageBlob(localCtx, item.cached, item.fp, item.src, item.slug)
      }
      if (!page) {
        const fallback = await notebookProcessedContent(
          localCtx,
          item.fp,
          opts,
          contextFps,
          { ...cacheOptions, read: false },
          processors,
        )
        page = fallback.page
      }
      const out = await emitNotebookPage(localCtx, page, allFiles, resources)
      ;(page.content as unknown[])[0] = undefined
      delete page.content[1].data.htmlAst
      item.page = undefined
      return out
    } catch (error) {
      console.error(styleText('red', `\n[emit:NotebookViewer] Error processing ${item.fp}:`), error)
      return undefined
    }
  })
  logBuildSpan(ctx.argv, 'notebook:render', `${prepared.length} pages`, perf.elapsedMs('render'))

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
  const PressKit = Component.PressKitConstructor()

  return {
    name: 'NotebookViewer',
    getQuartzComponents() {
      return [
        Head,
        Header,
        Headings,
        PressKit,
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
