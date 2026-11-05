import type { Root as MdRoot } from 'mdast'
import esbuild from 'esbuild'
import path from 'path'
import { read } from 'to-vfile'
import { styleText } from 'util'
import { VFile } from 'vfile'
import workerpool, { Promise as WorkerPromise } from 'workerpool'
import type { MarkdownContent, ProcessedContent } from '../plugins/vfile'
import type { BuildCtx, WorkerSerializableBuildCtx } from '../util/ctx'
import type { QuartzHtmlProcessor, QuartzMdProcessor } from './markdown-processors'
import { QuartzLogger } from '../util/log'
import { FilePath, QUARTZ, slugifyFilePath } from '../util/path'
import { logBuildSpan, PerfTimer } from '../util/perf'
import { updateReferenceCatalog } from '../util/references/context'
import { indexReferenceDocument } from '../util/references/document'
import {
  referenceResolutionSignature,
  resolveDocumentReferences,
} from '../util/references/markdown'
import { prepareReferenceContent } from '../util/references/preparation'
import { trace } from '../util/trace'
import { mdastToHastRoot } from './markdown-processors'
import { transformMarkdownSource } from './markdown-source'
import {
  cachedProcessedContent,
  cachedProcessors,
  recordHtmlReuseSignature,
  filePathForProcessedContent,
  storeProcessedContent,
  syncProcessedContentCacheSource,
} from './processed-content-cache'
export {
  cachedProcessors,
  PROCESSED_CONTENT_CACHE_VERSION,
  canReuseProcessedHtml,
  resetProcessedContentCache,
  restoreProcessedContentCache,
  restoredProcessedContent,
  reuseProcessedTitleOnlyChange,
  snapshotProcessedContentCache,
  titleOnlyFrontmatterChange,
} from './processed-content-cache'
export type { ProcessedContentSnapshot } from './processed-content-cache'
export { createHtmlProcessor, createMdProcessor, mdastToHastRoot } from './markdown-processors'
export type { QuartzHtmlProcessor, QuartzMdProcessor } from './markdown-processors'

function* chunks<T>(arr: T[], n: number) {
  for (let i = 0; i < arr.length; i += n) {
    yield arr.slice(i, i + n)
  }
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
        indexReferenceDocument(newAst, file)
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
        if (!(file instanceof VFile)) Object.setPrototypeOf(file, VFile.prototype)
        const perf = new PerfTimer()
        const hastPerf = new PerfTimer()
        resolveDocumentReferences(ctx, ast, file)
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

let lastProcessedContentCacheMisses = 0

export function processedContentCacheMisses(): number {
  return lastProcessedContentCacheMisses
}

async function processWithCatalog(
  ctx: BuildCtx,
  initialFiles: FilePath[],
  cachedContent: Map<FilePath, ProcessedContent>,
  signatures: Map<FilePath, string>,
  missSignatures: Map<FilePath, string>,
  parse: (files: FilePath[]) => Promise<MarkdownContent[]>,
  html: (markdown: MarkdownContent[]) => Promise<ProcessedContent[]>,
): Promise<ProcessedContent[]> {
  const markdown = initialFiles.length > 0 ? await parse(initialFiles) : []
  const refreshCatalog = () =>
    updateReferenceCatalog(ctx, [
      ...[...cachedContent.values()].map(([, file]) => file.data),
      ...markdown.map(([, file]) => file.data),
    ])
  refreshCatalog()
  await prepareReferenceContent(ctx, markdown, cachedContent, true)

  // Cached HTML retains authored requests. Reparse only consumers whose current lookup or
  // selected anchor differs after all changed documents have supplied their metadata.
  while (cachedContent.size > 0) {
    const stale = [...cachedContent].filter(
      ([, [, file]]) =>
        referenceResolutionSignature(ctx, file) !== file.data.referenceResolutionSignature,
    )
    if (stale.length === 0) break
    for (const [fp] of stale) {
      cachedContent.delete(fp)
      const signature = signatures.get(fp)
      if (signature !== undefined) missSignatures.set(fp, signature)
    }
    markdown.push(...(await parse(stale.map(([fp]) => fp))))
    refreshCatalog()
    await prepareReferenceContent(ctx, markdown, cachedContent, false)
  }
  lastProcessedContentCacheMisses = markdown.length
  return markdown.length > 0 ? html(markdown) : []
}

export async function parseMarkdown(
  ctx: BuildCtx,
  fps: FilePath[],
  dirtyPaths?: ReadonlySet<FilePath>,
): Promise<ProcessedContent[]> {
  const { argv } = ctx
  if (fps.length === 0) {
    updateReferenceCatalog(ctx, [])
    lastProcessedContentCacheMisses = 0
    return []
  }

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
  const signatures = new Map(cached.map(entry => [entry.fp, entry.signature]))
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
  lastProcessedContentCacheMisses = fpsToParse.length
  const concurrency = parseWorkerConcurrency(fpsToParse.length, ctx.argv.concurrency)

  let res: ProcessedContent[] = []
  log.start(`Parsing input files using ${concurrency} threads`)
  if (concurrency === 1) {
    try {
      const processors = cachedProcessors(ctx)
      res = await processWithCatalog(
        ctx,
        fpsToParse,
        cachedContent,
        signatures,
        missSignatures,
        files => createFileParser(ctx, files)(processors.md),
        markdown => createMarkdownParser(ctx, markdown)(processors.html),
      )
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
      referenceCatalogInputs: ctx.referenceCatalogInputs,
      incremental: ctx.incremental,
    }

    const heavyFps = fpsToParse.filter(isHeavyParseFile)
    const regularFps = fpsToParse.filter(fp => !isHeavyParseFile(fp))
    const scheduledChunks: FilePath[][] = [
      ...heavyFps.map(fp => [fp]),
      ...chunks(regularFps, HTML_PARSE_CHUNK_SIZE),
    ]

    try {
      const parsed = await processWithCatalog(
        ctx,
        fpsToParse,
        cachedContent,
        signatures,
        missSignatures,
        async files => {
          serializableCtx.referenceCatalogInputs = ctx.referenceCatalogInputs
          const batches =
            files === fpsToParse ? scheduledChunks : [...chunks(files, TEXT_PARSE_CHUNK_SIZE)]
          const promises: WorkerPromise<MarkdownContent[]>[] = batches.map(batch =>
            pool.exec('parseMarkdown', [serializableCtx, batch]),
          )
          return (await Promise.all(promises)).flat()
        },
        async markdown => {
          serializableCtx.referenceCatalogInputs = ctx.referenceCatalogInputs
          let processedFiles = 0
          const promises: WorkerPromise<ProcessedContent[]>[] = [
            ...chunks(markdown, HTML_PARSE_CHUNK_SIZE),
          ].map(batch => pool.exec('processHtml', [serializableCtx, batch]))
          const results = await Promise.all(
            promises.map(async promise => {
              const result = await promise
              processedFiles += result.length
              log.updateText(
                `markdown->html ${styleText('gray', `${processedFiles}/${markdown.length}`)}`,
              )
              return result
            }),
          )
          return results.flat()
        },
      )

      const parsedByFp = new Map<FilePath, ProcessedContent>()
      for (const content of parsed) {
        if (!(content[1] instanceof VFile)) Object.setPrototypeOf(content[1], VFile.prototype)
        const fp = filePathForProcessedContent(content)
        if (fp) parsedByFp.set(fp, content)
      }
      res = fps.flatMap(fp => {
        const content = parsedByFp.get(fp)
        return content ? [content] : []
      })
    } finally {
      await pool.terminate()
    }
  }

  const cacheHits = cachedContent.size
  if (useCache) {
    for (const content of res) {
      const fp = filePathForProcessedContent(content)
      if (!fp) continue
      const signature = missSignatures.get(fp)
      if (!signature) continue
      await storeProcessedContent(ctx, fp, signature, content, fileDependencies)
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
