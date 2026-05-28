import type { Element, ElementContent, Root as HtmlRoot } from 'hast'
import type { Parents as MdastParents, PhrasingContent, Root as MdRoot } from 'mdast'
import type { Handler, Options as ToHastOptions, State } from 'mdast-util-to-hast'
import esbuild from 'esbuild'
import matter from 'gray-matter'
import yaml from 'js-yaml'
import { toHast } from 'mdast-util-to-hast'
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

const processorCache = new WeakMap<BuildCtx, ProcessorCache>()

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

function recordHtmlReuseSignature(file: VFile): void {
  file.data.htmlReuseBody = markdownBody(file.value)
  file.data.htmlReuseFrontmatter = htmlReuseFrontmatterSignature(file.data.frontmatter)
}

export function canReuseProcessedHtml(file: VFile, previous: ProcessedContent): boolean {
  const previousFile = previous[1]
  const body = file.data.htmlReuseBody ?? markdownBody(file.value)
  const previousBody = previousFile.data.htmlReuseBody ?? markdownBody(previousFile.value)
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

function cachedProcessors(ctx: BuildCtx): ProcessorCache {
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
          build.onLoad({ filter: /\.scss$/ }, _ => ({ contents: '', loader: 'text' }))
          build.onLoad({ filter: /\.inline\.(ts|js)$/ }, _ => ({ contents: '', loader: 'text' }))
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
    trace(`\nFailed to parse notebook \`${file.path}\``, new Error(parsed.reason))
    return undefined
  }
  return notebookDocToFlatMdast(parsed)
}

export function createFileParser(
  ctx: BuildCtx,
  fps: FilePath[],
  htmlReuseCache?: ReadonlyMap<FilePath, ProcessedContent>,
) {
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

        const previous = htmlReuseCache?.get(file.data.relativePath)
        const reused = previous
          ? reuseMarkdownForTitleOnlyChange(ctx, file, previous, rawSource)
          : undefined
        if (reused) {
          res.push(reused)
          logBuildSpan(argv, 'markdown:reuse', `${fp} -> ${file.data.slug}`, perf.elapsedMs())
          continue
        }

        file.value = transformMarkdownSource(ctx, rawSource)

        const notebookAst = await parseNotebookFileToMdast({ path: file.path, value: file.value })
        const ast = notebookAst ?? processor.parse(file)
        const newAst = await processor.run(ast, file)
        recordHtmlReuseSignature(file)
        res.push([newAst, file])

        logBuildSpan(argv, 'markdown', `${fp} -> ${file.data.slug}`, perf.elapsedMs())
      } catch (err) {
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

export function createMarkdownParser(
  ctx: BuildCtx,
  mdContent: MarkdownContent[],
  htmlReuseCache?: ReadonlyMap<FilePath, ProcessedContent>,
) {
  return async (processor: QuartzHtmlProcessor) => {
    const res: ProcessedContent[] = []
    for (const [ast, file] of mdContent) {
      try {
        const perf = new PerfTimer()
        const relativePath = file.data.relativePath
        const previous = relativePath ? htmlReuseCache?.get(relativePath) : undefined
        if (previous && canReuseProcessedHtml(file, previous)) {
          res.push(reuseProcessedHtml(file, previous))
          logBuildSpan(ctx.argv, 'html', `${file.data.slug}`, perf.elapsedMs())
          continue
        }

        const hastPerf = new PerfTimer()
        const hast = mdastToHastRoot(ast)
        logBuildSpan(ctx.argv, 'ast:hast', `${file.data.slug}`, hastPerf.elapsedMs())

        const pluginPerf = new PerfTimer()
        const newAst = await processor.run(hast, file)
        logBuildSpan(ctx.argv, 'hast:plugins', `${file.data.slug}`, pluginPerf.elapsedMs())
        res.push([newAst, file])

        logBuildSpan(ctx.argv, 'html', `${file.data.slug}`, perf.elapsedMs())
      } catch (err) {
        trace(`\nFailed to process html \`${file.data.filePath}\``, err as Error)
      }
    }

    return res
  }
}

const clamp = (num: number, min: number, max: number) =>
  Math.min(Math.max(Math.round(num), min), max)

export async function parseMarkdown(
  ctx: BuildCtx,
  fps: FilePath[],
  htmlReuseCache?: ReadonlyMap<FilePath, ProcessedContent>,
): Promise<ProcessedContent[]> {
  const { argv } = ctx
  if (fps.length === 0) return []

  const perf = new PerfTimer()
  const log = new QuartzLogger(argv.verbose)
  const concurrency = parseWorkerConcurrency(fps.length, ctx.argv.concurrency)

  let res: ProcessedContent[] = []
  log.start(`Parsing input files using ${concurrency} threads`)
  if (concurrency === 1) {
    try {
      const processors = cachedProcessors(ctx)
      const mdRes = await createFileParser(ctx, fps, htmlReuseCache)(processors.md)
      res = await createMarkdownParser(ctx, mdRes, htmlReuseCache)(processors.html)
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
    const errorHandler = (err: any) => {
      console.error(err)
      process.exit(1)
    }

    const serializableCtx: WorkerSerializableBuildCtx = {
      buildId: ctx.buildId,
      argv: ctx.argv,
      allSlugs: ctx.allSlugs,
      allFiles: ctx.allFiles,
      incremental: ctx.incremental,
    }

    const textToMarkdownPromises: WorkerPromise<MarkdownContent[]>[] = []
    let processedFiles = 0
    for (const chunk of chunks(fps, TEXT_PARSE_CHUNK_SIZE)) {
      textToMarkdownPromises.push(pool.exec('parseMarkdown', [serializableCtx, chunk]))
    }

    const mdResults: Array<MarkdownContent[]> = await Promise.all(
      textToMarkdownPromises.map(async promise => {
        const result = await promise
        processedFiles += result.length
        log.updateText(`text->markdown ${styleText('gray', `${processedFiles}/${fps.length}`)}`)
        return result
      }),
    ).catch(errorHandler)

    const markdownToHtmlPromises: WorkerPromise<ProcessedContent[]>[] = []
    processedFiles = 0
    for (const mdChunk of chunks(mdResults.flat(), HTML_PARSE_CHUNK_SIZE)) {
      markdownToHtmlPromises.push(pool.exec('processHtml', [serializableCtx, mdChunk]))
    }
    const results: ProcessedContent[][] = await Promise.all(
      markdownToHtmlPromises.map(async promise => {
        const result = await promise
        processedFiles += result.length
        log.updateText(`markdown->html ${styleText('gray', `${processedFiles}/${fps.length}`)}`)
        return result
      }),
    ).catch(errorHandler)

    res = results.flat()
    await pool.terminate()
  }

  log.end(`Parsed ${res.length} Markdown files in ${perf.timeSince()}`)
  return res
}
