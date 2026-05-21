import { Root as HtmlRoot } from 'hast'
import { spawn } from 'node:child_process'
import fs from 'node:fs/promises'
import { availableParallelism } from 'node:os'
import path from 'node:path'
import { styleText } from 'node:util'
import { VFile } from 'vfile'
import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout, QuartzConfig } from '../../cfg'
import * as Component from '../../components'
import HeaderConstructor from '../../components/Header'
import {
  CuriusContent,
  CuriusFriends,
  CuriusNavigation,
  pageResources,
  renderPage,
} from '../../components/renderPage'
import notebookRuntimeScript from '../../components/scripts/notebook-runtime.inline'
import { createHtmlProcessor, createMdProcessor } from '../../processors/parse'
import { ChangeEvent, QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx, Argv } from '../../util/ctx'
import { glob } from '../../util/glob'
import { parseNotebook, notebookToMarkdown } from '../../util/notebook'
import { notebookRuntimeImportCandidates } from '../../util/notebook-runtime'
import { FilePath, FullSlug, joinSegments, pathToRoot, slugifyFilePath } from '../../util/path'
import { StaticResources } from '../../util/resources'
import { ProcessedContent } from '../vfile'
import { write } from './helpers'

type NotebookRenderMode = 'saved' | 'execute'

type NotebookRuntimeOptions = false | { enabled?: boolean; pyodideIndexUrl?: string }

type Options = {
  mode?: NotebookRenderMode
  executeTimeoutSeconds?: number
  allowErrors?: boolean
  runtime?: NotebookRuntimeOptions
}

type ResolvedOptions = {
  mode: NotebookRenderMode
  executeTimeoutSeconds: number
  allowErrors: boolean
  runtime: false | { enabled: true; pyodideIndexUrl: string }
}

type NotebookDependencies = { imports: Set<FilePath>; assets: Set<FilePath> }

const defaultOptions: ResolvedOptions = {
  mode: 'saved',
  executeTimeoutSeconds: 120,
  allowErrors: false,
  runtime: false,
}

const defaultPyodideIndexUrl = 'https://cdn.jsdelivr.net/pyodide/v0.29.4/full/'
const htmlResourceSrcPattern =
  /<(?:img|video|audio|iframe)\b[^>]*\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/gi
const markdownImageTargetPattern = /!\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)[^)]*\)/g

const notebookFiles = async (argv: Argv, cfg: QuartzConfig) => {
  return await glob('**/*.ipynb', argv.directory, [...cfg.configuration.ignorePatterns])
}

function normalizeMode(value: string | undefined): NotebookRenderMode | undefined {
  if (value === undefined) return undefined
  if (value === 'saved' || value === 'execute') return value
  if (value === 'clean') return 'saved'
  if (value === 'run' || value === 'full') return 'execute'
  throw new Error(`Invalid notebook render mode: ${value}`)
}

function resolveOptions(userOpts?: Options): ResolvedOptions {
  const runtime: ResolvedOptions['runtime'] =
    userOpts?.runtime === false ||
    userOpts?.runtime === undefined ||
    userOpts.runtime.enabled === false
      ? false
      : {
          enabled: true,
          pyodideIndexUrl: userOpts.runtime.pyodideIndexUrl ?? defaultPyodideIndexUrl,
        }

  return {
    ...defaultOptions,
    ...userOpts,
    mode: normalizeMode(process.env.QUARTZ_NOTEBOOK_MODE) ?? userOpts?.mode ?? defaultOptions.mode,
    runtime,
  }
}

const notebookPageLayout: FullPageLayout = {
  ...sharedPageComponents,
  beforeBody: [Component.ArticleTitle()],
  pageBody: Component.Content(),
  sidebar: defaultContentPageLayout.sidebar,
}

type NotebookPage = { fp: FilePath; slug: FullSlug; content: ProcessedContent }

function isNotebookPath(fp: FilePath): boolean {
  return path.extname(fp) === '.ipynb'
}

function sourceText(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(sourceText).join('')
  if (value === undefined || value === null) return ''
  return String(value)
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
  const notebook = parseNotebook(raw, fp)
  const imports = new Set<FilePath>()
  const assets = new Set<FilePath>()
  const dir = path.posix.dirname(fp)

  for (const cell of notebook.cells) {
    const source = sourceText(cell.source)
    if (cell.cell_type === 'code') {
      for (const name of notebookRuntimeImportCandidates(source)) {
        imports.add(path.posix.join(dir, `${name}.ipynb`) as FilePath)
      }
    }
    if (cell.cell_type !== 'markdown') continue

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

async function executeNotebook(src: FilePath, opts: ResolvedOptions): Promise<string> {
  const nbConvertArgs = [
    '--with',
    'jupyter',
    '--with',
    'nbconvert',
    '--with',
    'jupyter-contrib-nbextensions',
    '--with',
    'notebook<7',
    '--with',
    'ipykernel',
    '--with',
    'import-ipynb',
    '--from',
    'jupyter-core',
    'jupyter',
    'nbconvert',
    '--to',
    'notebook',
    '--execute',
    '--stdout',
    `--ExecutePreprocessor.timeout=${opts.executeTimeoutSeconds}`,
    '--log-level',
    '50',
  ]

  if (opts.allowErrors) {
    nbConvertArgs.push('--allow-errors')
  }

  nbConvertArgs.push(src)

  const command = process.env.CF_PAGES === '1' ? 'python' : 'uvx'
  const args =
    process.env.CF_PAGES === '1' ? ['-m', 'uv', 'tool', 'run', ...nbConvertArgs] : nbConvertArgs

  return await new Promise<string>((resolve, reject) => {
    const proc = spawn(command, args, { cwd: path.dirname(src), env: { ...process.env } })
    let stdout = ''
    let stderr = ''

    proc.stdout.on('data', chunk => {
      stdout += chunk
    })
    proc.stderr.on('data', chunk => {
      stderr += chunk
    })
    proc.on('error', reject)
    proc.on('exit', code => {
      if (code === 0) {
        resolve(stdout)
      } else {
        reject(new Error(`Notebook execution failed with code ${code}: ${stderr}`))
      }
    })
  })
}

async function notebookSource(ctx: BuildCtx, fp: FilePath, opts: ResolvedOptions): Promise<string> {
  const src = joinSegments(ctx.argv.directory, fp) as FilePath
  if (opts.mode === 'execute') {
    return await executeNotebook(src, opts)
  }

  return await fs.readFile(src, 'utf8')
}

async function notebookProcessedContent(
  ctx: BuildCtx,
  fp: FilePath,
  opts: ResolvedOptions,
): Promise<NotebookPage> {
  const src = joinSegments(ctx.argv.directory, fp) as FilePath
  const slug = slugifyFilePath(fp, true)
  const raw = await notebookSource(ctx, fp, opts)
  const notebook = parseNotebook(raw, src)
  let markdown = notebookToMarkdown(notebook, fp, {
    runtime: opts.runtime === false ? false : { ...opts.runtime, sourcePath: fp },
  }).trim()

  for (const plugin of ctx.cfg.plugins.transformers.filter(p => p.textTransform)) {
    markdown = plugin.textTransform!(ctx, markdown)
  }

  const file = new VFile({ path: src, value: markdown })
  file.data.filePath = src
  file.data.relativePath = fp
  file.data.slug = slug

  const mdProcessor = createMdProcessor(ctx)
  const htmlProcessor = createHtmlProcessor(ctx)
  const ast = mdProcessor.parse(file)
  const mdAst = await mdProcessor.run(ast, file)
  const htmlAst = (await htmlProcessor.run(mdAst, file)) as HtmlRoot
  if (file.data.frontmatter) {
    file.data.frontmatter.description = ''
    file.data.frontmatter.socialDescription = ''
  }
  file.data.description = ''

  return { fp, slug, content: [htmlAst, file] }
}

async function parseNotebookPages(
  ctx: BuildCtx,
  fps: FilePath[],
  maxWorkers: number,
  opts: ResolvedOptions,
): Promise<NotebookPage[]> {
  const pages: NotebookPage[] = []

  for (let start = 0; start < fps.length; start += maxWorkers) {
    const batch = fps.slice(start, start + maxWorkers)
    const results = await Promise.all(
      batch.map(async fp => {
        try {
          return {
            status: 'fulfilled' as const,
            page: await notebookProcessedContent(ctx, fp, opts),
          }
        } catch (error) {
          return { status: 'rejected' as const, fp, error }
        }
      }),
    )

    for (const result of results) {
      if (result.status === 'fulfilled') {
        pages.push(result.page)
      } else {
        console.error(
          styleText('red', `\n[emit:NotebookViewer] Error processing ${result.fp}:`),
          result.error,
        )
      }
    }
  }

  return pages
}

async function emitNotebookPage(
  ctx: BuildCtx,
  page: NotebookPage,
  allFiles: ProcessedContent[],
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
    allFiles: allFiles.map(([, pageFile]) => pageFile.data),
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
  const pages = await parseNotebookPages(localCtx, fps, maxWorkers, opts)
  const allFiles = [...content, ...pages.map(page => page.content)]

  for (const page of pages) {
    try {
      yield await emitNotebookPage(localCtx, page, allFiles, resources)
    } catch (error) {
      console.error(styleText('red', `\n[emit:NotebookViewer] Error processing ${page.fp}:`), error)
    }
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
        CuriusFriends,
        CuriusContent,
        CuriusNavigation,
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
      for (const changeEvent of notebookChangeEvents(changeEvents)) {
        if (changeEvent.type === 'delete') {
          await deleteNotebookPage(ctx, changeEvent.path)
        }
      }

      const fps = await notebookFiles(ctx.argv, ctx.cfg)
      const affected = await affectedNotebookFiles(ctx, fps, changeEvents)
      yield* emitNotebookPages(ctx, content, resources, affected, opts, fps)
    },
    async *emit(ctx, content, resources) {
      const { argv, cfg } = ctx
      const fps = await notebookFiles(argv, cfg)
      yield* emitNotebookPages(ctx, content, resources, fps, opts)
    },
    externalResources() {
      if (opts.runtime === false) return
      return {
        js: [
          {
            script: notebookRuntimeScript,
            loadTime: 'afterDOMReady',
            contentType: 'inline',
            moduleType: 'module',
          },
        ],
      }
    },
  }
}
