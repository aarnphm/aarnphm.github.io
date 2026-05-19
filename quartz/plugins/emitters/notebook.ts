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
import { createHtmlProcessor, createMdProcessor } from '../../processors/parse'
import { ChangeEvent, QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx, Argv } from '../../util/ctx'
import { glob } from '../../util/glob'
import { parseNotebook, notebookToMarkdown } from '../../util/notebook'
import { FilePath, FullSlug, joinSegments, pathToRoot, slugifyFilePath } from '../../util/path'
import { StaticResources } from '../../util/resources'
import { ProcessedContent } from '../vfile'
import { write } from './helpers'

type NotebookRenderMode = 'saved' | 'execute'

type Options = { mode?: NotebookRenderMode; executeTimeoutSeconds?: number; allowErrors?: boolean }

type ResolvedOptions = {
  mode: NotebookRenderMode
  executeTimeoutSeconds: number
  allowErrors: boolean
}

const defaultOptions: ResolvedOptions = {
  mode: 'saved',
  executeTimeoutSeconds: 120,
  allowErrors: false,
}

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
  return {
    ...defaultOptions,
    ...userOpts,
    mode: normalizeMode(process.env.QUARTZ_NOTEBOOK_MODE) ?? userOpts?.mode ?? defaultOptions.mode,
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
  let markdown = notebookToMarkdown(notebook, src).trim()

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
  const localCtx = notebookContext(ctx, fps)
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
      const notebookChanges = notebookChangeEvents(changeEvents)
      if (notebookChanges.length === 0) {
        return
      }

      for (const changeEvent of notebookChanges) {
        if (changeEvent.type === 'delete') {
          await deleteNotebookPage(ctx, changeEvent.path)
        }
      }

      const fps = await notebookFiles(ctx.argv, ctx.cfg)
      yield* emitNotebookPages(ctx, content, resources, fps, opts)
    },
    async *emit(ctx, content, resources) {
      const { argv, cfg } = ctx
      const fps = await notebookFiles(argv, cfg)
      yield* emitNotebookPages(ctx, content, resources, fps, opts)
    },
  }
}
