import { Repository } from '@napi-rs/simple-git'
import { Mutex } from 'async-mutex'
import chokidar from 'chokidar'
import { GlobbyFilterFunction, isGitIgnored } from 'globby'
import { minimatch } from 'minimatch'
import path from 'path'
import sourceMapSupport from 'source-map-support'
import { styleText } from 'util'
import type { ProcessedContent } from './plugins/vfile'
import type { ChangeEvent } from './types/plugin'
import cfg from '../quartz.config'
import { contentAssetClaims } from './plugins/emitters/assets'
import { resetWriteCache } from './plugins/emitters/helpers'
import { resetStaticFileCache, staticAssetClaims } from './plugins/emitters/static'
import { emitContent } from './processors/emit'
import { filterContentResult } from './processors/filter'
import { parseMarkdown, resetProcessedContentCache } from './processors/parse'
import { resetCopyFileCache } from './util/copy-file'
import { Argv, BuildCtx } from './util/ctx'
import { emitQuartzDevEvent } from './util/dev-events'
import { glob, toPosixPath } from './util/glob'
import {
  cleanOutputExcept,
  preservedOutputAssets,
  readOutputAssetManifest,
  writeCurrentOutputAssetManifest,
} from './util/output-assets'
import { FilePath, joinSegments, slugifyFilePath } from './util/path'
import { PerfTimer } from './util/perf'
import { randomIdNonSecure } from './util/random'
import { options } from './util/sourcemap'
import { trace } from './util/trace'

sourceMapSupport.install(options)

const markdownExtensions = new Set(['.md', '.base', '.canvas'])

const isMarkdownPath = (fp: string): boolean => markdownExtensions.has(path.extname(fp))

const syncCtxFiles = (ctx: BuildCtx, allFiles: FilePath[]) => {
  ctx.allFiles = allFiles
  ctx.allSlugs = allFiles.map(fp => slugifyFilePath(fp))
  delete ctx.trie
  delete ctx.renderData
}

async function syncOutputAssetClaims(ctx: BuildCtx, refreshStaticFiles = false): Promise<void> {
  if (refreshStaticFiles) {
    resetStaticFileCache()
  }
  ctx.outputAssetClaims = [
    ...contentAssetClaims(ctx),
    ...(await staticAssetClaims(ctx.argv.output, ctx.cfg.configuration.ignorePatterns)),
  ]
}

function processedContentPath(content: ProcessedContent): FilePath | undefined {
  const file = content[1]
  const fp = file.data.relativePath ?? file.data.filePath ?? file.path
  if (typeof fp === 'string' && fp.length > 0) return toPosixPath(fp) as FilePath
  return undefined
}

function processedContentByPath(content: ProcessedContent[]): ContentByPath {
  const entries = new Map<FilePath, ProcessedContent>()
  for (const item of content) {
    const fp = processedContentPath(item)
    if (fp) entries.set(fp, item)
  }
  return entries
}

function coalescePendingEvent(
  previous: ChangeEvent['type'] | undefined,
  next: ChangeEvent['type'],
): ChangeEvent['type'] {
  if (!previous) return next
  if (next === 'delete') return 'delete'
  if (previous === 'add') return 'add'
  if (previous === 'delete') return 'change'
  return next
}

function recordPendingEvent(buildData: BuildData, type: ChangeEvent['type'], fp: string) {
  const normalized = toPosixPath(fp) as FilePath
  buildData.pendingEvents.set(
    normalized,
    coalescePendingEvent(buildData.pendingEvents.get(normalized), type),
  )
}

function takePendingEvents(buildData: BuildData): PendingContentEvent[] {
  const events = [...buildData.pendingEvents].map(([path, type]) => ({ path, type }))
  buildData.pendingEvents.clear()
  return events
}

function buildChangeEvents(
  pendingEvents: PendingContentEvent[],
  previousContent: ContentByPath,
  parsedContent: ContentByPath,
  publishedContent: ContentByPath,
): ChangeEvent[] {
  const changeEvents: ChangeEvent[] = []
  for (const pendingEvent of pendingEvents) {
    const { path, type } = pendingEvent
    if (!isMarkdownPath(path)) {
      changeEvents.push({ path, type })
      continue
    }

    const current = publishedContent.get(path)
    const previous = previousContent.get(path)
    if (type === 'delete') {
      if (previous) {
        changeEvents.push({ path, type: 'delete', file: previous[1], previousFile: previous[1] })
      }
      continue
    }

    if (current) {
      changeEvents.push({
        path,
        type: previous ? 'change' : 'add',
        file: current[1],
        previousFile: previous?.[1],
      })
      continue
    }

    if (previous && parsedContent.has(path)) {
      changeEvents.push({ path, type: 'delete', file: previous[1], previousFile: previous[1] })
    }
  }

  return changeEvents
}

type BuildReason = 'initial' | 'source'
type RebuildQueue = { running: boolean; requested: boolean }
type PendingContentEvent = { path: FilePath; type: ChangeEvent['type'] }
type ContentByPath = Map<FilePath, ProcessedContent>
type BuildData = {
  ctx: BuildCtx
  ignored: GlobbyFilterFunction
  mut: Mutex
  contentByPath: ContentByPath
  pendingEvents: Map<FilePath, ChangeEvent['type']>
}

type WatchRuntime = { dispose(): Promise<void> }

function describeBuildError(err: unknown): string {
  if (err instanceof Error) return err.message
  if (typeof err === 'string') return err
  try {
    return JSON.stringify(err)
  } catch {
    return String(err)
  }
}

async function buildQuartz(
  argv: Argv,
  mut: Mutex,
  clientRefresh: () => void,
  reason: BuildReason = 'initial',
) {
  let gitCommitSha: string | undefined

  if (argv.serve || argv.watch) {
    try {
      const repo = Repository.discover(argv.directory)
      const head = repo.head()
      gitCommitSha = head?.target() || undefined
    } catch (e) {
      if (argv.verbose) {
        console.log(styleText('yellow', 'Warning: Unable to get git commit SHA'))
        console.error(e)
      }
    }
  }

  const ctx: BuildCtx = {
    buildId: randomIdNonSecure(),
    argv,
    cfg,
    allSlugs: [],
    allFiles: [],
    incremental: false,
    gitCommitSha,
  }

  const perf = new PerfTimer()
  const output = argv.output
  let contentByPath: ContentByPath = new Map()

  const pluginCount = Object.values(cfg.plugins).flat().length
  const pluginNames = (key: 'transformers' | 'filters' | 'emitters') =>
    cfg.plugins[key].map(plugin => plugin.name)
  if (argv.verbose) {
    console.log(`Loaded ${pluginCount} plugins`)
    console.log(`  Transformers: ${pluginNames('transformers').join(', ')}`)
    console.log(`  Filters: ${pluginNames('filters').join(', ')}`)
    console.log(`  Emitters: ${pluginNames('emitters').join(', ')}`)
  }

  const release = await mut.acquire()
  try {
    emitQuartzDevEvent({ type: 'build:start', epoch: ctx.buildId, reason })
    perf.addEvent('glob')
    const allFiles = await glob('**', argv.directory, cfg.configuration.ignorePatterns)
    const markdownPaths = allFiles.filter(isMarkdownPath).sort()
    console.log(
      `Found ${markdownPaths.length} input files from \`${argv.directory}\` in ${perf.timeSince('glob')}`,
    )

    const filePaths = markdownPaths.map(fp => joinSegments(argv.directory, fp) as FilePath)
    syncCtxFiles(ctx, allFiles)
    if (reason === 'source') {
      resetProcessedContentCache()
    }
    ctx.outputAssetManifest = await readOutputAssetManifest()
    await syncOutputAssetClaims(ctx, true)
    const outputAssetClaims = ctx.outputAssetClaims ?? []

    perf.addEvent('clean')
    emitQuartzDevEvent({ type: 'public:remove:start', epoch: ctx.buildId })
    ctx.outputAssetPreserved = await preservedOutputAssets(
      ctx.outputAssetManifest,
      outputAssetClaims,
    )
    await cleanOutputExcept(output, ctx.outputAssetPreserved)
    resetCopyFileCache()
    resetWriteCache()
    console.log(`Cleaned \`${output}\` in ${perf.timeSince('clean')}`)

    const parsedFiles = await parseMarkdown(ctx, filePaths)
    const filteredContent = filterContentResult(ctx, parsedFiles).published
    contentByPath = processedContentByPath(filteredContent)

    await emitContent(ctx, filteredContent)
    await writeCurrentOutputAssetManifest(ctx)
    delete ctx.outputAssetPreserved
    console.log(
      styleText('green', `Done processing ${markdownPaths.length} files in ${perf.timeSince()}`),
    )
    emitQuartzDevEvent({
      type: 'build:ready',
      epoch: ctx.buildId,
      files: markdownPaths.length,
      elapsedMs: perf.elapsedMs(),
    })
  } catch (err) {
    emitQuartzDevEvent({
      type: 'build:error',
      epoch: ctx.buildId,
      message: describeBuildError(err),
    })
    throw err
  } finally {
    release()
  }

  if (argv.watch) {
    ctx.incremental = true
    return startWatching(ctx, mut, clientRefresh, contentByPath)
  }
}

async function startWatching(
  ctx: BuildCtx,
  mut: Mutex,
  clientRefresh: () => void,
  contentByPath: ContentByPath,
): Promise<WatchRuntime> {
  const { argv } = ctx
  const ignored = await createIgnoredFilter(ctx)
  const buildData: BuildData = { ctx, mut, ignored, contentByPath, pendingEvents: new Map() }

  const watcher = chokidar.watch('.', {
    awaitWriteFinish: { stabilityThreshold: 250 },
    persistent: true,
    cwd: argv.directory,
    ignored: buildData.ignored,
    ignoreInitial: true,
  })

  const queue: RebuildQueue = { running: false, requested: false }
  watcher
    .on('add', fp => {
      if (buildData.ignored(fp)) return
      recordPendingEvent(buildData, 'add', fp)
      void requestRebuild(queue, clientRefresh, buildData)
    })
    .on('change', fp => {
      if (buildData.ignored(fp)) return
      recordPendingEvent(buildData, 'change', fp)
      void requestRebuild(queue, clientRefresh, buildData)
    })
    .on('unlink', fp => {
      if (buildData.ignored(fp)) return
      recordPendingEvent(buildData, 'delete', fp)
      void requestRebuild(queue, clientRefresh, buildData)
    })

  return {
    async dispose() {
      await watcher.close()
    },
  }
}

async function createIgnoredFilter(ctx: BuildCtx): Promise<GlobbyFilterFunction> {
  const { argv, cfg } = ctx
  const gitIgnoredMatcher = await isGitIgnored()
  return fp => {
    const rawPath = fp.toString()
    const pathStr = toPosixPath(
      path.isAbsolute(rawPath) ? path.relative(argv.directory, rawPath) : rawPath,
    )
    if (pathStr.startsWith('.git/')) return true
    if (pathStr.endsWith('.test.ts')) return true
    if (gitIgnoredMatcher(pathStr)) return true
    for (const pattern of cfg.configuration.ignorePatterns) {
      if (minimatch(pathStr, pattern)) {
        return true
      }
    }

    return false
  }
}

async function requestRebuild(
  queue: RebuildQueue,
  clientRefresh: () => void,
  buildData: BuildData,
) {
  queue.requested = true
  if (queue.running) return
  queue.running = true
  try {
    while (queue.requested) {
      queue.requested = false
      await rebuild(clientRefresh, buildData)
    }
  } finally {
    queue.running = false
  }
}

async function rebuild(clientRefresh: () => void, buildData: BuildData) {
  const { ctx, mut } = buildData
  const { argv } = ctx

  const release = await mut.acquire()
  let shouldRefresh = false
  let buildId = randomIdNonSecure()
  const pendingEvents = takePendingEvents(buildData)

  try {
    ctx.buildId = buildId
    emitQuartzDevEvent({ type: 'build:start', epoch: buildId, reason: 'content' })

    const perf = new PerfTimer()
    perf.addEvent('rebuild')
    console.log(styleText('yellow', 'Detected change, rebuilding...'))

    perf.addEvent('glob')
    const allFiles = await glob('**', argv.directory, ctx.cfg.configuration.ignorePatterns)
    const markdownPaths = allFiles.filter(isMarkdownPath).sort()
    console.log(
      `Found ${markdownPaths.length} input files from \`${argv.directory}\` in ${perf.timeSince('glob')}`,
    )

    const filePaths = markdownPaths.map(fp => joinSegments(argv.directory, fp) as FilePath)
    syncCtxFiles(ctx, allFiles)
    ctx.outputAssetManifest ??= await readOutputAssetManifest()
    await syncOutputAssetClaims(ctx)
    delete ctx.outputAssetPreserved

    const parsedFiles = await parseMarkdown(ctx, filePaths)
    const parsedByPath = processedContentByPath(parsedFiles)
    const processedFiles = filterContentResult(ctx, parsedFiles).published
    const processedByPath = processedContentByPath(processedFiles)
    const changeEvents = buildChangeEvents(
      pendingEvents,
      buildData.contentByPath,
      parsedByPath,
      processedByPath,
    )

    await emitContent(ctx, processedFiles, changeEvents)
    await writeCurrentOutputAssetManifest(ctx)
    buildData.contentByPath = processedByPath
    console.log(styleText('green', `Done rebuilding in ${perf.timeSince()}`))
    emitQuartzDevEvent({
      type: 'build:ready',
      epoch: buildId,
      files: markdownPaths.length,
      elapsedMs: perf.elapsedMs(),
    })
    shouldRefresh = ctx.buildId === buildId
  } catch (err) {
    emitQuartzDevEvent({ type: 'build:error', epoch: buildId, message: describeBuildError(err) })
    trace('Failed to rebuild Quartz content', err as Error)
  } finally {
    release()
  }

  if (shouldRefresh) {
    clientRefresh()
  }
}

export default async (argv: Argv, mut: Mutex, clientRefresh: () => void, reason?: BuildReason) => {
  try {
    return await buildQuartz(argv, mut, clientRefresh, reason)
  } catch (err) {
    trace('\nExiting Quartz due to a fatal error', err as Error)
  }
}
