import { Repository } from '@napi-rs/simple-git'
import { Mutex } from 'async-mutex'
import chokidar from 'chokidar'
import { GlobbyFilterFunction, isGitIgnored } from 'globby'
import { minimatch } from 'minimatch'
import path from 'path'
import sourceMapSupport from 'source-map-support'
import { styleText } from 'util'
import cfg from '../quartz.config'
import { contentAssetClaims } from './plugins/emitters/assets'
import { resetWriteCache } from './plugins/emitters/helpers'
import { resetStaticFileCache, staticAssetClaims } from './plugins/emitters/static'
import { ProcessedContent } from './plugins/vfile'
import { emitContent } from './processors/emit'
import { filterContentResult } from './processors/filter'
import { parseMarkdown, resetProcessedContentCache } from './processors/parse'
import { ChangeEvent } from './types/plugin'
import { Argv, BuildCtx } from './util/ctx'
import { emitQuartzDevEvent } from './util/dev-events'
import { isFlashcardPath } from './util/flashcards-path'
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

const isMarkdownPath = (fp: string): boolean =>
  markdownExtensions.has(path.extname(fp)) || isFlashcardPath(fp)

const syncCtxFiles = (ctx: BuildCtx, allFiles: FilePath[]) => {
  ctx.allFiles = allFiles
  ctx.allSlugs = allFiles.filter(fp => !isFlashcardPath(fp)).map(fp => slugifyFilePath(fp))
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

type BuildReason = 'initial' | 'source'
type RebuildQueue = { running: boolean; requested: boolean }
type PendingChanges = Map<FilePath, ChangeEvent['type']>
type BuildData = {
  ctx: BuildCtx
  ignored: GlobbyFilterFunction
  mut: Mutex
  contentMap: Map<FilePath, ProcessedContent>
  pending: PendingChanges
}

type WatchRuntime = { dispose(): Promise<void> }

const contentFilePath = (content: ProcessedContent): FilePath | undefined =>
  content[1].data.filePath as FilePath | undefined

function indexContent(content: ProcessedContent[]): Map<FilePath, ProcessedContent> {
  const map = new Map<FilePath, ProcessedContent>()
  for (const entry of content) {
    const fp = contentFilePath(entry)
    if (fp) map.set(fp, entry)
  }
  return map
}

function buildChangeEvents(
  ctx: BuildCtx,
  pending: PendingChanges,
  previous: ReadonlyMap<FilePath, ProcessedContent>,
  current: ReadonlyMap<FilePath, ProcessedContent>,
): ChangeEvent[] {
  const events: ChangeEvent[] = []
  for (const [relPath, fsType] of pending) {
    const fullPath = joinSegments(ctx.argv.directory, relPath) as FilePath
    const prev = previous.get(fullPath)
    const cur = fsType === 'delete' ? undefined : current.get(fullPath)
    if (prev && !cur) {
      events.push({ type: 'delete', path: relPath, file: prev[1], previousFile: prev[1] })
    } else if (!prev && cur) {
      events.push({ type: 'add', path: relPath, file: cur[1] })
    } else if (prev && cur) {
      events.push({ type: 'change', path: relPath, file: cur[1], previousFile: prev[1] })
    } else {
      events.push({ type: fsType, path: relPath })
    }
  }
  return events
}

function describeBuildError(err: unknown): string {
  if (err instanceof Error) return err.message
  if (typeof err === 'string') return err
  try {
    return JSON.stringify(err)
  } catch {
    return String(err)
  }
}

function resetGeneratedResourceState(ctx: BuildCtx): void {
  delete ctx.assetManifest
  delete ctx.extractedStaticResources
  delete ctx.pageResourceCacheBuildId
  delete ctx.pageResourceCache
}

async function cleanOutputForBuild(
  ctx: BuildCtx,
  perf: PerfTimer,
  refreshStaticFiles: boolean,
): Promise<void> {
  resetGeneratedResourceState(ctx)
  ctx.outputAssetManifest = await readOutputAssetManifest()
  await syncOutputAssetClaims(ctx, refreshStaticFiles)
  const outputAssetClaims = ctx.outputAssetClaims ?? []

  perf.addEvent('clean')
  emitQuartzDevEvent({ type: 'public:remove:start', epoch: ctx.buildId })
  ctx.outputAssetPreserved = await preservedOutputAssets(ctx.outputAssetManifest, outputAssetClaims)
  await cleanOutputExcept(ctx.argv.output, ctx.outputAssetPreserved)
  resetWriteCache()
  console.log(`Cleaned \`${ctx.argv.output}\` in ${perf.timeSince('clean')}`)
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

  const pluginCount = Object.values(cfg.plugins).flat().length
  const pluginNames = (key: 'transformers' | 'filters' | 'emitters') =>
    cfg.plugins[key].map(plugin => plugin.name)
  if (argv.verbose) {
    console.log(`Loaded ${pluginCount} plugins`)
    console.log(`  Transformers: ${pluginNames('transformers').join(', ')}`)
    console.log(`  Filters: ${pluginNames('filters').join(', ')}`)
    console.log(`  Emitters: ${pluginNames('emitters').join(', ')}`)
  }

  let initialContent: ProcessedContent[] = []
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
    await cleanOutputForBuild(ctx, perf, true)

    const parsedFiles = await parseMarkdown(ctx, filePaths)
    const filteredContent = filterContentResult(ctx, parsedFiles).published
    initialContent = filteredContent

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
    return startWatching(ctx, mut, clientRefresh, initialContent)
  }
}

async function startWatching(
  ctx: BuildCtx,
  mut: Mutex,
  clientRefresh: () => void,
  initialContent: ProcessedContent[],
): Promise<WatchRuntime> {
  const { argv } = ctx
  const ignored = await createIgnoredFilter(ctx)
  const buildData: BuildData = {
    ctx,
    mut,
    ignored,
    contentMap: indexContent(initialContent),
    pending: new Map(),
  }

  const watcher = chokidar.watch('.', {
    awaitWriteFinish: { stabilityThreshold: 250 },
    persistent: true,
    cwd: argv.directory,
    ignored: buildData.ignored,
    ignoreInitial: true,
  })

  const queue: RebuildQueue = { running: false, requested: false }
  const enqueue = (fp: string, type: ChangeEvent['type']) => {
    if (buildData.ignored(fp)) return
    buildData.pending.set(toPosixPath(fp) as FilePath, type)
    void requestRebuild(queue, clientRefresh, buildData)
  }
  watcher
    .on('add', fp => enqueue(fp, 'add'))
    .on('change', fp => enqueue(fp, 'change'))
    .on('unlink', fp => enqueue(fp, 'delete'))

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
      const pending: PendingChanges = new Map(buildData.pending)
      buildData.pending.clear()
      await rebuild(clientRefresh, buildData, pending)
    }
  } finally {
    queue.running = false
  }
}

async function rebuild(clientRefresh: () => void, buildData: BuildData, pending: PendingChanges) {
  const { ctx, mut, contentMap } = buildData
  const { argv } = ctx

  const release = await mut.acquire()
  let shouldRefresh = false
  const buildId = randomIdNonSecure()
  const incremental = ctx.incremental && pending.size > 0 && contentMap.size > 0

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

    const parsedFiles = await parseMarkdown(ctx, filePaths)
    const processedFiles = filterContentResult(ctx, parsedFiles).published
    const current = indexContent(processedFiles)
    const changeEvents = incremental ? buildChangeEvents(ctx, pending, contentMap, current) : []
    const dropsContentPage = changeEvents.some(event => event.type === 'delete' && event.file)

    if (incremental && !dropsContentPage) {
      await emitContent(ctx, processedFiles, changeEvents)
    } else {
      await cleanOutputForBuild(ctx, perf, true)
      await emitContent(ctx, processedFiles)
      await writeCurrentOutputAssetManifest(ctx)
      delete ctx.outputAssetPreserved
    }
    buildData.contentMap = current
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
