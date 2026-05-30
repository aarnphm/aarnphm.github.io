import { Repository } from '@napi-rs/simple-git'
import { Mutex } from 'async-mutex'
import chokidar from 'chokidar'
import { rm } from 'fs/promises'
import { GlobbyFilterFunction, isGitIgnored } from 'globby'
import { minimatch } from 'minimatch'
import path from 'path'
import sourceMapSupport from 'source-map-support'
import { styleText } from 'util'
import cfg from '../quartz.config'
import { resetWriteCache } from './plugins/emitters/helpers'
import { resetStaticFileCache } from './plugins/emitters/static'
import { emitContent } from './processors/emit'
import { filterContent } from './processors/filter'
import { parseMarkdown } from './processors/parse'
import { resetCopyFileCache } from './util/copy-file'
import { Argv, BuildCtx } from './util/ctx'
import { emitQuartzDevEvent } from './util/dev-events'
import { glob, toPosixPath } from './util/glob'
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

type BuildData = { ctx: BuildCtx; ignored: GlobbyFilterFunction; mut: Mutex }
type BuildReason = 'initial' | 'source'
type RebuildQueue = { running: boolean; requested: boolean }

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
    perf.addEvent('clean')
    emitQuartzDevEvent({ type: 'public:remove:start', epoch: ctx.buildId })
    await rm(output, { recursive: true, force: true })
    resetCopyFileCache()
    resetStaticFileCache()
    resetWriteCache()
    console.log(`Removed \`${output}\` in ${perf.timeSince('clean')}`)

    perf.addEvent('glob')
    const allFiles = await glob('**/*.*', argv.directory, cfg.configuration.ignorePatterns)
    const markdownPaths = allFiles.filter(isMarkdownPath).sort()
    console.log(
      `Found ${markdownPaths.length} input files from \`${argv.directory}\` in ${perf.timeSince('glob')}`,
    )

    const filePaths = markdownPaths.map(fp => joinSegments(argv.directory, fp) as FilePath)
    syncCtxFiles(ctx, allFiles)

    const parsedFiles = await parseMarkdown(ctx, filePaths)
    const filteredContent = filterContent(ctx, parsedFiles)

    await emitContent(ctx, filteredContent)
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
    return startWatching(ctx, mut, clientRefresh)
  }
}

async function startWatching(
  ctx: BuildCtx,
  mut: Mutex,
  clientRefresh: () => void,
): Promise<WatchRuntime> {
  const { argv } = ctx
  const ignored = await createIgnoredFilter(ctx)
  const buildData: BuildData = { ctx, mut, ignored }

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
      void requestRebuild(queue, clientRefresh, buildData)
    })
    .on('change', fp => {
      if (buildData.ignored(fp)) return
      void requestRebuild(queue, clientRefresh, buildData)
    })
    .on('unlink', fp => {
      if (buildData.ignored(fp)) return
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

  try {
    ctx.buildId = buildId
    emitQuartzDevEvent({ type: 'build:start', epoch: buildId, reason: 'content' })

    const perf = new PerfTimer()
    perf.addEvent('rebuild')
    console.log(styleText('yellow', 'Detected change, rebuilding...'))

    perf.addEvent('glob')
    const allFiles = await glob('**/*.*', argv.directory, ctx.cfg.configuration.ignorePatterns)
    const markdownPaths = allFiles.filter(isMarkdownPath).sort()
    console.log(
      `Found ${markdownPaths.length} input files from \`${argv.directory}\` in ${perf.timeSince('glob')}`,
    )

    const filePaths = markdownPaths.map(fp => joinSegments(argv.directory, fp) as FilePath)
    syncCtxFiles(ctx, allFiles)

    const parsedFiles = await parseMarkdown(ctx, filePaths)
    const processedFiles = filterContent(ctx, parsedFiles)

    await emitContent(ctx, processedFiles)
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
