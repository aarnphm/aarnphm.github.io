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
import { emitContent } from './processors/emit'
import { filterContent } from './processors/filter'
import { parseMarkdown } from './processors/parse'
import { Argv, BuildCtx } from './util/ctx'
import { glob, toPosixPath } from './util/glob'
import { resetLinkOrCopyFileCache } from './util/link-or-copy-file'
import { FilePath, joinSegments, slugifyFilePath } from './util/path'
import { flushBuildSpans, PerfTimer } from './util/perf'
import { randomIdNonSecure } from './util/random'
import { options } from './util/sourcemap'
import { trace } from './util/trace'

sourceMapSupport.install(options)

const markdownExtensions = new Set(['.md', '.base', '.canvas'])

const isMarkdownPath = (fp: string): boolean => markdownExtensions.has(path.extname(fp))

const normalizeWatchedPath = (directory: string, fp: string): string => {
  const pathStr = toPosixPath(path.isAbsolute(fp) ? path.relative(directory, fp) : fp)
  const directoryPath = toPosixPath(directory).replace(/\/$/, '')
  const directoryPrefix = `${directoryPath}/`
  return pathStr.startsWith(directoryPrefix) ? pathStr.slice(directoryPrefix.length) : pathStr
}

const isPathOrChild = (pathStr: string, prefix: string): boolean =>
  pathStr === prefix || pathStr.startsWith(`${prefix}/`)

const syncCtxFiles = (ctx: BuildCtx, allFiles: FilePath[]) => {
  ctx.allFiles = allFiles
  ctx.allSlugs = allFiles.map(fp => slugifyFilePath(fp))
  delete ctx.trie
  delete ctx.renderData
}

type FullBuildKind = 'initial' | 'rebuild'

async function buildQuartz(argv: Argv, mut: Mutex, clientRefresh: () => void) {
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

  const pluginCount = Object.values(cfg.plugins).flat().length
  const pluginNames = (key: 'transformers' | 'filters' | 'emitters') =>
    cfg.plugins[key].map(plugin => plugin.name)
  if (argv.verbose) {
    console.log(`Loaded ${pluginCount} plugins`)
    console.log(`  Transformers: ${pluginNames('transformers').join(', ')}`)
    console.log(`  Filters: ${pluginNames('filters').join(', ')}`)
    console.log(`  Emitters: ${pluginNames('emitters').join(', ')}`)
  }

  await runFullBuild(ctx, mut, 'initial')

  if (argv.watch) {
    ctx.incremental = true
    return startWatching(ctx, mut, clientRefresh)
  }
}

async function runFullBuild(
  ctx: BuildCtx,
  mut: Mutex,
  kind: FullBuildKind,
  shouldRun = () => true,
): Promise<boolean> {
  const { argv } = ctx
  const release = await mut.acquire()
  try {
    if (!shouldRun()) {
      return false
    }

    if (kind === 'rebuild') {
      ctx.buildId = randomIdNonSecure()
    }

    const perf = new PerfTimer()
    if (kind === 'rebuild') {
      perf.addEvent('rebuild')
      console.log(styleText('yellow', 'Detected change, rebuilding...'))
    }

    resetWriteCache()
    resetLinkOrCopyFileCache()
    ctx.cleanOutput = kind === 'initial'
    if (ctx.cleanOutput) {
      perf.addEvent('clean')
      await rm(argv.output, { recursive: true, force: true })
      console.log(`Removed \`${argv.output}\` in ${perf.timeSince('clean')}`)
    }

    perf.addEvent('glob')
    const allFiles = await glob('**/*.*', argv.directory, ctx.cfg.configuration.ignorePatterns)
    const markdownPaths = allFiles.filter(isMarkdownPath).sort()
    console.log(
      `Found ${markdownPaths.length} input files from \`${argv.directory}\` in ${perf.timeSince('glob')}`,
    )

    const filePaths = markdownPaths.map(fp => joinSegments(argv.directory, fp) as FilePath)
    syncCtxFiles(ctx, allFiles)

    const parsedFiles = await parseMarkdown(ctx, filePaths)
    const filteredContent = filterContent(ctx, parsedFiles)

    await emitContent(ctx, filteredContent)
    flushBuildSpans(argv)
    const doneText =
      kind === 'initial'
        ? `Done processing ${markdownPaths.length} files in ${perf.timeSince()}`
        : `Done rebuilding in ${perf.timeSince()}`
    console.log(styleText('green', doneText))
    return true
  } finally {
    ctx.cleanOutput = false
    release()
  }
}

async function startWatching(ctx: BuildCtx, mut: Mutex, clientRefresh: () => void) {
  const { argv } = ctx
  const ignored = await createIgnoredFilter(ctx)
  let disposed = false
  let rebuildInFlight = false
  let rebuildRequested = false

  const watcher = chokidar.watch('.', {
    awaitWriteFinish: { stabilityThreshold: 250 },
    persistent: true,
    cwd: argv.directory,
    ignored,
    ignoreInitial: true,
  })

  await new Promise<void>((resolve, reject) => {
    watcher.once('ready', resolve)
    watcher.once('error', reject)
  })

  const rebuild = async () => {
    if (disposed) return
    if (rebuildInFlight) {
      rebuildRequested = true
      return
    }

    rebuildInFlight = true
    try {
      let shouldRefresh = false
      do {
        rebuildRequested = false
        const didBuild = await runFullBuild(ctx, mut, 'rebuild', () => !disposed)
        shouldRefresh = shouldRefresh || didBuild
      } while (rebuildRequested && !disposed)

      if (shouldRefresh && !disposed) {
        clientRefresh()
      }
    } catch (err) {
      trace('Failed to rebuild Quartz content', err as Error)
    } finally {
      rebuildInFlight = false
    }
  }

  watcher.on('add', () => void rebuild())
  watcher.on('change', () => void rebuild())
  watcher.on('unlink', () => void rebuild())

  return async () => {
    disposed = true
    await watcher.close()
  }
}

async function createIgnoredFilter(ctx: BuildCtx): Promise<GlobbyFilterFunction> {
  const { argv, cfg } = ctx
  const gitIgnoredMatcher = await isGitIgnored()
  const outputPath = normalizeWatchedPath(argv.directory, argv.output)
  return fp => {
    const pathStr = normalizeWatchedPath(argv.directory, fp.toString())
    if (pathStr.startsWith('.git/')) return true
    if (isPathOrChild(pathStr, outputPath)) return true
    if (isPathOrChild(pathStr, '.quartz-cache')) return true
    if (isPathOrChild(pathStr, 'node_modules')) return true
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

export default async (argv: Argv, mut: Mutex, clientRefresh: () => void) => {
  try {
    return await buildQuartz(argv, mut, clientRefresh)
  } catch (err) {
    trace('\nExiting Quartz due to a fatal error', err as Error)
  }
}
