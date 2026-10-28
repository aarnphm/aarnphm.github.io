import { Mutex } from 'async-mutex'
import chokidar from 'chokidar'
import { GlobbyFilterFunction, isGitIgnored } from 'globby'
import { minimatch } from 'minimatch'
import { once } from 'node:events'
import { stat } from 'node:fs/promises'
import path from 'path'
import { styleText } from 'util'
import { AGENT_SKILLS_SOURCE_DIRECTORY } from '../plugins/emitters/agent-skills'
import { removeWritten } from '../plugins/emitters/helpers'
import { ProcessedContent } from '../plugins/vfile'
import { emitContent, emitPartialEmitter } from '../processors/emit'
import { filterContentResult } from '../processors/filter'
import {
  parseMarkdown,
  processedContentCacheMisses,
  snapshotProcessedContentCache,
} from '../processors/parse'
import { ChangeEvent } from '../types/plugin'
import { BuildCtx } from '../util/ctx'
import { emitQuartzDevEvent } from '../util/dev-events'
import { glob, toPosixPath } from '../util/glob'
import { FilePath, joinSegments } from '../util/path'
import { logBuildStage, PerfTimer } from '../util/perf'
import {
  captureBuildInputs,
  commitBuildCache,
  commitUnchangedParseBuildCache,
} from '../util/persistent-build-cache'
import { randomIdNonSecure } from '../util/random'
import {
  buildChangeEvents,
  deletedCourseOutputSlugs,
  derivePaths,
  describeBuildError,
  indexContent,
  isMarkdownPath,
  isTestPath,
  syncCtxFiles,
  type PendingChanges,
} from './context'
import { emitFullBuild, emitStagedChanges } from './emission'

type RebuildQueue = { running: boolean; requested: boolean }

type BuildData = {
  ctx: BuildCtx
  ignored: GlobbyFilterFunction
  mut: Mutex
  contentMap: Map<FilePath, ProcessedContent>
  pending: PendingChanges
  agentSkillsPending: PendingChanges
  disposed: boolean
}

export type WatchRuntime = { ready: boolean; dispose(): Promise<boolean> }

export async function startWatching(
  ctx: BuildCtx,
  mut: Mutex,
  clientRefresh: () => void,
  contentMap: Map<FilePath, ProcessedContent>,
): Promise<WatchRuntime> {
  const { argv } = ctx
  const stages = new PerfTimer()
  const ignored = await createIgnoredFilter(ctx)
  logBuildStage(argv, ctx.buildId, 'watch.ignore-filter', stages.elapsedMs())
  stages.addEvent('ready')
  const buildData: BuildData = {
    ctx,
    mut,
    ignored,
    contentMap,
    pending: new Map(),
    agentSkillsPending: new Map(),
    disposed: false,
  }

  const dependencies = ctx.cfg.plugins.transformers.flatMap(
    plugin => plugin.fileDependencies?.(ctx) ?? [],
  )
  const watcher = chokidar.watch(['.', ...dependencies.map(filename => path.resolve(filename))], {
    awaitWriteFinish: { stabilityThreshold: 250 },
    persistent: true,
    cwd: argv.directory,
    ignored: buildData.ignored,
    ignoreInitial: true,
  })

  const queue: RebuildQueue = { running: false, requested: false }
  const watchedSignatures = new Map<FilePath, string>()
  const watchedSignature = async (fp: FilePath): Promise<string> => {
    try {
      const info = await stat(path.join(argv.directory, fp))
      return `${info.mtimeMs}:${info.size}`
    } catch {
      return 'missing'
    }
  }
  const enqueue = async (fp: string, type: ChangeEvent['type']) => {
    if (buildData.disposed || buildData.ignored(fp)) return
    const normalized = toPosixPath(fp) as FilePath
    if (type === 'delete') {
      watchedSignatures.delete(normalized)
    } else {
      const signature = await watchedSignature(normalized)
      if (watchedSignatures.get(normalized) === signature) return
      watchedSignatures.set(normalized, signature)
    }
    if (buildData.disposed) return
    buildData.pending.set(normalized, type)
    void requestRebuild(queue, clientRefresh, buildData)
  }
  watcher
    .on('add', fp => void enqueue(fp, 'add'))
    .on('change', fp => void enqueue(fp, 'change'))
    .on('unlink', fp => void enqueue(fp, 'delete'))

  const agentSkillsWatcher = chokidar.watch(path.resolve(AGENT_SKILLS_SOURCE_DIRECTORY), {
    awaitWriteFinish: { stabilityThreshold: 250 },
    persistent: true,
    ignoreInitial: true,
  })
  const agentSkillsQueue: RebuildQueue = { running: false, requested: false }
  const enqueueAgentSkills = (fp: string, type: ChangeEvent['type']) => {
    if (buildData.disposed) return
    buildData.agentSkillsPending.set(toPosixPath(fp) as FilePath, type)
    void requestAgentSkillsRebuild(agentSkillsQueue, clientRefresh, buildData)
  }
  agentSkillsWatcher
    .on('add', fp => enqueueAgentSkills(fp, 'add'))
    .on('change', fp => enqueueAgentSkills(fp, 'change'))
    .on('unlink', fp => enqueueAgentSkills(fp, 'delete'))

  await Promise.all([
    once(watcher, 'ready').then(() =>
      logBuildStage(argv, ctx.buildId, 'watch.content-ready', stages.elapsedMs('ready')),
    ),
    once(agentSkillsWatcher, 'ready').then(() =>
      logBuildStage(argv, ctx.buildId, 'watch.skills-ready', stages.elapsedMs('ready')),
    ),
  ])

  return {
    ready: false,
    async dispose() {
      const hadPending =
        buildData.pending.size > 0 ||
        buildData.agentSkillsPending.size > 0 ||
        queue.running ||
        agentSkillsQueue.running
      buildData.disposed = true
      buildData.pending.clear()
      buildData.agentSkillsPending.clear()
      await Promise.all([watcher.close(), agentSkillsWatcher.close()])
      return hadPending
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
    if (isTestPath(pathStr)) return true
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
    while ((queue.requested || buildData.pending.size > 0) && !buildData.disposed) {
      queue.requested = false
      await rebuild(clientRefresh, buildData)
    }
  } finally {
    queue.running = false
  }
}

async function requestAgentSkillsRebuild(
  queue: RebuildQueue,
  clientRefresh: () => void,
  buildData: BuildData,
) {
  queue.requested = true
  if (queue.running) return
  queue.running = true
  try {
    while ((queue.requested || buildData.agentSkillsPending.size > 0) && !buildData.disposed) {
      queue.requested = false
      await rebuildAgentSkills(clientRefresh, buildData)
    }
  } finally {
    queue.running = false
  }
}

async function rebuildAgentSkills(clientRefresh: () => void, buildData: BuildData) {
  const { ctx, mut, contentMap } = buildData
  const release = await mut.acquire()
  const buildId = randomIdNonSecure()
  let shouldRefresh = false

  try {
    if (buildData.disposed || buildData.agentSkillsPending.size === 0) return
    const pending: PendingChanges = new Map(buildData.agentSkillsPending)
    buildData.agentSkillsPending.clear()
    globalThis.__quartzSourceEmissionSnapshot = undefined
    ctx.buildId = buildId
    emitQuartzDevEvent({ type: 'build:start', epoch: buildId, reason: 'content' })

    const perf = new PerfTimer()
    console.log(styleText('yellow', 'Detected agent skill change, rebuilding...'))
    const changeEvents = [...pending].map(([path, type]) => ({ path, type }))
    await emitPartialEmitter(ctx, [...contentMap.values()], changeEvents, 'AgentSkills')
    console.log(styleText('green', `Done rebuilding agent skills in ${perf.timeSince()}`))
    emitQuartzDevEvent({
      type: 'build:ready',
      epoch: buildId,
      files: ctx.allFiles.filter(isMarkdownPath).length,
      elapsedMs: perf.elapsedMs(),
    })
    shouldRefresh = ctx.buildId === buildId
  } catch (err) {
    emitQuartzDevEvent({ type: 'build:error', epoch: buildId, message: describeBuildError(err) })
    console.error('Failed to rebuild agent skills', err)
  } finally {
    release()
  }

  if (shouldRefresh) {
    clientRefresh()
  }
}

async function rebuild(clientRefresh: () => void, buildData: BuildData) {
  const { ctx, mut, contentMap } = buildData
  const { argv } = ctx

  const release = await mut.acquire()
  let shouldRefresh = false
  const buildId = randomIdNonSecure()

  try {
    if (buildData.disposed || buildData.pending.size === 0) return
    const pending: PendingChanges = new Map(buildData.pending)
    buildData.pending.clear()
    const incremental = ctx.incremental && pending.size > 0 && contentMap.size > 0
    globalThis.__quartzSourceEmissionSnapshot = undefined
    ctx.buildId = buildId
    emitQuartzDevEvent({ type: 'build:start', epoch: buildId, reason: 'content' })

    const perf = new PerfTimer()
    perf.addEvent('rebuild')
    console.log(styleText('yellow', 'Detected change, rebuilding...'))

    perf.addEvent('glob')
    const fileSetIsStable =
      pending.size > 0 &&
      ctx.allFiles.length > 0 &&
      [...pending.values()].every(type => type === 'change')
    const allFiles = fileSetIsStable
      ? ctx.allFiles
      : await glob('**', argv.directory, ctx.cfg.configuration.ignorePatterns)
    const { markdown: markdownPaths, files: filePaths } = derivePaths(argv, allFiles)
    console.log(
      `Found ${markdownPaths.length} input files from \`${argv.directory}\` in ${perf.timeSince('glob')}`,
    )

    syncCtxFiles(ctx, allFiles)

    const persistentInputs = await captureBuildInputs(ctx)
    const dirtyPaths = incremental
      ? new Set(
          [...pending.keys()].map(relPath => joinSegments(argv.directory, relPath) as FilePath),
        )
      : undefined
    const parsedFiles = await parseMarkdown(ctx, filePaths, dirtyPaths)
    const processedFiles = filterContentResult(ctx, parsedFiles).published
    const current = indexContent(processedFiles)
    const reuseParseBlob = Boolean(persistentInputs) && processedContentCacheMisses() === 0
    const parseSnapshot =
      persistentInputs && !reuseParseBlob ? snapshotProcessedContentCache(filePaths) : undefined
    const changeEvents = incremental ? buildChangeEvents(ctx, pending, contentMap, current) : []
    // Partial emit does not remove every output a deleted page derived, so a delete cleans `public`
    // and emits the site again. Course pages are leaves whose outputs sit at their own slug: remove
    // those and keep the rebuild scoped to the course.
    const droppedOutputs = changeEvents
      .filter(event => event.type === 'delete' && event.file)
      .map(deletedCourseOutputSlugs)

    if (incremental && droppedOutputs.every(slugs => slugs !== undefined)) {
      await emitStagedChanges(ctx, async () => {
        await Promise.all(
          droppedOutputs
            .flat()
            .flatMap(slug => [
              removeWritten(ctx, slug, '.html'),
              removeWritten(ctx, slug, '.md'),
              removeWritten(ctx, `${slug}-og-image`, '.webp'),
            ]),
        )
        await emitContent(ctx, processedFiles, changeEvents)
      })
    } else {
      await emitFullBuild(ctx, perf, processedFiles)
    }
    buildData.contentMap = current
    const published = [...current.keys()]
    if (
      !(reuseParseBlob && (await commitUnchangedParseBuildCache(ctx, persistentInputs, published)))
    ) {
      await commitBuildCache(
        ctx,
        persistentInputs,
        parseSnapshot ?? (persistentInputs ? snapshotProcessedContentCache(filePaths) : undefined),
        published,
      )
    }
    ctx.incremental = argv.watch
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
    ctx.incremental = false
    console.error('Failed to rebuild Quartz content', err)
  } finally {
    release()
  }

  if (shouldRefresh) {
    clientRefresh()
  }
}
