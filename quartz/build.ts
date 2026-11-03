import { Repository } from '@napi-rs/simple-git'
import { Mutex } from 'async-mutex'
import { stat } from 'node:fs/promises'
import sourceMapSupport from 'source-map-support'
import { styleText } from 'util'
import type { QuartzConfig } from './cfg'
import {
  describeBuildError,
  formatBuildError,
  indexContent,
  isMarkdownPath,
  syncCtxFiles,
} from './build/context'
import {
  cleanOutputForBuild,
  emitCourses,
  emitFullBuild,
  emitStagedChanges,
  type BuildScope,
} from './build/emission'
import { startWatching, type WatchRuntime } from './build/watch'
import { omitDisabledCourseContent } from './dev-features'
import { ProcessedContent } from './plugins/vfile'
import { emitPartialEmitter } from './processors/emit'
import { filterContentResult } from './processors/filter'
import {
  parseMarkdown,
  processedContentCacheMisses,
  restoredProcessedContent,
  snapshotProcessedContentCache,
} from './processors/parse'
import { ChangeEvent } from './types/plugin'
import { Argv, BuildCtx } from './util/ctx'
import { emitQuartzDevEvent } from './util/dev-events'
import { glob } from './util/glob'
import { readOutputAssetManifest } from './util/output-assets'
import { FilePath, joinSegments } from './util/path'
import { logBuildStage, PerfTimer } from './util/perf'
import {
  captureBuildInputs,
  commitBuildCache,
  commitResourceBuildCache,
  commitUnchangedParseBuildCache,
  resourceBuildCheckpoint,
  restoreBuildCache,
} from './util/persistent-build-cache'
import { randomIdNonSecure } from './util/random'
import { options } from './util/sourcemap'
import { trace } from './util/trace'

sourceMapSupport.install(options)

type BuildReason = 'initial' | 'source'

function gitCommitShaFor(argv: Argv): string | undefined {
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

  return gitCommitSha
}

export async function buildQuartz(
  config: QuartzConfig,
  argv: Argv,
  mut: Mutex,
  clientRefresh: () => void,
  reason: BuildReason = 'initial',
  scope: BuildScope = 'site',
) {
  const gitCommitSha = gitCommitShaFor(argv)

  const ctx: BuildCtx = {
    buildId: randomIdNonSecure(),
    argv,
    cfg: config,
    allSlugs: [],
    allFiles: [],
    incremental: false,
    gitCommitSha,
  }

  const perf = new PerfTimer()

  const pluginCount = Object.values(config.plugins).flat().length
  const pluginNames = (key: 'transformers' | 'filters' | 'emitters') =>
    config.plugins[key].map(plugin => plugin.name)
  if (argv.verbose) {
    console.log(`Loaded ${pluginCount} plugins`)
    console.log(`  Transformers: ${pluginNames('transformers').join(', ')}`)
    console.log(`  Filters: ${pluginNames('filters').join(', ')}`)
    console.log(`  Emitters: ${pluginNames('emitters').join(', ')}`)
  }

  const initialContent = new Map<FilePath, ProcessedContent>()
  let watching: WatchRuntime | undefined
  const release = await mut.acquire()
  try {
    emitQuartzDevEvent({ type: 'build:start', epoch: ctx.buildId, reason })
    logBuildStage(argv, ctx.buildId, 'build.lock', perf.elapsedMs())
    perf.addEvent('watchers')
    if (argv.watch) watching = await startWatching(ctx, mut, clientRefresh, initialContent)
    logBuildStage(argv, ctx.buildId, 'build.watchers', perf.elapsedMs('watchers'))
    perf.addEvent('glob')
    const discovered = await glob('**', argv.directory, config.configuration.ignorePatterns)
    const allFiles = omitDisabledCourseContent(argv, discovered)
    if (reason === 'initial' && allFiles.length !== discovered.length) {
      console.log(
        `Courses are off in dev (${discovered.length - allFiles.length} files skipped). Set courses to true in quartz/dev-features.ts, or pass --force.`,
      )
    }
    const markdownPaths = allFiles.filter(isMarkdownPath).sort()
    console.log(
      `Found ${markdownPaths.length} input files from \`${argv.directory}\` in ${perf.timeSince('glob')}`,
    )

    const filePaths = markdownPaths.map(fp => joinSegments(argv.directory, fp) as FilePath)
    syncCtxFiles(ctx, allFiles)
    if (scope === 'site' && !argv.watch) await cleanOutputForBuild(ctx, perf, true)

    logBuildStage(argv, ctx.buildId, 'build.glob-and-context', perf.elapsedMs('glob'))
    perf.addEvent('restore')
    const persistentInputs = scope !== 'courses' ? await captureBuildInputs(ctx) : undefined
    const restored =
      reason === 'initial' ? await restoreBuildCache(ctx, persistentInputs) : undefined
    logBuildStage(argv, ctx.buildId, 'build.cache-restore', perf.elapsedMs('restore'))
    if (restored?.unchanged) {
      const published = new Set(restored.published)
      for (const [fp, content] of indexContent(restoredProcessedContent(filePaths))) {
        if (published.has(fp)) initialContent.set(fp, content)
      }
      ctx.outputAssetManifest = await readOutputAssetManifest()
      ctx.incremental = true
      if (watching) watching.ready = true
      console.log(
        `Restored ${initialContent.size} published files; 0 parsed, 0 rendered, 0 output writes`,
      )
      emitQuartzDevEvent({
        type: 'build:ready',
        epoch: ctx.buildId,
        files: markdownPaths.length,
        elapsedMs: perf.elapsedMs(),
      })
      return watching
    }
    perf.addEvent('parse')
    const parsedFiles = await parseMarkdown(ctx, filePaths)
    logBuildStage(argv, ctx.buildId, 'build.parse', perf.elapsedMs('parse'))
    perf.addEvent('filter-index')
    const filteredContent = filterContentResult(ctx, parsedFiles).published
    for (const [fp, content] of indexContent(filteredContent)) initialContent.set(fp, content)
    logBuildStage(argv, ctx.buildId, 'build.filter-index', perf.elapsedMs('filter-index'))
    perf.addEvent('cache-snapshot')
    const reuseParseBlob = Boolean(persistentInputs) && processedContentCacheMisses() === 0
    const parseSnapshot =
      persistentInputs && !reuseParseBlob ? snapshotProcessedContentCache(filePaths) : undefined
    logBuildStage(argv, ctx.buildId, 'build.cache-snapshot', perf.elapsedMs('cache-snapshot'))

    perf.addEvent('emission')
    if (scope === 'courses') {
      globalThis.__quartzSourceEmissionSnapshot = undefined
      ctx.incremental = true
      await emitCourses(ctx, filteredContent)
    } else {
      await emitFullBuild(ctx, perf, filteredContent, scope)
    }
    logBuildStage(argv, ctx.buildId, 'build.emission', perf.elapsedMs('emission'))
    perf.addEvent('cache-commit')
    const published = [...initialContent.keys()]
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
    logBuildStage(argv, ctx.buildId, 'build.cache-commit', perf.elapsedMs('cache-commit'))
    ctx.incremental = argv.watch
    if (watching) watching.ready = true
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
    globalThis.__quartzSourceEmissionSnapshot = undefined
    emitQuartzDevEvent({
      type: 'build:error',
      epoch: ctx.buildId,
      message: describeBuildError(err),
    })
    if (!watching) throw err
    console.error(
      'Build failed; retaining the last output and watching for changes',
      formatBuildError(err),
    )
  } finally {
    release()
  }

  return watching
}

export { sourceRebuildScope } from './plugins/emitters/component-resources/change-classifier'
export { releaseModuleCaches } from './util/module-caches'

/**
 * Rebuilds what a source edit outside the server render path invalidates, without cleaning `public`
 * or re-emitting pages. Dev serves postscript.js and the stylesheets under stable names, so pages keep
 * pointing at the regenerated files.
 */
export async function buildSourceResources(
  argv: Argv,
  mut: Mutex,
  changedPaths: FilePath[],
  scope: { staticFiles: boolean; styles: boolean; scripts: boolean },
) {
  const { default: cfg } = await import('../quartz.config')
  const ctx: BuildCtx = {
    buildId: randomIdNonSecure(),
    argv,
    cfg,
    allSlugs: [],
    allFiles: [],
    incremental: true,
    gitCommitSha: gitCommitShaFor(argv),
  }
  const perf = new PerfTimer()
  const release = await mut.acquire()
  try {
    if (!scope.staticFiles && !scope.styles && !scope.scripts) {
      emitQuartzDevEvent({ type: 'build:start', epoch: ctx.buildId, reason: 'source' })
      emitQuartzDevEvent({
        type: 'build:ready',
        epoch: ctx.buildId,
        files: 0,
        elapsedMs: perf.elapsedMs(),
      })
      return true
    }
    globalThis.__quartzSourceEmissionSnapshot = undefined
    emitQuartzDevEvent({ type: 'build:start', epoch: ctx.buildId, reason: 'source' })
    syncCtxFiles(
      ctx,
      omitDisabledCourseContent(
        argv,
        await glob('**', argv.directory, cfg.configuration.ignorePatterns),
      ),
    )
    const persistentInputs = await captureBuildInputs(ctx)
    const checkpoint = await resourceBuildCheckpoint(ctx, persistentInputs)
    const changeEvents: ChangeEvent[] = await Promise.all(
      changedPaths.map(async fp => ({
        type: (await stat(fp).then(
          () => true,
          () => false,
        ))
          ? ('change' as const)
          : ('delete' as const),
        path: fp,
      })),
    )
    await emitStagedChanges(ctx, async () => {
      if (scope.staticFiles) await emitPartialEmitter(ctx, [], changeEvents, 'Static')
      if (scope.scripts || scope.styles) {
        await emitPartialEmitter(ctx, [], undefined, 'ComponentResources')
      }
      if (scope.scripts) {
        await emitPartialEmitter(ctx, [], undefined, 'LazyScripts')
      }
    })
    await commitResourceBuildCache(ctx, persistentInputs, checkpoint)
    console.log(styleText('green', `Rebuilt source resources in ${perf.timeSince()}`))
    emitQuartzDevEvent({
      type: 'build:ready',
      epoch: ctx.buildId,
      files: 0,
      elapsedMs: perf.elapsedMs(),
    })
    return true
  } catch (err) {
    emitQuartzDevEvent({
      type: 'build:error',
      epoch: ctx.buildId,
      message: describeBuildError(err),
    })
    console.error('Failed to rebuild Quartz source resources', formatBuildError(err))
    return false
  } finally {
    release()
  }
}

export default async (
  argv: Argv,
  mut: Mutex,
  clientRefresh: () => void,
  reason?: BuildReason,
  scope?: BuildScope,
) => {
  try {
    const { default: cfg } = await import('../quartz.config')
    return await buildQuartz(cfg, argv, mut, clientRefresh, reason, scope)
  } catch (err) {
    if (argv.watch) console.error('Build failed; waiting for the next change', err)
    else trace('\nExiting Quartz due to a fatal error', err as Error)
  }
}
