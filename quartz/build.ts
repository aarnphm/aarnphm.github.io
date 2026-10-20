import { Repository } from '@napi-rs/simple-git'
import { Mutex } from 'async-mutex'
import chokidar from 'chokidar'
import { GlobbyFilterFunction, isGitIgnored } from 'globby'
import { minimatch } from 'minimatch'
import { createHash } from 'node:crypto'
import { once } from 'node:events'
import { stat } from 'node:fs/promises'
import path from 'path'
import sourceMapSupport from 'source-map-support'
import { styleText } from 'util'
import type { QuartzConfig } from './cfg'
import { AGENT_SKILLS_SOURCE_DIRECTORY } from './plugins/emitters/agent-skills'
import { contentAssetClaims } from './plugins/emitters/assets'
import { removeWritten, resetWriteCache } from './plugins/emitters/helpers'
import { resetStaticFileCache, staticAssetClaims } from './plugins/emitters/static'
import { ProcessedContent } from './plugins/vfile'
import { emitContent, emitPartialEmitter } from './processors/emit'
import { filterContentResult } from './processors/filter'
import {
  parseMarkdown,
  processedContentCacheMisses,
  restoredProcessedContent,
  snapshotProcessedContentCache,
} from './processors/parse'
import { ChangeEvent } from './types/plugin'
import { defaultIoConcurrency, mapConcurrent } from './util/async-pool'
import { resetCopyFileCache } from './util/copy-file'
import { courseDirOf } from './util/course'
import { Argv, BuildCtx } from './util/ctx'
import { emitQuartzDevEvent } from './util/dev-events'
import { flashcardsSlug, isFlashcardPath } from './util/flashcards-path'
import { glob, toPosixPath } from './util/glob'
import {
  cleanOutputExcept,
  preservedOutputAssets,
  readOutputAssetManifest,
  relocateOutputAssets,
  writeCurrentOutputAssetManifest,
} from './util/output-assets'
import { FilePath, joinSegments, slugifyFilePath } from './util/path'
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
import { withStagedOutput } from './util/staged-output'
import { trace } from './util/trace'

sourceMapSupport.install(options)

const markdownExtensions = new Set(['.md', '.base', '.canvas'])
const testPathPattern =
  /(?:^|\/)(?:__(?:tests|specs)__|tests?|specs?)(?:\/|$)|(?:^|\/)(?:tests?|specs?|(?:test|spec)_[^/]+|[^/]+_(?:test|spec)|[^/]+\.(?:test|spec))\.[^/]+$/i

const isMarkdownPath = (fp: string): boolean =>
  markdownExtensions.has(path.extname(fp)) || isFlashcardPath(fp)

const isTestPath = (fp: string): boolean => testPathPattern.test(toPosixPath(fp))

let derivedPaths: { source: FilePath[]; markdown: FilePath[]; files: FilePath[] } | undefined

const derivePaths = (argv: Argv, allFiles: FilePath[]) => {
  if (derivedPaths?.source === allFiles) return derivedPaths
  const markdown = allFiles.filter(isMarkdownPath).sort()
  const files = markdown.map(fp => joinSegments(argv.directory, fp) as FilePath)
  derivedPaths = { source: allFiles, markdown, files }
  return derivedPaths
}

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
    ...(await contentAssetClaims(ctx)),
    ...(await staticAssetClaims(ctx.argv.output, ctx.cfg.configuration.ignorePatterns)),
  ]
}

type BuildReason = 'initial' | 'source'
/** `courses` keeps `public` and re-renders only the course pages, for a course source edit. */
type BuildScope = 'site' | 'courses' | { emitters: string[] }
type RebuildQueue = { running: boolean; requested: boolean }
type PendingChanges = Map<FilePath, ChangeEvent['type']>
type BuildData = {
  ctx: BuildCtx
  ignored: GlobbyFilterFunction
  mut: Mutex
  contentMap: Map<FilePath, ProcessedContent>
  pending: PendingChanges
  agentSkillsPending: PendingChanges
  disposed: boolean
}

type WatchRuntime = { ready: boolean; dispose(): Promise<boolean> }

type SourceEmissionSnapshot = {
  key: string
  inputs: string
  content: ProcessedContent[]
  projectedContent: ProcessedContent[]
  baseUrl: string | undefined
  emitters: string
  outputs: Map<string, Set<string>>
  resources: string
}
declare global {
  var __quartzSourceEmissionSnapshot: SourceEmissionSnapshot | undefined
}

function emissionKey(ctx: BuildCtx): string {
  return JSON.stringify([
    path.resolve(ctx.argv.directory),
    path.resolve(ctx.argv.output),
    ctx.gitCommitSha,
  ])
}

async function emissionInputs(ctx: BuildCtx): Promise<string | undefined> {
  const signatures = await mapConcurrent(ctx.allFiles, defaultIoConcurrency, async file => {
    return stat(path.join(ctx.argv.directory, file), { bigint: true }).then(
      info => `${file}:${info.size}:${info.mtimeNs}:${info.ctimeNs}`,
      () => undefined,
    )
  })
  if (signatures.some(signature => signature === undefined)) return undefined
  return createHash('sha256').update(signatures.sort().join('\n')).digest('hex')
}

async function matchingEmissionSnapshot(
  ctx: BuildCtx,
  content: ProcessedContent[],
  inputs: string | undefined,
): Promise<SourceEmissionSnapshot | undefined> {
  const snapshot = globalThis.__quartzSourceEmissionSnapshot
  if (
    !snapshot ||
    inputs === undefined ||
    snapshot.key !== emissionKey(ctx) ||
    snapshot.inputs !== inputs ||
    snapshot.baseUrl !== ctx.cfg.configuration.baseUrl ||
    snapshot.emitters !== JSON.stringify(ctx.cfg.plugins.emitters.map(emitter => emitter.name)) ||
    snapshot.content.length !== content.length ||
    snapshot.content.some((entry, index) => entry !== content[index])
  )
    return undefined
  const files = [...new Set([...snapshot.outputs.values()].flatMap(owned => [...owned]))]
  const present = await mapConcurrent(files, defaultIoConcurrency, async file => {
    if (path.isAbsolute(file) || file === '..' || file.startsWith(`..${path.sep}`)) return false
    return stat(path.join(ctx.argv.output, file)).then(
      info => info.isFile(),
      () => false,
    )
  })
  return present.every(Boolean) ? snapshot : undefined
}

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
  for (const [fullPath, content] of current) {
    const relPath = path.relative(ctx.argv.directory, fullPath) as FilePath
    if (pending.has(relPath) || previous.get(fullPath) === content) continue
    const prev = previous.get(fullPath)
    events.push({
      type: prev ? 'change' : 'add',
      path: relPath,
      file: content[1],
      previousFile: prev?.[1],
    })
  }
  return events
}

/**
 * Output slugs a deleted course page leaves in `public`: its page (ContentPage, FolderPage or PdfReader
 * writes it at the page slug), and the deck page of a `.fc`.
 */
function deletedCourseOutputSlugs(event: ChangeEvent): string[] | undefined {
  const data = event.file?.data
  if (!data?.slug || courseDirOf(data.slug) === null) return undefined
  const sourceSlug = data.flashcards?.sourceSlug
  return sourceSlug ? [data.slug, flashcardsSlug(sourceSlug)] : [data.slug]
}

/** Emitters that render course pages, and the client bundles their components feed. */
const courseEmitters = [
  'ContentPage',
  'FolderPage',
  'PdfReader',
  'FlashcardsPage',
  'CourseReviewPage',
]

async function emitCourses(ctx: BuildCtx, content: ProcessedContent[]): Promise<void> {
  // An add has no previous file, so no emitter skips the page as unchanged.
  const changeEvents: ChangeEvent[] = content.flatMap(([, file]) =>
    file.data.slug && file.data.filePath && courseDirOf(file.data.slug) !== null
      ? [
          {
            type: 'add' as const,
            path: path.relative(ctx.argv.directory, file.data.filePath) as FilePath,
            file,
          },
        ]
      : [],
  )
  await emitPartialEmitter(ctx, content, undefined, 'ComponentResources')
  await emitPartialEmitter(ctx, content, undefined, 'LazyScripts')
  for (const name of courseEmitters) await emitPartialEmitter(ctx, content, changeEvents, name)
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
  delete ctx.staticLeadingJs
  delete ctx.pageResourceCacheBuildId
  delete ctx.pageResourceCache
}

async function cleanOutputForBuild(
  ctx: BuildCtx,
  perf: PerfTimer,
  refreshStaticFiles: boolean,
  publicOutput?: string,
): Promise<void> {
  resetGeneratedResourceState(ctx)
  ctx.outputAssetManifest = await readOutputAssetManifest()
  if (publicOutput) relocateOutputAssets(ctx, publicOutput, ctx.argv.output)
  await syncOutputAssetClaims(ctx, refreshStaticFiles)
  const outputAssetClaims = ctx.outputAssetClaims ?? []

  perf.addEvent('clean')
  if (!publicOutput) emitQuartzDevEvent({ type: 'public:remove:start', epoch: ctx.buildId })
  ctx.outputAssetPreserved = await preservedOutputAssets(ctx.outputAssetManifest, outputAssetClaims)
  await cleanOutputExcept(ctx.argv.output, ctx.outputAssetPreserved)
  resetWriteCache()
  console.log(`Cleaned \`${ctx.argv.output}\` in ${perf.timeSince('clean')}`)
}

async function emitFullBuild(
  ctx: BuildCtx,
  perf: PerfTimer,
  content: ProcessedContent[],
  scope: BuildScope = 'site',
): Promise<void> {
  const publicArgv = ctx.argv
  const stages = new PerfTimer()
  const inputs = publicArgv.watch ? await emissionInputs(ctx) : ''
  logBuildStage(publicArgv, ctx.buildId, 'emission.inputs', stages.elapsedMs())
  stages.addEvent('snapshot')
  const previous =
    typeof scope === 'object' ? await matchingEmissionSnapshot(ctx, content, inputs) : undefined
  let selected =
    typeof scope === 'object' &&
    previous &&
    scope.emitters.length > 0 &&
    scope.emitters.every(name => previous.outputs.has(name))
      ? {
          emitters: new Set(scope.emitters),
          resourceSignature: previous.resources,
          projectedContent: previous.projectedContent,
        }
      : undefined
  if (selected && previous) {
    const requested = selected.emitters
    const selectedPaths = new Set(
      [...requested].flatMap(name => [...(previous.outputs.get(name) ?? [])]),
    )
    const sharesOutput = [...previous.outputs].some(
      ([name, files]) => !requested.has(name) && [...files].some(file => selectedPaths.has(file)),
    )
    if (sharesOutput) selected = undefined
  }
  logBuildStage(publicArgv, ctx.buildId, 'emission.snapshot', stages.elapsedMs('snapshot'))
  globalThis.__quartzSourceEmissionSnapshot = undefined
  let resources = ''
  let projectedContent: ProcessedContent[] | undefined
  stages.addEvent('transaction')
  if (publicArgv.watch) ctx.emittedFiles = new Map()
  if (publicArgv.watch) {
    await withStagedOutput(
      publicArgv.output,
      async stage => {
        ctx.argv = { ...publicArgv, output: stage }
        const setup = new PerfTimer()
        try {
          if (selected) {
            console.log(
              `Reusing output; emitting ${[...selected.emitters].join(', ')} and resources`,
            )
            resetGeneratedResourceState(ctx)
            ctx.outputAssetManifest = await readOutputAssetManifest()
            relocateOutputAssets(ctx, publicArgv.output, stage)
            await syncOutputAssetClaims(ctx, true)
            ctx.outputAssetPreserved = await preservedOutputAssets(
              ctx.outputAssetManifest,
              ctx.outputAssetClaims ?? [],
            )
            resetWriteCache()
            ctx.incremental = true
          } else {
            await cleanOutputForBuild(ctx, perf, true, publicArgv.output)
          }
          logBuildStage(publicArgv, ctx.buildId, 'emission.setup', setup.elapsedMs())
          const emitted = await emitContent(ctx, content, undefined, selected)
          resources = emitted.resources
          projectedContent = emitted.projectedContent
          if (selected && previous && ctx.emittedFiles) {
            const unselectedPaths = new Set(
              [...previous.outputs]
                .filter(([name]) => !ctx.emittedFiles?.has(name))
                .flatMap(([, files]) => [...files]),
            )
            const newOverlap = [...selected.emitters].some(name =>
              [...(ctx.emittedFiles?.get(name) ?? [])].some(file => unselectedPaths.has(file)),
            )
            if (newOverlap) {
              console.log('New output ownership overlaps; emitting all owners')
              resetGeneratedResourceState(ctx)
              ctx.emittedFiles.clear()
              const emitted = await emitContent(ctx, content)
              resources = emitted.resources
              projectedContent = emitted.projectedContent
            }
          }
          if (selected && previous && ctx.emittedFiles) {
            const owners = new Map([...previous.outputs, ...ctx.emittedFiles])
            const retained = new Set([...owners.values()].flatMap(files => [...files]))
            for (const name of ctx.emittedFiles.keys()) {
              for (const file of previous.outputs.get(name) ?? []) {
                if (!retained.has(file)) await removeWritten(ctx, file, '')
              }
            }
          }
        } finally {
          relocateOutputAssets(ctx, stage, publicArgv.output)
          ctx.argv = publicArgv
          resetWriteCache()
          delete ctx.outputAssetPreserved
        }
      },
      ['models'],
      (stage, elapsedMs) => logBuildStage(publicArgv, ctx.buildId, stage, elapsedMs),
    )
  } else {
    await emitContent(ctx, content)
  }
  logBuildStage(publicArgv, ctx.buildId, 'emission.transaction', stages.elapsedMs('transaction'))
  stages.addEvent('manifest')
  await writeCurrentOutputAssetManifest(ctx)
  logBuildStage(publicArgv, ctx.buildId, 'emission.manifest', stages.elapsedMs('manifest'))
  stages.addEvent('save-snapshot')
  delete ctx.outputAssetPreserved
  if (publicArgv.watch && ctx.emittedFiles && inputs !== undefined && projectedContent) {
    globalThis.__quartzSourceEmissionSnapshot = {
      key: emissionKey(ctx),
      inputs,
      content,
      projectedContent,
      baseUrl: ctx.cfg.configuration.baseUrl,
      emitters: JSON.stringify(ctx.cfg.plugins.emitters.map(emitter => emitter.name)),
      outputs: new Map([...(selected && previous ? previous.outputs : []), ...ctx.emittedFiles]),
      resources,
    }
    delete ctx.emittedFiles
  }
  logBuildStage(
    publicArgv,
    ctx.buildId,
    'emission.save-snapshot',
    stages.elapsedMs('save-snapshot'),
  )
}

async function emitStagedChanges(ctx: BuildCtx, emit: () => Promise<void>): Promise<void> {
  const argv = ctx.argv
  ctx.outputAssetManifest ??= await readOutputAssetManifest()
  await withStagedOutput(
    argv.output,
    async stage => {
      ctx.argv = { ...argv, output: stage }
      relocateOutputAssets(ctx, argv.output, stage)
      resetWriteCache()
      resetCopyFileCache()
      try {
        await emit()
      } finally {
        relocateOutputAssets(ctx, stage, argv.output)
        ctx.argv = argv
        resetWriteCache()
        resetCopyFileCache()
      }
    },
    ['models'],
    (stage, elapsedMs) => logBuildStage(argv, ctx.buildId, stage, elapsedMs),
  )
  await writeCurrentOutputAssetManifest(ctx)
}

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
    const allFiles = await glob('**', argv.directory, config.configuration.ignorePatterns)
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
    console.error('Build failed; retaining the last output and watching for changes', err)
  } finally {
    release()
  }

  return watching
}

async function startWatching(
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
    globalThis.__quartzSourceEmissionSnapshot = undefined
    emitQuartzDevEvent({ type: 'build:start', epoch: ctx.buildId, reason: 'source' })
    syncCtxFiles(ctx, await glob('**', argv.directory, cfg.configuration.ignorePatterns))
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
    console.error('Failed to rebuild Quartz source resources', err)
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
