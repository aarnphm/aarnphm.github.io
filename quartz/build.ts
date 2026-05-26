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
import { getStaticResourcesFromPlugins } from './plugins'
import { ProcessedContent } from './plugins/vfile'
import { emitContent } from './processors/emit'
import { filterContentResult, type FilterContentResult } from './processors/filter'
import { parseMarkdown } from './processors/parse'
import { ChangeEvent, QuartzEmitterPluginInstance } from './types/plugin'
import { affectedComponentPageEmitters } from './util/component-source'
import { Argv, BuildCtx } from './util/ctx'
import { glob, toPosixPath } from './util/glob'
import { FilePath, joinSegments, slugifyFilePath } from './util/path'
import { PerfTimer } from './util/perf'
import { randomIdNonSecure } from './util/random'
import { options } from './util/sourcemap'
import { trace } from './util/trace'

sourceMapSupport.install(options)

type ContentMap = Map<FilePath, { type: 'markdown'; content: ProcessedContent } | { type: 'other' }>

const markdownExtensions = new Set(['.md', '.base', '.canvas'])

const isMarkdownPath = (fp: string): boolean => markdownExtensions.has(path.extname(fp))

const syncCtxFiles = (ctx: BuildCtx, contentMap: ContentMap) => {
  ctx.allFiles = Array.from(contentMap.keys())
  ctx.allSlugs = ctx.allFiles.map(fp => slugifyFilePath(fp))
}

const seedContentMap = (allFiles: FilePath[], content: ProcessedContent[]): ContentMap => {
  const contentMap: ContentMap = new Map()
  for (const filePath of allFiles) {
    if (!isMarkdownPath(filePath)) {
      contentMap.set(filePath, { type: 'other' })
    }
  }

  for (const item of content) {
    const relativePath = item[1].data.relativePath
    if (relativePath) {
      contentMap.set(relativePath, { type: 'markdown', content: item })
    }
  }

  return contentMap
}

const markdownContentByPath = (content: ProcessedContent[]): Map<FilePath, ProcessedContent> => {
  const byPath = new Map<FilePath, ProcessedContent>()
  for (const item of content) {
    const relativePath = item[1].data.relativePath
    if (relativePath) {
      byPath.set(relativePath, item)
    }
  }
  return byPath
}

const publishedMarkdownContent = (contentMap: ContentMap): ProcessedContent[] =>
  Array.from(contentMap.values()).flatMap(file => (file.type === 'markdown' ? [file.content] : []))

type BuildData = {
  ctx: BuildCtx
  ignored: GlobbyFilterFunction
  mut: Mutex
  contentMap: ContentMap
  changesSinceLastBuild: Record<FilePath, ChangeEvent['type']>
  lastBuildMs: number
}

type WatchHandoff = {
  ctx: BuildCtx
  contentMap: ContentMap
  changesSinceLastBuild: Record<FilePath, ChangeEvent['type']>
  pendingChanges: ChangeEvent[]
}

type WatchRuntime = { dispose(): Promise<void>; handoff(): WatchHandoff }

type EmitResult = FilePath[] | AsyncGenerator<FilePath> | null

async function countEmittedFiles(
  ctx: BuildCtx,
  emitterName: string,
  emitted: EmitResult,
): Promise<number> {
  if (emitted === null) return 0

  if (Symbol.asyncIterator in emitted) {
    let emittedFiles = 0
    for await (const file of emitted) {
      emittedFiles++
      if (ctx.argv.verbose) {
        console.log(`[emit:${emitterName}] ${file}`)
      }
    }
    return emittedFiles
  }

  if (ctx.argv.verbose) {
    for (const file of emitted) {
      console.log(`[emit:${emitterName}] ${file}`)
    }
  }
  return emitted.length
}

async function emitFromPlugin(
  ctx: BuildCtx,
  emitter: QuartzEmitterPluginInstance,
  content: ProcessedContent[],
  resources: ReturnType<typeof getStaticResourcesFromPlugins>,
  changeEvents: ChangeEvent[],
): Promise<number> {
  const emitFn = emitter.partialEmit ?? emitter.emit
  const emitted = await emitFn(ctx, content, resources, changeEvents)
  return countEmittedFiles(ctx, emitter.name, emitted)
}

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
  perf.addEvent('clean')
  await rm(output, { recursive: true, force: true })
  console.log(`Removed \`${output}\` in ${perf.timeSince('clean')}`)

  perf.addEvent('glob')
  const allFiles = await glob('**/*.*', argv.directory, cfg.configuration.ignorePatterns)
  const markdownPaths = allFiles
    .filter(fp => fp.endsWith('.md') || fp.endsWith('.base') || fp.endsWith('.canvas'))
    .sort()
  console.log(
    `Found ${markdownPaths.length} input files from \`${argv.directory}\` in ${perf.timeSince('glob')}`,
  )

  const filePaths = markdownPaths.map(fp => joinSegments(argv.directory, fp) as FilePath)
  ctx.allFiles = allFiles
  ctx.allSlugs = allFiles.map(fp => slugifyFilePath(fp as FilePath))

  const parsedFiles = await parseMarkdown(ctx, filePaths)
  const filteredContent = filterContentResult(ctx, parsedFiles).published
  const contentMap = seedContentMap(allFiles, filteredContent)
  syncCtxFiles(ctx, contentMap)

  await emitContent(ctx, filteredContent)
  console.log(
    styleText('green', `Done processing ${markdownPaths.length} files in ${perf.timeSince()}`),
  )
  release()

  if (argv.watch) {
    ctx.incremental = true
    return startWatching(ctx, mut, filteredContent, clientRefresh)
  }
}

async function startWatching(
  ctx: BuildCtx,
  mut: Mutex,
  initialContent: ProcessedContent[],
  clientRefresh: () => void,
): Promise<WatchRuntime> {
  const contentMap = seedContentMap(ctx.allFiles, initialContent)
  return startWatchingContentMap(ctx, mut, contentMap, clientRefresh, {})
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

async function startWatchingContentMap(
  ctx: BuildCtx,
  mut: Mutex,
  contentMap: ContentMap,
  clientRefresh: () => void,
  changesSinceLastBuild: Record<FilePath, ChangeEvent['type']>,
): Promise<WatchRuntime> {
  const { argv } = ctx
  const ignored = await createIgnoredFilter(ctx)
  const buildData: BuildData = {
    ctx,
    mut,
    contentMap,
    ignored,
    changesSinceLastBuild,
    lastBuildMs: 0,
  }

  const watcher = chokidar.watch('.', {
    awaitWriteFinish: { stabilityThreshold: 250 },
    persistent: true,
    cwd: argv.directory,
    ignored: buildData.ignored,
    ignoreInitial: true,
  })

  const changes: ChangeEvent[] = []
  watcher
    .on('add', fp => {
      if (buildData.ignored(fp)) return
      changes.push({ path: fp as FilePath, type: 'add' })
      void rebuild(changes, clientRefresh, buildData)
    })
    .on('change', fp => {
      if (buildData.ignored(fp)) return
      changes.push({ path: fp as FilePath, type: 'change' })
      void rebuild(changes, clientRefresh, buildData)
    })
    .on('unlink', fp => {
      if (buildData.ignored(fp)) return
      changes.push({ path: fp as FilePath, type: 'delete' })
      void rebuild(changes, clientRefresh, buildData)
    })

  return {
    async dispose() {
      await watcher.close()
    },
    handoff() {
      return { ctx, contentMap, changesSinceLastBuild, pendingChanges: changes.slice() }
    },
  }
}

async function rebuild(changes: ChangeEvent[], clientRefresh: () => void, buildData: BuildData) {
  const { ctx, contentMap, mut, changesSinceLastBuild } = buildData
  const { argv, cfg } = ctx

  const buildId = randomIdNonSecure()
  ctx.buildId = buildId
  buildData.lastBuildMs = new Date().getTime()
  const numChangesInBuild = changes.length
  const release = await mut.acquire()

  // if there's another build after us, release and let them do it
  if (ctx.buildId !== buildId) {
    release()
    return
  }

  const perf = new PerfTimer()
  perf.addEvent('rebuild')
  console.log(styleText('yellow', 'Detected change, rebuilding...'))

  // update changesSinceLastBuild
  for (const change of changes) {
    changesSinceLastBuild[change.path] = change.type
  }
  const pendingChanges = Object.entries(changesSinceLastBuild) as Array<
    [FilePath, ChangeEvent['type']]
  >

  const staticResources = getStaticResourcesFromPlugins(ctx)
  const pathsToParse: FilePath[] = []
  for (const [fp, type] of pendingChanges) {
    if (type === 'delete' || !isMarkdownPath(fp)) continue
    const fullPath = joinSegments(argv.directory, toPosixPath(fp)) as FilePath
    pathsToParse.push(fullPath)
  }

  const previousMarkdown = new Map<FilePath, ProcessedContent>()
  for (const [fp, file] of contentMap) {
    if (file.type === 'markdown') {
      previousMarkdown.set(fp, file.content)
    }
  }

  const parsed = await parseMarkdown(ctx, pathsToParse)
  const parsedResult: FilterContentResult =
    parsed.length > 0 ? filterContentResult(ctx, parsed) : { published: [], removed: [] }
  const publishedByPath = markdownContentByPath(parsedResult.published)

  const changeEvents: ChangeEvent[] = []
  for (const [path, type] of pendingChanges) {
    if (isMarkdownPath(path)) {
      const previous = previousMarkdown.get(path)

      if (type === 'delete') {
        contentMap.delete(path)
        if (previous) {
          changeEvents.push({ type: 'delete', path, file: previous[1] })
        }
        continue
      }

      const published = publishedByPath.get(path)
      if (published) {
        contentMap.set(path, { type: 'markdown', content: published })
        changeEvents.push({ type: previous ? 'change' : 'add', path, file: published[1] })
        continue
      }

      contentMap.delete(path)
      if (previous) {
        changeEvents.push({ type: 'delete', path, file: previous[1] })
      }
      continue
    }

    if (type === 'delete') {
      contentMap.delete(path)
    } else if (type === 'add') {
      contentMap.set(path, { type: 'other' })
    }
    changeEvents.push({ type, path })
  }

  syncCtxFiles(ctx, contentMap)
  const processedFiles = publishedMarkdownContent(contentMap)

  if (changeEvents.length === 0) {
    console.log(`Emitted 0 files to \`${argv.output}\` in ${perf.timeSince('rebuild')}`)
    console.log(styleText('green', `Done rebuilding in ${perf.timeSince()}`))
    for (const [fp] of pendingChanges) {
      delete changesSinceLastBuild[fp]
    }
    changes.splice(0, numChangesInBuild)
    release()
    return
  }

  let emittedFiles = 0
  for (const emitter of cfg.plugins.emitters) {
    emittedFiles += await emitFromPlugin(
      ctx,
      emitter,
      processedFiles,
      staticResources,
      changeEvents,
    )
  }

  console.log(`Emitted ${emittedFiles} files to \`${argv.output}\` in ${perf.timeSince('rebuild')}`)
  console.log(styleText('green', `Done rebuilding in ${perf.timeSince()}`))
  for (const [fp] of pendingChanges) {
    delete changesSinceLastBuild[fp]
  }
  changes.splice(0, numChangesInBuild)
  clientRefresh()
  release()
}

async function emitComponentSourceChanges(
  ctx: BuildCtx,
  contentMap: ContentMap,
  changeEvents: ChangeEvent[],
): Promise<number> {
  syncCtxFiles(ctx, contentMap)
  const content = publishedMarkdownContent(contentMap)
  const resources = getStaticResourcesFromPlugins(ctx)
  const emitters = ctx.cfg.plugins.emitters
  const pageEmitterPlan = affectedComponentPageEmitters(ctx, emitters, changeEvents)
  let emittedFiles = 0

  const componentResources = emitters.find(emitter => emitter.name === 'ComponentResources')
  if (componentResources) {
    emittedFiles += await emitFromPlugin(ctx, componentResources, content, resources, changeEvents)
  }

  for (const emitter of emitters) {
    if (emitter.name === 'ComponentResources') continue
    if (!emitter.getQuartzComponents) continue
    if (!pageEmitterPlan.all && !pageEmitterPlan.names.has(emitter.name)) continue
    emittedFiles += await countEmittedFiles(
      ctx,
      emitter.name,
      await emitter.emit(ctx, content, resources),
    )
  }

  return emittedFiles
}

export async function rebuildQuartzSource(
  state: WatchHandoff,
  mut: Mutex,
  clientRefresh: () => void,
  changeEvents: ChangeEvent[],
): Promise<WatchRuntime> {
  const { ctx, contentMap, changesSinceLastBuild } = state
  ctx.cfg = cfg
  ctx.incremental = true
  ctx.buildId = randomIdNonSecure()

  const release = await mut.acquire()
  const perf = new PerfTimer()
  perf.addEvent('rebuild')
  console.log(styleText('yellow', 'Detected change, rebuilding...'))

  let emittedFiles = 0
  try {
    emittedFiles = await emitComponentSourceChanges(ctx, contentMap, changeEvents)
  } finally {
    release()
  }

  console.log(
    `Emitted ${emittedFiles} files to \`${ctx.argv.output}\` in ${perf.timeSince('rebuild')}`,
  )
  console.log(styleText('green', `Done rebuilding in ${perf.timeSince()}`))
  clientRefresh()
  return startWatchingContentMap(ctx, mut, contentMap, clientRefresh, changesSinceLastBuild)
}

export default async (argv: Argv, mut: Mutex, clientRefresh: () => void) => {
  try {
    return await buildQuartz(argv, mut, clientRefresh)
  } catch (err) {
    trace('\nExiting Quartz due to a fatal error', err as Error)
  }
}
