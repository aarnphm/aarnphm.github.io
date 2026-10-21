import path from 'node:path'
import { styleText } from 'node:util'
import type { ChangeEvent, QuartzEmitterPluginInstance } from '../types/plugin'
import type { FilePath } from '../util/path'
import { removeWritten } from '../plugins/emitters/helpers'
import { pruneCitationCaches } from '../plugins/stores/citation-cache'
import { ProcessedContent } from '../plugins/vfile'
import { defaultEmitterConcurrency, mapConcurrent } from '../util/async-pool'
import { BuildCtx, deckDataFor } from '../util/ctx'
import { flashcardPages, flashcardPageChanges } from '../util/flashcards-routing'
import { QuartzLogger } from '../util/log'
import { recordOutput, withOutputOwner } from '../util/output-receipt-context'
import { logBuildSpan, logBuildStage, PerfTimer } from '../util/perf'
import { getStaticResourcesFromPlugins } from '../util/static-resources'
import { trace } from '../util/trace'
import { transclusionChangeEvents } from '../util/transclusion-dependencies'
import { trackedWahooChangeEvents } from '../util/wahoo-tracking'

type EmitterOutput = Promise<FilePath[]> | AsyncGenerator<FilePath> | null

async function runEmitterOutput(
  emitter: QuartzEmitterPluginInstance,
  ctx: BuildCtx,
  log: QuartzLogger,
  output: EmitterOutput,
) {
  let emittedFiles = 0
  const perf = new PerfTimer()
  const spanArgv = ctx.emittedFiles ? { verbose: true } : ctx.argv
  try {
    const emitted = await output
    if (emitted === null) {
      logBuildSpan(spanArgv, `emit:${emitter.name}`, `${emittedFiles} files`, perf.elapsedMs())
      return emittedFiles
    }

    if (Symbol.asyncIterator in emitted) {
      for await (const file of emitted) {
        recordOutput(file, false)
        emittedFiles++
        ctx.emittedFiles?.get(emitter.name)?.add(path.relative(ctx.argv.output, file))
        if (ctx.argv.verbose) {
          console.log(`[emit:${emitter.name}] ${file}`)
        } else {
          log.updateText(`${emitter.name} -> ${styleText('gray', file)}`)
        }
      }
    } else {
      emittedFiles += emitted.length
      for (const file of emitted) {
        recordOutput(file, false)
        ctx.emittedFiles?.get(emitter.name)?.add(path.relative(ctx.argv.output, file))
        if (ctx.argv.verbose) {
          console.log(`[emit:${emitter.name}] ${file}`)
        } else {
          log.updateText(`${emitter.name} -> ${styleText('gray', file)}`)
        }
      }
    }
  } catch (err) {
    if (ctx.argv.watch)
      throw new Error(`Failed to emit from plugin ${emitter.name}`, { cause: err })
    trace(`Failed to emit from plugin \`${emitter.name}\``, err as Error)
  }
  logBuildSpan(spanArgv, `emit:${emitter.name}`, `${emittedFiles} files`, perf.elapsedMs())
  return emittedFiles
}

function runEmitter(
  emitter: QuartzEmitterPluginInstance,
  ctx: BuildCtx,
  content: ProcessedContent[],
  staticResources: ReturnType<typeof getStaticResourcesFromPlugins>,
  log: QuartzLogger,
) {
  ctx.emittedFiles?.set(emitter.name, new Set())
  ctx.decks = deckDataFor(content)
  return withOutputOwner(ctx, content, emitter.name, 'replace', () =>
    runEmitterOutput(emitter, ctx, log, emitter.emit(ctx, content, staticResources)),
  )
}

function runPartialEmitter(
  emitter: QuartzEmitterPluginInstance,
  ctx: BuildCtx,
  content: ProcessedContent[],
  staticResources: ReturnType<typeof getStaticResourcesFromPlugins>,
  log: QuartzLogger,
  changeEvents: ChangeEvent[],
) {
  ctx.decks = deckDataFor(content)
  return withOutputOwner(
    ctx,
    content,
    emitter.name,
    emitter.partialEmit ? 'patch' : 'replace',
    () => {
      const output = emitter.partialEmit
        ? emitter.partialEmit(ctx, content, staticResources, changeEvents)
        : emitter.emit(ctx, content, staticResources)
      return runEmitterOutput(emitter, ctx, log, output)
    },
  )
}

export async function emitPartialEmitter(
  ctx: BuildCtx,
  content: ProcessedContent[],
  changeEvents: ChangeEvent[] | undefined,
  emitterName: string,
): Promise<void> {
  const emitter = ctx.cfg.plugins.emitters.find(candidate => candidate.name === emitterName)
  if (!emitter) throw new Error(`missing emitter: ${emitterName}`)

  if (changeEvents) {
    const projected = flashcardPageChanges(content, changeEvents, ctx.cfg.configuration.baseUrl)
    content = projected.content
    changeEvents = projected.changes
  } else {
    content = flashcardPages(content, ctx.cfg.configuration.baseUrl)
  }
  const perf = new PerfTimer()
  const log = new QuartzLogger(ctx.argv.verbose)
  const staticResources = getStaticResourcesFromPlugins(ctx, content)
  log.start(``)
  // Without change events the emitter regenerates everything it owns.
  const emittedFiles = changeEvents
    ? await runPartialEmitter(emitter, ctx, content, staticResources, log, changeEvents)
    : await runEmitter(emitter, ctx, content, staticResources, log)
  log.end(`Emitted ${emittedFiles} files to \`${ctx.argv.output}\` in ${perf.timeSince()}`)
}

const RESOURCE_INDEPENDENT_EMITTERS = new Set<string>([
  'CustomOgImages',
  'LazyScripts',
  'LLMText',
  'Assets',
  'Static',
  'AgentSkills',
  'AliasRedirects',
  'Bibliography',
  'SemanticIndex',
  'Favicon',
  'NotebookPagesManifest',
  'CNAME',
])

const EMITTER_PRIORITY = new Map<string, number>(
  [
    'CustomOgImages',
    'LazyScripts',
    'LLMText',
    'Assets',
    'Static',
    'AgentSkills',
    'AliasRedirects',
    'Bibliography',
    'SemanticIndex',
    'Favicon',
    'NotebookPagesManifest',
    'CNAME',
    'ArenaPage',
    'NotebookViewer',
    'Strava',
    'ContentPage',
    'ContentIndex',
    'StreamIndex',
    'FolderPage',
    'TagPage',
    'BaseViewPage',
    'FlashcardsPage',
    'PdfReader',
    'SlidesPage',
    'Masonry',
    'CanvasPage',
    'CourseReviewPage',
    'StreamPage',
    'NotFoundPage',
    'DisabledPage',
    'EmailEmitter',
  ].map((name, index) => [name, index]),
)

function sortEmittersByPriority(
  emitters: readonly QuartzEmitterPluginInstance[],
): QuartzEmitterPluginInstance[] {
  return emitters
    .map((emitter, index) => ({
      emitter,
      index,
      priority: EMITTER_PRIORITY.get(emitter.name) ?? EMITTER_PRIORITY.size + index,
    }))
    .sort((left, right) => left.priority - right.priority || left.index - right.index)
    .map(entry => entry.emitter)
}

export async function emitContent(
  ctx: BuildCtx,
  content: ProcessedContent[],
  inputChangeEvents?: ChangeEvent[],
  sourceScope?: {
    emitters: ReadonlySet<string>
    resourceSignature: string
    projectedContent?: ProcessedContent[]
  },
) {
  const { argv, cfg } = ctx
  const stages = new PerfTimer()
  await pruneCitationCaches(content)
  logBuildStage(argv, ctx.buildId, 'emit.citation-pruning', stages.elapsedMs())
  stages.addEvent('dependencies')
  let changeEvents =
    inputChangeEvents &&
    transclusionChangeEvents(
      content,
      trackedWahooChangeEvents(content, inputChangeEvents, argv.directory),
      argv.directory,
    )
  logBuildStage(argv, ctx.buildId, 'emit.change-dependencies', stages.elapsedMs('dependencies'))
  stages.addEvent('projection')
  if (changeEvents) {
    const projected = flashcardPageChanges(content, changeEvents, cfg.configuration.baseUrl)
    content = projected.content
    changeEvents = projected.changes
    for (const slug of projected.removed) {
      await removeWritten(ctx, slug, '.html')
      await removeWritten(ctx, slug, '.md')
      await removeWritten(ctx, `${slug}-og-image`, '.webp')
    }
  } else {
    content = sourceScope?.projectedContent ?? flashcardPages(content, cfg.configuration.baseUrl)
  }
  const perf = new PerfTimer()
  const log = new QuartzLogger(ctx.argv.verbose)
  logBuildStage(argv, ctx.buildId, 'emit.flashcard-projection', stages.elapsedMs('projection'))
  const partialChanges = ctx.incremental ? changeEvents : undefined

  log.start(``)

  let emittedFiles = 0
  const failures: unknown[] = []
  const staticResources = getStaticResourcesFromPlugins(ctx, content)
  const componentResources = cfg.plugins.emitters.find(
    emitter => emitter.name === 'ComponentResources',
  )
  let componentResourcesFailed = false
  const runComponentResources = async () => {
    try {
      const count = componentResources
        ? partialChanges
          ? await runPartialEmitter(
              componentResources,
              ctx,
              content,
              staticResources,
              log,
              partialChanges,
            )
          : await runEmitter(componentResources, ctx, content, staticResources, log)
        : 0
      logBuildStage(argv, ctx.buildId, 'emit.component-resources', perf.elapsedMs())
      return count
    } catch (error) {
      componentResourcesFailed = true
      failures.push(error)
      return 0
    }
  }

  const computeResourceSignature = () =>
    JSON.stringify([
      [...(ctx.assetManifest ?? [])].sort(([a], [b]) => a.localeCompare(b)),
      [...(ctx.extractedStaticResources ?? [])].sort(([a], [b]) => a.localeCompare(b)),
      ctx.staticLeadingJs,
      staticResources.css,
      staticResources.js,
    ])

  let resourceSignature = ''
  if (sourceScope) {
    emittedFiles += await runComponentResources()
    stages.addEvent('other-emitters')
    resourceSignature = computeResourceSignature()
    const changesExternalResources = cfg.plugins.emitters.some(
      emitter => sourceScope.emitters.has(emitter.name) && emitter.externalResources,
    )
    const scoped = !changesExternalResources && sourceScope.resourceSignature === resourceSignature
    if (!scoped) console.log('Resource identities changed; emitting all owners')
    const otherEmitters = sortEmittersByPriority(
      cfg.plugins.emitters.filter(
        emitter =>
          emitter.name !== 'ComponentResources' &&
          (!scoped || emitter.name === 'LazyScripts' || sourceScope.emitters.has(emitter.name)),
      ),
    )
    const counts = await mapConcurrent(otherEmitters, defaultEmitterConcurrency, async emitter => {
      if (componentResourcesFailed) return 0
      try {
        return await (partialChanges
          ? runPartialEmitter(emitter, ctx, content, staticResources, log, partialChanges)
          : runEmitter(emitter, ctx, content, staticResources, log))
      } catch (error) {
        failures.push(error)
        return 0
      }
    })
    if (failures.length > 0) throw new AggregateError(failures, 'Failed to emit Quartz output')
    emittedFiles += counts.reduce((sum, count) => sum + count, 0)
    logBuildStage(argv, ctx.buildId, 'emit.other-owners', stages.elapsedMs('other-emitters'))
  } else {
    stages.addEvent('other-emitters')
    const componentResourcesPromise = runComponentResources()
    const otherEmitters = sortEmittersByPriority(
      cfg.plugins.emitters.filter(emitter => emitter.name !== 'ComponentResources'),
    )
    const counts = await mapConcurrent(otherEmitters, defaultEmitterConcurrency, async emitter => {
      if (!RESOURCE_INDEPENDENT_EMITTERS.has(emitter.name)) {
        await componentResourcesPromise
        if (componentResourcesFailed) return 0
      }
      try {
        return await (partialChanges
          ? runPartialEmitter(emitter, ctx, content, staticResources, log, partialChanges)
          : runEmitter(emitter, ctx, content, staticResources, log))
      } catch (error) {
        failures.push(error)
        return 0
      }
    })
    const componentResourcesCount = await componentResourcesPromise
    if (failures.length > 0) throw new AggregateError(failures, 'Failed to emit Quartz output')
    emittedFiles += componentResourcesCount + counts.reduce((sum, count) => sum + count, 0)
    resourceSignature = computeResourceSignature()
    logBuildStage(argv, ctx.buildId, 'emit.other-owners', stages.elapsedMs('other-emitters'))
  }

  log.end(`Emitted ${emittedFiles} files to \`${argv.output}\` in ${perf.timeSince()}`)
  return { resources: resourceSignature, projectedContent: content }
}
