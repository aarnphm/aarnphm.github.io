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
import { logBuildSpan, PerfTimer } from '../util/perf'
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
  try {
    const emitted = await output
    if (emitted === null) {
      logBuildSpan(ctx.argv, `emit:${emitter.name}`, `${emittedFiles} files`, perf.elapsedMs())
      return emittedFiles
    }

    if (Symbol.asyncIterator in emitted) {
      for await (const file of emitted) {
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
  logBuildSpan(ctx.argv, `emit:${emitter.name}`, `${emittedFiles} files`, perf.elapsedMs())
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
  return runEmitterOutput(emitter, ctx, log, emitter.emit(ctx, content, staticResources))
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
  const output = emitter.partialEmit
    ? emitter.partialEmit(ctx, content, staticResources, changeEvents)
    : emitter.emit(ctx, content, staticResources)
  return runEmitterOutput(emitter, ctx, log, output)
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
  const staticResources = getStaticResourcesFromPlugins(ctx)
  log.start(``)
  // Without change events the emitter regenerates everything it owns.
  const emittedFiles = changeEvents
    ? await runPartialEmitter(emitter, ctx, content, staticResources, log, changeEvents)
    : await runEmitter(emitter, ctx, content, staticResources, log)
  log.end(`Emitted ${emittedFiles} files to \`${ctx.argv.output}\` in ${perf.timeSince()}`)
}

export async function emitContent(
  ctx: BuildCtx,
  content: ProcessedContent[],
  inputChangeEvents?: ChangeEvent[],
  sourceScope?: { emitters: ReadonlySet<string>; resourceSignature: string },
) {
  const { argv, cfg } = ctx
  await pruneCitationCaches(content)
  let changeEvents =
    inputChangeEvents &&
    transclusionChangeEvents(
      content,
      trackedWahooChangeEvents(content, inputChangeEvents, argv.directory),
      argv.directory,
    )
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
    content = flashcardPages(content, cfg.configuration.baseUrl)
  }
  const perf = new PerfTimer()
  const log = new QuartzLogger(ctx.argv.verbose)
  const partialChanges = ctx.incremental ? changeEvents : undefined

  log.start(``)

  let emittedFiles = 0
  const staticResources = getStaticResourcesFromPlugins(ctx)
  const componentResources = cfg.plugins.emitters.find(
    emitter => emitter.name === 'ComponentResources',
  )
  if (componentResources) {
    emittedFiles += partialChanges
      ? await runPartialEmitter(
          componentResources,
          ctx,
          content,
          staticResources,
          log,
          partialChanges,
        )
      : await runEmitter(componentResources, ctx, content, staticResources, log)
  }

  const resourceSignature = JSON.stringify([
    [...(ctx.assetManifest ?? [])].sort(([a], [b]) => a.localeCompare(b)),
    [...(ctx.extractedStaticResources ?? [])].sort(([a], [b]) => a.localeCompare(b)),
    ctx.staticLeadingJs,
    staticResources.css,
    staticResources.js,
  ])
  const changesExternalResources = cfg.plugins.emitters.some(
    emitter => sourceScope?.emitters.has(emitter.name) && emitter.externalResources,
  )
  const scoped =
    sourceScope && !changesExternalResources && sourceScope.resourceSignature === resourceSignature
  if (sourceScope && !scoped) console.log('Resource identities changed; emitting all owners')
  const otherEmitters = cfg.plugins.emitters.filter(
    emitter =>
      emitter.name !== 'ComponentResources' &&
      (!scoped || emitter.name === 'LazyScripts' || sourceScope.emitters.has(emitter.name)),
  )
  const failures: unknown[] = []
  const counts = await mapConcurrent(otherEmitters, defaultEmitterConcurrency, async emitter => {
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

  log.end(`Emitted ${emittedFiles} files to \`${argv.output}\` in ${perf.timeSince()}`)
  return resourceSignature
}
