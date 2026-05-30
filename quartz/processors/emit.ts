import { styleText } from 'node:util'
import type { ChangeEvent, QuartzEmitterPluginInstance } from '../types/plugin'
import type { FilePath } from '../util/path'
import { ProcessedContent } from '../plugins/vfile'
import { defaultEmitterConcurrency, mapConcurrent } from '../util/async-pool'
import { BuildCtx } from '../util/ctx'
import { QuartzLogger } from '../util/log'
import { logBuildSpan, PerfTimer } from '../util/perf'
import { getStaticResourcesFromPlugins } from '../util/static-resources'
import { trace } from '../util/trace'

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
        if (ctx.argv.verbose) {
          console.log(`[emit:${emitter.name}] ${file}`)
        } else {
          log.updateText(`${emitter.name} -> ${styleText('gray', file)}`)
        }
      }
    } else {
      emittedFiles += emitted.length
      for (const file of emitted) {
        if (ctx.argv.verbose) {
          console.log(`[emit:${emitter.name}] ${file}`)
        } else {
          log.updateText(`${emitter.name} -> ${styleText('gray', file)}`)
        }
      }
    }
  } catch (err) {
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
  const output = emitter.partialEmit
    ? emitter.partialEmit(ctx, content, staticResources, changeEvents)
    : emitter.emit(ctx, content, staticResources)
  return runEmitterOutput(emitter, ctx, log, output)
}

export async function emitContent(
  ctx: BuildCtx,
  content: ProcessedContent[],
  changeEvents?: ChangeEvent[],
) {
  const { argv, cfg } = ctx
  const perf = new PerfTimer()
  const log = new QuartzLogger(ctx.argv.verbose)
  const usePartialEmit = ctx.incremental && changeEvents !== undefined

  log.start(``)

  let emittedFiles = 0
  const staticResources = getStaticResourcesFromPlugins(ctx)
  const componentResources = cfg.plugins.emitters.find(
    emitter => emitter.name === 'ComponentResources',
  )
  if (componentResources) {
    emittedFiles += usePartialEmit
      ? await runPartialEmitter(
          componentResources,
          ctx,
          content,
          staticResources,
          log,
          changeEvents,
        )
      : await runEmitter(componentResources, ctx, content, staticResources, log)
  }

  const otherEmitters = cfg.plugins.emitters.filter(
    emitter => emitter.name !== 'ComponentResources',
  )
  const counts = await mapConcurrent(otherEmitters, defaultEmitterConcurrency, emitter => {
    return usePartialEmit
      ? runPartialEmitter(emitter, ctx, content, staticResources, log, changeEvents)
      : runEmitter(emitter, ctx, content, staticResources, log)
  })
  emittedFiles += counts.reduce((sum, count) => sum + count, 0)

  log.end(`Emitted ${emittedFiles} files to \`${argv.output}\` in ${perf.timeSince()}`)
}
