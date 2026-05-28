import { styleText } from 'node:util'
import type { QuartzEmitterPluginInstance } from '../types/plugin'
import { getStaticResourcesFromPlugins } from '../plugins'
import { ProcessedContent } from '../plugins/vfile'
import { defaultEmitterConcurrency, mapConcurrent } from '../util/async-pool'
import { BuildCtx } from '../util/ctx'
import { QuartzLogger } from '../util/log'
import { logBuildSpan, PerfTimer } from '../util/perf'
import { trace } from '../util/trace'

async function runEmitter(
  emitter: QuartzEmitterPluginInstance,
  ctx: BuildCtx,
  content: ProcessedContent[],
  staticResources: ReturnType<typeof getStaticResourcesFromPlugins>,
  log: QuartzLogger,
) {
  let emittedFiles = 0
  const perf = new PerfTimer()
  try {
    const emitted = await emitter.emit(ctx, content, staticResources)
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

export async function emitContent(ctx: BuildCtx, content: ProcessedContent[]) {
  const { argv, cfg } = ctx
  const perf = new PerfTimer()
  const log = new QuartzLogger(ctx.argv.verbose)

  log.start(``)

  let emittedFiles = 0
  const staticResources = getStaticResourcesFromPlugins(ctx)
  const componentResources = cfg.plugins.emitters.find(
    emitter => emitter.name === 'ComponentResources',
  )
  if (componentResources) {
    emittedFiles += await runEmitter(componentResources, ctx, content, staticResources, log)
  }

  const otherEmitters = cfg.plugins.emitters.filter(
    emitter => emitter.name !== 'ComponentResources',
  )
  const counts = await mapConcurrent(otherEmitters, defaultEmitterConcurrency, emitter =>
    runEmitter(emitter, ctx, content, staticResources, log),
  )
  emittedFiles += counts.reduce((sum, count) => sum + count, 0)

  log.end(`Emitted ${emittedFiles} files to \`${argv.output}\` in ${perf.timeSince()}`)
}
