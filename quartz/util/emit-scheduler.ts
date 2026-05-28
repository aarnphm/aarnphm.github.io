import { getStaticResourcesFromPlugins } from '../plugins'
import { ProcessedContent } from '../plugins/vfile'
import { type ChangeEvent, type QuartzEmitterPluginInstance } from '../types/plugin'
import { defaultEmitterConcurrency, mapConcurrent } from './async-pool'
import { BuildCtx } from './ctx'
import { FilePath } from './path'
import { logBuildSpan, PerfTimer } from './perf'
import { pageTitlePatchEvents } from './title-patch'

type EmitResult = FilePath[] | AsyncGenerator<FilePath> | null

const titleOnlyEmitterNames = new Set([
  'ContentPage',
  'ContentIndex',
  'TagPage',
  'FolderPage',
  'BaseViewPage',
  'Masonry',
  'StreamPage',
  'StreamIndex',
  'ArenaPage',
  'SlidesPage',
  'CanvasPage',
  'NotebookViewer',
  'CustomOgImages',
])

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

export async function emitFromPlugin(
  ctx: BuildCtx,
  emitter: QuartzEmitterPluginInstance,
  content: ProcessedContent[],
  resources: ReturnType<typeof getStaticResourcesFromPlugins>,
  changeEvents: ChangeEvent[],
): Promise<number> {
  const perf = new PerfTimer()
  const emitFn = emitter.partialEmit ?? emitter.emit
  const emitted = await emitFn(ctx, content, resources, changeEvents)
  const emittedFiles = await countEmittedFiles(ctx, emitter.name, emitted)
  logBuildSpan(ctx.argv, `emit:${emitter.name}`, `${emittedFiles} files`, perf.elapsedMs())
  return emittedFiles
}

export async function emitAllFromPlugin(
  ctx: BuildCtx,
  emitter: QuartzEmitterPluginInstance,
  content: ProcessedContent[],
  resources: ReturnType<typeof getStaticResourcesFromPlugins>,
): Promise<number> {
  const perf = new PerfTimer()
  const emitted = await emitter.emit(ctx, content, resources)
  const emittedFiles = await countEmittedFiles(ctx, emitter.name, emitted)
  logBuildSpan(ctx.argv, `emit:${emitter.name}`, `${emittedFiles} files`, perf.elapsedMs())
  return emittedFiles
}

export async function emitChangedContent(
  ctx: BuildCtx,
  emitters: QuartzEmitterPluginInstance[],
  content: ProcessedContent[],
  resources: ReturnType<typeof getStaticResourcesFromPlugins>,
  changeEvents: ChangeEvent[],
): Promise<number> {
  const componentResources = emitters.find(emitter => emitter.name === 'ComponentResources')
  const titleOnly = pageTitlePatchEvents(changeEvents) !== undefined
  const otherEmitters = emitters
    .filter(emitter => emitter.name !== 'ComponentResources')
    .filter(emitter => !titleOnly || titleOnlyEmitterNames.has(emitter.name))
  let emittedFiles = 0

  if (componentResources) {
    emittedFiles += await emitFromPlugin(ctx, componentResources, content, resources, changeEvents)
  }

  const counts = titleOnly
    ? await Promise.all(
        otherEmitters.map(emitter =>
          emitFromPlugin(ctx, emitter, content, resources, changeEvents),
        ),
      )
    : await mapConcurrent(otherEmitters, defaultEmitterConcurrency, emitter =>
        emitFromPlugin(ctx, emitter, content, resources, changeEvents),
      )
  return emittedFiles + counts.reduce((sum, count) => sum + count, 0)
}
