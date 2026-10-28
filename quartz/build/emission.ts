import { createHash } from 'node:crypto'
import { stat } from 'node:fs/promises'
import path from 'path'
import { contentAssetClaims } from '../plugins/emitters/assets'
import { removeWritten, resetWriteCache } from '../plugins/emitters/helpers'
import { resetStaticFileCache, staticAssetClaims } from '../plugins/emitters/static'
import { ProcessedContent } from '../plugins/vfile'
import { emitContent, emitPartialEmitter } from '../processors/emit'
import { ChangeEvent } from '../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../util/async-pool'
import { resetCopyFileCache } from '../util/copy-file'
import { courseDirOf } from '../util/course'
import { BuildCtx } from '../util/ctx'
import { emitQuartzDevEvent } from '../util/dev-events'
import {
  cleanOutputExcept,
  preservedOutputAssets,
  readOutputAssetManifest,
  relocateOutputAssets,
  writeCurrentOutputAssetManifest,
} from '../util/output-assets'
import { FilePath } from '../util/path'
import { logBuildStage, PerfTimer } from '../util/perf'
import { withStagedOutput } from '../util/staged-output'

async function syncOutputAssetClaims(ctx: BuildCtx, refreshStaticFiles = false): Promise<void> {
  if (refreshStaticFiles) {
    resetStaticFileCache()
  }
  ctx.outputAssetClaims = [
    ...(await contentAssetClaims(ctx)),
    ...(await staticAssetClaims(ctx.argv.output, ctx.cfg.configuration.ignorePatterns)),
  ]
}

/** `courses` keeps `public` and re-renders only the course pages, for a course source edit. */
export type BuildScope = 'site' | 'courses' | { emitters: string[] }

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

/** Emitters that render course pages, and the client bundles their components feed. */
const courseEmitters = [
  'ContentPage',
  'FolderPage',
  'PdfReader',
  'FlashcardsPage',
  'CourseReviewPage',
]

export async function emitCourses(ctx: BuildCtx, content: ProcessedContent[]): Promise<void> {
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

function resetGeneratedResourceState(ctx: BuildCtx): void {
  delete ctx.assetManifest
  delete ctx.extractedStaticResources
  delete ctx.staticLeadingJs
  delete ctx.pageResourceCacheBuildId
  delete ctx.pageResourceCache
}

export async function cleanOutputForBuild(
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

export async function emitFullBuild(
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

export async function emitStagedChanges(ctx: BuildCtx, emit: () => Promise<void>): Promise<void> {
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
