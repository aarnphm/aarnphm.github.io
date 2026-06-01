import type { ChangeEvent } from '../../types/plugin'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { glob } from '../../util/glob'
import { emitOutputAsset, removeOutputAsset, type OutputAssetClaim } from '../../util/output-assets'
import { FilePath, QUARTZ, joinSegments } from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'

const staticPath = joinSegments(QUARTZ, 'static')
const staticPathPrefix = `${staticPath}/`
let staticFileCache: FilePath[] | undefined

function staticRelativePath(changeEvent: ChangeEvent): FilePath | undefined {
  if (!changeEvent.path.startsWith(staticPathPrefix)) {
    return undefined
  }

  const relativePath = changeEvent.path.slice(staticPathPrefix.length)
  return relativePath.length > 0 ? (relativePath as FilePath) : undefined
}

function staticAssetClaim(output: string, fp: FilePath): OutputAssetClaim {
  const src = joinSegments(staticPath, fp) as FilePath
  const dest = joinSegments(output, 'static', fp) as FilePath
  return { owner: 'quartz-static', source: src, output: dest }
}

export function resetStaticFileCache(): void {
  staticFileCache = undefined
}

async function staticFiles(ignorePatterns: string[]): Promise<FilePath[]> {
  staticFileCache ??= await glob('**', staticPath, ignorePatterns)
  return staticFileCache
}

export async function staticAssetClaims(output: string, ignorePatterns: string[]) {
  const fps = await staticFiles(ignorePatterns)
  return fps.map(fp => staticAssetClaim(output, fp))
}

export const Static: QuartzEmitterPlugin = () => ({
  name: 'Static',
  async *emit(ctx, _content, _resources) {
    const { argv, cfg } = ctx
    const globPerf = new PerfTimer()
    const fps = await staticFiles(cfg.configuration.ignorePatterns)
    logBuildSpan(argv, 'static:glob', `${fps.length} files`, globPerf.elapsedMs())
    const copyPerf = new PerfTimer()
    const files = await mapConcurrent(fps, defaultIoConcurrency, fp =>
      emitOutputAsset(ctx, staticAssetClaim(argv.output, fp)),
    )
    logBuildSpan(argv, 'static:copy', `${fps.length} files`, copyPerf.elapsedMs())
    for (const file of files) {
      yield file
    }
  },
  async *partialEmit(ctx, _content, _resources, changeEvents) {
    const { argv } = ctx
    for (const changeEvent of changeEvents) {
      const fp = staticRelativePath(changeEvent)
      if (!fp) {
        continue
      }
      resetStaticFileCache()

      if (changeEvent.type === 'delete') {
        const dest = joinSegments(argv.output, 'static', fp) as FilePath
        await removeOutputAsset(ctx, dest)
        continue
      }

      yield emitOutputAsset(ctx, staticAssetClaim(argv.output, fp))
    }
  },
})
