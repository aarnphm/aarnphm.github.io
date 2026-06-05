import path from 'path'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { Argv, BuildCtx } from '../../util/ctx'
import { isFlashcardPath } from '../../util/flashcards-path'
import { glob } from '../../util/glob'
import { emitOutputAsset, removeOutputAsset, type OutputAssetClaim } from '../../util/output-assets'
import { FilePath, joinSegments, slugifyFilePath } from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'

function shouldIgnoreAssetFile(argv: Argv, fp: FilePath): boolean {
  const ext = path.extname(fp).toLowerCase()
  if (ext === '.md' || ext === '.base' || isFlashcardPath(fp)) return true
  return (process.env.CF_PAGES === '1' || argv.watch) && ['.pdf', '.ddl', '.mat'].includes(ext)
}

export function contentAssetFiles(ctx: BuildCtx): FilePath[] {
  return ctx.allFiles.filter(fp => !shouldIgnoreAssetFile(ctx.argv, fp))
}

function contentAssetClaim(argv: Argv, fp: FilePath): OutputAssetClaim {
  const src = joinSegments(argv.directory, fp) as FilePath
  const name = slugifyFilePath(fp)
  const output = joinSegments(argv.output, name) as FilePath
  return { owner: 'content-asset', source: src, output }
}

export function contentAssetClaims(ctx: BuildCtx): OutputAssetClaim[] {
  return contentAssetFiles(ctx).map(fp => contentAssetClaim(ctx.argv, fp))
}

const filesToCopy = async (ctx: BuildCtx): Promise<FilePath[]> => {
  const { argv, cfg } = ctx
  const perf = new PerfTimer()
  if (ctx.allFiles.length > 0) {
    const fps = contentAssetFiles(ctx)
    logBuildSpan(argv, 'assets:scan', `${fps.length} files`, perf.elapsedMs())
    return fps
  }

  const patterns = [
    '**/*.md',
    '**/*.base',
    '**/*.fc',
    '**/*.flashcards',
    ...cfg.configuration.ignorePatterns,
  ]

  if (process.env.CF_PAGES === '1' || argv.watch) {
    patterns.push('**/*.pdf', '**.ddl', '**.mat')
  }

  const fps = await glob('**', argv.directory, patterns)
  logBuildSpan(argv, 'assets:glob', `${fps.length} files`, perf.elapsedMs())
  return fps
}

export const Assets: QuartzEmitterPlugin = () => {
  return {
    name: 'Assets',
    async *emit(ctx) {
      const { argv } = ctx
      const fps = await filesToCopy(ctx)
      const perf = new PerfTimer()
      const files = await mapConcurrent(fps, defaultIoConcurrency, fp =>
        emitOutputAsset(ctx, contentAssetClaim(argv, fp)),
      )
      logBuildSpan(argv, 'assets:copy', `${fps.length} files`, perf.elapsedMs())
      for (const file of files) {
        yield file
      }
    },
    async *partialEmit(ctx, _content, _resources, changeEvents) {
      for (const changeEvent of changeEvents) {
        if (shouldIgnoreAssetFile(ctx.argv, changeEvent.path)) continue

        const claim = contentAssetClaim(ctx.argv, changeEvent.path)

        if (changeEvent.type === 'add' || changeEvent.type === 'change') {
          yield emitOutputAsset(ctx, claim)
        } else if (changeEvent.type === 'delete') {
          await removeOutputAsset(ctx, claim.output)
        }
      }
    },
  }
}
