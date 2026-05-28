import fs from 'node:fs/promises'
import path from 'path'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { batchHardlinkRelativeFiles } from '../../util/batch-hardlink'
import { Argv, BuildCtx } from '../../util/ctx'
import { glob } from '../../util/glob'
import { linkOrCopyFile, useLocalDevLinks } from '../../util/link-or-copy-file'
import { FilePath, joinSegments, slugifyFilePath } from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'

const devAssetLinkConcurrency = 64

type DevAssetPlan = { stable: FilePath[]; slugged: FilePath[] }

const filesToCopy = async (ctx: BuildCtx) => {
  const { argv, cfg } = ctx
  const perf = new PerfTimer()
  const patterns = ['**/*.md', '**/*.base', ...cfg.configuration.ignorePatterns]

  if (process.env.CF_PAGES === '1' || argv.watch) {
    patterns.push('**/*.pdf', '**.ddl', '**.mat')
  }

  const fps = await glob('**', argv.directory, patterns)
  logBuildSpan(argv, 'assets:glob', `${fps.length} files`, perf.elapsedMs())
  return fps
}

const copyFile = async (argv: Argv, fp: FilePath) => {
  const src = joinSegments(argv.directory, fp) as FilePath

  const name = slugifyFilePath(fp)
  const dest = joinSegments(argv.output, name) as FilePath

  return linkOrCopyFile(src, dest, { symlink: useLocalDevLinks(argv) })
}

export function planDevAssetLinks(fps: FilePath[]): DevAssetPlan {
  const stable: FilePath[] = []
  const slugged: FilePath[] = []
  for (const fp of fps) {
    if (String(slugifyFilePath(fp)) === fp) {
      stable.push(fp)
    } else {
      slugged.push(fp)
    }
  }
  return { stable, slugged }
}

async function batchLinkStableAssets(ctx: BuildCtx, fps: FilePath[]): Promise<FilePath[]> {
  const { argv } = ctx
  const perf = new PerfTimer()
  try {
    const files = await batchHardlinkRelativeFiles(argv.directory, argv.output, fps)
    logBuildSpan(argv, 'assets:hardlink', `${fps.length} files`, perf.elapsedMs())
    return files
  } catch {
    const files = await mapConcurrent(fps, devAssetLinkConcurrency, fp =>
      linkOrCopyFile(
        joinSegments(argv.directory, fp) as FilePath,
        joinSegments(argv.output, fp) as FilePath,
        { hardLink: true },
      ),
    )
    logBuildSpan(argv, 'assets:hardlink:fallback', `${fps.length} files`, perf.elapsedMs())
    return files
  }
}

async function emitDevAssets(ctx: BuildCtx, fps: FilePath[]): Promise<FilePath[]> {
  const planPerf = new PerfTimer()
  const plan = planDevAssetLinks(fps)
  logBuildSpan(ctx.argv, 'assets:plan', `${fps.length} files`, planPerf.elapsedMs())
  const symlinkPerf = new PerfTimer()
  const [stableFiles, sluggedFiles] = await Promise.all([
    batchLinkStableAssets(ctx, plan.stable),
    mapConcurrent(plan.slugged, devAssetLinkConcurrency, fp => copyFile(ctx.argv, fp)).then(
      files => {
        logBuildSpan(
          ctx.argv,
          'assets:symlink',
          `${plan.slugged.length} files`,
          symlinkPerf.elapsedMs(),
        )
        return files
      },
    ),
  ])
  return [...stableFiles, ...sluggedFiles]
}

export const Assets: QuartzEmitterPlugin = () => {
  return {
    name: 'Assets',
    async *emit(ctx) {
      const { argv } = ctx
      const fps = await filesToCopy(ctx)
      let files: FilePath[]
      if (useLocalDevLinks(argv)) {
        files = await emitDevAssets(ctx, fps)
      } else {
        const perf = new PerfTimer()
        files = await mapConcurrent(fps, defaultIoConcurrency, fp => copyFile(argv, fp))
        logBuildSpan(argv, 'assets:copy', `${fps.length} files`, perf.elapsedMs())
      }
      for (const file of files) {
        yield file
      }
    },
    async *partialEmit(ctx, _content, _resources, changeEvents) {
      for (const changeEvent of changeEvents) {
        const ext = path.extname(changeEvent.path)
        if (ext === '.md') continue

        if (changeEvent.type === 'add' || changeEvent.type === 'change') {
          yield copyFile(ctx.argv, changeEvent.path)
        } else if (changeEvent.type === 'delete') {
          const name = slugifyFilePath(changeEvent.path)
          const dest = joinSegments(ctx.argv.output, name) as FilePath
          await fs.unlink(dest)
        }
      }
    },
  }
}
