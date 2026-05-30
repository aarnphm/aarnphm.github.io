import fs from 'node:fs/promises'
import path from 'path'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { copyFile as copyFileFast } from '../../util/copy-file'
import { Argv, BuildCtx } from '../../util/ctx'
import { glob } from '../../util/glob'
import { FilePath, joinSegments, slugifyFilePath } from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'

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
  return copyFileFast(src, dest)
}

export const Assets: QuartzEmitterPlugin = () => {
  return {
    name: 'Assets',
    async *emit(ctx) {
      const { argv } = ctx
      const fps = await filesToCopy(ctx)
      const perf = new PerfTimer()
      const files = await mapConcurrent(fps, defaultIoConcurrency, fp => copyFile(argv, fp))
      logBuildSpan(argv, 'assets:copy', `${fps.length} files`, perf.elapsedMs())
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
