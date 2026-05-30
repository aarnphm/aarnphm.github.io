import fs from 'node:fs/promises'
import type { ChangeEvent } from '../../types/plugin'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { copyFile as copyFileFast } from '../../util/copy-file'
import { glob } from '../../util/glob'
import { FilePath, QUARTZ, joinSegments } from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'

const staticPath = joinSegments(QUARTZ, 'static')
const staticPathPrefix = `${staticPath}/`

function staticRelativePath(changeEvent: ChangeEvent): FilePath | undefined {
  if (!changeEvent.path.startsWith(staticPathPrefix)) {
    return undefined
  }

  const relativePath = changeEvent.path.slice(staticPathPrefix.length)
  return relativePath.length > 0 ? (relativePath as FilePath) : undefined
}

async function copyStaticFile(output: string, fp: FilePath): Promise<FilePath> {
  const src = joinSegments(staticPath, fp) as FilePath
  const dest = joinSegments(output, 'static', fp) as FilePath
  return copyFileFast(src, dest)
}

export const Static: QuartzEmitterPlugin = () => ({
  name: 'Static',
  async *emit({ argv, cfg }, _content, _resources) {
    const outputStaticPath = joinSegments(argv.output, 'static')
    const globPerf = new PerfTimer()
    const fps = await glob('**', staticPath, cfg.configuration.ignorePatterns)
    logBuildSpan(argv, 'static:glob', `${fps.length} files`, globPerf.elapsedMs())
    await fs.mkdir(outputStaticPath, { recursive: true })
    const copyPerf = new PerfTimer()
    const files = await mapConcurrent(fps, defaultIoConcurrency, fp =>
      copyStaticFile(argv.output, fp),
    )
    logBuildSpan(argv, 'static:copy', `${fps.length} files`, copyPerf.elapsedMs())
    for (const file of files) {
      yield file
    }
  },
  async *partialEmit({ argv }, _content, _resources, changeEvents) {
    for (const changeEvent of changeEvents) {
      const fp = staticRelativePath(changeEvent)
      if (!fp) {
        continue
      }

      if (changeEvent.type === 'delete') {
        const dest = joinSegments(argv.output, 'static', fp) as FilePath
        await fs.rm(dest, { force: true })
        continue
      }

      yield copyStaticFile(argv.output, fp)
    }
  },
})
