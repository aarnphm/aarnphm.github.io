import fs from 'node:fs/promises'
import type { ChangeEvent } from '../../types/plugin'
import type { Argv } from '../../util/ctx'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { batchHardlinkRelativeFiles } from '../../util/batch-hardlink'
import { glob } from '../../util/glob'
import { linkOrCopyFile, useLocalDevLinks } from '../../util/link-or-copy-file'
import { FilePath, QUARTZ, joinSegments } from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'

const staticPath = joinSegments(QUARTZ, 'static')
const staticPathPrefix = `${staticPath}/`
const devStaticLinkConcurrency = 64

function staticRelativePath(changeEvent: ChangeEvent): FilePath | undefined {
  if (!changeEvent.path.startsWith(staticPathPrefix)) {
    return undefined
  }

  const relativePath = changeEvent.path.slice(staticPathPrefix.length)
  return relativePath.length > 0 ? (relativePath as FilePath) : undefined
}

async function copyStaticFile(output: string, fp: FilePath, symlink: boolean): Promise<FilePath> {
  const src = joinSegments(staticPath, fp) as FilePath
  const dest = joinSegments(output, 'static', fp) as FilePath
  return linkOrCopyFile(src, dest, { symlink })
}

async function linkStaticFiles(argv: Argv, fps: FilePath[]): Promise<FilePath[]> {
  const outputStaticPath = joinSegments(argv.output, 'static')
  const perf = new PerfTimer()
  try {
    const files = await batchHardlinkRelativeFiles(staticPath, outputStaticPath, fps)
    logBuildSpan(argv, 'static:hardlink', `${fps.length} files`, perf.elapsedMs())
    return files
  } catch {
    const files = await mapConcurrent(fps, devStaticLinkConcurrency, fp =>
      copyStaticFile(argv.output, fp, true),
    )
    logBuildSpan(argv, 'static:hardlink:fallback', `${fps.length} files`, perf.elapsedMs())
    return files
  }
}

export const Static: QuartzEmitterPlugin = () => ({
  name: 'Static',
  async *emit({ argv, cfg }, _content, _resources) {
    const outputStaticPath = joinSegments(argv.output, 'static')
    const globPerf = new PerfTimer()
    const fps = await glob('**', staticPath, cfg.configuration.ignorePatterns)
    logBuildSpan(argv, 'static:glob', `${fps.length} files`, globPerf.elapsedMs())
    await fs.mkdir(outputStaticPath, { recursive: true })
    const symlink = useLocalDevLinks(argv)
    let files: FilePath[]
    if (symlink) {
      files = await linkStaticFiles(argv, fps)
    } else {
      const copyPerf = new PerfTimer()
      files = await mapConcurrent(fps, defaultIoConcurrency, fp =>
        copyStaticFile(argv.output, fp, false),
      )
      logBuildSpan(argv, 'static:copy', `${fps.length} files`, copyPerf.elapsedMs())
    }
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

      yield copyStaticFile(argv.output, fp, useLocalDevLinks(argv))
    }
  },
})
