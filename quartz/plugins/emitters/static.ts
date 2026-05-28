import fs from 'node:fs/promises'
import type { ChangeEvent } from '../../types/plugin'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { glob } from '../../util/glob'
import { linkOrCopyFile } from '../../util/link-or-copy-file'
import { FilePath, QUARTZ, joinSegments } from '../../util/path'

const staticPath = joinSegments(QUARTZ, 'static')
const staticPathPrefix = `${staticPath}/`

function staticRelativePath(changeEvent: ChangeEvent): FilePath | undefined {
  if (!changeEvent.path.startsWith(staticPathPrefix)) {
    return undefined
  }

  const relativePath = changeEvent.path.slice(staticPathPrefix.length)
  return relativePath.length > 0 ? (relativePath as FilePath) : undefined
}

async function copyStaticFile(output: string, fp: FilePath, hardLink: boolean): Promise<FilePath> {
  const src = joinSegments(staticPath, fp) as FilePath
  const dest = joinSegments(output, 'static', fp) as FilePath
  return linkOrCopyFile(src, dest, { hardLink })
}

export const Static: QuartzEmitterPlugin = () => ({
  name: 'Static',
  async *emit({ argv, cfg }, _content, _resources) {
    const outputStaticPath = joinSegments(argv.output, 'static')
    const fps = await glob('**', staticPath, cfg.configuration.ignorePatterns)
    await fs.mkdir(outputStaticPath, { recursive: true })
    const hardLink = argv.watch && process.env.CF_PAGES !== '1'
    const files = await mapConcurrent(fps, defaultIoConcurrency, fp =>
      copyStaticFile(argv.output, fp, hardLink),
    )
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

      yield copyStaticFile(argv.output, fp, argv.watch && process.env.CF_PAGES !== '1')
    }
  },
})
