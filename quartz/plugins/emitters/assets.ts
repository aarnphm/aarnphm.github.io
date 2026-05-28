import fs from 'node:fs/promises'
import path from 'path'
import { QuartzConfig } from '../../cfg'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { Argv } from '../../util/ctx'
import { glob } from '../../util/glob'
import { linkOrCopyFile } from '../../util/link-or-copy-file'
import { FilePath, joinSegments, slugifyFilePath } from '../../util/path'

const filesToCopy = async (argv: Argv, cfg: QuartzConfig) => {
  const patterns = ['**/*.md', '**/*.base', ...cfg.configuration.ignorePatterns]

  if (process.env.CF_PAGES === '1' || argv.watch) {
    patterns.push('**/*.pdf', '**.ddl', '**.mat')
  }

  return await glob('**', argv.directory, patterns)
}

const copyFile = async (argv: Argv, fp: FilePath) => {
  const src = joinSegments(argv.directory, fp) as FilePath

  const name = slugifyFilePath(fp)
  const dest = joinSegments(argv.output, name) as FilePath

  return linkOrCopyFile(src, dest, { hardLink: argv.watch && process.env.CF_PAGES !== '1' })
}

export const Assets: QuartzEmitterPlugin = () => {
  return {
    name: 'Assets',
    async *emit({ argv, cfg }) {
      const fps = await filesToCopy(argv, cfg)
      const files = await mapConcurrent(fps, defaultIoConcurrency, fp => copyFile(argv, fp))
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
