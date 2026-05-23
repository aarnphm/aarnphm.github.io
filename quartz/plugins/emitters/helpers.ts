import fs from 'fs'
import path from 'path'
import { Readable } from 'stream'
import type { BuildCtx } from '../../util/ctx'
import type { FilePath, FullSlug } from '../../util/path'
import { joinSegments } from '../../util/path'

type WriteOptions = {
  ctx: BuildCtx
  slug: FullSlug | string
  ext: `.${string}` | ''
  content: string | Buffer | Readable
}

export const write = async ({ ctx, slug, ext, content }: WriteOptions): Promise<FilePath> => {
  const pathToPage = joinSegments(ctx.argv.output, slug + ext) as FilePath
  const dir = path.dirname(pathToPage)
  await fs.promises.mkdir(dir, { recursive: true })
  await fs.promises.writeFile(pathToPage, content)
  return pathToPage
}
