import { globby } from 'globby'
import fs from 'node:fs/promises'
import path from 'path'
import type { BuildCtx } from '../../../util/ctx'
import type { FilePath } from '../../../util/path'
import { assetSlugForContent } from '../../../util/asset-manifest'
import { isFullSlug, joinSegments } from '../../../util/path'
import { write } from '../helpers'
import { emojiAssetSourceDir } from './asset-paths'

export async function writeEmojiAssets(ctx: BuildCtx): Promise<FilePath[]> {
  const files = await globby([`${emojiAssetSourceDir}/**/*.json`])
  return await Promise.all(
    files.map(async file => {
      const rel = path.relative(emojiAssetSourceDir, file).split(path.sep).join('/')
      const slug = joinSegments('static', 'scripts', 'emoji', rel.slice(0, -'.json'.length))
      if (!isFullSlug(slug)) throw new Error(`invalid emoji asset slug ${slug}`)
      const content = await fs.readFile(file)
      return await write({
        ctx,
        slug: assetSlugForContent(ctx, slug, '.json', content),
        ext: '.json',
        content,
      })
    }),
  )
}
