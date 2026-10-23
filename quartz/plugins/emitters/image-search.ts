import { globby } from 'globby'
import path from 'node:path'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { removeWritten, write } from './helpers'
import { imageSearchOutput } from './image-search-publication'

export const ImageSearch: QuartzEmitterPlugin = () => {
  const emit: ReturnType<QuartzEmitterPlugin>['emit'] = async function* (ctx, content) {
    const publicIndex = Object.fromEntries(
      content.flatMap(([, file]) =>
        file.data.slug
          ? [
              [
                file.data.slug,
                { ...file.data.frontmatter, protected: file.data.frontmatter?.protected },
              ],
            ]
          : [],
      ),
    )
    let output: Awaited<ReturnType<typeof imageSearchOutput>>
    try {
      output = await imageSearchOutput({
        contentDir: ctx.argv.directory,
        publicationOnly: true,
        ignorePatterns: ctx.cfg.configuration.ignorePatterns,
        publicIndex,
      })
    } catch (error) {
      // Fail closed: an old index can contain images whose last public reference was removed.
      console.warn('[ImageSearch] Withholding invalid image index:', error)
    }
    const files = output?.files ?? new Map<string, Buffer>()
    for (const [slug, bytes] of files) yield await write({ ctx, slug, ext: '', content: bytes })
    const obsolete = await globby('**/*', {
      cwd: path.join(ctx.argv.output, 'embeddings/images'),
      onlyFiles: true,
    })
    for (const file of obsolete) {
      const slug = `embeddings/images/${file}`
      if (!files.has(slug)) await removeWritten(ctx, slug, '')
    }
  }
  return { name: 'ImageSearch', emit, partialEmit: emit }
}
