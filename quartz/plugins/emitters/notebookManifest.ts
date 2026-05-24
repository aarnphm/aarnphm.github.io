import { QuartzEmitterPlugin } from '../../types/plugin'
import { glob } from '../../util/glob'
import { FilePath, slugifyFilePath } from '../../util/path'
import { write } from './helpers'

const MANIFEST_NAME = 'notebook-pages.json'

async function notebookFilePaths(
  argv: { directory: string },
  cfg: { configuration: { ignorePatterns: string[] } },
): Promise<FilePath[]> {
  return (await glob('**/*.ipynb', argv.directory, cfg.configuration.ignorePatterns)) as FilePath[]
}

export const NotebookPagesManifest: QuartzEmitterPlugin = () => ({
  name: 'NotebookPagesManifest',
  async *emit(ctx) {
    const fps = await notebookFilePaths(ctx.argv, ctx.cfg)
    const slugs = fps.map(fp => slugifyFilePath(fp, true)).sort()
    const payload = JSON.stringify({ slugs }, null, 2)
    yield write({
      ctx,
      content: payload,
      slug: MANIFEST_NAME.replace(/\.json$/, '') as ReturnType<typeof slugifyFilePath>,
      ext: '.json',
    })
  },
})

export const notebookManifestFilename = MANIFEST_NAME
