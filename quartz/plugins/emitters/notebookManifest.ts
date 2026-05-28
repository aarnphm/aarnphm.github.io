import type { BuildCtx } from '../../util/ctx'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { FilePath, slugifyFilePath } from '../../util/path'
import { write } from './helpers'

const MANIFEST_NAME = 'notebook-pages.json'

function isNotebookPath(fp: FilePath): boolean {
  return fp.endsWith('.ipynb')
}

function changesNotebookMembership(changeEvents: readonly { path: FilePath; type: string }[]) {
  return changeEvents.some(
    changeEvent => isNotebookPath(changeEvent.path) && changeEvent.type !== 'change',
  )
}

async function writeNotebookManifest(ctx: BuildCtx): Promise<FilePath> {
  const fps = ctx.allFiles.filter(isNotebookPath)
  const slugs = fps.map(fp => slugifyFilePath(fp, true)).sort()
  const payload = JSON.stringify({ slugs }, null, 2)
  return write({
    ctx,
    content: payload,
    slug: MANIFEST_NAME.replace(/\.json$/, '') as ReturnType<typeof slugifyFilePath>,
    ext: '.json',
  })
}

export const NotebookPagesManifest: QuartzEmitterPlugin = () => ({
  name: 'NotebookPagesManifest',
  async *emit(ctx) {
    yield writeNotebookManifest(ctx)
  },
  async *partialEmit(ctx, _content, _resources, changeEvents) {
    if (!changesNotebookMembership(changeEvents)) return
    yield writeNotebookManifest(ctx)
  },
})

export const notebookManifestFilename = MANIFEST_NAME
