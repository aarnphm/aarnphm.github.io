import { globby } from 'globby'
import { readFile, readdir, rm } from 'node:fs/promises'
import path from 'node:path'
import type { QuartzEmitterPlugin } from '../../types/plugin'
import { readPdfIndex, validatePdfVectorShard } from '../../util/pdf-search'
import { removeWritten, write } from './helpers'

async function publication(content: string) {
  const root = path.join(content, 'embeddings/pdf')
  const payload = await readFile(path.join(root, 'manifest.json'))
  const index = readPdfIndex(JSON.parse(payload.toString()))
  const immutable = await readFile(path.join(root, index.generation, 'manifest.json'))
  if (!immutable.equals(payload)) throw new Error('PDF manifests disagree')
  const files = new Map<string, Buffer>()
  for (const shard of index.shards) {
    const bytes = await readFile(path.join(content, shard.path))
    await validatePdfVectorShard(Uint8Array.from(bytes).buffer, shard)
    files.set(shard.path.slice(1), bytes)
  }
  files.set(`embeddings/pdf/${index.generation}/manifest.json`, payload)
  files.set('embeddings/pdf/manifest.json', payload)
  return { files, generation: index.generation }
}

export const PdfSearch: QuartzEmitterPlugin = () => {
  const emit: ReturnType<QuartzEmitterPlugin>['emit'] = async function* (ctx) {
    let current: Awaited<ReturnType<typeof publication>>
    try {
      current = await publication(ctx.argv.directory)
    } catch (error) {
      // A batch can replace its source generation during a build. Keep the last verified output.
      if (!(error instanceof Error && 'code' in error && error.code === 'ENOENT'))
        console.warn('[PdfSearch] Preserving the previous verified index:', error)
      return
    }
    for (const [slug, bytes] of current.files)
      yield await write({ ctx, slug, ext: '', content: bytes })
    const root = path.join(ctx.argv.output, 'embeddings/pdf')
    for (const entry of await readdir(root, { withFileTypes: true })) {
      if (
        !entry.isDirectory() ||
        !/^[a-f\d]{64}$/.test(entry.name) ||
        entry.name === current.generation
      )
        continue
      const directory = path.join(root, entry.name)
      for (const file of await globby('**/*', {
        cwd: directory,
        onlyFiles: true,
        followSymbolicLinks: false,
      }))
        await removeWritten(ctx, `embeddings/pdf/${entry.name}/${file}`, '')
      await rm(directory, { recursive: true, force: true })
    }
  }
  return { name: 'PdfSearch', emit, partialEmit: emit }
}
