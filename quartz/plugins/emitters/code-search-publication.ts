import { mkdir, readdir, rm } from 'node:fs/promises'
import path from 'node:path'
import type { CodeSearchManifest, CodeSourceShard, CodeUnit } from '../../util/code-search-types'
import type { BuildCtx } from '../../util/ctx'
import { codeSourceId } from '../transformers/code-search-blocks'
import { removeWritten, write } from './helpers'

export async function* publishCodeSearch(
  ctx: BuildCtx,
  units: CodeUnit[],
  bodies: ReadonlyMap<string, string>,
) {
  units.sort((a, b) => a.id.localeCompare(b.id))
  const generation = codeSourceId(JSON.stringify(units))
  // A content revert can reuse a directory pruned after an earlier publication.
  await mkdir(path.join(ctx.argv.output, 'static/code-search', generation), { recursive: true })
  const index: CodeSearchManifest = {
    kind: 'quartz-code-search-v1',
    version: 1,
    generation,
    units,
    shards: [],
    coverage: {
      owners: new Set(units.map(unit => unit.ownerSlug)).size,
      units: units.length,
      sources: bodies.size,
    },
  }
  const languages = new Map<string, Set<string>>()
  for (const unit of units) {
    const ids = languages.get(unit.language) ?? new Set<string>()
    ids.add(unit.sourceId)
    languages.set(unit.language, ids)
  }
  const publicationCtx = { ...ctx, incremental: true }
  for (const [language, ids] of [...languages].sort(([a], [b]) => a.localeCompare(b))) {
    let sources: CodeSourceShard['sources'] = []
    let bytes = 0
    let ordinal = 0
    const flush = async () => {
      const payload: CodeSourceShard = {
        kind: 'quartz-code-sources-v1',
        generation,
        language,
        sources,
      }
      const content = JSON.stringify(payload)
      const slug = `static/code-search/${generation}/${codeSourceId(language).slice(0, 12)}-${ordinal++}`
      index.shards.push({
        language,
        path: `/${slug}.json`,
        sha256: codeSourceId(content),
        byteLength: Buffer.byteLength(content),
        sourceIds: sources.map(source => source.id),
      })
      return write({ ctx: publicationCtx, slug, ext: '.json', content })
    }
    for (const id of [...ids].sort()) {
      const text = bodies.get(id)
      if (text === undefined) throw new Error(`Missing code source ${id}`)
      const size = Buffer.byteLength(JSON.stringify(text))
      if (sources.length && bytes + size > 256 * 1024) {
        yield await flush()
        sources = []
        bytes = 0
      }
      sources.push({ id, text })
      bytes += size
    }
    if (sources.length) yield await flush()
  }
  yield await write({
    ctx: publicationCtx,
    slug: 'static/code-search/manifest',
    ext: '.json',
    content: JSON.stringify(index),
  })
  const root = path.join(ctx.argv.output, 'static/code-search')
  for (const entry of await readdir(root, { withFileTypes: true })) {
    if (!entry.isDirectory() || !/^[a-f0-9]{64}$/.test(entry.name) || entry.name === generation)
      continue
    const directory = path.join(root, entry.name)
    for (const file of await readdir(directory))
      await removeWritten(ctx, `static/code-search/${entry.name}/${file}`, '')
    await rm(directory, { recursive: true, force: true })
  }
}
