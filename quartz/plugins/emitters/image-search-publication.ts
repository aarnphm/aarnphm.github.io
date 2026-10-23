import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import {
  collectImageSearchInventory,
  type ImageInventoryOptions,
} from '../../scripts/image-search-inventory'
import {
  readImageManifest,
  validateImageVectorShard,
  type ImageSearchManifest,
} from '../../util/image-search'
import { isRecord } from '../../util/type-guards'

const sha256 = (bytes: Uint8Array | string) => createHash('sha256').update(bytes).digest('hex')

/** Revalidate publication independently of the last inference run. Only live rows leave the vault. */
export async function imageSearchOutput(
  options: ImageInventoryOptions,
): Promise<{ manifest: ImageSearchManifest; files: Map<string, Buffer> } | undefined> {
  let raw: unknown
  try {
    raw = JSON.parse(
      await readFile(path.join(options.contentDir, 'embeddings/images/manifest.json'), 'utf8'),
    )
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') return undefined
    throw error
  }
  const source = readImageManifest(raw)
  if (!isRecord(raw) || !isRecord(raw.sourceDigests))
    throw new Error('Image source digests are missing; regenerate image embeddings.')
  const sourceDigests = raw.sourceDigests
  const inventory = await collectImageSearchInventory(options)
  const eligible = new Map(
    inventory.assets.filter(asset => asset.public).map(asset => [asset.assetId, asset]),
  )
  const allVectors = new Float32Array(source.rows * source.dims)
  for (const shard of source.vectors.shards) {
    const bytes = await readFile(path.join(options.contentDir, shard.path))
    const view = await validateImageVectorShard(Uint8Array.from(bytes).buffer, shard, source.dims)
    allVectors.set(view, shard.rowOffset * source.dims)
  }
  const ids: string[] = []
  const retainedDigests: Record<string, string> = {}
  const parts: Buffer[] = []
  for (let row = 0; row < source.ids.length; row++) {
    const id = source.ids[row]
    const asset = eligible.get(id)
    if (!asset || sourceDigests[id] !== asset.contentId) continue
    ids.push(id)
    retainedDigests[id] = asset.contentId
    const vector = allVectors.subarray(row * source.dims, (row + 1) * source.dims)
    parts.push(Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength))
  }
  const payload = Buffer.concat(parts)
  const generation = sha256(
    JSON.stringify({ source: source.generation, ids, sourceDigests: retainedDigests }),
  )
  const root = `embeddings/images/${generation}`
  const shards = ids.length
    ? [
        {
          path: `/${root}/vectors-000.f32`,
          rows: ids.length,
          rowOffset: 0,
          byteLength: payload.byteLength,
          byteStride: source.dims * 4,
          sha256: sha256(payload),
        },
      ]
    : []
  const manifest = readImageManifest({
    ...source,
    generation,
    ids,
    rows: ids.length,
    sourceDigests: retainedDigests,
    vectors: { dtype: 'fp32', rows: ids.length, dims: source.dims, shards },
  })
  const files = new Map<string, Buffer>()
  if (ids.length) files.set(`${root}/vectors-000.f32`, payload)
  const json = Buffer.from(JSON.stringify({ ...manifest, sourceDigests: retainedDigests }))
  files.set(`${root}/manifest.json`, json)
  files.set('embeddings/images/manifest.json', json)
  return { manifest, files }
}
