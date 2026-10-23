import type { SemanticDocument, SemanticHit } from './semantic-search'
import { isSearchExcluded } from './search-exclude'
import { searchScopeIncludes } from './search-scope'
import { isRecord } from './type-guards'

export const IMAGE_EMBEDDING_REVISION = 'daa72c51243991dfcaf9f9137d2c573d8f7790c0'
export const IMAGE_MANIFEST_URL = '/embeddings/images/manifest.json'

export interface ImageVectorShard {
  path: string
  rows: number
  rowOffset: number
  byteLength: number
  byteStride: number
  sha256: string
}

export interface ImageSearchManifest {
  version: number
  modality: 'image'
  model: 'google/embeddinggemma-2'
  modelRevision: string
  onnxRevision: string
  encoder?: { repo: string; dtype: string; device: string; revision: string }
  preprocessing: Record<string, unknown> & { revision: string }
  generation: string
  dims: number
  dtype: 'fp32'
  normalized: true
  rows: number
  ids: string[]
  sourceDigests?: Record<string, string>
  vectors: { dtype: 'fp32'; rows: number; dims: number; shards: ImageVectorShard[] }
}

const digest = /^[a-f\d]{64}$/
const positiveInteger = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value > 0

export function imageManifestUrl(index: Pick<ImageSearchManifest, 'generation'>): string {
  return `/embeddings/images/${index.generation}/manifest.json`
}

export function readImageManifest(data: unknown): ImageSearchManifest {
  if (
    !isRecord(data) ||
    data.version !== 1 ||
    data.modality !== 'image' ||
    data.model !== 'google/embeddinggemma-2' ||
    typeof data.modelRevision !== 'string' ||
    !data.modelRevision.trim() ||
    data.onnxRevision !== IMAGE_EMBEDDING_REVISION ||
    !isRecord(data.preprocessing) ||
    typeof data.preprocessing.revision !== 'string' ||
    !data.preprocessing.revision.trim() ||
    typeof data.generation !== 'string' ||
    !digest.test(data.generation) ||
    data.dims !== 768 ||
    data.dtype !== 'fp32' ||
    data.normalized !== true ||
    typeof data.rows !== 'number' ||
    !Number.isSafeInteger(data.rows) ||
    data.rows < 0 ||
    !Array.isArray(data.ids) ||
    data.ids.length !== data.rows ||
    !data.ids.every(
      (id): id is string =>
        typeof id === 'string' &&
        searchScopeIncludes('images', { slug: id, fileName: id, filePath: id }) &&
        !isSearchExcluded(id),
    ) ||
    new Set(data.ids).size !== data.ids.length ||
    !isRecord(data.vectors) ||
    data.vectors.dtype !== 'fp32' ||
    data.vectors.rows !== data.rows ||
    data.vectors.dims !== data.dims ||
    !Array.isArray(data.vectors.shards)
  )
    throw new Error('Invalid or incompatible image embedding generation.')

  const shards: ImageVectorShard[] = []
  let rowOffset = 0
  const prefix = `/embeddings/images/${data.generation}/`
  for (const shard of data.vectors.shards) {
    if (
      !isRecord(shard) ||
      typeof shard.path !== 'string' ||
      !shard.path.startsWith(prefix) ||
      /(?:\.{2}|[\\?#%])/.test(shard.path) ||
      !positiveInteger(shard.rows) ||
      shard.rowOffset !== rowOffset ||
      shard.byteStride !== data.dims * 4 ||
      shard.byteLength !== shard.rows * data.dims * 4 ||
      typeof shard.sha256 !== 'string' ||
      !digest.test(shard.sha256)
    )
      throw new Error('Invalid image embedding shard layout.')
    shards.push({
      path: shard.path,
      rows: shard.rows,
      rowOffset,
      byteLength: shard.byteLength,
      byteStride: shard.byteStride,
      sha256: shard.sha256,
    })
    rowOffset += shard.rows
  }
  if (rowOffset !== data.rows) throw new Error('Incomplete image embedding shard inventory.')
  let encoder: ImageSearchManifest['encoder']
  if (data.encoder !== undefined) {
    if (
      !isRecord(data.encoder) ||
      typeof data.encoder.repo !== 'string' ||
      !data.encoder.repo.trim() ||
      typeof data.encoder.dtype !== 'string' ||
      !data.encoder.dtype.trim() ||
      typeof data.encoder.device !== 'string' ||
      !data.encoder.device.trim() ||
      typeof data.encoder.revision !== 'string' ||
      !data.encoder.revision.trim()
    )
      throw new Error('Invalid image encoder provenance.')
    encoder = {
      repo: data.encoder.repo,
      dtype: data.encoder.dtype,
      device: data.encoder.device,
      revision: data.encoder.revision,
    }
  }
  let sourceDigests: Record<string, string> | undefined
  if (data.sourceDigests !== undefined) {
    if (!isRecord(data.sourceDigests) || Object.keys(data.sourceDigests).length !== data.ids.length)
      throw new Error('Invalid image source digest inventory.')
    const entries: [string, string][] = []
    for (const id of data.ids) {
      const value = data.sourceDigests[id]
      if (
        !Object.hasOwn(data.sourceDigests, id) ||
        typeof value !== 'string' ||
        !digest.test(value)
      )
        throw new Error('Invalid image source digest inventory.')
      entries.push([id, value])
    }
    sourceDigests = Object.fromEntries(entries)
  }
  return {
    version: 1,
    modality: 'image',
    model: data.model,
    modelRevision: data.modelRevision,
    onnxRevision: data.onnxRevision,
    ...(encoder ? { encoder } : {}),
    preprocessing: { ...data.preprocessing, revision: data.preprocessing.revision },
    generation: data.generation,
    dims: data.dims,
    dtype: 'fp32',
    normalized: true,
    rows: data.rows,
    ids: data.ids,
    ...(sourceDigests ? { sourceDigests } : {}),
    vectors: { dtype: 'fp32', rows: data.rows, dims: data.dims, shards },
  }
}

export async function validateImageVectorShard(
  bytes: ArrayBuffer,
  shard: ImageVectorShard,
  dims: number,
): Promise<Float32Array> {
  if (bytes.byteLength !== shard.byteLength || bytes.byteLength !== shard.rows * dims * 4)
    throw new Error(`Image shard ${shard.path} length mismatch.`)
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))
  const checksum = Array.from(hash, byte => byte.toString(16).padStart(2, '0')).join('')
  if (checksum !== shard.sha256) throw new Error(`Image shard ${shard.path} checksum mismatch.`)
  const view = new Float32Array(bytes)
  for (let row = 0; row < shard.rows; row++) {
    let norm = 0
    for (let dimension = 0; dimension < dims; dimension++) {
      const value = view[row * dims + dimension]
      norm += value * value
    }
    if (!Number.isFinite(norm) || Math.abs(norm - 1) > 0.01)
      throw new Error(`Image shard ${shard.path} contains an invalid unit vector at row ${row}.`)
  }
  return view
}

export function imageSearchResults(
  hits: readonly SemanticHit[],
  index: Pick<ImageSearchManifest, 'ids'>,
): SemanticDocument[] {
  const scores = new Map<string, number>()
  for (const hit of hits) {
    const slug = index.ids[hit.id]
    if (!slug || !Number.isFinite(hit.score)) continue
    scores.set(slug, Math.max(scores.get(slug) ?? -Infinity, hit.score))
  }
  return Array.from(scores, ([slug, score]) => ({ slug, score })).sort(
    (a, b) => b.score - a.score || a.slug.localeCompare(b.slug),
  )
}
