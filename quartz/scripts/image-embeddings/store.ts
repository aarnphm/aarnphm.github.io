import { createHash, randomUUID } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import sharp from 'sharp'
import {
  IMAGE_EMBEDDING_REVISION,
  readImageManifest,
  type ImageSearchManifest,
  type ImageVectorShard,
} from '../../util/image-search'
import { isRecord } from '../../util/type-guards'

const transformerPackage: unknown = JSON.parse(
  readFileSync(
    new URL('../package.json', import.meta.resolve('@huggingface/transformers')),
    'utf8',
  ),
)
if (!isRecord(transformerPackage) || typeof transformerPackage.version !== 'string')
  throw new Error('Cannot identify the installed Transformers.js version.')

export const IMAGE_ENCODER_CONTRACT = {
  model: 'google/embeddinggemma-2',
  repo: 'onnx-community/embeddinggemma-2-ONNX',
  modelRevision: IMAGE_EMBEDDING_REVISION,
  onnxRevision: IMAGE_EMBEDDING_REVISION,
  dtype: 'q4',
  device: 'cpu',
  dims: 768,
  preprocessing: {
    revision: 'garden-image-rgb-v1',
    orientation: 'exif',
    alphaBackground: '#ffffff',
    colourspace: 'srgb',
    frame: 'first',
    processorRevision: IMAGE_EMBEDDING_REVISION,
    processorOptions: 'pinned-defaults',
    transformerVersion: transformerPackage.version,
    sharpVersion: sharp.versions.sharp,
    vipsVersion: sharp.versions.vips,
  },
}

export interface ImageEncoderContract {
  model: string
  repo: string
  modelRevision: string
  onnxRevision: string
  dtype: string
  device: string
  dims: number
  preprocessing: Record<string, unknown> & { revision: string }
}
export type ImageEmbeddingRow = { id: string; vector: Float32Array }
export const sha256 = (input: string | Uint8Array) =>
  createHash('sha256').update(input).digest('hex')

export function normalizeImageVector(values: ArrayLike<number>): Float32Array {
  if (values.length !== 768)
    throw new Error(`Image encoder must return 768 dimensions, received ${values.length}.`)
  const result = Float32Array.from(values)
  const norm = Math.hypot(...result)
  if (!Number.isFinite(norm) || norm === 0)
    throw new Error('Image encoder returned a degenerate or nonfinite vector.')
  return result.map(value => value / norm)
}

function vectorBytes(vector: Float32Array): Buffer {
  if (
    vector.length !== 768 ||
    Math.abs(Math.hypot(...vector) - 1) > 0.01 ||
    vector.some(value => !Number.isFinite(value))
  )
    throw new Error('Expected a finite unit image vector with 768 dimensions.')
  return Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength)
}

export async function atomicWrite(file: string, data: string | Uint8Array): Promise<void> {
  await mkdir(path.dirname(file), { recursive: true })
  const temporary = `${file}.${randomUUID()}.tmp`
  try {
    await writeFile(temporary, data)
    await rename(temporary, file)
  } finally {
    await rm(temporary, { force: true })
  }
}

export function imageVectorCache(cacheDir: string, contract: ImageEncoderContract) {
  const fingerprint = sha256(JSON.stringify(contract))
  const pathFor = (contentId: string) => {
    if (!/^[a-f\d]{64}$/.test(contentId)) throw new Error('Invalid image content digest.')
    return path.join(cacheDir, fingerprint, `${contentId}.json`)
  }
  return {
    fingerprint,
    pathFor,
    async read(contentId: string): Promise<Float32Array | undefined> {
      let raw: unknown
      try {
        raw = JSON.parse(await readFile(pathFor(contentId), 'utf8'))
      } catch (error) {
        if (
          error instanceof Error &&
          ('code' in error ? error.code === 'ENOENT' : error instanceof SyntaxError)
        )
          return undefined
        throw error
      }
      if (
        !isRecord(raw) ||
        raw.version !== 1 ||
        typeof raw.sha256 !== 'string' ||
        !Array.isArray(raw.values) ||
        !raw.values.every((value): value is number => typeof value === 'number')
      )
        return undefined
      const vector = Float32Array.from(raw.values)
      try {
        if (sha256(vectorBytes(vector)) !== raw.sha256) return undefined
        return vector
      } catch {
        return undefined
      }
    },
    async write(contentId: string, vector: Float32Array): Promise<void> {
      const checksum = sha256(vectorBytes(vector))
      await atomicWrite(
        pathFor(contentId),
        JSON.stringify({ version: 1, sha256: checksum, values: Array.from(vector) }),
      )
    },
  }
}

export async function publishImageGeneration(options: {
  outDir: string
  rows: readonly ImageEmbeddingRow[]
  contract: ImageEncoderContract
  shardSize?: number
  sourceDigests?: Readonly<Record<string, string>>
}): Promise<ImageSearchManifest> {
  const { outDir, rows, contract, shardSize = 1024, sourceDigests } = options
  if (!Number.isSafeInteger(shardSize) || shardSize < 1)
    throw new Error('Invalid image shard size.')
  const encoder = {
    repo: contract.repo,
    dtype: contract.dtype,
    device: contract.device,
    revision: contract.modelRevision,
  }
  const digest = createHash('sha256').update(
    JSON.stringify({ contract, ids: rows.map(row => row.id), shardSize, sourceDigests, encoder }),
  )
  for (const row of rows) digest.update(vectorBytes(row.vector))
  const generation = digest.digest('hex')
  const shards: ImageVectorShard[] = []
  const payloads: { file: string; payload: Buffer }[] = []
  for (let offset = 0; offset < rows.length; offset += shardSize) {
    const selected = rows.slice(offset, offset + shardSize)
    const payload = Buffer.concat(selected.map(row => vectorBytes(row.vector)))
    const file = `vectors-${String(shards.length).padStart(3, '0')}.f32`
    payloads.push({ file, payload })
    shards.push({
      path: `/embeddings/images/${generation}/${file}`,
      rows: selected.length,
      rowOffset: offset,
      byteLength: payload.byteLength,
      byteStride: 768 * 4,
      sha256: sha256(payload),
    })
  }
  const manifest = readImageManifest({
    version: 1,
    modality: 'image',
    model: contract.model,
    modelRevision: contract.modelRevision,
    encoder,
    onnxRevision: contract.onnxRevision,
    preprocessing: contract.preprocessing,
    generation,
    dims: 768,
    dtype: 'fp32',
    normalized: true,
    rows: rows.length,
    ids: rows.map(row => row.id),
    sourceDigests,
    vectors: { dtype: 'fp32', rows: rows.length, dims: 768, shards },
  })
  const serialized = `${JSON.stringify(manifest)}\n`
  for (const { file, payload } of payloads)
    await atomicWrite(path.join(outDir, generation, file), payload)
  await atomicWrite(path.join(outDir, generation, 'manifest.json'), serialized)
  await atomicWrite(path.join(outDir, 'manifest.json'), serialized)
  return manifest
}
