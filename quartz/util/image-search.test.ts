import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import test from 'node:test'
import {
  IMAGE_EMBEDDING_REVISION,
  imageManifestUrl,
  imageSearchResults,
  readImageManifest,
  validateImageVectorShard,
} from './image-search'

function fixture() {
  const values = new Float32Array(768 * 2)
  values[0] = 1
  values[768 + 1] = 1
  const bytes = values.buffer
  const generation = 'a'.repeat(64)
  return {
    bytes,
    manifest: {
      version: 1,
      modality: 'image',
      model: 'google/embeddinggemma-2',
      modelRevision: IMAGE_EMBEDDING_REVISION,
      onnxRevision: IMAGE_EMBEDDING_REVISION,
      preprocessing: { revision: 'sharp-rgb-first-frame-v1' },
      generation,
      dims: 768,
      dtype: 'fp32',
      normalized: true,
      rows: 2,
      ids: ['photos/one.webp', 'photos/two.webp'],
      vectors: {
        dtype: 'fp32',
        rows: 2,
        dims: 768,
        shards: [
          {
            path: `/embeddings/images/${generation}/vectors.f32`,
            rows: 2,
            rowOffset: 0,
            byteLength: bytes.byteLength,
            byteStride: 768 * 4,
            sha256: createHash('sha256').update(new Uint8Array(bytes)).digest('hex'),
          },
        ],
      },
    },
  }
}

test('an image generation preserves path identities and uses its immutable manifest', () => {
  const { manifest } = fixture()
  const parsed = readImageManifest(JSON.parse(JSON.stringify(manifest)))
  assert.deepEqual(parsed.ids, manifest.ids)
  assert.equal(imageManifestUrl(parsed), `/embeddings/images/${manifest.generation}/manifest.json`)
  assert.deepEqual(
    imageSearchResults(
      [
        { id: 1, score: 0.8 },
        { id: 0, score: 0.5 },
      ],
      [parsed.ids[1], parsed.ids[0]],
    ),
    [
      { slug: 'photos/two.webp', score: 0.8 },
      { slug: 'photos/one.webp', score: 0.5 },
    ],
  )
})

test('an incompatible image encoder or generation fails before row mapping', () => {
  const { manifest } = fixture()
  for (const change of [
    { model: 'other/model' },
    { dims: 512 },
    { normalized: false },
    { dtype: 'fp16' },
    { modality: 'text' },
    { generation: '../old' },
    { modelRevision: '' },
    { onnxRevision: 'other' },
    { preprocessing: {} },
    { ids: ['photos/one.webp'] },
    { ids: ['photos/one.webp', 'photos/one.webp'] },
    { ids: ['photos/one.webp', 'private/note.md'] },
  ])
    assert.throws(() => readImageManifest({ ...manifest, ...change }))
})

test('shard layout requires a complete ordered generation with integrity metadata', () => {
  const { manifest } = fixture()
  const shard = manifest.vectors.shards[0]
  for (const change of [
    { rows: 1 },
    { rowOffset: 1 },
    { byteLength: 1 },
    { byteStride: 4 },
    { path: 'https://example.org/vectors.f32' },
    { sha256: undefined },
  ])
    assert.throws(() =>
      readImageManifest({
        ...manifest,
        vectors: { ...manifest.vectors, shards: [{ ...shard, ...change }] },
      }),
    )
})

test('actual binary bytes must match the digest, length and finite unit-vector contract', async () => {
  const { manifest, bytes } = fixture()
  const parsed = readImageManifest(manifest)
  const shard = parsed.vectors.shards[0]
  const decoded = await validateImageVectorShard(bytes, shard, parsed.dims)
  assert.equal(decoded.length, 1536)
  assert.equal(decoded[769], 1)
  const corrupt = bytes.slice(0)
  new Uint8Array(corrupt)[0] ^= 1
  await assert.rejects(validateImageVectorShard(corrupt, shard, parsed.dims), /checksum/)
  await assert.rejects(validateImageVectorShard(bytes.slice(4), shard, parsed.dims), /length/)
  for (const value of [Number.NaN, 2, 0]) {
    const invalid = bytes.slice(0)
    new Float32Array(invalid)[0] = value
    const sha256 = createHash('sha256').update(new Uint8Array(invalid)).digest('hex')
    await assert.rejects(validateImageVectorShard(invalid, { ...shard, sha256 }, 768), /vector/)
  }
})

test('hits without a row identity or a finite score never become image search results', () => {
  assert.deepEqual(
    imageSearchResults(
      [
        { id: 0, score: 0.8 },
        { id: 0, score: 0.4 },
        { id: 1, score: Number.NaN },
        { id: 9, score: 1 },
      ],
      ['photos/one.webp', 'photos/one.webp', 'photos/two.webp', undefined],
    ),
    [{ slug: 'photos/one.webp', score: 0.8 }],
  )
})

test('source digests preserve complete exact row ownership when provided', () => {
  const { manifest } = fixture()
  const sourceDigests = Object.fromEntries(manifest.ids.map(id => [id, 'b'.repeat(64)]))
  assert.deepEqual(readImageManifest({ ...manifest, sourceDigests }).sourceDigests, sourceDigests)
  assert.equal(readImageManifest(manifest).sourceDigests, undefined)
  for (const value of [
    null,
    [],
    {},
    { [manifest.ids[0]]: 'b'.repeat(64) },
    { ...sourceDigests, unexpected: 'b'.repeat(64) },
    { ...sourceDigests, [manifest.ids[0]]: 'broken' },
  ])
    assert.throws(() => readImageManifest({ ...manifest, sourceDigests: value }))
})

test('encoder provenance preserves the exact local inference implementation', () => {
  const { manifest } = fixture()
  const encoder = {
    repo: 'mlx-community/embeddinggemma-2-4bit',
    dtype: 'q4',
    device: 'mlx-metal',
    revision: 'local-encoder-revision',
  }
  assert.deepEqual(readImageManifest({ ...manifest, encoder }).encoder, encoder)
  assert.equal(readImageManifest(manifest).encoder, undefined)
  for (const value of [
    null,
    [],
    {},
    { ...encoder, repo: '' },
    { ...encoder, dtype: 4 },
    { ...encoder, device: ' ' },
    { ...encoder, revision: undefined },
  ])
    assert.throws(() => readImageManifest({ ...manifest, encoder: value }))
})
