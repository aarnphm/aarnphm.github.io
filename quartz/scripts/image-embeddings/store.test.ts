import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { readImageManifest, validateImageVectorShard } from '../../util/image-search'
import {
  IMAGE_ENCODER_CONTRACT,
  imageVectorCache,
  normalizeImageVector,
  publishImageGeneration,
} from './store'

const vector = (dimension: number) => {
  const result = new Float32Array(768)
  result[dimension] = 1
  return result
}

test('image generation publishes complete immutable shards before the identical stable manifest', async () => {
  const outDir = await mkdtemp(path.join(os.tmpdir(), 'image-publish-'))
  try {
    const rows = [
      { id: 'images/cat.webp', vector: vector(0) },
      { id: 'images/dog.png', vector: vector(1) },
    ]
    const manifest = await publishImageGeneration({
      outDir,
      rows,
      shardSize: 1,
      contract: IMAGE_ENCODER_CONTRACT,
      sourceDigests: { 'images/cat.webp': 'a'.repeat(64), 'images/dog.png': 'b'.repeat(64) },
    })
    const stable = await readFile(path.join(outDir, 'manifest.json'), 'utf8')
    assert.equal(
      await readFile(path.join(outDir, manifest.generation, 'manifest.json'), 'utf8'),
      stable,
    )
    const decoded = readImageManifest(JSON.parse(stable))
    assert.deepEqual(JSON.parse(stable).encoder, {
      repo: IMAGE_ENCODER_CONTRACT.repo,
      dtype: IMAGE_ENCODER_CONTRACT.dtype,
      device: IMAGE_ENCODER_CONTRACT.device,
      revision: IMAGE_ENCODER_CONTRACT.modelRevision,
    })
    assert.deepEqual(JSON.parse(stable).sourceDigests, {
      'images/cat.webp': 'a'.repeat(64),
      'images/dog.png': 'b'.repeat(64),
    })
    assert.deepEqual(
      decoded.ids,
      rows.map(row => row.id),
    )
    for (const shard of decoded.vectors.shards) {
      const file = path.join(outDir, shard.path.slice('/embeddings/images/'.length))
      const bytes = new Uint8Array(await readFile(file))
      const actual = await validateImageVectorShard(bytes.buffer, shard, 768)
      assert.deepEqual(actual, rows[shard.rowOffset].vector)
    }
    const repeated = await publishImageGeneration({
      outDir,
      rows,
      shardSize: 1,
      contract: IMAGE_ENCODER_CONTRACT,
      sourceDigests: { 'images/cat.webp': 'a'.repeat(64), 'images/dog.png': 'b'.repeat(64) },
    })
    assert.equal(repeated.generation, manifest.generation)
    await assert.rejects(
      publishImageGeneration({
        outDir,
        rows: [{ id: 'images/fail.webp', vector: new Float32Array(768) }],
        contract: IMAGE_ENCODER_CONTRACT,
      }),
      /unit|degenerate/i,
    )
    assert.equal(await readFile(path.join(outDir, 'manifest.json'), 'utf8'), stable)
    const changed = await publishImageGeneration({
      outDir,
      rows: rows.slice(1),
      contract: IMAGE_ENCODER_CONTRACT,
    })
    assert.notEqual(changed.generation, manifest.generation)
    assert.deepEqual(changed.ids, ['images/dog.png'])
    await assert.rejects(readFile(path.join(outDir, manifest.generation, 'manifest.json')), {
      code: 'ENOENT',
    })
  } finally {
    await rm(outDir, { recursive: true, force: true })
  }
})

test('vector cache resumes by bytes and full encoder contract and rejects damaged data', async () => {
  const cacheDir = await mkdtemp(path.join(os.tmpdir(), 'image-cache-'))
  try {
    const contentId = 'a'.repeat(64)
    const first = imageVectorCache(cacheDir, IMAGE_ENCODER_CONTRACT)
    assert.equal(await first.read(contentId), undefined)
    await first.write(contentId, vector(4))
    const resumed = imageVectorCache(cacheDir, IMAGE_ENCODER_CONTRACT)
    assert.deepEqual(await resumed.read(contentId), vector(4))
    const cached = JSON.parse(await readFile(first.pathFor(contentId), 'utf8'))
    cached.values[4] = 0
    cached.values[5] = 1
    await writeFile(first.pathFor(contentId), JSON.stringify(cached))
    assert.equal(await resumed.read(contentId), undefined)
    await first.write(contentId, vector(4))
    const changed = imageVectorCache(cacheDir, {
      ...IMAGE_ENCODER_CONTRACT,
      preprocessing: { ...IMAGE_ENCODER_CONTRACT.preprocessing, revision: 'different' },
    })
    assert.equal(await changed.read(contentId), undefined)
    await writeFile(first.pathFor(contentId), new Uint8Array([0, 1, 2]))
    assert.equal(await resumed.read(contentId), undefined)
  } finally {
    await rm(cacheDir, { recursive: true, force: true })
  }
})

test('encoder vectors require finite 768 values and normalize the retained dimensions', () => {
  const input = vector(2)
  input[2] = 4
  assert.deepEqual(normalizeImageVector(input), vector(2))
  assert.throws(() => normalizeImageVector(new Float32Array(768)), /degenerate/i)
  assert.throws(() => normalizeImageVector(new Float32Array(12)), /768/)
  input[3] = NaN
  assert.throws(() => normalizeImageVector(input), /finite|degenerate/i)
})
