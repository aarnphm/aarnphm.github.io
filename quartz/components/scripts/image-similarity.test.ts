import assert from 'node:assert/strict'
import { once } from 'node:events'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import {
  IMAGE_ENCODER_CONTRACT,
  publishImageGeneration,
} from '../../scripts/image-embeddings/store'
import { createImageSimilarity } from './image-similarity'

// Exercise real static HTTP output. Similarity must need neither a model nor a query encoder.
test('stored-image similarity verifies bytes, excludes its seed, caches vectors, and cancels', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'image-similarity-'))
  const unit = (first: number, second: number) => {
    const vector = new Float32Array(768)
    vector[0] = first
    vector[1] = second
    return vector
  }
  const manifest = await publishImageGeneration({
    outDir: path.join(root, 'embeddings/images'),
    contract: IMAGE_ENCODER_CONTRACT,
    rows: [
      { id: 'images/seed.webp', vector: unit(1, 0) },
      { id: 'images/near.webp', vector: unit(0.8, 0.6) },
      { id: 'images/far.webp', vector: unit(0, 1) },
    ],
  })
  const requests: string[] = []
  let corrupt = false
  const server = createServer(async (request, response) => {
    const pathname = new URL(request.url ?? '/', 'http://localhost').pathname
    requests.push(pathname)
    try {
      const bytes = await readFile(path.join(root, pathname))
      if (corrupt && pathname.endsWith('.f32')) bytes[0] ^= 1
      response.end(bytes)
    } catch {
      response.writeHead(404).end()
    }
  })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const address = server.address()
  assert.ok(address && typeof address !== 'string')
  const url = `http://127.0.0.1:${address.port}/embeddings/images/manifest.json`
  const service = createImageSimilarity(url)
  try {
    assert.deepEqual(
      [...(await service.ids())],
      ['images/seed.webp', 'images/near.webp', 'images/far.webp'],
    )
    assert.equal(requests.length, 1)
    const hits = await service.search('images/seed.webp', 2)
    assert.deepEqual(
      hits.map(hit => hit.slug),
      ['images/near.webp', 'images/far.webp'],
    )
    assert.ok(Math.abs(hits[0].score - 0.8) < 0.00001)
    assert.equal(requests.length, 2)
    assert.deepEqual(requests, [
      '/embeddings/images/manifest.json',
      manifest.vectors.shards[0].path,
    ])
    await service.search('images/near.webp', 1)
    assert.equal(requests.length, 2)
    await assert.rejects(service.search('images/absent.webp', 2), /not indexed/)
    const controller = new AbortController()
    controller.abort()
    await assert.rejects(service.search('images/seed.webp', 2, controller.signal), {
      name: 'AbortError',
    })
    service.cancel()
    assert.equal((await service.search('images/seed.webp', 1)).length, 1)
    service.dispose()
    await assert.rejects(service.search('images/seed.webp', 1), /disposed/)
    corrupt = true
    const broken = createImageSimilarity(url)
    await assert.rejects(broken.search('images/seed.webp', 2), /checksum mismatch/)
    corrupt = false
    assert.equal((await broken.search('images/seed.webp', 1)).length, 1)
    broken.dispose()
    const stale = createImageSimilarity(url)
    await stale.ids()
    await publishImageGeneration({
      outDir: path.join(root, 'embeddings/images'),
      contract: IMAGE_ENCODER_CONTRACT,
      rows: [
        { id: 'images/seed.webp', vector: unit(1, 0) },
        { id: 'images/near.webp', vector: unit(0.6, 0.8) },
      ],
    })
    await rm(path.join(root, 'embeddings/images', manifest.generation), { recursive: true })
    await assert.rejects(stale.search('images/seed.webp', 1), /unavailable: 404/)
    assert.equal((await stale.search('images/seed.webp', 1))[0].slug, 'images/near.webp')
    stale.dispose()
  } finally {
    service.dispose()
    await new Promise<void>((resolve, reject) =>
      server.close(error => (error ? reject(error) : resolve())),
    )
    await rm(root, { recursive: true, force: true })
  }
})
