import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtemp, mkdir, open, readFile, readdir, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import sharp from 'sharp'
import { IMAGE_ENCODER_CONTRACT, imageVectorCache, sha256 } from './store'

test(
  'full image publication retains current cache entries and excludes another cache writer',
  { timeout: 120000 },
  async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'image-cache-retention-'))
    const content = path.join(root, 'content')
    const cacheDir = path.join(root, 'cache')
    const output = path.join(root, 'output')
    const config = path.join(root, 'quartz.config.ts')
    const index = path.join(root, 'index.json')
    const cache = imageVectorCache(cacheDir, IMAGE_ENCODER_CONTRACT)
    const old = imageVectorCache(cacheDir, { ...IMAGE_ENCODER_CONTRACT, modelRevision: 'older' })
    const run = () =>
      spawnSync(
        process.execPath,
        [
          '--import',
          'tsx',
          'quartz/scripts/build-image-embeddings.ts',
          '--backend',
          'onnx',
          '--content',
          content,
          '--cache',
          cacheDir,
          '--out',
          output,
          '--config',
          config,
          '--public-index',
          index,
          '--report',
          path.join(root, 'report.json'),
        ],
        { encoding: 'utf8', timeout: 30000, env: { ...process.env, HF_HUB_OFFLINE: '1' } },
      )
    try {
      await mkdir(content)
      await writeFile(config, 'const configuration = { ignorePatterns: [] }')
      await writeFile(index, JSON.stringify({ note: { title: 'Published' } }))
      const ids: string[] = []
      for (const [dimension, name] of ['red', 'blue'].entries()) {
        const bytes = await sharp({
          create: { width: 8, height: 8, channels: 3, background: name },
        })
          .png()
          .toBuffer()
        const id = sha256(bytes)
        ids.push(id)
        await writeFile(path.join(content, `${name}.png`), bytes)
        const vector = new Float32Array(768)
        vector[dimension] = 1
        await cache.write(id, vector)
        await old.write(id, vector)
      }
      const external = path.join(root, 'external')
      await mkdir(external)
      await writeFile(path.join(external, 'keep'), 'keep')
      const linked = path.join(cacheDir, 'a'.repeat(64))
      await symlink(external, linked, 'dir')
      await writeFile(path.join(content, 'note.md'), '![[red.png]]\n![[blue.png]]')
      const lock = await open(path.join(cacheDir, '.build.lock'), 'wx')
      try {
        assert.notEqual(run().status, 0)
        assert.equal((await readdir(path.join(cacheDir, old.fingerprint))).length, 2)
      } finally {
        await lock.close()
        await rm(path.join(cacheDir, '.build.lock'))
      }
      let result = run()
      assert.equal(result.status, 0, result.stderr)
      await assert.rejects(readdir(path.join(cacheDir, old.fingerprint)), { code: 'ENOENT' })
      assert.equal((await readdir(path.join(cacheDir, cache.fingerprint))).length, 2)
      await writeFile(path.join(content, 'note.md'), '![[red.png]]')
      result = run()
      assert.equal(result.status, 0, result.stderr)
      assert.deepEqual(await readdir(path.join(cacheDir, cache.fingerprint)), [`${ids[0]}.json`])
      assert.equal(await readFile(path.join(external, 'keep'), 'utf8'), 'keep')
      const report = JSON.parse(await readFile(path.join(root, 'report.json'), 'utf8'))
      assert.equal(report.encoded, 0)
      assert.equal(report.status, 'published')
      console.log(
        JSON.stringify({
          currentCacheEntries: 1,
          obsoleteContractsRemoved: true,
          unusedVectorsRemoved: true,
          cacheWriterExcluded: true,
          inferenceRuns: 0,
        }),
      )
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  },
)
