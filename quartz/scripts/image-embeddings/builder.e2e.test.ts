import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { lstat, mkdtemp, mkdir, readFile, readdir, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import sharp from 'sharp'
import { readImageManifest, validateImageVectorShard } from '../../util/image-search'
import { isRecord } from '../../util/type-guards'
import { IMAGE_ENCODER_CONTRACT, imageVectorCache, sha256 } from './store'

test('inventory-only CLI records local coverage without loading an encoder or publishing private paths', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'image-cli-'))
  try {
    const content = path.join(root, 'content')
    await mkdir(path.join(content, 'private'), { recursive: true })
    const pixels = await sharp({
      create: { width: 8, height: 8, channels: 3, background: '#ff0000' },
    })
      .png()
      .toBuffer()
    await writeFile(path.join(content, 'red.png'), pixels)
    await writeFile(path.join(content, 'private', 'red.png'), pixels)
    await writeFile(path.join(content, 'note.md'), '![[red.png]]')
    await writeFile(path.join(content, 'private', 'secret.md'), '![[red.png]]')
    const publicIndex = path.join(root, 'contentIndex.json')
    await writeFile(publicIndex, JSON.stringify({ note: { title: 'Published note' } }))
    const config = path.join(root, 'quartz.config.ts')
    await writeFile(config, "const configuration = { ignorePatterns: ['private'] }")
    const reportPath = path.join(root, 'report.json')
    const command = spawnSync(
      process.execPath,
      [
        '--import',
        'tsx',
        'quartz/scripts/build-image-embeddings.ts',
        '--inventory-only',
        '--content',
        content,
        '--public-index',
        publicIndex,
        '--config',
        config,
        '--report',
        reportPath,
        '--out',
        path.join(root, 'output'),
        '--cache',
        path.join(root, 'cache'),
      ],
      { encoding: 'utf8' },
    )
    assert.equal(command.status, 0, command.stderr)
    const report = JSON.parse(await readFile(reportPath, 'utf8'))
    assert.equal(report.status, 'inventory')
    assert.equal(report.coverage.localAssets, 2)
    assert.equal(report.coverage.publicAssets, 1)
    assert.equal(report.coverage.excludedAssets, 1)
    assert.equal(report.encoded, 0)
    await assert.rejects(readFile(path.join(root, 'output', 'manifest.json')), { code: 'ENOENT' })
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test(
  'builder CLI validates replacements, excludes overlapping writers, and prunes only obsolete generations',
  { timeout: 120_000 },
  async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'image-retention-'))
    const content = path.join(root, 'content')
    const outDir = path.join(root, 'output')
    const reportPath = path.join(root, 'report.json')
    const config = path.join(root, 'quartz.config.ts')
    const publicIndex = path.join(root, 'contentIndex.json')
    const cacheDir = path.join(root, 'cache')
    const lockPath = path.join(outDir, '.publish.lock')
    const run = (destination = outDir) =>
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
          '--public-index',
          publicIndex,
          '--config',
          config,
          '--report',
          reportPath,
          '--out',
          destination,
          '--cache',
          cacheDir,
          '--shard-size',
          '1',
        ],
        { encoding: 'utf8', timeout: 30_000, env: { ...process.env, HF_HUB_OFFLINE: '1' } },
      )
    const validate = async (directory = outDir) => {
      const bytes = await readFile(path.join(directory, 'manifest.json'), 'utf8')
      const manifest = readImageManifest(JSON.parse(bytes))
      assert.equal(
        await readFile(path.join(directory, manifest.generation, 'manifest.json'), 'utf8'),
        bytes,
      )
      for (const shard of manifest.vectors.shards) {
        const file = path.join(directory, shard.path.slice('/embeddings/images/'.length))
        await validateImageVectorShard(
          Uint8Array.from(await readFile(file)).buffer,
          shard,
          manifest.dims,
        )
      }
      return manifest
    }
    const publish = async (destination = outDir) => {
      const command = run(destination)
      assert.equal(command.status, 0, command.stderr)
      const report: unknown = JSON.parse(await readFile(reportPath, 'utf8'))
      assert.ok(isRecord(report))
      assert.equal(report.status, 'published')
      assert.equal(report.encoded, 0)
      return validate(destination)
    }
    try {
      await mkdir(content)
      await writeFile(config, 'const configuration = { ignorePatterns: [] }')
      await writeFile(publicIndex, JSON.stringify({ note: { title: 'Published note' } }))
      const cache = imageVectorCache(cacheDir, IMAGE_ENCODER_CONTRACT)
      for (const [dimension, name] of ['red', 'blue'].entries()) {
        const pixels = await sharp({
          create: { width: 8, height: 8, channels: 3, background: name },
        })
          .png()
          .toBuffer()
        await writeFile(path.join(content, `${name}.png`), pixels)
        const vector = new Float32Array(768)
        vector[dimension] = 1
        await cache.write(sha256(pixels), vector)
      }
      await writeFile(path.join(content, 'note.md'), '![[red.png]]')
      const first = await publish()
      const initial = await readFile(path.join(outDir, 'manifest.json'), 'utf8')
      assert.deepEqual(first.ids, ['red.png'])

      await writeFile(path.join(content, 'note.md'), '![[red.png]]\n![[blue.png]]')
      const blue = new Float32Array(768)
      blue[1] = 1
      await cache.write(sha256(await readFile(path.join(content, 'blue.png'))), blue)
      const next = await publish(path.join(root, 'probe-output'))
      const blocked = path.join(outDir, next.generation)
      await writeFile(blocked, 'block generation writes')
      assert.notEqual(run().status, 0)
      assert.equal(await readFile(path.join(outDir, 'manifest.json'), 'utf8'), initial)
      await validate()
      await assert.rejects(lstat(lockPath), { code: 'ENOENT' })
      await rm(blocked)

      const owner = 'another image publisher\n'
      await writeFile(lockPath, owner, { flag: 'wx' })
      const competing = run()
      assert.notEqual(competing.status, 0)
      assert.match(competing.stderr, /EEXIST|locked/)
      assert.equal(await readFile(lockPath, 'utf8'), owner)
      assert.equal(await readFile(path.join(outDir, 'manifest.json'), 'utf8'), initial)
      await validate()
      await rm(lockPath)

      const obsolete = path.join(outDir, 'a'.repeat(64))
      await mkdir(obsolete)
      await writeFile(path.join(obsolete, 'partial.f32'), 'abandoned generation')
      await mkdir(path.join(outDir, 'notes'))
      await writeFile(path.join(outDir, 'notes', 'keep.txt'), 'unrelated directory')
      await writeFile(path.join(outDir, 'b'.repeat(64)), 'unrelated file')
      const external = path.join(root, 'external')
      await mkdir(external)
      await writeFile(path.join(external, 'keep.txt'), 'symlink target')
      const linked = path.join(outDir, 'c'.repeat(64))
      await symlink(external, linked, 'dir')
      const second = await publish()
      assert.equal(second.generation, next.generation)
      assert.deepEqual(second.ids, ['blue.png', 'red.png'])
      await assert.rejects(lstat(path.join(outDir, first.generation)), { code: 'ENOENT' })
      await assert.rejects(lstat(obsolete), { code: 'ENOENT' })
      await assert.rejects(lstat(lockPath), { code: 'ENOENT' })
      assert.equal(
        await readFile(path.join(outDir, 'notes', 'keep.txt'), 'utf8'),
        'unrelated directory',
      )
      assert.equal(await readFile(path.join(outDir, 'b'.repeat(64)), 'utf8'), 'unrelated file')
      assert.ok((await lstat(linked)).isSymbolicLink())
      assert.equal(await readFile(path.join(external, 'keep.txt'), 'utf8'), 'symlink target')

      await mkdir(obsolete)
      assert.equal((await publish()).generation, second.generation)
      await assert.rejects(lstat(obsolete), { code: 'ENOENT' })
      await writeFile(path.join(content, 'note.md'), 'No image references remain.')
      const empty = await publish()
      assert.equal(empty.rows, 0)
      assert.equal(empty.vectors.shards.length, 0)
      await assert.rejects(lstat(path.join(outDir, second.generation)), { code: 'ENOENT' })
      assert.equal((await readdir(path.join(cacheDir, cache.fingerprint))).length, 0)
      assert.equal((await readdir(content)).length, 3)
      console.log(
        JSON.stringify({
          retention: {
            first: first.generation,
            second: second.generation,
            empty: empty.generation,
            remaining: await readdir(outDir),
            inferenceRuns: 0,
            failurePreservedActive: true,
            competingWriterRejected: true,
          },
        }),
      )
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  },
)
