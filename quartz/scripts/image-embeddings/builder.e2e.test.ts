import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import sharp from 'sharp'

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
