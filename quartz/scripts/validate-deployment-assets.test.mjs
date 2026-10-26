import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, truncateSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const script = fileURLToPath(new URL('./validate-deployment-assets.mjs', import.meta.url))
const limit = 25 * 1024 * 1024
const pixel = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=',
  'base64',
)

function fixture(run) {
  const root = mkdtempSync(path.join(os.tmpdir(), 'deployment-assets-'))
  try {
    run(root)
  } finally {
    rmSync(root, { recursive: true })
  }
}

function validate(root) {
  return spawnSync(process.execPath, [script, root], { encoding: 'utf8' })
}

test('accepts a real PNG and leaves input bytes unchanged', () =>
  fixture(root => {
    writeFileSync(path.join(root, 'pixel.png'), pixel)
    const result = validate(root)
    assert.equal(result.status, 0, result.stderr)
    const report = JSON.parse(result.stdout)
    assert.equal(report.images, 1)
    assert.equal(report.largest.bytes, pixel.length)
  }))

test('rejects the exact 25 MiB boundary and larger image files', () =>
  fixture(root => {
    const image = path.join(root, 'oversize.PNG')
    writeFileSync(image, pixel)
    for (const size of [limit, limit + 1]) {
      truncateSync(image, size)
      const result = validate(root)
      assert.equal(result.status, 1)
      assert.match(result.stderr, /oversize.PNG/)
      assert.match(result.stderr, /25 MiB/)
    }
    truncateSync(image, limit - 1)
    assert.equal(validate(root).status, 0)
  }))

test('rejects unresolved image and Hinterland Markdown LFS pointers', () =>
  fixture(root => {
    mkdirSync(path.join(root, 'nested'))
    const pointer = `version https://git-lfs.github.com/spec/v1\noid sha256:${'a'.repeat(64)}\nsize 1234\n`
    for (const name of ['image.webp', 'q.md']) {
      const file = path.join(root, 'nested', name)
      writeFileSync(file, pointer)
      const result = validate(root)
      assert.equal(result.status, 1)
      assert.match(result.stderr, /unresolved Git LFS pointer/)
      assert.ok(result.stderr.includes(name))
      rmSync(file)
    }
  }))

test('fails when a required asset directory is missing', () =>
  fixture(root => {
    const result = validate(path.join(root, 'missing'))
    assert.equal(result.status, 1)
    assert.match(result.stderr, /missing/)
  }))
