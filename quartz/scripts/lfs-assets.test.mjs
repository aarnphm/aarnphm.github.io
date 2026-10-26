import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const script = fileURLToPath(new URL('./lfs-assets.mjs', import.meta.url))
const pixel = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=',
  'base64',
)

test('real Git/LFS staging preserves unrelated staged content and audit verifies image bytes', () => {
  const repo = mkdtempSync(path.join(os.tmpdir(), 'lfs-assets-'))
  try {
    const git = (...args) => {
      const result = spawnSync('git', args, { cwd: repo, encoding: 'utf8' })
      assert.equal(result.status, 0, result.stderr)
      return result.stdout
    }
    git('init', '--quiet')
    git('lfs', 'install', '--local', '--skip-smudge')
    mkdirSync(path.join(repo, 'content/thoughts/images'), { recursive: true })
    mkdirSync(path.join(repo, 'content/hinterland/app'), { recursive: true })
    writeFileSync(path.join(repo, 'content/hinterland/app/q.md'), 'hydrated note\n')
    writeFileSync(
      path.join(repo, '.gitattributes'),
      'content/thoughts/images/**/*.png filter=lfs diff=lfs merge=lfs -text\n',
    )
    writeFileSync(path.join(repo, 'content/thoughts/images/pixel.png'), pixel)
    writeFileSync(path.join(repo, 'unrelated.txt'), 'preserve this staged blob\n')
    git('add', '--', 'unrelated.txt')
    const unrelated = git('rev-parse', ':unrelated.txt').trim()
    const run = command =>
      spawnSync(process.execPath, [script, command, '--repo', repo], { encoding: 'utf8' })
    const before = run('audit')
    assert.notEqual(before.status, 0)
    const staged = run('stage-images')
    assert.equal(staged.status, 0, staged.stderr)
    assert.equal(git('rev-parse', ':unrelated.txt').trim(), unrelated)
    assert.match(
      git('show', ':content/thoughts/images/pixel.png'),
      /^version https:\/\/git-lfs.github.com\/spec\/v1\n/,
    )
    assert.deepEqual(readFileSync(path.join(repo, 'content/thoughts/images/pixel.png')), pixel)
    const audit = run('audit')
    assert.equal(audit.status, 0, audit.stderr)
    assert.equal(JSON.parse(audit.stdout).images, 1)
    writeFileSync(path.join(repo, 'content/thoughts/images/pixel.png'), 'changed bytes')
    const changed = run('audit')
    assert.notEqual(changed.status, 0)
    assert.match(changed.stderr, /hash|bytes|differs/)
  } finally {
    rmSync(repo, { recursive: true, force: true })
  }
})

test('unknown commands fail without touching the repository', () => {
  const result = spawnSync(process.execPath, [script, 'rewrite-history'], { encoding: 'utf8' })
  assert.notEqual(result.status, 0)
  assert.match(result.stderr, /usage|command/i)
})
