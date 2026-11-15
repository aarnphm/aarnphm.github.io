import yaml from 'js-yaml'
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../../', import.meta.url))
const artifacts = path.join(root, 'docs/verification/intake-policy/artifacts')
fs.mkdirSync(artifacts, { recursive: true })
const run = fs.mkdtempSync(path.join(artifacts, 'render-'))
const cli = path.join(root, 'harness/library-intake/intake.mjs')
const cover = 'library/posters/the-odyssey.jpg'
const coverPath = path.join(root, 'content', cover)
const checksum = () => createHash('sha256').update(fs.readFileSync(coverPath)).digest('hex')
const book = {
  id: `Intake policy fixture ${path.basename(run)}`,
  title: `Intake policy fixture ${path.basename(run)}`,
  author: ['Fixture Author'],
  year: 2000,
  date: '2026-11-14',
  description: 'A local renderer verification fixture.',
  category: ['p/fiction'],
  subcategory: ['fiction'],
  tags: ['literature'],
  status: 'to read',
  posters: `[[${cover}]]`,
  external: 'https://publisher.example/book',
  body: 'Read alongside [[library/Heart of Darkness]].',
}

function render(name, overrides) {
  const input = path.join(run, `${name}.json`)
  fs.writeFileSync(input, JSON.stringify({ ...book, ...overrides }, null, 2) + '\n')
  const result = spawnSync(process.execPath, [cli, 'render', input], {
    cwd: root,
    encoding: 'utf8',
  })
  assert.ifError(result.error)
  fs.writeFileSync(path.join(run, `${name}.md`), result.stdout)
  fs.writeFileSync(
    path.join(run, `${name}-result.json`),
    JSON.stringify(
      {
        command: [process.execPath, cli, 'render', input],
        status: result.status,
        stderr: result.stderr,
      },
      null,
      2,
    ) + '\n',
  )
  return result
}

test('render keeps the cover, bibliographic metadata, and useful reading notes', () => {
  const before = checksum()
  const result = render('book-with-cover', {})
  assert.equal(result.status, 0, result.stdout + result.stderr)
  const [, metadata, body] = result.stdout.split('---')
  const saved = yaml.load(metadata)
  assert.equal(saved.posters, book.posters)
  assert.equal(saved.external, book.external)
  assert.deepEqual(saved.author, book.author)
  assert.equal(body.trim(), book.body)
  assert.equal(checksum(), before)
  assert.equal(fs.existsSync(path.join(root, 'content/library', `${book.id}.md`)), false)
})

const rejected = [
  ['source-field', { source: 'https://www.instagram.com/reel/example/' }],
  ['credit-field', { source_rating_by: 'creator' }],
  ['source-line', { body: 'Source: [recommendation](https://example.com/reel), 00:12.' }],
  ['bibliography-line', { body: 'Bibliography: [publisher](https://publisher.example/book).' }],
  ['reel-citation', { body: 'From [a reading reel](https://www.instagram.com/reel/example/).' }],
  ['captured-image', { body: '![[thoughts/images/captured-slide.jpg]]' }],
  ['archived-reel', { body: '[[thoughts/images/instagram/capture.mp4|Archived reel]]' }],
  ['captured-poster', { posters: '[[thoughts/images/captured-slide.jpg]]' }],
]
for (const [name, input] of rejected) {
  test(`render rejects ${name}`, () => {
    const result = render(name, input)
    assert.equal(result.status, 1, result.stdout + result.stderr)
    assert.ok(JSON.parse(result.stdout).errors.length > 0)
  })
}
