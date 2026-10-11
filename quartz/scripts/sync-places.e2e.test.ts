import matter from 'gray-matter'
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { test } from 'node:test'

test('places CLI preserves authored data and is byte-stable on repeat runs', async () => {
  const artifacts = path.resolve('docs/verification/arena-places/artifacts')
  await mkdir(artifacts, { recursive: true })
  const root = await mkdtemp(path.join(artifacts, 'sync-'))
  await mkdir(path.join(root, 'content/places'), { recursive: true })
  const source = `## places

- https://example.com/existing -- Existing
  - [meta]:
    - date: 10/11/2026
    - tags: [studio]
    - coord: [43.5, -79.5]
    - address: 1 Example Street
- https://example.com/new -- New
  - [meta]:
    - date: 10/11/2026
    - tags: [workshop]
    - address: 2 Example Street
`
  const file = path.join(root, 'content/places/authored.md')
  const original = matter.stringify('\nAuthored body.\n', {
    title: 'Existing',
    id: 'Existing',
    url: 'https://example.com/existing',
    category: ['[[personal category]]'],
    aliases: ['Keep me'],
    rating: 2,
    modified: '2020-01-01',
    custom: { keep: true },
  })
  await writeFile(file, original)
  await writeFile(path.join(root, 'content/are.na.md'), source)
  const args = [
    path.resolve('node_modules/tsx/dist/cli.mjs'),
    path.resolve('quartz/scripts/sync-places.ts'),
  ]
  const first = execFileSync(process.execPath, args, { cwd: root, encoding: 'utf8' })
  const updated = await readFile(file, 'utf8')
  const parsed = matter(updated)
  assert.deepEqual(parsed.data.aliases, ['Keep me'])
  assert.deepEqual(parsed.data.custom, { keep: true })
  assert.deepEqual(parsed.data.category, ['[[personal category]]', '[[places to go]]'])
  assert.equal(parsed.data.rating, 2)
  assert.equal(parsed.data.address, '1 Example Street')
  assert.equal(parsed.content, matter(original).content)
  assert.deepEqual(parsed.data.coordinates, ['43.5', '-79.5'])
  const newFile = path.join(root, 'content/places/New.md')
  const created = await readFile(newFile, 'utf8')
  assert.equal(matter(created).data.address, '2 Example Street')
  await new Promise(resolve => setTimeout(resolve, 1100))
  const second = execFileSync(process.execPath, args, { cwd: root, encoding: 'utf8' })
  assert.equal(await readFile(file, 'utf8'), updated)
  assert.equal(await readFile(newFile, 'utf8'), created)
  assert.match(second, /created: 0, updated: 0, skipped: 2/)
  await writeFile(
    path.join(root, 'result.json'),
    JSON.stringify(
      { command: [process.execPath, ...args], cwd: root, first, second, passed: true },
      null,
      2,
    ),
  )
})
