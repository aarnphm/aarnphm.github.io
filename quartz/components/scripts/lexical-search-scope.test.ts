import assert from 'node:assert/strict'
import test from 'node:test'
import type { ContentDetails } from '../../plugins'
import { FilePath } from '../../util/path'
import { LexicalSearchClient } from './lexical-search'

function document(fileName: string): ContentDetails {
  return {
    slug: fileName,
    title: 'cat',
    fileName: fileName as FilePath,
    filePath: `content/${fileName}` as FilePath,
    content: 'cat',
    links: [],
    aliases: [],
    tags: [],
    layout: 'default',
  }
}

test('real lexical queries apply the scope before the result limit', async () => {
  const attachments = Object.fromEntries(
    Array.from({ length: 15 }, (_, id) => [`cat-${id}.png`, document(`cat-${id}.png`)]),
  )
  const data = { ...attachments, note: document('cat.md'), pdf: document('cat.pdf') }
  const client = new LexicalSearchClient()
  const spec = { query: 'cat', limit: 2, index: ['title'] as const }
  const notes = await client.search(data, { ...spec, index: [...spec.index], scope: 'notes' })
  assert.deepEqual(notes[0]?.result, [15])
  const images = await client.search(data, { ...spec, index: [...spec.index], scope: 'images' })
  assert.equal(images[0]?.result.length, 2)
  assert.ok(images[0]?.result.every(id => id < 15))
  const pdf = await client.search(data, { ...spec, index: [...spec.index], scope: 'pdf' })
  assert.deepEqual(pdf[0]?.result, [16])
  const all = await client.search(data, { ...spec, index: [...spec.index], scope: 'all' })
  assert.equal(all[0]?.result.length, 2)
})

test('image scope excludes course material before the lexical result limit', async () => {
  const courseImages = Object.fromEntries(
    Array.from({ length: 15 }, (_, id) => [
      `courses/cat-${id}.png`,
      document(`courses/cat-${id}.png`),
    ]),
  )
  const data = { ...courseImages, 'images/cat.png': document('images/cat.png') }
  const client = new LexicalSearchClient()
  const images = await client.search(data, {
    query: 'cat',
    limit: 2,
    index: ['title'],
    scope: 'images',
  })
  assert.deepEqual(images[0]?.result, [15])
  const all = await client.search(data, { query: 'cat', limit: 2, index: ['title'], scope: 'all' })
  assert.deepEqual(all[0]?.result, [0, 1])
})
